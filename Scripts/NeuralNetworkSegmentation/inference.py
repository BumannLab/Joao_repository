from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
import tifffile as tiff
import torch
import torch.nn.functional as F
from monai.inferers import Inferer, SlidingWindowInferer
from monai.inferers.utils import dense_patch_slices
from monai.utils import PytorchPadMode, ensure_tuple_rep, fall_back_tuple
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.measure import label, regionprops

from Scripts.utils import AXES
from Scripts.NeuralNetworkSegmentation.utils import get_logger
from Scripts.NeuralNetworkSegmentation.models.unet import is_model_2d
from compute_direction_and_cut_overlap import (
    compute_overlap_pixels,
    cut_overlap,
)
from Scripts.Voxel_Analysis.voxel_cluster_and_intensity_thresholding import (
    build_cluster_objects,
    build_cluster_mask,
    build_file_report
)

FIRST_TIME_OPENING_TSV = True

logger = get_logger("Predictor")


class SlidingWindowHaloInferer(SlidingWindowInferer):
    """Sliding window inferer that runs each patch with extra context, then discards it.

    Every core patch of size ``roi_size`` is enlarged by ``halo_shape`` on every side
    before being sent to the network. Once the network returns a prediction for the
    enlarged patch, the halo is cropped back off and only the core region is written
    into the output. Core patches tile the image edge-to-edge with no overlap, so
    stitching is a direct placement rather than a blend.

    Args:
        roi_size: spatial size of the core patch (excluding halo).
        halo_shape: extra context added on every side of each patch before inference,
            and cropped off afterwards. Int or per-dimension sequence.
        sw_batch_size: number of windows to run through the network at once.
        padding_mode: how to fill the halo at the volume's true outer border, where no
            real neighboring data exists. Interior patches always get real neighboring
            pixels regardless of this setting; it only affects the outermost edge.
            Defaults to ``"reflect"`` (mirror padding).
        cval: fill value used when ``padding_mode`` is ``"constant"``.
        sw_device: device the network runs on. Defaults to the input's device.
        device: device the stitched output is assembled on. Defaults to the input's device.
        progress: whether to show a tqdm progress bar over the windows.
    """

    def __init__(
        self,
        roi_size: Sequence[int] | int,
        halo_shape: Sequence[int] | int,
        sw_batch_size: int = 1,
        padding_mode: PytorchPadMode | str = PytorchPadMode.REFLECT,
        cval: float = 0.0,
        sw_device: torch.device | str | None = None,
        device: torch.device | str | None = None,
        progress: bool = False,
    ) -> None:
        super().__init__(
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            padding_mode=padding_mode,
            cval=cval,
            sw_device=sw_device,
            device=device,
            progress=progress,
        )
        self.halo_shape = halo_shape

    def __call__(
        self,
        inputs: torch.Tensor,
        network: Callable[..., torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        num_spatial_dims = inputs.ndim - 2
        image_size = tuple(inputs.shape[2:])
        roi_size = fall_back_tuple(self.roi_size, image_size)
        halo_shape = ensure_tuple_rep(self.halo_shape, num_spatial_dims)

        device = kwargs.pop("device", self.device) or inputs.device
        sw_device = kwargs.pop("sw_device", self.sw_device) or inputs.device

        # Pad once up front so edge patches get real (not synthetic-zero) halo context.
        pad = []
        for h in reversed(halo_shape):
            pad.extend([h, h])
        padded_inputs = F.pad(inputs, pad, mode=PytorchPadMode(self.padding_mode).value, value=self.cval)

        # scan_interval == roi_size -> core tiles are adjacent and never overlap
        core_slices = dense_patch_slices(image_size, roi_size, roi_size)

        batch_size = inputs.shape[0]
        windows_range = range(0, len(core_slices), self.sw_batch_size)
        output: torch.Tensor | None = None

        for b in range(batch_size):
            for start in tqdm(windows_range) if self.progress else windows_range:
                batch_slices = core_slices[start : start + self.sw_batch_size]

                halo_windows = []
                for core_slice in batch_slices:
                    halo_slice = tuple(slice(s.start, s.stop + 2 * h) for s, h in zip(core_slice, halo_shape))
                    halo_windows.append(padded_inputs[(slice(b, b + 1), slice(None)) + halo_slice])

                window_batch = torch.cat(halo_windows).to(sw_device)
                pred_batch = network(window_batch, *args, **kwargs)

                if pred_batch.shape[2:] != window_batch.shape[2:]:
                    raise ValueError(
                        "SlidingWindowHalo requires a shape-preserving network: got input patch "
                        f"spatial shape {tuple(window_batch.shape[2:])} but output spatial shape "
                        f"{tuple(pred_batch.shape[2:])}."
                    )

                if output is None:
                    output = torch.zeros(
                        (batch_size, pred_batch.shape[1], *image_size), dtype=pred_batch.dtype, device=device
                    )

                for i, core_slice in enumerate(batch_slices):
                    core_shape = tuple(s.stop - s.start for s in core_slice)
                    crop = tuple(slice(h, h + c) for h, c in zip(halo_shape, core_shape))
                    cropped = pred_batch[(slice(i, i + 1), slice(None)) + crop]
                    output[(slice(b, b + 1), slice(None)) + core_slice] = cropped.to(device)

        return output


class AbstractPredictor:
    """Base class for predictors.

    Args:
        model: Segmentation model.
        output_dir: Directory where the predictions will be saved.
        inferer: Built from monai inferers, requires:
            - window_shape
            - window_stride
            - batch_size to run window slices
            - device
            - halo_shape, optional
        device: Device to run the model on (CPU, CUDA, MPS).
        save_segmentation: If True the segmentation will be saved instead of the probability maps. Default: False.
        performance_metric: Performance metric to be used for evaluation. Default: None.
        loader
        """

    def __init__(
        self,
        model: nn.Module,
        output_dir: str,
        inferer: Inferer, # Add something to relate to monai inferer
        device: str,
        out_channels: int,
        overlap: int = 0, # Integer that spans from 0 to 100
        save_segmentation: bool = False,
        save_tsv: bool = False,
        performance_metric: str | None = None,
    ):
        self.model = model
        self.output_dir = output_dir
        assert device in ["cpu", "cuda"], f"Unsupported device: {device}"
        self.device = device
        self.out_channels = out_channels
        self.inferer = inferer
        self.overlap = overlap
        self.save_segmentation = save_segmentation
        self.save_tsv = save_tsv
        self.performance_metric = performance_metric

    def __call__(self, test_loader: DataLoader) -> Any:
        """Run the model prediction on the test_loader and save the results in the output_dir.

        If the performance_metric is provided, the predictions will be evaluated against the ground truth
        and returned as a tensor.

        Args:
            test_loader: DataLoader for the test dataset.

        Returns:
            Evaluation metric result if performance_metric is provided, otherwise None.
        """
        raise NotImplementedError


class StandardPredictor(AbstractPredictor):
    """Standard predictor that applies the model on the given dataset and saves the result as Tiff file.

    Predictions from the network are kept in memory.
    """

    def __init__(
        self,
        model: nn.Module,
        output_dir: str,
        inferer: Inferer,
        device: torch.device | str,
        out_channels: int,
        overlap: int = 0, 
        save_segmentation: bool = False,
        save_tsv: bool = False,
        performance_metric: str | None = None,
    ):
        super().__init__(
            model,
            output_dir,
            inferer,
            device,
            out_channels,
            overlap,
            save_segmentation,
            save_tsv,
            performance_metric,
        )

    def __call__(self, test_loader: DataLoader) -> Any:
        logger.info(f"Processing '{test_loader.dataset.samples}'...")

        has_mask = test_loader.dataset.has_mask

        # Sets the module in evaluation mode explicitly
        # It is necessary for batchnorm/dropout layers if present as well as final Sigmoid/Softmax to be applied
        self.model.eval()
        logger.info(f"Running inference on {len(test_loader)} batches")
       
        # Run predictions on the entire input dataset
        with torch.no_grad():
            for batch in tqdm(test_loader):
                input = batch[0] # Retains Batch dimension which is necessary for the monai inferer

                # send batch to device
                # generally this is an effective choice for enhancing transfer speed,
                # see: https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
                input = input.to(self.device, non_blocking=True)

                if is_model_2d(self.model):
                    # remove the singleton z-dimension from the input
                    input = torch.squeeze(input, dim=-3)

                    # forward pass
                    prediction = self.inferer(input, self.model, return_logits = False)

                    # # add the singleton z-dimension to the output and input
                    prediction = torch.unsqueeze(prediction, dim=-3)
                    input = torch.unsqueeze(input, dim=-3)
                else:
                    # forward pass
                    prediction = self.inferer(input, self.model, return_logits = False)

                # convert to numpy array
                prediction = prediction.cpu().numpy()
                prediction = prediction[0]

                if self.save_segmentation:

                    if prediction.shape[0] == 1:
                        # if single channel, binarize
                        prediction = prediction > 0.5    
                    
                    else:
                        # use the argmax of the prediction
                        prediction = np.argmax(prediction, axis=0, keepdims=True)

                    prediction = prediction.astype("uint16")
                
                else:
                    prediction = prediction.astype("float32")

                input = input.cpu().numpy()
                input = np.squeeze(input, axis=0)

                if self.performance_metric is not None:
                    
                    assert has_mask is True, "No mask was found during the test evaluation"
                    
                    gt, tile_name = batch[1][0], batch[2][0]

                    if isinstance(gt, torch.Tensor):
                        gt = gt.cpu().numpy()               
                    
                    # create metric
                    assert self.performance_metric in ["dice", "mean_iou"], (
                        f"Unsupported performance metric: {self.performance_metric}, only dice and mean_iou are supported"
                    )

                    if self.performance_metric == "dice":
                        score = dice_score(prediction, gt, n_classes=max(self.out_channels, 2))
                        
                    else:
                        score = mean_iou(prediction, gt, n_classes=max(self.out_channels, 2)) # binary model: 1 output channel = 2 classes (bg + fg)

                    input = (input * 65535).astype(np.uint16)

                    save_prediction(input, prediction, test_loader.dataset.samples[0].file_name, self.output_dir, gt)
                    
                    if self.save_tsv:
                        num_channels = input.shape[1]
                        build_connected_components_clusters(prediction, input, tile_name, self.output_dir, num_channels, "", 0)

                    return score

                elif has_mask is False:
                    raw_full, tile_name, direction_remove_overlap = batch[1][0], batch[2][0], batch[3][0]

                    if self.save_segmentation:
                        raw_full = raw_full.astype("uint16")

                        if 1 in prediction: 
                            has_bacteria = "positive_"

                        else:
                            has_bacteria = "negative_"

                        save_prediction(raw_full, prediction, f"{has_bacteria}{tile_name}.tif", self.output_dir)

                    else:
                        raw_full = raw_full.astype("float32")
                        raw_full = raw_full / 65535
                        has_bacteria = ""

                        save_prediction(raw_full, prediction, f"{has_bacteria}{tile_name}.tif", self.output_dir)

                    if self.save_tsv:
                        num_channels = raw_full.shape[0]
                        build_connected_components_clusters(prediction, raw_full, tile_name, self.output_dir, num_channels, direction_remove_overlap, self.overlap)

                    return 


def save_prediction(input: np.ndarray, pred: np.ndarray, name: str, odirname: Path, target = None) -> None:

    odirname = Path(odirname)

    odirname.mkdir(parents=True, exist_ok=True)

    concat = np.concatenate((input, pred), axis=0)

    if target is not None:
        target = target.astype(np.uint16)
        concat = np.concatenate((concat, target), axis=0)

    axes = AXES[-concat.ndim:]

    concat = np.transpose(concat, axes = (1, 0, 2, 3))

    tiff.imwrite(
        f"{odirname}/{name}",
        concat,
        imagej=True,
        metadata={"mode": "color", "axes": axes},
    )


def mean_iou(pred: np.ndarray, gt: np.ndarray, n_classes: int, avg: bool = False) -> list[float] | float:
    """
    Compute the mean Intersection over Union (IoU) for the given predictions and ground truth.
    Args:
        pred: predicted segmentation
        gt: ground truth segmentation
        n_classes: number of classes
        avg: if True, return the mean IoU, otherwise return the IoU for each class
    Returns:
        mean IoU
    """
    # convert to numpy arrays
    pred = pred.astype("uint16")
    gt = gt.astype("uint16")
    assert pred.shape == gt.shape, f"Predictions and ground truth have different shapes: {pred.shape} != {gt.shape}"

    # compute IoU per class, background (0) included
    per_class_iou = []
    for c in range(0, n_classes):
        intersection = np.logical_and(gt == c, pred == c).sum()
        union = np.logical_or(gt == c, pred == c).sum()
        # class absent from both gt and prediction (e.g. a correct negative tile): IoU is undefined,
        # use NaN so it is skipped by np.nanmean instead of rewarding or penalizing it
        iou = np.nan if union == 0 else intersection / union
        per_class_iou.append(iou)

    if avg:
        return np.nanmean(per_class_iou)
    
    return per_class_iou


def dice_score(pred: np.ndarray, gt: np.ndarray, n_classes: int, avg: bool = False) -> list[float] | float:
    """
    Compute the Dice score for the given predictions and ground truth (label maps of shape 1,D,H,W).
    The label maps are one-hot encoded into n_classes channels (background included) so the Dice
    is computed per class.
    If avg is True, return the mean Dice score, otherwise return the Dice score for each channel/class.
    """
    assert pred.shape == gt.shape, f"Predictions and ground truth have different shapes: {pred.shape} != {gt.shape}"
    # one-hot encode: 1,D,H,W label map -> C,D,H,W boolean masks, one channel per class
    classes = np.arange(n_classes).reshape(-1, 1, 1, 1)
    pred = pred[0] == classes
    gt = gt[0] == classes
    # compute Dice score
    per_class_dice = []
    for c_pred, c_gt in zip(pred, gt, strict=True):
        intersection = np.logical_and(c_gt, c_pred).sum()
        union = c_gt.sum() + c_pred.sum()
        # class absent from both gt and prediction (e.g. a correct negative tile): Dice is undefined,
        # use NaN so it is skipped by np.nanmean instead of rewarding or penalizing it
        dice = np.nan if union == 0 else 2 * intersection / union
        per_class_dice.append(dice)
    if avg:
        return np.nanmean(per_class_dice)
    return per_class_dice


def build_connected_components_clusters(prediction, raw_input, tile_name, output_dir, n_channels, directions_remove_overlap, image_overlap):
    global FIRST_TIME_OPENING_TSV

    prediction = np.transpose(prediction, axes=(1, 0, 2, 3))
    raw_input = np.transpose(raw_input, axes=(1, 0, 2, 3))

    prediction_depth, _, prediction_height, prediction_width = prediction.shape

    final_prediction_mask = np.zeros((prediction_depth, prediction_height, prediction_width), dtype=np.uint16)

    pixels_to_remove_h, pixels_to_remove_w = compute_overlap_pixels(
        image_overlap, prediction_height, prediction_width, rounding_mode="ceil"
    )

    prediction_label = label(np.squeeze(prediction, axis=1), connectivity=3)

    prediction_label_no_overlap = cut_overlap(
        prediction_label,
        pixels_to_remove_h,
        pixels_to_remove_w,
        directions_remove_overlap,
    )

    all_clusters_props = regionprops(prediction_label_no_overlap)

    cluster_objects = build_cluster_objects(
        all_clusters_props, tile_name, raw_input, pixels_to_remove_w, pixels_to_remove_h, directions_remove_overlap
    )

    if not cluster_objects:

        no_intensity = "\t".join("-1" for _ in range(n_channels))

        voxel_cluster_info = f"{tile_name}\t-1\t-1\t-1\t-1\t{no_intensity}\n"

    else:
        final_prediction_mask = build_cluster_mask(final_prediction_mask, cluster_objects)

        voxel_cluster_info = build_file_report(cluster_objects)

    # first write of this run truncates any report left over from a previous run, later tiles append
    open_mode = "w" if FIRST_TIME_OPENING_TSV else "a"

    with open(f"{output_dir}/voxel_clusters_report.tsv", open_mode) as f:

        if FIRST_TIME_OPENING_TSV == True:
            channel_cols = "\t".join(
            f"Channel_{i + 1}_Mean_Intensity" for i in range(n_channels)
            )
            header = f"Image\tVolume\tX\tY\tZ\t{channel_cols}\n"
            f.write(header)
            FIRST_TIME_OPENING_TSV = False

        f.write(voxel_cluster_info)

    return voxel_cluster_info


