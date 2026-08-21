from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from pathlib import Path

import torch
from tqdm import tqdm
import tifffile as tiff
import numpy as np

from Scripts.utils import AXES
from Scripts.NeuralNetworkSegmentation.utils import get_logger
from Scripts.NeuralNetworkSegmentation.models.unet import is_model_2d


logger = get_logger("UNetPredictor")

def save_prediction(input: np.ndarray, pred: np.ndarray, name: str, odirname: Path, target = None) -> None:

    odirname = odirname
    odirname.mkdir(parents=True, exist_ok=True)

    concat = np.concatenate((input, pred), axis=1)

    if target:
        concat = np.concatenate((concat, target), axis=1)

    axes = AXES[-concat.ndim:]

    tiff.imwrite(
        f"{odirname}/{name}",
        concat,
        imagej=True,
        metadata={"mode": "color", "axes": axes},
    )

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
        save_segmentation: bool = False,
        performance_metric: str | None = None,
    ):
        self.model = model
        self.output_dir = output_dir
        assert device in ["cpu", "cuda"], f"Unsupported device: {device}"
        self.device = device
        self.inferer = inferer
        self.save_segmentation = save_segmentation
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
        device: TorchDevice,
        save_segmentation: bool = False,
        performance_metric: str | None = None,
    ):
        super().__init__(
            model,
            output_dir,
            inferer,
            device,
            save_segmentation,
            performance_metric,
        )

    def __call__(self, test_loader: DataLoader) -> Any:
        logger.info(f"Processing '{test_loader.dataset.samples}'...")

        has_mask = self.loader.dataset.has_mask

        # Sets the module in evaluation mode explicitly
        # It is necessary for batchnorm/dropout layers if present as well as final Sigmoid/Softmax to be applied
        self.model.eval()
        logger.info(f"Running inference on {len(test_loader)} batches")
       
        # Run predictions on the entire input dataset
        with torch.no_grad():
            for batch in tqdm(test_loader):
                input = batch[0]

                # send batch to device
                # generally this is an effective choice for enhancing transfer speed,
                # see: https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
                input = input.to(self.device, non_blocking=True)

                if is_model_2d(self.model):
                    # remove the singleton z-dimension from the input
                    input = torch.squeeze(input, dim=-3)
                    # forward pass
                    prediction = self.inferer(input, self.model, return_logits = False)
                    # add the singleton z-dimension to the output
                    prediction = torch.unsqueeze(prediction, dim=-3)
                else:
                    # forward pass
                    prediction = self.inferer(input, self.model, return_logits = False)

                # convert to numpy array
                prediction = prediction.cpu().numpy()

                if self.save_segmentation:
                    # if single channel, binarize
                    pred = pred[0] > 0.5
                    pred = pred.astype("uint16")
                    pred[pred > 0] = 65535
                else:
                    pred = pred.astype("float32")

            if self.performance_metric is not None:
                assert has_mask is True, "No mask was found during the test evaluation"
                gt, tile_name = batch[1], batch[2]

                if isinstance(gt, torch.Tensor):
                    gt = gt.cpu().numpy()
                
                # create metric
                assert self.performance_metric in ["dice", "mean_iou"], (
                    f"Unsupported performance metric: {self.performance_metric}, only dice and mean_iou are supported"
                )
                if self.performance_metric == "dice":
                    score = dice_score(prediction, gt)
                else:
                    score = mean_iou(prediction, gt, n_classes=input.shape[0]) # num classes is usually one

                save_prediction(input, prediction, test_loader.dataset.samples[0].file_name, self.save_dir, gt)

                return score

            elif has_mask is False:
                raw_full, tile_name = batch[1], batch[2]

                if isinstance(raw_full, torch.Tensor):
                    gt = gt.cpu().numpy()

                if self.save_segmentation:
                    gt = gt.astype("uint32")

                    if 65535 in prediction: 
                        has_bacteria = "positive_"

                    else:
                        has_bacteria = "negative_"

                else:
                    gt = gt.astype("float32")
                    has_bacteria = ""

                save_prediction(raw_full, prediction, f"{has_bacteria}{tile_name}.tif", self.save_dir)

                return 


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

    # compute IoU, skip 0: background
    per_class_iou = []
    for c in range(1, n_classes):
        intersection = np.logical_and(gt == c, pred == c).sum()
        union = np.logical_or(gt == c, pred == c).sum()
        iou = intersection / union
        per_class_iou.append(iou)

    if avg:
        return np.mean(per_class_iou)
    return per_class_iou


def dice_score(pred: np.ndarray, gt: np.ndarray, avg: bool = False) -> list[float] | float:
    """
    Compute the Dice score for the given predictions and ground truth.
    If avg is True, return the mean Dice score, otherwise return the Dice score for each channel/class.
    """
    # convert to numpy arrays
    pred = pred.astype("uint16")
    gt = gt.astype("uint16")
    assert pred.shape == gt.shape, f"Predictions and ground truth have different shapes: {pred.shape} != {gt.shape}"
    # compute Dice score
    per_class_dice = []
    for c_pred, c_gt in zip(pred, gt, strict=True):
        intersection = np.logical_and(c_gt, c_pred).sum()
        union = c_gt.sum() + c_pred.sum()
        dice = 2 * intersection / union
        per_class_dice.append(dice)
    if avg:
        return np.mean(per_class_dice)
    return per_class_dice
