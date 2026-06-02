from pathlib import Path
from dataclasses import dataclass

import torch
from tqdm import tqdm
import tifffile as tiff
import numpy as np
from skimage.measure import label, regionprops
from monai.inferers import SlidingWindowInferer

from utils import get_images_path, read_image, AXES
from ZOrganized_scripts.NeuralNetworkSegmentationConfig.models.unet import UNet

IMAGE_DIR = "/data/joao/mTBData/20251127/full111/1"
CHANNEL_MARKERS = "SYBRGold,CD68AL555,LAMAL647,DAPI"
CHANNELS_TO_USE = [0]
CLUSTER_SIZE_THRESHOLD = 5

WINDOW_SIZE = (21, 144, 144)
WINDOW_STRIDE = (21, 72, 72)

TORCH_MODEL_PATH = "/data/joao/Materials/Joao_repository/mTB_segmentation/center_crop_no_translation/model.pt"
NUM_BLOCKS = 3

SAVE_MASKS = True

ODIRNAME = "NeuralNetworkSegmentation/inference/test_inference"

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


@dataclass(frozen=True)
class AxisCrop:
    local_start: int
    local_stop: int
    global_start: int
    global_stop: int


@dataclass
class BacteriaCluster:
    Tile_Name: str
    Area: int
    Z_centroid: float
    Y_centroid: float
    X_centroid: float
    Pixels_coords: np.typing.NDArray
    Mean_Intensities: np.typing.NDArray | None = None

    def get_channel_intensity(self, image: np.ndarray) -> np.ndarray:
        z_indices = self.Pixels_coords[:, 0]
        y_indices = self.Pixels_coords[:, 1]
        x_indices = self.Pixels_coords[:, 2]
        return image[z_indices, :, y_indices, x_indices]

    def cluster_mean_intensities(self, image: np.ndarray) -> None:
        intensities = self.get_channel_intensity(image)
        mean_intensities = np.mean(intensities, axis=1)
        frac = mean_intensities - np.floor(mean_intensities)

        # Match the rounding behavior used in the 2D script.
        mean_intensities = np.where(frac > 0.5, np.ceil(mean_intensities), np.round(mean_intensities, 1))
        self.Mean_Intensities = mean_intensities


def write_header(channel_markers, fname):
    channel_cols = "\t".join(
            f"{marker}_Mean_Intensity" for marker in channel_markers.split(",")
        )
    header = f"Image\tArea\tZ\tY\tX\t{channel_cols}\n"
    fname.write(header)


def load_channels_image(image, channels_to_use):

    image = np.take(image, indices=channels_to_use, axis=-3)

    return image


def pad_3d_image_depth(image: np.ndarray, num_blocks: int) -> np.ndarray:
    """Match the z-depth to the nearest multiple required by the 3D UNet."""
    depth_multiple = 2 ** num_blocks
    depth = image.shape[0]
    remainder = depth % depth_multiple

    if remainder == 0:
        return image

    trim_voxels = remainder
    pad_voxels = depth_multiple - remainder

    if trim_voxels <= pad_voxels:
        trim_top = trim_voxels // 2
        trim_bottom = trim_voxels - trim_top
        return image[trim_top:depth - trim_bottom]

    pad_top = pad_voxels // 2
    pad_bottom = pad_voxels - pad_top
    pad_width = [(0, 0)] * image.ndim
    pad_width[0] = (pad_top, pad_bottom)
    
    return np.pad(image, pad_width, mode="constant", constant_values=0)


def get_window_starts(axis_size: int, window_size: int, stride: int) -> list[int]:
    if window_size >= axis_size:
        return [0]

    starts = list(range(0, axis_size - window_size + 1, stride))

    if starts[-1] + window_size < axis_size:
        starts.append(axis_size - window_size)

    return starts


def compute_axis_crop(axis_start, axis_size, image_size, overlap):
    half_overlap = overlap // 2
    axis_end = axis_start + axis_size

    crop_before = 0 if axis_start == 0 else half_overlap
    crop_after = 0 if axis_end == image_size else half_overlap

    local_start = crop_before
    local_stop = axis_size - crop_after
    global_start = axis_start + crop_before
    global_stop = axis_end - crop_after

    return AxisCrop(
        local_start=local_start,
        local_stop=local_stop,
        global_start=global_start,
        global_stop=global_stop,
    )


def compute_window_crop(starts, window_size, image_shape, overlap):

    crops = []
    for axis_start, axis_size, image_size, overlap in zip(
        starts,
        window_size,
        image_shape,
        overlap,
    ):
        crop = compute_axis_crop(axis_start, axis_size, image_size, overlap)
        crops.append(crop)

    return tuple(crops)


def process_image(image: np.ndarray, device) -> torch.Tensor:
    
    image = torch.from_numpy(image / 65535).to(torch.float32).to(device)
    image = torch.permute(image, (1, 0, 2, 3))

    return image.unsqueeze(0)


def pad_image(image: np.ndarray, num_blocks: int) -> np.ndarray:
    axis_multiple = 2 ** num_blocks
    
    depth, _, height, width = image.shape
    
    depth_remainder = depth % axis_multiple
    height_remainder = height % axis_multiple
    width_remainder = width % axis_multiple

    if all([remainder == 0 for remainder in [depth_remainder, height_remainder, width_remainder]]):
        return image, ((0, 0), (0, 0), (0, 0), (0, 0))

    depth_pad_total = axis_multiple - depth_remainder if depth_remainder != 0 else 0
    depth_pad_back = round(depth_pad_total / 2)
    depth_pad_front = depth_pad_total - depth_pad_back

    height_pad_total = axis_multiple - height_remainder if height_remainder != 0 else 0
    height_pad_top = round(height_pad_total / 2)
    height_pad_bottom = height_pad_total - height_pad_top
    
    width_pad_total = axis_multiple - width_remainder if width_remainder != 0 else 0
    width_pad_left = round(width_pad_total / 2)
    width_pad_right = width_pad_total - width_pad_left 

    padding = (
        (depth_pad_front, depth_pad_back),
        (0, 0),
        (height_pad_top, height_pad_bottom),
        (width_pad_left, width_pad_right)
    )

    padded_image = np.pad(image, padding)

    return padded_image, padding


def unpad_image(image: np.ndarray, padding) -> np.ndarray:

    depth_pad_front, depth_pad_back = padding[0]
    height_pad_top, height_pad_bottom = padding[2]
    width_pad_left, width_pad_right = padding[3]

    depth_start = depth_pad_front
    depth_end = image.shape[0] - depth_pad_back
    
    height_start = height_pad_top
    height_end = image.shape[2] - height_pad_bottom
    
    width_start = width_pad_left
    width_end = image.shape[3] - width_pad_right

    return image[depth_start:depth_end, :, height_start:height_end, width_start:width_end]


def apply_model(
    image: np.ndarray,
    model: torch.nn.Module,
    num_blocks: int,
    window_size: tuple[int, int, int],
    window_stride: tuple[int, int, int],
    device
) -> np.ndarray:

    depth, _, height, width = image.shape

    segmented_image = np.zeros([*image.shape], dtype=np.uint16)

    window_depth, window_height, window_width = window_size
    stride_depth, stride_height, stride_width = window_stride

    depth_starts = get_window_starts(depth, window_depth, stride_depth)
    height_starts = get_window_starts(height, window_height, stride_height)
    width_starts = get_window_starts(width, window_width, stride_width)

    for depth_start in depth_starts:
        depth_end = depth_start + window_depth

        for height_start in height_starts:
            height_end = height_start + window_height

            for width_start in width_starts:
                width_end = width_start + window_width

                window = image[
                    :,
                    depth_start:depth_end,
                    height_start:height_end,
                    width_start:width_end,
                ]

                padded_window, padding = pad_image(window, num_blocks)
                
                with torch.no_grad():
                    window_pred = model(process_image(padded_window, device))
                    window_pred = torch.argmax(window_pred, dim=1)
                    window_pred = torch.permute(window_pred, (1, 0, 2, 3)) # Convert from CxDxHxW into DxCxHxW
                
                unpadded_window_pred = unpad_image(window_pred.cpu().numpy(), padding)

                starts = (depth_start, height_start, width_start)
                overlap = tuple(
                    size - step for size, step in zip(window_size, window_stride)
                )    

                crops = compute_window_crop(starts, window_size, (depth, height, width), overlap)
                
                depth_crop, height_crop, width_crop = crops
                
                unpadded_window_pred =  unpadded_window_pred[
                    :,
                    depth_crop.local_start:depth_crop.local_stop,
                    height_crop.local_start:height_crop.local_stop,
                    width_crop.local_start:width_crop.local_stop,
                ].astype(np.uint16)
                unpadded_window_pred[unpadded_window_pred > 0] = 65535

                segmented_image[
                    :,
                    depth_crop.global_start:depth_crop.global_stop,
                    height_crop.global_start:height_crop.global_stop,
                    width_crop.global_start:width_crop.global_stop,
                ] = unpadded_window_pred
    
    return segmented_image


def apply_model_2(image, model, sliding_window, device):

    with torch.no_grad():
    
        image_tensor = process_image(image, device)
        pred = sliding_window(inputs=image_tensor, predictor=model)
        
        pred = torch.argmax(pred, dim=1)
        pred = torch.permute(pred, (1, 0, 2, 3)) # Convert from CxDxHxW into DxCxHxW

    pred = pred.cpu().numpy().astype(np.uint16)
    pred[pred > 0] = 65535

    return pred


def get_prediction_clusters(pred: np.ndarray, cluster_size_threshold: int):
    pred = pred > 0
    pred = np.squeeze(pred, axis=1)

    pred_label = label(pred, connectivity=pred.ndim)
    predicted_clusters = regionprops(pred_label)

    return [
        cluster
        for cluster in predicted_clusters
        if cluster.area > cluster_size_threshold
    ]


def build_cluster_report(cluster: BacteriaCluster) -> str:
    file_content = "\t".join(
        [
            cluster.Tile_Name,
            str(cluster.Area),
            str(cluster.Z_centroid),
            str(cluster.Y_centroid),
            str(cluster.X_centroid),
            *cluster.Mean_Intensities.astype(str),
        ]
    )

    return file_content + "\n"


def save_prediction(image_og: np.ndarray, pred: np.ndarray, name: str, odirname: Path) -> None:

    odirname = odirname / "pred_masks"
    odirname.mkdir(parents=True, exist_ok=True)

    concat = np.concatenate((image_og, pred), axis=1)
    axes = AXES[-concat.ndim:]

    tiff.imwrite(
        f"{odirname}/{name}.tif",
        concat,
        imagej=True,
        metadata={"mode": "color", "axes": axes},
    )


def main(
        image_dir,
        channel_markers,
        channels_to_use,
        cluster_size_threshold,
        window_size,
        window_stride,
        save_pred,
        torch_model,
        num_blocks,
        odirname,
        device
    ):
    
    odirname = Path(odirname)
    odirname.mkdir(parents=True, exist_ok=True)

    images_path = get_images_path(image_dir)[210:]

    model = UNet(in_channels=len(channels_to_use), out_channels=2, channels=32, num_blocks=num_blocks, use_batchnorm=True, is_3d=True)
    model.to(device)
    model.load_state_dict(torch.load(torch_model, map_location=device, weights_only=True))
    model.eval()

    sliding_window = SlidingWindowInferer(
        roi_size=window_size,
        mode="gaussian",
        overlap=0.5,
    )

    mtb_pos_fname = odirname / "mTB_position.tsv"

    with open(mtb_pos_fname, "w") as mtb_pos_out:
        write_header(channel_markers, mtb_pos_out)

        for path in tqdm(images_path):

            image_name = path.stem
            image_og = read_image(path)

            image = load_channels_image(image_og, channels_to_use)

            # pred = apply_model(image, model, num_blocks, window_size, window_stride, device)

            pred = apply_model_2(image, model, sliding_window, device)

            predicted_clusters = get_prediction_clusters(pred, cluster_size_threshold)

            if not predicted_clusters:
                no_intensity = "\t".join("-1" for _ in range(image_og.shape[0]))
                cluster_info = f"{image_name}\t-1\t-1\t-1\t-1\t{no_intensity}\n"
                mtb_pos_out.write(cluster_info)
            else:
                for cluster in predicted_clusters:
                    cluster_obj = BacteriaCluster(
                        image_name,
                        cluster.area,
                        round(cluster.centroid[0], 1),
                        round(cluster.centroid[1], 1),
                        round(cluster.centroid[2], 1),
                        cluster.coords,
                    )

                    cluster_obj.cluster_mean_intensities(image_og)
                    cluster_info = build_cluster_report(cluster_obj)
                    mtb_pos_out.write(cluster_info)

            if save_pred:
                save_prediction(image_og, pred, image_name, odirname)

if __name__ == "__main__":
    main(
        IMAGE_DIR,
        CHANNEL_MARKERS,
        CHANNELS_TO_USE,
        CLUSTER_SIZE_THRESHOLD,
        WINDOW_SIZE,
        WINDOW_STRIDE,
        SAVE_MASKS,
        TORCH_MODEL_PATH,
        NUM_BLOCKS,
        ODIRNAME,
        DEVICE
    )