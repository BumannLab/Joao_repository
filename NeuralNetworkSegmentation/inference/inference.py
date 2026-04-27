from pathlib import Path
from dataclasses import dataclass

import torch
from tqdm import tqdm
import tifffile as tiff
import numpy as np
from skimage.measure import label, regionprops

from utils import get_images_path, read_image, AXES
from overlap_cropping import OverlapCropper, AxisCrop
from NeuralNetworkSegmentation.models.unet3d import UNet3D
from NeuralNetworkSegmentation.models.unet2d import UNet2D

IMAGE_DIR = "/data/joao/mTBData/20251127/full111/1"
IS_3D = True
CHANNEL_MARKERS = "SYBRGold,CD68AL555,LAMAL647,DAPI"
CHANNELS_TO_USE = [0]
CLUSTER_SIZE_THRESHOLD = 5

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


def layout_3d(image, num_blocks):
    
    image = pad_3d_image_depth(image, num_blocks)

    image = np.transpose(image, (1, 0, 2, 3))

    return image


def get_window_starts(axis_size: int, window_size: int, stride: int) -> list[int]:
    if window_size >= axis_size:
        return [0]

    starts = list(range(0, axis_size - window_size + 1, stride))

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

    return image.unsqueeze(0)


def apply_model(
    image: np.ndarray,
    model: torch.nn.Module,
    num_blocks: int,
    window_size: tuple[int, int, int],
    window_stride: tuple[int, int, int],
    device
) -> np.ndarray:

    _, depth, height, width = image.shape

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

                with torch.no_grad():
                    window_pred = model(process_image(window, device))
                    window_pred = torch.argmax(window_pred, dim=1)
                
                starts = (depth_start, height_start, width_start)
                overlap = tuple(
                    size - step for size, step in zip(window_size, window_stride)
                )    

                crops = compute_window_crop(starts, window_size, (depth, height, width), overlap)
                
                depth_crop, height_crop, width_crop = crops
                
                window_pred =  window_pred[
                    :,
                    depth_crop.local_start:depth_crop.local_stop,
                    height_crop.local_start:height_crop.local_stop,
                    width_crop.local_start:width_crop.local_stop,
                ].cpu().numpy().astype(np.uint16)
                window_pred[window_pred > 0] = 65535

                segmented_image[
                    :,
                    depth_crop.global_start:depth_crop.global_stop,
                    height_crop.global_start:height_crop.global_stop,
                    width_crop.global_start:width_crop.global_stop,
                ] = window_pred

    segmented_image = np.transpose(segmented_image, (1, 0, 2, 3))

    return segmented_image


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


def save_prediction(image_og: np.ndarray, pred: np.ndarray, name: str, num_blocks, odirname: Path) -> None:
    image_og = layout_3d(image_og, num_blocks)
    image_og = np.transpose(image_og, (1, 0, 2, 3))
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
        is_3d,
        channel_markers,
        channels_to_use,
        cluster_size_threshold,
        save_pred,
        torch_model,
        num_blocks,
        odirname,
        device
    ):
    odirname = Path(odirname)
    odirname.mkdir(parents=True, exist_ok=True)


    images_path = get_images_path(image_dir)

    model = UNet3D(in_channels=len(channels_to_use), out_channels=2, channels=32, num_blocks=num_blocks, use_batchnorm=True)
    model.to(device)
    model.load_state_dict(torch.load(torch_model, map_location=device, weights_only=True))
    model.eval()

    mtb_pos_fname = odirname / "mTB_position.tsv"

    with open(mtb_pos_fname, "w") as mtb_pos_out:
        write_header(channel_markers, mtb_pos_out)

        for path in tqdm(images_path):

            image_name = path.stem
            image_og = read_image(path)

            image = load_channels_image(image_og, channels_to_use)

            if is_3d:
                image = layout_3d(image, num_blocks)

            pred = apply_model(image, model, num_blocks, (24, 576, 576), (24, 288, 288), device)

            # predicted_clusters = get_prediction_clusters(pred, cluster_size_threshold)

            # if not predicted_clusters:
            #     no_intensity = "\t".join("-1" for _ in range(image_og.shape[0]))
            #     cluster_info = f"{image_name}\t-1\t-1\t-1\t-1\t{no_intensity}\n"
            #     mtb_pos_out.write(cluster_info)
            # else:
            #     for cluster in predicted_clusters:
            #         cluster_obj = BacteriaCluster(
            #             image_name,
            #             cluster.area,
            #             round(cluster.centroid[0], 1),
            #             round(cluster.centroid[1], 1),
            #             round(cluster.centroid[2], 1),
            #             cluster.coords,
            #         )

            #         cluster_obj.cluster_mean_intensities(image_og)
            #         cluster_info = build_cluster_report(cluster_obj)
            #         mtb_pos_out.write(cluster_info)

            if save_pred:
                save_prediction(image_og, pred, image_name, num_blocks, odirname)

if __name__ == "__main__":
    main(
        IMAGE_DIR,
        IS_3D,
        CHANNEL_MARKERS,
        CHANNELS_TO_USE,
        CLUSTER_SIZE_THRESHOLD,
        SAVE_MASKS,
        TORCH_MODEL_PATH,
        NUM_BLOCKS,
        ODIRNAME,
        DEVICE
    )