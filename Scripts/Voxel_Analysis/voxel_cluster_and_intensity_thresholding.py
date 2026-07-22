import operator as op
from itertools import repeat
from multiprocessing import Pool
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm
from skimage.measure import label, regionprops
import tifffile as tiff

from utils import get_images_path, slice_channel_threshold, read_image
from compute_direction_and_cut_overlap import (
    get_all_tiles_directions_to_remove_overlap,
    compute_overlap_pixels,
    cut_overlap,
)

IMAGE_DIR = "/data/joao/mTBData/B2022.10246/SYBRGold_CD68AL555_LAMAL647_DAPI"
ODIRNAME = "/data/joao/Materials/Joao_repository/Results/B2022.10246/SYBRGold_CD68AL555_LAMAL647_DAPI/SYBRGold_GET800/"

CHANNEL_MARKERS = "SYBRGold,CD68AL555,LAMAL647,DAPI"
CHANNEL_THRESHOLD_CONDITIONS = "SYBRGold>=800"
MEAN_CLUSTER_THRESHOLD_CONDITIONS = "DAPI<=250,SYBRGold>=1000"

CLUSTER_SIZE_THRESHOLD = 5

SAVE_TILE_WITH_MASK = False
MIP_ON_SAVE = False

IMAGE_OVERLAP = 5

NUM_PROCESSERS = 15

COMPARISON_OPERATORS = {">": op.gt, ">=": op.ge, "<": op.lt, "<=": op.le}


@dataclass
class Cluster_3D:
    Tile_Name: str
    Volume: int
    X_centroid: int
    Y_centroid: int
    Z_centroid: int
    Pixels_coords: np.typing.NDArray
    Mean_Intensities: np.typing.NDArray = None

    def correct_positions(self, pixels_to_remove_w, pixels_to_remove_h, directions):

        for direction in directions:
            if direction == "top":

                self.Y_centroid += pixels_to_remove_h
                self.Pixels_coords[:, -2] += pixels_to_remove_h

            elif direction == "right":

                pass

            elif direction == "left":

                self.X_centroid += pixels_to_remove_w
                self.Pixels_coords[:, -1] += pixels_to_remove_w

    def get_channel_intensity(self, image):

        x_indixes = self.Pixels_coords[:, -1]
        y_indixes = self.Pixels_coords[:, -2]
        z_indexes = self.Pixels_coords[:, -3]

        pixel_intensities = image[z_indexes, :, y_indixes, x_indixes]

        return pixel_intensities

    def cluster_mean_intensities(self, image):

        intensities = self.get_channel_intensity(image)

        mean_intensities = np.mean(intensities, axis=0)

        frac = mean_intensities - np.floor(mean_intensities)

        # Similar behaviour to round python function
        mean_intensities = np.where(frac > 0.5, np.ceil(mean_intensities), np.round(mean_intensities, 1))

        self.Mean_Intensities = mean_intensities


def build_out_file_sufix(threshold_conditions, prefix = None):

    if ">=" in threshold_conditions:
        threshold_conditions = threshold_conditions.replace(">=", "_get_")
    if "<=" in threshold_conditions:
        threshold_conditions = threshold_conditions.replace("<=", "_let_")
    if ">" in threshold_conditions:
        threshold_conditions = threshold_conditions.replace(">", "_gt_")
    if "<" in threshold_conditions:
        threshold_conditions = threshold_conditions.replace("<", "_lt_")

    if prefix is not None:
        threshold_conditions = prefix + threshold_conditions.replace(",", prefix)
    
    return threshold_conditions


def voxel_channel_intensity_threshold(image, channel, operator, threshold):

    return COMPARISON_OPERATORS[operator](image[:, channel], threshold)


def multi_channel_threshold_mask(image, voxel_channel_thresholds):
    image_masks = []

    for channel_threshold in voxel_channel_thresholds:

        channel, operator, threshold = channel_threshold

        channel_mask = voxel_channel_intensity_threshold(
            image, channel, operator, threshold
        )

        image_masks.append(channel_mask)

    final_mask = np.logical_and.reduce(image_masks)

    return final_mask


def filter_mean_threshold_clusters(cluster_objects, mean_cluster_thresholds):

    if not mean_cluster_thresholds:
        return cluster_objects

    filtered_clusters = [
        cluster
        for cluster in cluster_objects
        if all(
            COMPARISON_OPERATORS[operator](cluster.Mean_Intensities[channel], threshold)
            for channel, operator, threshold in mean_cluster_thresholds
        )
    ]
    return filtered_clusters


def filter_size_clusters(cluster_objects, cluster_size_threshold: int = 5):
    filtered_clusters = [
        cluster
        for cluster in cluster_objects
        if cluster.Volume > cluster_size_threshold
    ]
    return filtered_clusters


def concatenate_image_mask(image, threshold_mask):
    
    if threshold_mask.ndim == 3:
        threshold_mask = threshold_mask[:, np.newaxis, ...]

    threshold_mask = np.where(threshold_mask, 65535.0, 0.0)

    image = np.concatenate((image, threshold_mask), axis=1)

    return image


def write_matrix_tiff(name, image, threshold_mask, odirname, do_max_proj):

    image = concatenate_image_mask(image, threshold_mask)

    axes = "ZCYX"
    img_folder = "Masked_tiles"

    if do_max_proj:
        image  = np.max(image, axis=0)
        axes = axes[1:]
        img_folder = img_folder + "_max_projection"

    odirname = f"{odirname}/{img_folder}"

    odirname = Path(odirname)
    odirname.mkdir(parents=True, exist_ok=True)
    
    tiff.imwrite(
        f"{odirname}/{name}.tif",
        image.astype(np.uint16),
        imagej=True,
        metadata={"mode": "color", "axes": axes},
        )


def build_file_report(cluster_objects):
    file_content = list()
    
    for cluster in cluster_objects:

        content = "\t".join(
            [
                cluster.Tile_Name,
                str(cluster.Volume),
                str(cluster.X_centroid),
                str(cluster.Y_centroid),
                str(cluster.Z_centroid),
                *(str(intensity) for intensity in cluster.Mean_Intensities),
            ]
        )

        file_content.append(content)

    return "\n".join(file_content) + "\n"


def build_cluster_objects(
    all_clusters_props, img_name, image, pixels_to_remove_w, pixels_to_remove_h, directions_remove_overlap
):
    cluster_objects = list()

    for cluster in all_clusters_props:

        cluster_obj = Cluster_3D(
            img_name,
            cluster.area,
            round(cluster.centroid[-1], 1),
            round(cluster.centroid[-2], 1),
            round(cluster.centroid[-3], 1),
            cluster.coords
        )

        cluster_obj.correct_positions(
            pixels_to_remove_w, pixels_to_remove_h, directions_remove_overlap
        )

        cluster_obj.cluster_mean_intensities(image)
        cluster_objects.append(cluster_obj)

    return cluster_objects


def build_voxel_clusters(args):

    (
        image_path,
        num_channels,
        directions_remove_overlap,
        image_overlap,
        channel_threshold_conditions,
        cluster_mean_thresholds_conditions,
        cluster_size_threshold,
        save_tile_with_mask,
        mip_on_save,
        odirname,
    ) = args

    img_name = image_path.stem
    
    image = read_image(image_path)

    if image is None:
        print("As a result this image will be classified as having no clusters.")
        no_intensity = "\t".join("-1" for _ in range(num_channels))
        return f"{img_name}\t-1\t-1\t-1\t-1\t{no_intensity}\n"

    if image.ndim == 3: #CHW -> DCHW
        image = image[np.newaxis]

    img_height, img_width = image.shape[-2:]

    pixels_to_remove_h, pixels_to_remove_w = compute_overlap_pixels(
        image_overlap, img_height, img_width, rounding_mode="ceil"
    )

    final_thresholded_mask = multi_channel_threshold_mask(
        image, channel_threshold_conditions
    )

    mask_label = label(final_thresholded_mask, connectivity=3)

    mask_label_no_overlap = cut_overlap(
        mask_label,
        pixels_to_remove_h,
        pixels_to_remove_w,
        directions_remove_overlap,
    )

    all_clusters_props = regionprops(mask_label_no_overlap)

    cluster_objects = build_cluster_objects(
        all_clusters_props, img_name, image, pixels_to_remove_w, pixels_to_remove_h, directions_remove_overlap
    )

    cluster_objects = filter_mean_threshold_clusters(
        cluster_objects, cluster_mean_thresholds_conditions
    )

    cluster_objects = filter_size_clusters(cluster_objects, cluster_size_threshold)

    if not cluster_objects:
        no_intensity = "\t".join("-1" for _ in range(num_channels))
        return f"{img_name}\t-1\t-1\t-1\t-1\t{no_intensity}\n"

    voxel_cluster_info = build_file_report(cluster_objects)
    
    if save_tile_with_mask:
        write_matrix_tiff(img_name, image, mask_label, odirname, mip_on_save)

    return voxel_cluster_info


def main(
    image_dir,
    channel_threshold_conditions,
    image_overlap,
    odirname,
    channel_markers = None,
    save_tile_with_mask = True,
    mip_on_save = False,
    mean_cluster_threshold_conditions = None,
    cluster_size_threshold = None,
    num_processers: int = 1,
):

    odirname = Path(odirname)
    odirname.mkdir(parents=True, exist_ok=True)

    images_path = get_images_path(image_dir, ".nd2", ".tif", ".tiff")

    all_directions_remove_overlap = get_all_tiles_directions_to_remove_overlap(
        images_path
    )

    channel_markers = channel_markers.split(",")

    sliced_channel_thresholds = slice_channel_threshold(
        channel_threshold_conditions, channel_markers
    )

    renamed_channel_threshold_conditions = build_out_file_sufix(channel_threshold_conditions, "_")

    voxel_cluster_report_fname = f"{odirname}/voxel_clusters_size_over{cluster_size_threshold}{renamed_channel_threshold_conditions}"

    sliced_cluster_mean_thresholds = []

    if mean_cluster_threshold_conditions:
        sliced_cluster_mean_thresholds = slice_channel_threshold(
                mean_cluster_threshold_conditions, channel_markers
        )

        renamed_mean_cluster_threshold_condition = build_out_file_sufix(mean_cluster_threshold_conditions, "_mean_")

        voxel_cluster_report_fname += renamed_mean_cluster_threshold_condition

    voxel_cluster_report_fname += ".tsv"

    with open(voxel_cluster_report_fname, "w") as voxel_cluster_out:
        channel_cols = "\t".join(
            f"{marker}_Mean_Intensity" for marker in channel_markers
        )
        header = f"Image\tVolume\tX\tY\tZ\t{channel_cols}\n"
        voxel_cluster_out.write(header)

        with Pool(processes=num_processers, maxtasksperchild=1) as pool:
            
            args_iter = zip(
                images_path,
                repeat(len(channel_markers)),
                all_directions_remove_overlap,
                repeat(image_overlap),
                repeat(sliced_channel_thresholds),
                repeat(sliced_cluster_mean_thresholds),
                repeat(cluster_size_threshold),
                repeat(save_tile_with_mask),
                repeat(mip_on_save),
                repeat(odirname),
            )

            for voxel_cluster_info in tqdm(
                pool.imap(build_voxel_clusters, args_iter, chunksize=1),
                total=len(images_path),
            ):

                voxel_cluster_out.write(voxel_cluster_info)
                

if __name__ == "__main__":

    main(
        image_dir=IMAGE_DIR,
        channel_markers=CHANNEL_MARKERS,
        channel_threshold_conditions=CHANNEL_THRESHOLD_CONDITIONS,
        mean_cluster_threshold_conditions = MEAN_CLUSTER_THRESHOLD_CONDITIONS,
        cluster_size_threshold = CLUSTER_SIZE_THRESHOLD,
        save_tile_with_mask = SAVE_TILE_WITH_MASK,
        mip_on_save = MIP_ON_SAVE,
        image_overlap = IMAGE_OVERLAP,
        odirname = ODIRNAME,
        num_processers = NUM_PROCESSERS,
    )
