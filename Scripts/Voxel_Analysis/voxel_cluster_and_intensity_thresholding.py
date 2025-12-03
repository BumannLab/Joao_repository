from itertools import repeat
from multiprocessing import Pool
from dataclasses import dataclass
from pathlib import Path

import nd2
import numpy as np
from tqdm import tqdm
from skimage.measure import label, regionprops

from utils import slice_channel_threshold, get_images_path
from compute_direction_and_cut_overlap import get_all_tiles_directions_to_remove_overlap, compute_overlap_pixels, cut_overlap


Z_STACK_IMAGE_DIR = "M:/mTB Data/4CHANNELsTAINING_JoaoSamples" # "Joao_repository/Data/img_mTB_zstack_voxel_analysis"
ODIRNAME = "Joao_repository/New_Results/CD68AL555_SYBRGold_DAPI_CD68AL647/Voxel_Analysis/"

CHANNEL_MARKERS = "CD68AL555,SYBRGold,DAPI,CD68AL647"
CHANNEL_THRESHOLD_CONDITIONS = "2>=1000" # Maybe change this to be SYBRGold >= 1000
CLUSTER_SIZE_THRESHOLD = 5

IMAGE_OVERLAP = 5
NUM_PROCESSERS = 1


@dataclass
class Tile:
    Name: str
    Matrix: np.typing.NDArray[np.uint16]


@dataclass
class Cluster_3D:
    Tile_Info: Tile
    Volume: int
    X_centroid: int
    Y_centroid: int
    Z_centroid: int
    Pixels_coords: np.typing.NDArray[np.int64]


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

        return


    def get_channel_intensity(self):

        x_indixes = self.Pixels_coords[:, -1]
        y_indixes = self.Pixels_coords[:, -2]
        z_indexes = self.Pixels_coords[:, -3]

        pixel_intensities = self.Tile_Info.Matrix[z_indexes, :, y_indixes, x_indixes]

        return pixel_intensities
    

    def obtain_cluster_summed_intensities(self):

        intensities = self.get_channel_intensity()

        sum_intensities = np.sum(intensities, axis=0)

        return sum_intensities.astype(str)
    

def build_out_file_sufix(threshold_conditions):

    if ">=" in threshold_conditions:
        threshold_conditions = threshold_conditions.replace(">=", "_greater_equal_")
    if "<=" in threshold_conditions:
        threshold_conditions = threshold_conditions.replace("<=", "_less_equal_")
    if ">" in threshold_conditions:
        threshold_conditions = threshold_conditions.replace(">", "_greater_")
    if "<" in threshold_conditions:
        threshold_conditions = threshold_conditions.replace("<", "_less_")

    threshold_conditions = "channel" + threshold_conditions.replace(",", "_channel")

    return threshold_conditions


def voxel_channel_intensity_threshold(image, channel, operator, threshold):

    if operator == ">":
        voxel_coi_intensity_mask = image[:, channel] > threshold
    elif operator == ">=":
        voxel_coi_intensity_mask = image[:, channel] >= threshold
    elif operator == "<":
        voxel_coi_intensity_mask = image[:, channel] < threshold
    elif operator == "<=":
        voxel_coi_intensity_mask = image[:, channel] <= threshold

    return voxel_coi_intensity_mask


def multi_channel_threshold_mask(image, thresholds):
    image_masks = []

    for channel_threshold in thresholds:

        channel, operator, threshold = channel_threshold

        channel_mask = voxel_channel_intensity_threshold(
            image,
            channel,
            operator,
            threshold 
        )

        image_masks.append(channel_mask)

    final_mask = np.logical_and.reduce(image_masks)

    return final_mask


def find_voxel_clusters(intensity_thresholded_mask):

    mask_label = label(intensity_thresholded_mask, connectivity = 3)
    clusters_properties = regionprops(mask_label)

    return clusters_properties


def filter_size_clusters(all_clusters_props, cluster_size_threshold: int = 5):
    filtered_clusters = [
        cluster for cluster in all_clusters_props if cluster.area > cluster_size_threshold
    ]
    return filtered_clusters


def build_file_report(cluster_objects):
    file_content = list()

    for cluster in cluster_objects:

        content = "\t".join([
            cluster.Tile_Info.Name,
            str(cluster.Volume),
            str(cluster.X_centroid), 
            str(cluster.Y_centroid),
            str(cluster.Z_centroid),
            *cluster.obtain_cluster_summed_intensities(),
        ])

        file_content.append(content)
        
    return "\n".join(file_content) + "\n"


def build_voxel_clusters(args):

    image_path, directions_remove_overlap, image_overlap, channel_threshold_conditions, cluster_size_threshold = args

    img_name = image_path.split("/")[-1].split(".")[0]

    image = nd2.imread(image_path)

    tile = Tile(img_name, image)

    img_height, img_width = image.shape[-2:]

    pixels_to_remove_h, pixels_to_remove_w = compute_overlap_pixels(image_overlap, img_height, img_width)

    image = cut_overlap(image, pixels_to_remove_h, pixels_to_remove_w, directions_remove_overlap)

    final_thresholded_mask = multi_channel_threshold_mask(image, channel_threshold_conditions)

    all_clusters_props = find_voxel_clusters(final_thresholded_mask)

    filtered_clusters_props = filter_size_clusters(all_clusters_props, cluster_size_threshold)

    if not filtered_clusters_props:
        return f"{img_name}\t-1\t-1\t-1\t-1\t{"\t".join("-1" for _ in range(image.shape[-3]))}\n"
    
    cluster_objects = list()
    
    for cluster in filtered_clusters_props:

        cluster_obj = Cluster_3D(tile,
                                cluster.area,
                                cluster.centroid[-1],
                                cluster.centroid[-2],
                                cluster.centroid[-3],
                                cluster.coords
                                )
        
        cluster_obj.correct_positions(pixels_to_remove_w, pixels_to_remove_h, directions_remove_overlap)
        cluster_objects.append(cluster_obj)

    voxel_cluster_info = build_file_report(cluster_objects)

    return voxel_cluster_info


def main(z_stack_image_dir, channel_markers, channel_threshold_conditions, cluster_size_threshold, image_overlap, odirname, num_processers: int = 1):

    odirname = Path(odirname)
    odirname.mkdir(parents= True, exist_ok=True)

    z_stack_images_path = get_images_path(z_stack_image_dir, ".nd2")

    all_directions_remove_overlap = get_all_tiles_directions_to_remove_overlap(z_stack_images_path)

    sliced_channel_thresholds = slice_channel_threshold(channel_threshold_conditions)

    renamed_threshold_conditions = build_out_file_sufix(channel_threshold_conditions)

    voxel_cluster_report_fname = f"{odirname}/voxel_clusters_size_over{cluster_size_threshold}_{renamed_threshold_conditions}_test2.tsv"

    with open(voxel_cluster_report_fname, "w") as voxel_cluster_out:

        header = (f"Image\tVolume\tX\tY\tZ\t{"\t".join(f"{marker}_Summed_Intensity" for marker in channel_markers.split(","))}\n")
        voxel_cluster_out.write(header)
        
        with Pool(processes=num_processers) as pool:

            args_iter = zip(
                z_stack_images_path,
                all_directions_remove_overlap,
                repeat(image_overlap),
                repeat(sliced_channel_thresholds),
                repeat(cluster_size_threshold)
            )

            for voxel_cluster_info in tqdm(pool.imap(build_voxel_clusters,
                            args_iter), total=len(z_stack_images_path)):

                voxel_cluster_out.write(voxel_cluster_info)


if __name__ == "__main__":

    main(Z_STACK_IMAGE_DIR,
        CHANNEL_MARKERS,
        CHANNEL_THRESHOLD_CONDITIONS,
        CLUSTER_SIZE_THRESHOLD,
        IMAGE_OVERLAP,
        ODIRNAME,
        NUM_PROCESSERS)
