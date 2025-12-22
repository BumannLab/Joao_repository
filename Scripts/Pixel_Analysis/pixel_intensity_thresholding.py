# TODO: Add CLI, use click

import time
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from dataclasses import dataclass

import tifffile as tiff
import numpy as np
from tqdm import tqdm
from skimage import measure

from compute_direction_and_cut_overlap import get_all_tiles_directions_to_remove_overlap, compute_overlap_pixels, cut_overlap
from utils import get_images_path


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

        pixel_intensities = self.Tile_Info.Matrix[:, y_indixes, x_indixes]

        return pixel_intensities
    

    def obtain_cluster_mean_intensities(self):

        intensities = self.get_channel_intensity()

        sum_intensities = np.mean(intensities, axis=0)

        return sum_intensities.astype(str)


IMAGE_DIR = "U:/RG Bumann/ScientificData/Group/DataTansfer/New_mTB_Data_4Channel/MAX"
ODIRNAME = "Joao_repository/New_Results/CD68AL555_SYBRGold_DAPI_CD68AL647/Pixel_Analysis"

CHANNEL_MARKERS = "CD68AL555,SYBRGold,DAPI,CD68AL647"
CHANNEL_OF_INTEREST = "SYBRGold"  # Channel of Interest == coi
PIXEL_INTENSITY_THRESHOLD = 1000
CLUSTER_SIZE_THRESHOLD = 5

TILE_OVERLAP = 5
NUM_PROCESSORS = 10


def build_cluster_report_header(channel_markers: list[str]):
    """
    Input:
        - num_channels: Number of channels present in all images.

    Output:
        - A tab-separated string representing the header row to be written into the cluster report file.

    Method:
        - Constructs a list of channel-specific column names (e.g., Channel_1_Summed_Intensity, ...)
        and combines them with fixed metadata columns (Image, Area, X_position, Y_position).
    """
    ch_cols = [f"{channel_name}_Summed_Intensity" for channel_name in channel_markers]
    return "Image\tArea\tX_position\tY_position\t" + "\t".join(ch_cols) + "\n"


def build_pixel_report_header(channel_markers: list[str], channel_of_interest: str):
    """
    Input:
        - num_channels: Number of channels present in all images.
        - channel_of_interest: Channel of interest to produce the ratios.

    Output:
        - A tab-separated string representing the header row to be written into the pixel report file.

    Method:
        - Constructs two lists of channel-specific column names: (e.g., Channel_1, ...) and (e.g., Channel_1/Channel_of_interest)
        and combines them with fixed metadata columns (Image, X_position, Y_position).
    """
    
    ch_cols = [f"{marker}_Intensity" for marker in channel_markers]
    
    ratio_cols = [
        f"{marker}_Intenstity/{channel_of_interest}_Intensity"
        for marker in channel_markers
            if marker != channel_of_interest
    ]

    return "Image\tX_position\tY_position\t" + "\t".join(ch_cols + ratio_cols) + "\n"


def find_pixel_clusters(
    image: np.ndarray, channel_of_interest: int, intensity_threshold: int = 1000
):
    """
    Input:
        - image: Image with the intensity values in each pixel for all channels.
        - channel_of_interest: Channel of interest to threshold the pixels.
        - intensity_threshold: Threshold to extract the pixels that are above it.

    Output:
        - cluster_properties: Contains clusters of pixels that are above or equal to the threshold
        in the channel of interest of the image


    Method:
        - Extracts information from the image relative to the channel of interest, applies the threshold to the image
        where each pixel will be labeled as true if it meets the condition (>=threshold) and false otherwise. Then it
        produces a label for each pixel that belongs to the same cluster and then extracts all the information from the image, 
        such as area, centroid x and y positions.
    """

    coi_image = image[channel_of_interest]
    mask = coi_image >= intensity_threshold
    labeled = measure.label(mask, connectivity=2)
    clusters_properties = measure.regionprops(labeled)
    return clusters_properties


def filter_clusters(all_clusters, cluster_size_threshold: int):
    """
    Input:
        - all_clusters: Contains information for all clusters that have a pixel intensity above or equal than the threshold
        - cluster_size_threshold: The threshold to extract all clusters that are above it.
    Output: 
        - filtered_clusters: List with all the information about cluster with sie greater than threshold

    Method:
        - Extracts clusters that have an area/size greater than the threshold
    """

    filtered_clusters = [
        cluster for cluster in all_clusters if cluster.area > cluster_size_threshold
    ]
    return filtered_clusters


def build_cluster_report(image_name, image, cluster_report, corrected_centroid_positios):
    file_content = [0] * len(cluster_report)

    for i in range(len(cluster_report)):
        summed_intensities = get_summed_intensities(image, cluster_report[i].coords)

        y, x = corrected_centroid_positios[i]

        row = [
            image_name,
            str(cluster_report[i].area),
            str(x),
            str(y),
            *summed_intensities,
        ]

        file_content[i] = "\t".join(row)

    return "\n".join(file_content) + "\n"


def get_all_pixel_positions(pixel_clusters):
    pixel_coords = np.vstack([cluster.coords for cluster in pixel_clusters])
    return pixel_coords


def correct_pixel_positions(pixel_coords, pixels_to_remove_w, pixels_to_remove_h, directions):

    for direction in directions:
        if direction == "top":
            pixel_coords[:, 0] += pixels_to_remove_h
        elif direction == "right":
            pass
        elif direction == "left":
            pixel_coords[:, 1] += pixels_to_remove_w

    return pixel_coords


def normalize_by_coi_intensity(
    pixel_intensities: np.ndarray, num_channels: int, channel_of_interest: int
):
    """
    TODO
    """
    coi_intensity = pixel_intensities[:, [channel_of_interest]]

    non_coi_channels = [
        channel for channel in range(num_channels) if channel != channel_of_interest
    ]
    non_coi_channels_intensity = pixel_intensities[:, non_coi_channels]

    normalized_ratios = non_coi_channels_intensity / coi_intensity

    return normalized_ratios.astype(np.float32)


def build_pixel_report_tsv(
    image_name: str, pixel_coords, pixel_intensity, pixel_ratio
): 
    """
    TODO
    """
    file_content = [0] * len(pixel_coords)

    for i in range(len(pixel_coords)):

        pos_y = pixel_coords[:, 0][i]
        pos_x = pixel_coords[:, 1][i]

        raw_intensity = pixel_intensity[i].astype(str)
        normalized_intensity = pixel_ratio[i].astype(str)

        row = [
            image_name,
            str(pos_x),
            str(pos_y),
            *raw_intensity,
            *normalized_intensity,
        ]

        file_content[i] = "\t".join(row)

    return "\n".join(file_content) + "\n"


def analyse_clusters_and_pixels(args):
        image_path, num_channels, channel_of_interest, pixel_intensity_threshold, cluster_size_threshold, overlap, directions = args
        
        image = tiff.imread(image_path)

        img_height, img_width = image.shape[-2:]

        pixels_to_remove_h, pixels_to_remove_w = compute_overlap_pixels(overlap, img_height, img_width, rounding_mode="ceil")

        image = cut_overlap(image, pixels_to_remove_h, pixels_to_remove_w, directions)

        img_name = image_path.split("\\")[-1].split(".")[0]

        clusters_high_intensity = find_pixel_clusters(
            image, channel_of_interest, pixel_intensity_threshold
        )
        filtered_clusters_high_intensity = filter_clusters(
            clusters_high_intensity, cluster_size_threshold
        )

        if not filtered_clusters_high_intensity:
                return []
        
        corrected_cluster_positions = correct_cluster_positions(filtered_clusters_high_intensity, pixels_to_remove_w, pixels_to_remove_h, directions)

        clusters_info = build_cluster_report(
            img_name, image, filtered_clusters_high_intensity, corrected_cluster_positions
        )

        pixel_coords = get_all_pixel_positions(filtered_clusters_high_intensity)

        pixel_intensity = get_channel_intensity(image, pixel_coords)
        pixel_ratios = normalize_by_coi_intensity(
            pixel_intensity, num_channels, channel_of_interest
        )

        corrected_pixel_coords = correct_pixel_positions(pixel_coords, pixels_to_remove_w, pixels_to_remove_h, directions)
        
        pixel_info = build_pixel_report_tsv(
            img_name, corrected_pixel_coords, pixel_intensity, pixel_ratios
        )

        return clusters_info, pixel_info


def main(
    image_dir,
    channel_markers,
    channel_of_interest,
    pixel_intensity_threshold,
    cluster_size_threshold,
    overlap,
    odirname,
    num_processers: int = 1
):

    odirname = Path(odirname)
    odirname.mkdir(parents=True, exist_ok=True)

    channel_markers = channel_markers.split(",")
    channel_of_interest_idx = channel_markers.index(channel_of_interest)

    num_channels = len(channel_markers)

    images_path = get_images_path(image_dir, ".tif", ".tiff")

    all_directions_remove_overlap = get_all_tiles_directions_to_remove_overlap(images_path)

    pixel_report_fname = f"{odirname}/pixels_intensity_over{pixel_intensity_threshold}_test.tsv"
    cluster_report_fname = f"{odirname}/clusters_size{cluster_size_threshold}plus_intensity_over{pixel_intensity_threshold}_test.tsv"

    with open(pixel_report_fname, "w") as pixel_out, open(
        cluster_report_fname, "w"
    ) as cluster_out:
        
        pixel_header = build_pixel_report_header(
            channel_markers, channel_of_interest
        )
        cluster_header = build_cluster_report_header(channel_markers)

        pixel_out.write(pixel_header)
        cluster_out.write(cluster_header)
        
        with Pool(processes=num_processers) as pool:

            args_iter = zip(
                images_path,
                repeat(num_channels),
                repeat(channel_of_interest_idx),
                repeat(pixel_intensity_threshold),
                repeat(cluster_size_threshold),
                repeat(overlap),
                all_directions_remove_overlap
            )

            for result in tqdm(pool.imap(analyse_clusters_and_pixels,
                            args_iter), total=len(images_path)):
                
                if not result:
                    continue
                
                clusters_info, pixels_info = result

                cluster_out.write(clusters_info)
                pixel_out.write(pixels_info)


if __name__ == "__main__":
    start_time = time.time()

    main(
        IMAGE_DIR,
        CHANNEL_MARKERS,
        CHANNEL_OF_INTEREST,
        PIXEL_INTENSITY_THRESHOLD,
        CLUSTER_SIZE_THRESHOLD,
        TILE_OVERLAP,
        ODIRNAME,
        NUM_PROCESSORS
    )

    print(f"Finished in {time.time() - start_time} seconds")


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


Z_STACK_IMAGE_DIR = "/data/joao/mTBData/4CHANNELsTAINING_JoaoSamples"
ODIRNAME = "Results/Previous_Destaining/CD68AL555_SYBRGold_DAPI_CD68AL647/Voxel_Analysis/"

CHANNEL_MARKERS = "CD68AL555,SYBRGold,DAPI,CD68AL647"
CHANNEL_THRESHOLD_CONDITIONS = "SYBRGold>=1000"
CLUSTER_SIZE_THRESHOLD = 5

IMAGE_OVERLAP = 5
NUM_PROCESSERS = 20


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
    

    def obtain_cluster_mean_intensities(self):

        intensities = self.get_channel_intensity()

        sum_intensities = np.mean(intensities, axis=0)

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

    threshold_conditions = threshold_conditions.replace(",", "_")

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
            *cluster.obtain_cluster_mean_intensities(),
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
        no_intensity = '\t'.join('-1' for _ in range(image.shape[-3]))
        return f"{img_name}\t-1\t-1\t-1\t-1\t{no_intensity}\n"
    
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


def main(image_dir, channel_markers, channel_threshold_conditions, channel_of_interest, cluster_size_threshold, image_overlap, odirname, num_processers: int = 1):

    odirname = Path(odirname)
    odirname.mkdir(parents= True, exist_ok=True)

    channel_markers = channel_markers.split(",")

    images_path = get_images_path(image_dir, ".tif", ".tiff")

    all_directions_remove_overlap = get_all_tiles_directions_to_remove_overlap(images_path)

    sliced_channel_thresholds = slice_channel_threshold(channel_threshold_conditions, channel_markers)

    renamed_threshold_conditions = build_out_file_sufix(channel_threshold_conditions)

    pixel_cluster_report_fname = f"{odirname}/pixel_clusters_size_over{cluster_size_threshold}_{renamed_threshold_conditions}.tsv"
    pixel_report_fname = f"{odirname}/pixels_{renamed_threshold_conditions}.tsv"

    with open(pixel_report_fname, "w") as pixel_out, open(
        pixel_cluster_report_fname, "w"
    ) as cluster_out:

        pixel_header = build_pixel_report_header(
            channel_markers, channel_of_interest
        )
        cluster_header = build_cluster_report_header(channel_markers)

        pixel_out.write(pixel_header)
        cluster_out.write(cluster_header)
        
        with Pool(processes=num_processers) as pool:

            args_iter = zip(
                images_path,
                all_directions_remove_overlap,
                repeat(image_overlap),
                repeat(sliced_channel_thresholds),
                repeat(cluster_size_threshold)
            )

            for voxel_cluster_info in tqdm(pool.imap(build_voxel_clusters,
                            args_iter), total=len(images_path)):

                voxel_cluster_out.write(voxel_cluster_info)


if __name__ == "__main__":

    main(Z_STACK_IMAGE_DIR,
        CHANNEL_MARKERS,
        CHANNEL_THRESHOLD_CONDITIONS,
        CLUSTER_SIZE_THRESHOLD,
        IMAGE_OVERLAP,
        ODIRNAME,
        NUM_PROCESSERS)
