from itertools import repeat
from multiprocessing import Pool
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm
from skimage.measure import label, regionprops

from utils import slice_channel_threshold, get_images_path, read_tiff_image
from compute_direction_and_cut_overlap import get_all_tiles_directions_to_remove_overlap, compute_overlap_pixels, cut_overlap


IMAGE_DIR = "U:/RG Bumann/ScientificData/Group/DataTansfer/New_mTB_Data_4Channel/MAX"
ODIRNAME = "Results/Test/Pixel_Analysis/"

CHANNEL_MARKERS = "CD68AL555,SYBRGold,DAPI,CD68AL647"
CHANNEL_THRESHOLD_CONDITIONS = "SYBRGold>=500,SYBRGold<=800"
CLUSTER_SIZE_THRESHOLD = 5

IMAGE_OVERLAP = 5
NUM_PROCESSERS = 20


@dataclass
class Tile:
    Name: str
    Matrix: np.typing.NDArray[np.uint16]


@dataclass
class Cluster_2D:
    Tile_Info: Tile
    Area: int
    X_centroid: float
    Y_centroid: float
    Pixels_coords: np.typing.NDArray[np.int64]


    def __repr__(self):
        return f"Cluster_2D({self.Tile_Info.Name}, {self.Area}, {self.X_centroid}, {self.Y_centroid}"


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

        mean_intensities = np.mean(intensities, axis=1).astype(np.float16)

        return mean_intensities.astype(str)


    def get_pixel_intensity(self, x, y):

        pixel_intensity = self.Tile_Info.Matrix[:, y, x]

        return pixel_intensity.astype(str)


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
    ch_cols = [f"{channel_name}_Mean_Intensity" for channel_name in channel_markers]
    return "Image\tArea\tX\tY\t" + "\t".join(ch_cols) + "\n"


def build_pixel_report_header(channel_markers: list[str]):
    """
    Input:
        - channel_markers: Number of channels present in all images.

    Output:
        - A tab-separated string representing the header row to be written into the pixel report file.

    Method:
        - Constructs two lists of channel-specific column names: (e.g., Channel_1, ...)
        and combines it with fixed metadata columns (Image, X_position, Y_position).
    """
    
    ch_cols = [f"{marker}_Intensity" for marker in channel_markers]
    
    return "Image\tX\tY\t" + "\t".join(ch_cols) + "\n"


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
        voxel_coi_intensity_mask = image[channel] > threshold
    elif operator == ">=":
        voxel_coi_intensity_mask = image[channel] >= threshold
    elif operator == "<":
        voxel_coi_intensity_mask = image[channel] < threshold
    elif operator == "<=":
        voxel_coi_intensity_mask = image[channel] <= threshold

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

    mask_label = label(intensity_thresholded_mask, connectivity = 2)
    clusters_properties = regionprops(mask_label)

    return clusters_properties


def filter_size_clusters(all_clusters_props, cluster_size_threshold: int = 5):
    filtered_clusters = [
        cluster for cluster in all_clusters_props if cluster.area > cluster_size_threshold
    ]
    return filtered_clusters


def build_file_report(cluster_objects):
    cluster_file_content = list()
    pixel_file_content = list()

    for cluster in cluster_objects:
    
        cluster_content = "\t".join([
            cluster.Tile_Info.Name,
            str(cluster.Area),
            str(cluster.X_centroid), 
            str(cluster.Y_centroid),
            *cluster.obtain_cluster_mean_intensities(),
        ])

        cluster_file_content.append(cluster_content)

        for pixel in cluster.Pixels_coords:

            x_pos, y_pos = pixel[1], pixel[0]

            pixel_content = "\t".join([
                cluster.Tile_Info.Name,
                str(x_pos), 
                str(y_pos),
                *cluster.get_pixel_intensity(x_pos, y_pos),
            ])

            pixel_file_content.append(pixel_content)
        
    return "\n".join(cluster_file_content) + "\n", "\n".join(pixel_file_content) + "\n"

def analyse_clusters_and_pixels(args):

    image_path, num_channels, directions_remove_overlap, image_overlap, channel_threshold_conditions, cluster_size_threshold = args

    img_name = image_path.split("/")[-1].split(".")[0]

    try: image = read_tiff_image(image_path)
    except Exception as e:
        print(f"Error loading {image_path}. As a result this image will be classified as having no clusters.")
        print(f"The error in question is :\n{e}")
        no_intensity = '\t'.join('-1' for _ in range(num_channels))
        return (f"{img_name}\t-1\t-1\t-1\t{no_intensity}\n", "")

    tile = Tile(img_name, image)

    img_height, img_width = image.shape[-2:]

    pixels_to_remove_h, pixels_to_remove_w = compute_overlap_pixels(image_overlap, img_height, img_width, rounding_mode="ceil")

    image = cut_overlap(image, pixels_to_remove_h, pixels_to_remove_w, directions_remove_overlap)

    final_thresholded_mask = multi_channel_threshold_mask(image, channel_threshold_conditions)

    all_clusters_props = find_voxel_clusters(final_thresholded_mask)

    filtered_clusters_props = filter_size_clusters(all_clusters_props, cluster_size_threshold)

    if not filtered_clusters_props:
        no_intensity = '\t'.join('-1' for _ in range(num_channels))
        return (f"{img_name}\t-1\t-1\t-1\t{no_intensity}\n", "")
    
    cluster_objects = list()
    
    for cluster in filtered_clusters_props:

        cluster_obj = Cluster_2D(tile,
                                cluster.area,
                                round(cluster.centroid[-1], 1),
                                round(cluster.centroid[-2], 1),
                                cluster.coords
                                )
        
        cluster_obj.correct_positions(pixels_to_remove_w, pixels_to_remove_h, directions_remove_overlap)
        cluster_objects.append(cluster_obj)

    cluster_info, pixel_info = build_file_report(cluster_objects)

    return cluster_info, pixel_info


def main(image_dir, channel_markers, channel_threshold_conditions, cluster_size_threshold, image_overlap, odirname, num_processers: int = 1):

    odirname = Path(odirname)
    odirname.mkdir(parents= True, exist_ok=True)

    channel_markers = channel_markers.split(",")

    images_path = get_images_path(image_dir, ".tif", ".tiff")

    all_directions_remove_overlap = get_all_tiles_directions_to_remove_overlap(images_path)

    sliced_channel_thresholds = slice_channel_threshold(channel_threshold_conditions, channel_markers)

    renamed_threshold_conditions = build_out_file_sufix(channel_threshold_conditions)

    cluster_report_fname = f"{odirname}/pixel_clusters_size_over{cluster_size_threshold}_{renamed_threshold_conditions}.tsv"
    pixel_report_fname = f"{odirname}/pixels_{renamed_threshold_conditions}.tsv"

    with open(pixel_report_fname, "w") as pixel_out, open(
        cluster_report_fname, "w"
    ) as cluster_out:

        pixel_header = build_pixel_report_header(
            channel_markers
        )
        cluster_header = build_cluster_report_header(channel_markers)

        pixel_out.write(pixel_header)
        cluster_out.write(cluster_header)
        
        with Pool(processes=num_processers, maxtasksperchild=1) as pool:

            args_iter = zip(
                images_path,
                repeat(len(channel_markers)),
                all_directions_remove_overlap,
                repeat(image_overlap),
                repeat(sliced_channel_thresholds),
                repeat(cluster_size_threshold)
            )

            for result in tqdm(pool.imap(analyse_clusters_and_pixels,
                            args_iter), total=len(images_path)):
                
                cluster_info, pixel_info = result

                cluster_out.write(cluster_info)
                pixel_out.write(pixel_info)


if __name__ == "__main__":

    main(IMAGE_DIR,
        CHANNEL_MARKERS,
        CHANNEL_THRESHOLD_CONDITIONS,
        CLUSTER_SIZE_THRESHOLD,
        IMAGE_OVERLAP,
        ODIRNAME,
        NUM_PROCESSERS)