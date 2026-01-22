import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import tifffile as tiff
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from utils import read_tiff_image, slice_channel_threshold
from compute_direction_and_cut_overlap import compute_overlap_pixels, get_all_tiles_position_snake


VOXEL_CLUSTERS_FILE = "/data/joao/Materials/Joao_repository/Results/Old_Destaining/SYBRGold_CD68AL555_LAMAL647_DAPI/Voxel_Analysis/SYBRGold1000_Threshold/voxel_clusters_size_over5_SYBRGold_greater_equal_1000.tsv"
BINDOWN_IMAGE_FILE = "/data/joao/mTBData/LAM1AL647_CD68AL555_SYBRGold_DAPI/5xbindown_stitched/stitched_01_bin5x.tif" # Preferably the sybrgold bindown channel

TILE_HEIGHT = 2304
TILE_WIDTH = 2304
TILE_OVERLAP = 5

BINDOWN_BY = 5

MAXIMUM_DISPLAYED_VALUE = 800

USE_FILTERS = True
FILTERS = "DAPI<=250"

ODIRFILENAME = "/data/joao/Materials/Joao_repository/Results/Old_Destaining/SYBRGold_CD68AL555_LAMAL647_DAPI/Voxel_Analysis/SYBRGold1000_Threshold/DAPI_mean_lt250_SYBRGold_mean_gt1000_bindown5x"

@dataclass
class Cluster:
    X_centroid: float
    Y_centroid: float

    def bindowned_rescaled_centroid_pos(self, tile_column, tile_row, tile_height, tile_width,
                                        pixels_remove_width, pixels_remove_height, bindown_by):
        
        # Since each tile is being downscaled first and then added to the bindown overview. It will first compute how much pixels to remove
        # based on overlap (e.g 2304 * 0.05 (5% overlap) = 115.2 ~ 115, it will round based on the proximity to the previous or next integer). Then the bindown
        # is done (e.g. 2304 - 115 = 2189 / 5 (bindown by 5) = 437.8 ~ 438, same rounding principle as before)

        # In case in the future each tile is stitched together and then downscaling its applied to the final stitched version, change this function
        bindown_tile_width = round((tile_width - pixels_remove_width) / bindown_by) * tile_column
        bindown_tile_height = round((tile_height - pixels_remove_height) / bindown_by) * tile_row
        
        bindown_x_centroid = round(self.X_centroid / bindown_by)
        bindown_y_centroid = round(self.Y_centroid / bindown_by)

        rescaled_bindown_x = bindown_x_centroid + bindown_tile_width
        rescaled_bindown_y = bindown_y_centroid + bindown_tile_height

        self.X_centroid = rescaled_bindown_x
        self.Y_centroid = rescaled_bindown_y


def read_cluster_data(file_path, use_filter, filters = None):

    def filter_data(row, filters):
        
        pass_threshold = True
        
        for filter in filters:
            channel, operator, threshold = filter
            
            value = float(row[channel + "_Mean_Intensity"])

            if operator == ">=":
                pass_threshold = value >= threshold
            elif operator == ">":
                pass_threshold = value > threshold
            elif operator == "<=":
                pass_threshold = value <= threshold
            elif operator == "<":
                pass_threshold = value < threshold

            if not pass_threshold:
                break
        
        return pass_threshold

    cluster_data = defaultdict(list)

    with open(file_path, mode='r') as infile:
        reader = csv.DictReader(infile, delimiter="\t")
        
        for rows in reader:
            tile_name = rows["Image"]

            _ = cluster_data[tile_name]

            if use_filter:
                if not filter_data(rows, filters):
                    continue

            cluster = Cluster(float(rows["X"]), float(rows["Y"]))
            cluster_data[tile_name].append(cluster)

    return cluster_data


def resize_cluster_positions(cluster_data, all_tiles_positions, tile_height, tile_width, pixels_to_remove_h, pixels_to_remove_w, bindown_by):

    resize_filtered_cluster_positions = list()
    
    for tile_pos , tile_cluster in enumerate(cluster_data.items()):

        _, clusters_info = tile_cluster

        tile_column, tile_row = all_tiles_positions[tile_pos]

        for cluster in clusters_info:

            if cluster.X_centroid == float(-1) and cluster.Y_centroid == float(-1):
                break

            cluster.bindowned_rescaled_centroid_pos(tile_column,
                                                        tile_row,
                                                        tile_height,
                                                        tile_width,
                                                        pixels_to_remove_h,
                                                        pixels_to_remove_w,
                                                        bindown_by)
            
            resize_filtered_cluster_positions.append(cluster)

    return resize_filtered_cluster_positions


def plot_bindown_with_voxel_clusters_png_overview(bindown_image, cluster_data, maximum_displayed_value, odirfilename):
    
    bindown_image = bindown_image * 255 / maximum_displayed_value

    datax = [cluster.X_centroid for cluster in cluster_data]
    datay = [cluster.Y_centroid for cluster in cluster_data]

    plt.figure(figsize=(16, 8), frameon=False)

    plt.imshow(bindown_image, cmap='gray', vmin=0, vmax=255, interpolation='nearest', aspect='auto', origin='upper')
    plt.scatter(datax, datay, marker = "x", s = 2.5, linewidths = 0.7, c ="magenta")
    
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(
        f"{odirfilename}.png",
        dpi=300,
        bbox_inches="tight",
    )


def plot_bindown_with_voxel_clusters_tiff_overview(image_shape, cluster_data, odirfilename):
        
        img = Image.fromarray(
            np.zeros(image_shape, dtype=np.uint16)
        )

        draw = ImageDraw.Draw(img)

        font = ImageFont.load_default(size = 30)

        text = "x"

        bbox = draw.textbbox((0, 0), text, font=font)

        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        for cluster in cluster_data:
            
            x, y = cluster.X_centroid, cluster.Y_centroid
            
            draw.text(
                (int(x - text_w / 2), int(y - text_h / 2)),
                text,
                fill=65535,
                font=font
            )

        tiff.imwrite(f"{odirfilename}.tiff", np.array(img))


def main(voxel_clusters_file, bindown_image_file, tile_height, tile_width, tile_overlap, bindown_by, maximum_displayed_value, odirfilename, use_filters, filters = None):
    
    odirname = Path(odirfilename).parent
    odirname.mkdir(parents= True, exist_ok=True)
    
    if use_filters:
        filters = slice_channel_threshold(filters)

    cluster_data = read_cluster_data(voxel_clusters_file, use_filters, filters) 

    all_tiles_positions = get_all_tiles_position_snake(cluster_data.keys())

    pixels_to_remove_h, pixels_to_remove_w = compute_overlap_pixels(tile_overlap, tile_height, tile_width, rounding_mode="nearest")

    resized_cluster_data = resize_cluster_positions(cluster_data, all_tiles_positions, tile_height, tile_width, pixels_to_remove_h, pixels_to_remove_w, bindown_by)

    print(len(resized_cluster_data))
    
    bindown_image = read_tiff_image(bindown_image_file).astype(float)

    plot_bindown_with_voxel_clusters_png_overview(bindown_image, resized_cluster_data, maximum_displayed_value, odirfilename)
    print("png done")

    plot_bindown_with_voxel_clusters_tiff_overview(bindown_image.shape, resized_cluster_data, odirfilename)
    print("tiff done")



if __name__ == "__main__":
    main(VOXEL_CLUSTERS_FILE,
         BINDOWN_IMAGE_FILE,
         TILE_HEIGHT,
         TILE_WIDTH,
         TILE_OVERLAP,
         BINDOWN_BY,
         MAXIMUM_DISPLAYED_VALUE,
         ODIRFILENAME,
         USE_FILTERS,
         FILTERS)
    