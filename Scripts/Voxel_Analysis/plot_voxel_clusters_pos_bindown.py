import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import tifffile as tiff
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from utils import read_tiff_image
from compute_direction_and_cut_overlap import compute_overlap_pixels, get_all_tiles_position_corrected


VOXEL_CLUSTERS_FILE = "Joao_repository/Results/SYBRGold_CD68AL555_LAMAL647_DAPI/Full_scan/Voxel_Analysis/voxel_clusters_size_over5_SYBRGold_greater_equal_500.tsv"
BINDOWN_IMAGE_FILE = "M:/Joao/5xbindown_stitched/stitched_01_bin5x.tif"

TILE_HEIGHT = 2304
TILE_WIDTH = 2304
TILE_OVERLAP = 5

BINDOWN_BY = 5

MAXIMUM_DISPLAYED_VALUE = 400

ALLOW_RANGE = True # Range will be applied only on the sybrgold channel using clusters with a dapi mean intensity under 250
RANGE_CATEGORIES = [500, 800, 1000] # e.g 500,800,1000 -> [500, 800[ ; [800, 1000[ ; [1000, 1000+[

@dataclass
class Cluster:
    X_centroid: float
    Y_centroid: float
    Range_Categorie: int = 65535

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

        return rescaled_bindown_x, rescaled_bindown_y, self.Range_Categorie
        

def obtain_range_limits(range_categories):
    range_limits = []
    
    for i in range(len(range_categories)):
        
        try: range_limits.append((range_categories[i], range_categories[i+1]))
        except: range_limits.append((range_categories[i], 65536))

    return range_limits


def read_cluster_data(file_path, allow_range, range_limits):

    cluster_data = defaultdict(list)

    with open(file_path, mode='r') as infile:
        reader = csv.DictReader(infile, delimiter="\t")

        for rows in reader:
            tile_name = rows["Image"]

            if allow_range:

                sybr_gold_mean_intensity = float(rows["SYBRGold_Mean_Intensity"])
                dapi_mean_intensity = float(rows["DAPI_Mean_Intensity"])

                for idx, min_max_range in enumerate(range_limits):
                
                    min_intensity, max_intensity = min_max_range
                    if dapi_mean_intensity <= 250:
                        if min_intensity <= sybr_gold_mean_intensity < max_intensity: 
                            cluster_pos = Cluster(float(rows["X"]), float(rows["Y"]), idx + 1)

                    else:
                        cluster_pos = Cluster(float(rows["X"]), float(rows["Y"]))
                        break
            
            else:
                cluster_pos = Cluster(float(rows["X"]), float(rows["Y"]))

            cluster_data[tile_name].append(cluster_pos)

    return cluster_data


def resize_cluster_positions(cluster_data, all_tiles_positions, tile_height, tile_width, pixels_to_remove_h, pixels_to_remove_w, bindown_by):

    bindowned_cluster_positions = list()
    
    for tile_pos , tile_cluster_pos in enumerate(cluster_data.items()):

        _, clusters_positions = tile_cluster_pos

        tile_column, tile_row = all_tiles_positions[tile_pos]

        for cluster_pos in clusters_positions:

            if cluster_pos == Cluster(-1, -1, 65535):
                print(tile_pos, cluster_pos)
                break

            rescaled_pos = cluster_pos.bindowned_rescaled_centroid_pos(tile_column,
                                                                tile_row,
                                                                tile_height,
                                                                tile_width,
                                                                pixels_to_remove_h,
                                                                pixels_to_remove_w,
                                                                bindown_by)

            bindowned_cluster_positions.append(rescaled_pos)

    return bindowned_cluster_positions


def plot_bindown_with_voxel_clusters_png_overview(bindown_image, cluster_positions, maximum_displayed_value, out_path):
    
    bindown_image = bindown_image * 255 / maximum_displayed_value

    datax = [positions[0] for positions in cluster_positions]
    datay = [positions[1] for positions in cluster_positions]

    plt.figure(figsize=(16, 8), frameon=False)

    plt.imshow(bindown_image, cmap='gray', vmin=0, vmax=255, interpolation='nearest', aspect='auto', origin='upper')
    plt.scatter(datax, datay, marker = "x", s = 2.5, linewidths = 0.7, c ="magenta")
    
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(
        f"Joao_repository/Results/SYBRGold_CD68AL555_LAMAL647_DAPI/Full_scan/Voxel_Analysis/Plots/SYBRGold_above_500_cluster_pos_in_bindown5x_with_ranges.png",
        dpi=300,
        bbox_inches="tight",
    )


def plot_bindown_with_voxel_clusters_tiff_overview(image_shape, cluster_positions, out_path):

    img = Image.fromarray(
        np.zeros(image_shape, dtype=np.uint16)
    )

    draw = ImageDraw.Draw(img)

    font = ImageFont.load_default(size = 30)

    text = "x"

    for cluster_pos in cluster_positions:

        x, y, range_category = cluster_pos
        
        bbox = draw.textbbox((0, 0), text, font=font)

        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        draw.text(
            (int(x - text_w / 2), int(y - text_h / 2)),
            text,
            fill=range_category,
            font=font
        )

    tiff.imwrite(f"Joao_repository/Results/SYBRGold_CD68AL555_LAMAL647_DAPI/Full_scan/Voxel_Analysis/Plots/SYBRGold_above_500_cluster_pos_in_bindown5x_with_ranges.tiff", np.array(img))


def main(voxel_clusters_file, bindown_image_file, tile_height, tile_width, tile_overlap, bindown_by, maximum_displayed_value, allow_range, range_categories):
    
    odirname = Path(odirname)
    odirname.mkdir(parents= True, exist_ok=True)

    range_limits = obtain_range_limits(range_categories)

    cluster_pos_data = read_cluster_data(voxel_clusters_file, allow_range, range_limits)

    all_tiles_positions = get_all_tiles_position_corrected(cluster_pos_data.keys())

    pixels_to_remove_h, pixels_to_remove_w = compute_overlap_pixels(tile_overlap, tile_height, tile_width, rounding_mode="nearest")

    resized_cluster_pos = resize_cluster_positions(cluster_pos_data, all_tiles_positions, tile_height, tile_width, pixels_to_remove_h, pixels_to_remove_w, bindown_by)

    bindown_image = read_tiff_image(bindown_image_file).astype(float)

    plot_bindown_with_voxel_clusters_png_overview(bindown_image, resized_cluster_pos, maximum_displayed_value, "")

    plot_bindown_with_voxel_clusters_tiff_overview(bindown_image.shape, resized_cluster_pos, "")


if __name__ == "__main__":
    main(VOXEL_CLUSTERS_FILE,
         BINDOWN_IMAGE_FILE,
         TILE_HEIGHT,
         TILE_WIDTH,
         TILE_OVERLAP,
         BINDOWN_BY,
         MAXIMUM_DISPLAYED_VALUE,
         ALLOW_RANGE,
         RANGE_CATEGORIES)