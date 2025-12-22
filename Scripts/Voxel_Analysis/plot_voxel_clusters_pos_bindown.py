import csv
from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt

from utils import read_tiff_image
from compute_direction_and_cut_overlap import compute_overlap_pixels, get_all_tiles_position_corrected


VOXEL_CLUSTERS_FILE = "Joao_repository/New_Results/SYBRGold_CD68AL555_LAMAL647_DAPI/Full_scan/Voxel_Analysis/voxel_clusters_size_over5_channel1_greater_equal_500.tsv"
BINDOWN_IMAGE_FILE = "M:/Joao/5xbindown_stitched/stitched_01_bin5x.tif"

TILE_HEIGHT = 2304
TILE_WIDTH = 2304
TILE_OVERLAP = 5

BINDOWN_BY = 5

MAXIMUM_DISPLAYED_VALUE = 400

ALLOW_RANGE = True
RANGE_CATEGORIES = [500, 800, 1000] # e.g 500,800,1000 -> [500, 800[ ; [800, 1000[ ; [1000, 1000+[

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

        return rescaled_bindown_x, rescaled_bindown_y
        

def read_cluster_data(file_path):

    cluster_data = defaultdict(list)

    with open(file_path, mode='r') as infile:
        reader = csv.DictReader(infile, delimiter="\t")

        for rows in reader:
            tile_name = rows["Image"]

            cluster_pos = Cluster(float(rows["X"]), float(rows["Y"]))

            cluster_data[tile_name].append(cluster_pos)

    return cluster_data


def resize_cluster_positions(cluster_data, all_tiles_positions, tile_height, tile_width, pixels_to_remove_h, pixels_to_remove_w, bindown_by):

    bindowned_cluster_positions = list()
    
    for tile_pos , tile_cluster_pos in enumerate(cluster_data.items()):

        _, clusters_positions = tile_cluster_pos

        tile_column, tile_row = all_tiles_positions[tile_pos]

        for cluster_pos in clusters_positions:

            if cluster_pos == Cluster(-1, -1):
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


def plot_bindown_with_voxel_clusters(bindown_path, cluster_positions, maximum_displayed_value):

    image = read_tiff_image(bindown_path).astype(float)

    image = image * 255 / maximum_displayed_value

    datax = [positions[0] for positions in cluster_positions]
    datay = [positions[1] for positions in cluster_positions]

    plt.figure(figsize=(16, 8), frameon=False)

    plt.imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='nearest', aspect='auto', origin='upper')
    plt.scatter(datax, datay, marker = "x", s = 2.5, linewidths = 0.7, c ="magenta")
    
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(
        f"Joao_Repository/New_Results/Test.png",
        dpi=300,
        bbox_inches="tight",
    )
    

def main(voxel_clusters_file, bindown_image_file, tile_height, tile_width, tile_overlap, bindown_by, maximum_displayed_value):

    cluster_pos_data = read_cluster_data(voxel_clusters_file)

    all_tiles_positions = get_all_tiles_position_corrected(cluster_pos_data.keys())

    pixels_to_remove_h, pixels_to_remove_w = compute_overlap_pixels(tile_overlap, tile_height, tile_width)

    pixels_to_remove_h, pixels_to_remove_w = pixels_to_remove_h - 1, pixels_to_remove_w - 1

    resized_cluster_pos = resize_cluster_positions(cluster_pos_data, all_tiles_positions, tile_height, tile_width, pixels_to_remove_h, pixels_to_remove_w, bindown_by)

    plot_bindown_with_voxel_clusters(bindown_image_file, resized_cluster_pos, maximum_displayed_value)


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