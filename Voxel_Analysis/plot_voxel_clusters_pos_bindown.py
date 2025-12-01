import csv
from collections import defaultdict

import matplotlib.pyplot as plt

from utils import read_tiff_image
from compute_direction_and_cut_overlap import compute_overlap_pixels, get_all_tiles_position_corrected


VOXEL_CLUSTERS_FILE =  "Joao_repository/New_Results/CD68AL555_SYBRGold_DAPI_CD68AL647/Voxel_Analysis/voxel_clusters_size_over5_channel2_greater_equal_1000_channel3_less_equal_250_test.tsv"  # "Joao_repository/Results/test_data.tsv"
BINDOWN_IMAGE_FILE = "M:/Joao/5xbindown_stitched/stitched_01_bin5x.tif"

TILE_HEIGHT = 2304
TILE_WIDTH = 2304
TILE_OVERLAP = 5

BINDOWN_BY = 5

MAXIMUM_DISPLAYED_VALUE = 400


def read_cluster_data(file_path):

    cluster_data = defaultdict(list)

    with open(file_path, mode='r') as infile:
        reader = csv.DictReader(infile, delimiter="\t")

        for rows in reader:
            tile_name = rows["Image"]

            if rows["Volume"] == "0":
                cluster_data[tile_name].append((-1, -1))
            else: # once we change the normal script to write -1 when there is no bacteria found and run it replace the previous tuple by the tuple present down
                cluster_pos = (float(rows["X"]), float(rows["Y"]))

            cluster_data[tile_name].append(cluster_pos)

    return cluster_data


def bindowned_rescaled_cluster_x_y_pos(cluster_pos, tile_height, tile_width, tile_column, tile_row, 
                                       pixels_remove_width, pixels_remove_height, bindown_by):

    x, y = cluster_pos

    if tile_row > 0:
     rescaled_y = y - pixels_remove_height + tile_height + ((tile_height - pixels_remove_height) * (tile_row - 1))

    else:
        rescaled_y = y

    if tile_row % 2 == 0:
        rescaled_x = x - pixels_remove_width + tile_width + ((tile_width - pixels_remove_width) * (tile_column - 1))

    elif tile_row % 2 != 0:
        rescaled_x = x + ((tile_width - pixels_remove_width) * (tile_column))

    return rescaled_x / bindown_by, rescaled_y / bindown_by


def resize_cluster_positions(cluster_data, all_tiles_positions, tile_height, tile_width, pixels_to_remove_h, pixels_to_remove_w, bindown_by):

    bindowned_cluster_positions = list()
    
    for tile_pos , tile_cluster_pos in enumerate(cluster_data.items()):

        tile_name, clusters_positions = tile_cluster_pos

        tile_column, tile_row = all_tiles_positions[tile_pos]

        for cluster_pos in clusters_positions:

            if cluster_pos == (-1, -1):
                break

            rescaled_pos = bindowned_rescaled_cluster_x_y_pos(cluster_pos,
                                                                tile_height,
                                                                tile_width,
                                                                tile_column,
                                                                tile_row,
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


# def plot_bindown_with_voxel_clusters(bindown_path, cluster_positions, maximum_displayed_value):

#     image = read_tiff_image(bindown_path).astype(float)

#     image = image * 255 / maximum_displayed_value

#     datax = [positions[0] for positions in cluster_positions]
#     datay = [positions[1] for positions in cluster_positions]

#     fig, ax = plt.subplots(figsize=(16, 8))

#     ax.imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='nearest', aspect='auto', origin='upper')
#     ax.scatter(datax, datay, marker = "x", s = 2.5, linewidths = 0.7, c ="magenta")

#     ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    
#     ax.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
#     ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
#     fig.tight_layout()
#     fig.savefig(
#         f"Joao_Repository/New_Results/Test.png",
#         dpi=300,
#         bbox_inches="tight",
#     )
    
    

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
         MAXIMUM_DISPLAYED_VALUE)