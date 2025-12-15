import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tifffile as tiff

from utils import get_images_path
from compute_direction_and_cut_overlap import compute_overlap_pixels, get_all_tiles_position_corrected


TILES_PATH = "U:/RG Bumann/ScientificData/Group/DataTansfer/New_mTB_Data_4Channel/MAX"
BINDOWN_IMAGE_FILE = "M:/Joao/5xbindown_stitched/stitched_01_bin5x.tif"

TILE_HEIGHT = 2304
TILE_WIDTH = 2304
TILE_OVERLAP = 5

BINDOWN_BY = 5

OUT_PATH = "M:/Joao/5xbindown_stitched/labeled_stitched_bin5x.tif"


def adjusted_tile_center_pos(all_tiles_positions, tile_height, tile_width, pixels_to_remove_h, pixels_to_remove_w, bindown_by):

    adjusted_tile_center_positions = list()
    
    for tile_pos in all_tiles_positions:
        
        tile_column, tile_row = tile_pos

        tile_center_x = ((tile_width / 2) + (((tile_width - pixels_to_remove_w) * (tile_column)))) / bindown_by

        tile_center_y = ((tile_height / 2) + (((tile_height - pixels_to_remove_h) * (tile_row)))) / bindown_by

        adjusted_tile_center_positions.append((tile_center_x, tile_center_y))

    return adjusted_tile_center_positions


def tile_centres_text_layer(image_shape, tile_center_positions, out_path):

    img = Image.fromarray(
        np.zeros(image_shape, dtype=np.uint16)
    )

    draw = ImageDraw.Draw(img)

    font = ImageFont.load_default(size = 50)

    for tile_number, (x, y) in enumerate(tile_center_positions):
        
        text = str(tile_number + 1)

        # Measure text size for true centering
        bbox = draw.textbbox((0, 0), text, font=font)

        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        draw.text(
            (int(x - text_w / 2), int(y - text_h / 2)),
            text,
            fill=65535,
            font=font
        )

    tiff.imwrite(out_path, np.array(img))


def main(tiles_path, tile_height, tile_width, tile_overlap, bindown_img_path, bindown_by, out_path):

    images_paths = get_images_path(tiles_path, ".tif")

    all_tiles_positions = get_all_tiles_position_corrected(images_paths)

    pixels_to_remove_h, pixels_to_remove_w = compute_overlap_pixels(tile_overlap, tile_height, tile_width, rounding_mode="nearest")

    adjusted_tile_center_positions = adjusted_tile_center_pos(all_tiles_positions, tile_height, tile_width, pixels_to_remove_h, pixels_to_remove_w, bindown_by)

    bindown_img_shape = tiff.imread(bindown_img_path).shape

    tile_centres_text_layer(bindown_img_shape, adjusted_tile_center_positions, out_path)


if __name__ == "__main__":
    main(TILES_PATH,
         TILE_HEIGHT,
         TILE_WIDTH,
         TILE_OVERLAP,
         BINDOWN_IMAGE_FILE,
         BINDOWN_BY,
         OUT_PATH)