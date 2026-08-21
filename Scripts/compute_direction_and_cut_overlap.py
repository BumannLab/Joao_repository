from math import ceil, floor
import re
from pathlib import Path


def convert_row_numerical(row: str):
    row_numerical = 0

    for char in row:
        assert ord(char) <= ord("T"), f"Row characters only span from A to T, {char} is not included"
        row_numerical = (row_numerical * 20) + (ord(char) - ord("A"))
    
    return row_numerical


def obtain_tile_position(tile_name):
    """
    Returns (col, row)
    """

    if not isinstance(tile_name, Path):
        tile_name = Path(tile_name)
    
    tile_info = tile_name.stem

    tile_pos = tile_info.split(".")[0].split("_")[-1]

    match = re.search(r'([A-Za-z]+)(\d+)', tile_pos)
    
    if match is None:
        raise ValueError(f"No row and col found in {tile_info}. Expected to find e.g. AA01")
    
    row, column = match.groups()

    # Convert column to numerical since they are like this: "AA" -> 0, "BA" -> 26
    # and remove 1 from row to be 0-based 
    return int(column) - 1, convert_row_numerical(row) # col, row


def get_all_tiles_position_snake(tile_names):

    all_tiles = []
    
    for name in tile_names:
        
        position = obtain_tile_position(name)
        
        assert position not in all_tiles, "Found duplicate tile with same position"
        
        all_tiles.append(position)
    
    return all_tiles


def obtain_direction_to_remove_overlap(tile_pos, last_col):

    tile_col, tile_row = tile_pos
    
    if tile_row == 0 and tile_col == 0:
        return [""]

    directions = []

    if tile_row > 0:
        directions.append("top")

    if tile_row % 2 == 0:
        if tile_col > 0:
            directions.append("left")
    else:
        if tile_col < last_col:
            directions.append("right")

    return directions
    

def get_all_tiles_directions_to_remove_overlap(tile_names: list[str]):
    all_tiles_positions = get_all_tiles_position_snake(tile_names)

    if not all_tiles_positions:
        return []

    last_col = max(col for col, _ in all_tiles_positions)

    all_tiles_directions = [obtain_direction_to_remove_overlap(tile_pos, last_col) for tile_pos in all_tiles_positions]

    return all_tiles_directions


def compute_overlap_pixels(overlap, height, width, rounding_mode = "nearest"):
    """
    rounding_mode: ceil, floor, nearest
    """

    pixels_to_remove_h = height * (overlap / 100)
    pixels_to_remove_w = width  * (overlap / 100)

    if rounding_mode == "ceil":

        pixels_to_remove_h = ceil(pixels_to_remove_h)
        pixels_to_remove_w = ceil(pixels_to_remove_w)

    elif rounding_mode == "floor":

        pixels_to_remove_h = floor(pixels_to_remove_h)
        pixels_to_remove_w = floor(pixels_to_remove_w)

    elif rounding_mode == "nearest":

        pixels_to_remove_h = round(pixels_to_remove_h)
        pixels_to_remove_w = round(pixels_to_remove_w)

    return pixels_to_remove_h, pixels_to_remove_w


def cut_overlap(image, pixels_to_remove_h, pixels_to_remove_w, directions_to_cut):
    if pixels_to_remove_h == 0 and pixels_to_remove_w == 0:
        return image

    idx = [slice(None)] * image.ndim

    for direction in directions_to_cut:
        if direction == "top" and pixels_to_remove_h > 0:
            idx[-2] = slice(pixels_to_remove_h, None)
        elif direction == "left" and pixels_to_remove_w > 0:
            idx[-1] = slice(pixels_to_remove_w, None)
        elif direction == "right" and pixels_to_remove_w > 0:
            idx[-1] = slice(None, -pixels_to_remove_w)

    return image[tuple(idx)]
