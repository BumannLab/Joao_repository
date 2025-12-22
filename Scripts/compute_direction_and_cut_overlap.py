from math import ceil, floor
import re


def convert_row_numerical(row: str):
    row_numerical = 0

    for char in row:
        row_numerical = (row_numerical * 26) + (ord(char) - ord("A"))
    
    return row_numerical


def obtain_tile_position(tile_name):

    tile_info = tile_name.split("_")[-1]

    match = re.search(r'([A-Za-z]+)(\d+)', tile_info)
    row, column = match.groups()

    # Convert column to numerical since they are like this: "AA" -> 0, "BA" -> 26
    # and remove 1 from row to be 0-based 
    return int(column) - 1, convert_row_numerical(row)


def get_all_tiles_position_corrected(tile_names):
    all_tiles_positions = list()

    row_count = 0
    column_count = 0

    previous_row = 0
    previous_column = 0
    
    for name in tile_names:
        column, row = obtain_tile_position(name)

        if row != previous_row:
            row_count += 1

        if column > previous_column:
            column_count += 1
        
        elif column < previous_column:
            column_count -= 1

        all_tiles_positions.append((column_count, row_count))

        previous_column = column
        previous_row = row

    return all_tiles_positions

        
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
    all_tiles_positions = get_all_tiles_position_corrected(tile_names)

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

    idx = [slice(None)] * image.ndim

    for direction in directions_to_cut:

        if direction == "top":
            idx[-2] = slice(pixels_to_remove_h, None)
        
        elif direction == "left":
            idx[-1] = slice(pixels_to_remove_w, None)
        
        elif direction == "right":
            idx[-1] = slice(-pixels_to_remove_w)
    
    return image[tuple(idx)]
