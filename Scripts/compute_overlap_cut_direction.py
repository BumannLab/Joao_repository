from math import ceil
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

    all_tiles_positions = [obtain_tile_position(name) for name in tile_names]

    last_col = max(col for col, _ in all_tiles_positions)

    all_tiles_directions = [obtain_direction_to_remove_overlap(tile_pos, last_col) for tile_pos in all_tiles_positions]

    return all_tiles_directions

