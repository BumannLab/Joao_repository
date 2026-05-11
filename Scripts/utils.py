from glob import glob
import re
from pathlib import Path
from math import ceil, floor
import re
from enum import StrEnum

import tifffile as tiff
import nd2
import numpy as np

NUM_CHANNELS = 4
IMAGE_WIDTH = 2304
IMAGE_HEIGHT = 2304

AXES = "TZCYX"

class RoundingMode(StrEnum):
    NEAREST = "nearest"
    CEIL = "ceil"
    FLOOR = "floor"


class Tile:
    def __init__(self, tile_path, overlap, direction_to_remove_overlap = [], rounding_mode = "nearest"):
        self.tile_path = tile_path
        self.direction_rm_overlap = direction_to_remove_overlap
        self.overlap = overlap # Overlap is from 0 - 100 not from 0 - 1
        self.rounding_mode = RoundingMode(rounding_mode)


    def compute_overlap_pixels(self, height, width):
        """
        rounding_mode: ceil, floor, nearest
        """

        pixels_to_remove_h = height * (self.overlap / 100)
        pixels_to_remove_w = width  * (self.overlap / 100)

        if self.rounding_mode == "ceil":

            pixels_to_remove_h = ceil(pixels_to_remove_h)
            pixels_to_remove_w = ceil(pixels_to_remove_w)

        elif self.rounding_mode == "floor":

            pixels_to_remove_h = floor(pixels_to_remove_h)
            pixels_to_remove_w = floor(pixels_to_remove_w)

        elif self.rounding_mode == "nearest":

            pixels_to_remove_h = round(pixels_to_remove_h)
            pixels_to_remove_w = round(pixels_to_remove_w)

        return pixels_to_remove_h, pixels_to_remove_w


    def remove_overlap(self, tile): 

        if self.overlap == 0:
            return tile

        idx = [slice(None)] * tile.ndim

        height, width = tile.shape[-2:]

        pixels_to_remove_h, pixels_to_remove_w = self.compute_overlap_pixels(height, width)

        for direction in self.direction_rm_overlap:

            if direction == "top":
                idx[-2] = slice(pixels_to_remove_h, None)
            
            elif direction == "left":
                idx[-1] = slice(pixels_to_remove_w, None)
            
            elif direction == "right":
                idx[-1] = slice(-pixels_to_remove_w)
        
        return self.tile[tuple(idx)]


def get_images_path(image_dir, *formats):
    
    if not formats:
        formats = (".tif", ".tiff", ".nd2")

    images_path = []    
    
    for format in formats:

        paths = Path(image_dir).glob(f"*{format}")

        images_path.extend([Path(p.as_posix()) for p in paths])

    return sorted(images_path)


def read_tiff_image(path, inc_metadata=False):

    if inc_metadata:

        with tiff.TiffFile(path) as img:
            return img.asarray(), img.imagej_metadata
    
    return tiff.imread(path)


def read_nd2_image(path, inc_metadata=False):
    
    if inc_metadata:

        with nd2.ND2File(path) as img:
            return img.asarray(), img.metadata    
    
    return nd2.imread(path)


def read_image(path, inc_metadata = False):

    if path.suffix == ".nd2":
        try: 
            tpl = read_nd2_image(path, inc_metadata)
        
        except ValueError as e: 
            print(f"There was a problem loading {path}.\nThe error in question was {e}.") # Return error
            return
            # tpl = nd2.rescue_nd2(path, (IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS), 'uint16')
            # TODO: add metadata for rescue files if possible for nd2 files
        
    elif path.suffix in [".tif", ".tiff"]:
        try: 
            tpl = read_tiff_image(path, inc_metadata)
        
        except ValueError as e: 
            print(f"There was a problem loading {path}.\nThe error in question was {e}.") # Return error
            return
    
    else:
        print(f"File extension is neither a tif neither an nd2.")
        return
        
    return tpl


def slice_channel_threshold(threshold_conditions, channel_markers = None):

    channel_threshold_conditions = []
    
    for channel_threshold in threshold_conditions.split(","):

        try: channel, operator, threshold = re.split(r'(<=|>=|<|>)', channel_threshold)
        except: raise ValueError(f"The operator in \"{channel_threshold}\" is not valid.")

        if channel_markers:
            conditions = (channel_markers.index(channel), operator, int(threshold))  # channel will go from 1- to 0-based

        else:
            conditions = (channel, operator, int(threshold))
            
        channel_threshold_conditions.append(conditions)

    return channel_threshold_conditions


def create_luts(rgb_values):
    
    val_range = np.arange(256, dtype=np.uint8)
    lut = np.zeros((3, 256), dtype=np.uint8)
    lut[0, :] = (val_range.astype(np.uint16) * rgb_values[0] // 255).astype(np.uint8)
    lut[1, :] = (val_range.astype(np.uint16) * rgb_values[1] // 255).astype(np.uint8)
    lut[2, :] = (val_range.astype(np.uint16) * rgb_values[2] // 255).astype(np.uint8)

    return lut
