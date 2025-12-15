from glob import glob
import re
from pathlib import Path

import tifffile as tiff
import nd2

def get_images_path(image_dir, *formats):
    
    images_path = []
    
    for format in formats:

        paths = Path(image_dir).glob(f"*{format}")

        images_path.extend([p.as_posix() for p in paths])

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


def slice_channel_threshold(threshold_conditions, channel_markers):

    channel_threshold_conditions = []
    
    for channel_threshold in threshold_conditions.split(","):

        try: channel, operator, threshold = re.split(r'(<=|>=|<|>)', channel_threshold)
        except: raise ValueError(f"The operator in \"{channel_threshold}\" is not valid.")

        channel_threshold_conditions.append((channel_markers.index(channel), operator, int(threshold))) # channel will go from 1- to 0-based

    return channel_threshold_conditions