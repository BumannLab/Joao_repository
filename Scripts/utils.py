import glob
import re

import tifffile as tiff

def get_images_path(image_dir, *formats):
    
    images_path = []
    
    for format in formats:
        images_path.extend(glob.glob(f"{image_dir}/*{format}"))

    return sorted(images_path)


def read_tiff_image(path, inc_metada=False):

    if inc_metada:

        with tiff.TiffFile(path) as tif:
            return tif.asarray(), tif.imagej_metadata
        
    return tiff.imread(path)


def slice_channel_threshold(threshold_conditions):

    channel_threshold_conditions = []
    
    for channel_threshold in threshold_conditions.split(","):

        try: channel, operator, threshold = re.split(r'(<=|>=|<|>)', channel_threshold)
        except: raise ValueError(f"The operator in \"{channel_threshold}\" is not valid.")
        
        channel_threshold_conditions.append((int(channel) - 1, operator, int(threshold))) # channel will go from 1- to 0-based

    return channel_threshold_conditions