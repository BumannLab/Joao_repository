import glob
import re

import tifffile as tiff
import nd2

def get_images_path(image_dir, *formats):
    
    images_path = []
    
    for format in formats:
        images_path.extend(glob.glob(f"{image_dir}/*{format}"))

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

read_nd2_image("U:/RG Bumann/ScientificData/Users/João_rodrig0012/João_fromMicroscopy/Slide1/Captured Spinning Disk (1)DAPI SD.nd2", inc_metadata=True)

def slice_channel_threshold(threshold_conditions):

    channel_threshold_conditions = []
    
    for channel_threshold in threshold_conditions.split(","):

        try: channel, operator, threshold = re.split(r'(<=|>=|<|>)', channel_threshold)
        except: raise ValueError(f"The operator in \"{channel_threshold}\" is not valid.")
        
        channel_threshold_conditions.append((int(channel) - 1, operator, int(threshold))) # channel will go from 1- to 0-based

    return channel_threshold_conditions