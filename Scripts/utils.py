from glob import glob
import re
from pathlib import Path

import tifffile as tiff
import nd2
import numpy as np

AXES = "TZCYX"


def get_images_path(image_dir, *formats):
    
    if not formats:
        formats = (".tif", ".nd2")

    images_path = []    
    
    for format in formats:

        paths = Path(image_dir).glob(f"*{format}")

        images_path.extend([Path(p.as_posix()) for p in paths])

    return sorted(images_path)


def _read_tiff_array(path):
    try:
        return tiff.imread(path)
    except AttributeError as error:
        if "newbyteorder" not in str(error):
            raise

        with tiff.TiffFile(path) as img:
            return img.series[0].asarray()


def read_tiff_image(path, inc_metadata=False):

    if inc_metadata:

        with tiff.TiffFile(path) as img:
            try:
                image = img.asarray()
            except AttributeError as error:
                if "newbyteorder" not in str(error):
                    raise

                image = img.series[0].asarray()

            return image, img.imagej_metadata
    
    return _read_tiff_array(path)


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


# path = "/Data_HDD/joao/mtb_data/SYBRGold_CD68_LAM_DAPI_22_04_26/0025_AA25.nd2"

# import nd2
# import numpy as np
# import matplotlib.pyplot as plt

# with open(path, 'rb') as fh:
#     frames = nd2.rescue_nd2(fh, (2304, 2304, 4), 'uint16')
#     ary = np.stack([frame for frame in frames])
#     print(ary.shape)

# ary_t = np.transpose(ary, (0, 3, 1, 2))

# max_ary = np.max(ary_t, axis=0)


# def normalize_image(image: np.ndarray) -> np.ndarray:
#     image = image.astype(np.float32)
#     min_value = image.min()
#     max_value = image.max()

#     if max_value == min_value:
#         return np.zeros_like(image, dtype=np.float32)

#     return (image - min_value) / (max_value - min_value)

# for i in range(max_ary.shape[0]):
#     normalized_image = normalize_image(max_ary[i])

#     plt.figure(figsize=(12,4))

#     plt.imshow(normalized_image, cmap="gray", vmin=0, vmax=1)
#     plt.show()
#     plt.close()
