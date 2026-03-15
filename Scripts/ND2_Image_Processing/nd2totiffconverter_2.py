from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff


import time
from multiprocessing import Pool
from pathlib import Path
from itertools import repeat

import nd2
import tifffile as tiff
import numpy as np
from tqdm import tqdm

from utils import get_images_path, read_image, AXES
from ND2_Image_Processing.nd2_metadata_retrieval import *


IMAGE_DIR = "/data/joao/mTBData/B2022.10246/SYBRGold_CD68AL555_LAMAL647_DAPI"
OUT_DIR = ""
NUM_PROCESSORS = 24


def nd2totiff_converter(args_iter):
        image_path, odirname = args_iter
        
        image, metadata = nd2.imread(image_path, inc_metadata=True)

        axes = AXES[-image.ndim:]

        channel_luts = extract_channel_colors_as_luts(metadata)
        voxel_size = extract_voxel_size(metadata)
        resolution = get_resolution(voxel_size)

        out_metadata = {
             "LUTs":channel_luts,
             "mode": "color",
             "axes": axes,
             "spacing": voxel_size["PhysicalSizeZ"],
             "unit": "micron"
        }

        out_path = f"{odirname}/{image_path.stem}.tif"

        tiff.imwrite(
            out_path,
            image,
            imagej=True,
            metadata=out_metadata,
            resolution=resolution
            )


def main(image_dir, odirname, num_processers: int = 1):

    odirname = Path(odirname)
    odirname.mkdir(exist_ok=True)

    images_path = get_images_path(image_dir)
    
    with Pool(processes=num_processers) as pool:

        args_iter = zip(
            images_path,
            repeat(odirname)
        )

        for _ in tqdm(pool.imap(nd2totiff_converter,
                        args_iter), total=len(images_path)):
        
            continue
             
             
if __name__ == "__main__":
    start_time = time.time()

    main(IMAGE_DIR, OUT_DIR, NUM_PROCESSORS)

    print(f"Finished in {time.time() - start_time} seconds")