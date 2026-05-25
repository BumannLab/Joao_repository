import time
from multiprocessing import Pool
from pathlib import Path
from itertools import repeat

import tifffile as tiff
import numpy as np
from tqdm import tqdm

from utils import get_images_path, read_image
from ND2_Image_Processing.nd2_metadata_retrieval import *

# TODO: Load metadata for one image and recycle it for all of the rest


IMAGE_DIR = "/data/joao/mTBData/B2020.20260_scanned_after_1_month/SYBRGold_CD68_LAM_DAPI_after_1_month"
NUM_PROCESSORS = 15


def write_tiff(path, image, metadata, resolution = None):
    tiff.imwrite(
        path,
        image,
        imagej=True,
        metadata=metadata,
        resolution=resolution,
        )


def analyse_clusters_and_pixels(args_iter):
        
        image_path, odirname = args_iter
        
        tpl = read_image(image_path, inc_metadata=True)

        if tpl is None:
            return 
        
        image, metadata = tpl

        image = np.max(image, axis=0)

        out_metadata = {
        "mode": "color",
        "axes": "CYX",
        "unit": "micron"
        }

        out_path = f"{odirname}/MAX_{image_path.stem}.tif"

        if metadata is None: # The file was corrupted so metadata will not be retrieved
            write_tiff(out_path, image, out_metadata)
            return 
        
        channel_luts = extract_channel_colors_as_luts(metadata)
        voxel_size = extract_voxel_size(metadata)
        resolution = get_resolution(voxel_size)

        out_metadata["spacing"] = voxel_size["PhysicalSizeZ"]
        out_metadata["LUTs"] = channel_luts

        write_tiff(out_path, image, out_metadata, resolution)


def main(image_dir, num_processers: int = 1):
    image_dir = Path(image_dir)

    odirname = image_dir / "MAX"
    odirname.mkdir(exist_ok=True)

    images_path = get_images_path(image_dir)
    
    with Pool(processes=num_processers, maxtasksperchild=1) as pool:

        args_iter = zip(
            images_path,
            repeat(odirname)
        )

        for _ in tqdm(pool.imap(analyse_clusters_and_pixels,
                        args_iter, chunksize=1), total=len(images_path)):
        
            continue
             
             
if __name__ == "__main__":
    start_time = time.time()

    main(IMAGE_DIR, NUM_PROCESSORS)

    print(f"Finished in {time.time() - start_time} seconds")
