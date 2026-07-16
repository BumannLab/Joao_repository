import time
from multiprocessing import Pool
from pathlib import Path
from itertools import repeat

import tifffile as tiff
import numpy as np
from tqdm import tqdm
import click

from utils import get_images_path, read_image
from ND2_Image_Processing.nd2_metadata_retrieval import *


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
     

@click.command()
@click.option("--tiles_dir", help="Folder containing the raw nd2 files")
@click.option("--threads", type=int, default=1, help="Number of threads")
def main(tiles_dir, threads):
    
    tiles_dir = Path(tiles_dir)

    odirname = tiles_dir / "MAX"
    odirname.mkdir(exist_ok=True)

    images_path = get_images_path(tiles_dir)
    
    with Pool(processes=threads, maxtasksperchild=1) as pool:

        args_iter = zip(
            images_path,
            repeat(odirname)
        )

        for _ in tqdm(pool.imap(analyse_clusters_and_pixels,
                        args_iter, chunksize=1), total=len(images_path)):
        
            continue
             
             
if __name__ == "__main__":
    start_time = time.time()

    main()

    print(f"Finished in {time.time() - start_time} seconds")
