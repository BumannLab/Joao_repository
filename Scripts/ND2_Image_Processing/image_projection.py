import time
from multiprocessing import Pool
from pathlib import Path
from itertools import repeat

import tifffile as tiff
import numpy as np
from tqdm import tqdm
import click

from utils import get_images_path, read_image, AXES
from ND2_Image_Processing.nd2_metadata_retrieval import *

# TODO: Load metadata for one image and recycle it for all of the rest
# TODO: Check whether shape is helpful to record
# TODO: Handle missing metadata extraction
# TODO: can rescue nd2 file work when the given number of channels is not given? Get shape and pass it to read_file


IMAGE_DIR = "/home/Test_augmentations"
NUM_PROCESSORS = 15


def write_tiff(path, image, metadata = None):

    resolution = metadata.pop("resolution", None) if metadata else None

    tiff.imwrite(
        path,
        image,
        imagej=True,
        metadata=metadata,
        resolution=resolution,
        )


def analyse_clusters_and_pixels(args_iter):
        
        image_path, metadata, odirname = args_iter
        
        image = read_image(image_path)

        if image is None:
            print(f"Got an empty image in {image_path}")
            return 
        
        image = np.max(image, axis=0)

        metadata["axes"] = AXES[-image.ndim:]

        out_path = f"{odirname}/MAX_{image_path.stem}.tif"
        
        write_tiff(out_path, image, metadata)
        


<<<<<<< Updated upstream
        write_tiff(out_path, image, out_metadata, resolution)        
     
=======
def get_metadata(images_path):

    out_metadata = {
        "mode": "color",
        "unit": "micron" # TODO: Is this always micron? Can i extract it from the image itself?
        }

    for image_path in images_path:
        tpl = read_image(image_path, inc_metadata=True)
        
        if tpl is not None:
        
            _, metadata = tpl

            if metadata is not None:
                channel_luts = extract_channel_colors_as_luts(metadata)
                voxel_size = extract_voxel_size(metadata)
                resolution = get_resolution(voxel_size)

                out_metadata["spacing"] = voxel_size["PhysicalSizeZ"]
                out_metadata["LUTs"] = channel_luts
                out_metadata["resolution"] = resolution
            
            else: 
                continue

            return out_metadata
        
    print("Warning: no valid image found, metadata will be incomplete")
    
    return out_metadata

>>>>>>> Stashed changes

@click.command()
@click.option("--tiles_dir", help="Folder containing the raw nd2 files")
@click.option("--threads", type=int, default=1, help="Number of threads")
def main(tiles_dir, threads):
    
    tiles_dir = Path(tiles_dir)

    odirname = tiles_dir / "MAX"
    odirname.mkdir(exist_ok=True)

<<<<<<< Updated upstream
    images_path = get_images_path(tiles_dir)
=======
    images_path = get_images_path(image_dir)

    metadata = get_metadata(images_path)
>>>>>>> Stashed changes
    
    with Pool(processes=threads, maxtasksperchild=1) as pool:

        args_iter = zip(
            images_path,
            repeat(metadata),
            repeat(odirname),
        )

        for _ in tqdm(pool.imap(analyse_clusters_and_pixels,
                        args_iter, chunksize=1), total=len(images_path)):
        
            continue
             
             
if __name__ == "__main__":
    start_time = time.time()

    main()

    print(f"Finished in {time.time() - start_time} seconds")
