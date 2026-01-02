import time
from multiprocessing import Pool
from pathlib import Path
from itertools import repeat

import tifffile as tiff
import numpy as np
from tqdm import tqdm

from utils import get_images_path, read_nd2_image


IMAGE_DIR = "U:/RG Bumann/ScientificData/Users/João_rodrig0012"
ODIRNAME = "Joao_repository/New_Results/CD68AL555_SYBRGold_DAPI_CD68AL647/Pixel_Analysis"

METHOD = "MAX" # MAX | MEAN

NUM_PROCESSORS = 10


def apply_function_to_image(image, image_function):

    if image_function == "MAX":
        image = np.max(image, axis=0)

    elif image_function == "MEAN":
        image = np.mean(image, axis=0)

    return image


def analyse_clusters_and_pixels(args):
        image_path, method, odirname = args

        image_name = image_path.split("/")[-1].replace(".nd2", "")
        
        image, metadata, metadata_xml = read_nd2_image(image_path, inc_metadata=True)

        print(metadata_xml)

        print(metadata.channels[0].channel.color.as_abgr_u4())
        print(metadata.channels[0].channel.color.as_hex())
        print(metadata.channels[1].channel.color.as_hex())
        print(metadata.channels[2].channel.color.as_hex())
        print(metadata.channels[3].channel.color.as_hex())
        

        image = apply_function_to_image(image, method)

        out_path = f"{odirname}/{method}_{image_name}.tif"

        tiff.imwrite(
            out_path,
            image,
            imagej=True,
            metadata=metadata_xml
            )


def main(
    image_dir,
    method,
    odirname,
    num_processers: int = 1
):

    odirname = Path(odirname)
    odirname.mkdir(parents=True, exist_ok=True)

    method = method.upper()

    images_path = get_images_path(image_dir, ".nd2")
    
    with Pool(processes=num_processers) as pool:

        args_iter = zip(
            images_path,
            repeat(method),
            repeat(odirname)
        )

        for _ in tqdm(pool.imap(analyse_clusters_and_pixels,
                        args_iter), total=len(images_path)):
            continue
             
             
if __name__ == "__main__":
    start_time = time.time()

    main(
        IMAGE_DIR,
        METHOD,
        ODIRNAME,
        NUM_PROCESSORS
    )

    print(f"Finished in {time.time() - start_time} seconds")