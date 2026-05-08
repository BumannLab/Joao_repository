from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import get_images_path, read_image

IMAGE_FOLDER = "/data/joao/mTBData/B2020.20260/SYBRGold_CD68_LAM_DAPI_2"
OUT_FOLDER = "Test_script"

NUM_PROCESSORS = 20

def compute_channel_bins(image_path: Path) -> np.ndarray:
    
    image = read_image(image_path)

    n_channels = image.shape[-3]
    max_intensity = 65535 # for 16bit images

    bins = np.zeros((n_channels, max_intensity + 1))

    for i in range(n_channels):

        channel = image.take(i, axis=-3)

        bins[i] = np.bincount(channel.flatten(), minlength=max_intensity + 1)

    return bins


def write_bins(sample_bins: np.ndarray, odirname: Path) -> None:

    sample_bins = sample_bins.T

    np.savetxt(f"{odirname}/channel_pixel_intensities.csv", sample_bins, delimiter = ',')


def main(image_folder,
         out_folder,
         num_processors):

    odirname = Path(out_folder)
    odirname.mkdir(parents=True, exist_ok=True)

    image_paths = get_images_path(image_folder)

    sample_bins = None

    with Pool(processes=num_processors, maxtasksperchild=1) as pool:
        
        for bins in tqdm(
            (pool.imap(compute_channel_bins, image_paths, chunksize=1))
            , total=len(image_paths)):

            sample_bins = bins if sample_bins is None else bins + sample_bins

        write_bins(sample_bins, odirname)


if __name__ == "__main__":
    main(
        IMAGE_FOLDER,
        OUT_FOLDER,
        NUM_PROCESSORS=1
    )
