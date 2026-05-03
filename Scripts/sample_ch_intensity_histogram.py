from multiprocessing import Pool
from pathlib import Path

import numpy as np

def get_images_path(path):
    return

def read_image(image_path):
    return

IMAGE_FOLDER = ""
OUT_FOLDER = ""

NUM_PROCESSORS = 15

def compute_channel_bins(image_path):
    
    image = read_image(image_path)

    n_channels = image.shape[-3]

    bins = np.zeros(n_channels, 65535)

    for i in range(n_channels):

        channel = image[i]

        bins[i] = np.bincount(channel, minlength=65535)

    return bins



def main(image_folder,
         out_folder,
         num_processors):

    odirname = Path(out_folder)
    odirname.mkdir(parents=True, exist_ok=True)

    image_paths = get_images_path(image_folder)

    sample_bins = None

    with Pool(processes=num_processors, maxtasksperchild=1) as pool:
        
        bins = pool.imap(compute_channel_bins, zip(image_paths), chunksize=1)

        sample_bins = bins if sample_bins is None else sample_bins = bins + sample_bins


if __name__ == "__main__":
    main(IMAGE_FOLDER)
