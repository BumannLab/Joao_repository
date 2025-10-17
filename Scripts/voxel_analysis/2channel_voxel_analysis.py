from dataclasses import dataclass
import glob
from itertools import repeat
from multiprocessing import Pool
import os
from pathlib import Path

import nd2
import numpy as np
from tqdm import tqdm


@dataclass
class Voxel:
    channel_intensity: list[int, int]


IMAGE_DIR = "Joao_repository/img_mTB_zstack_voxel_analysis"
CHANNEL_OF_INTEREST_1 = 2 # 1-based
CHANNEL_OF_INTEREST_1_THRESHOLDING = 500
CHANNEL_OF_INTEREST_2 = 3 # 1-based
VOXEL_REPORT_FNAME = "voxel_2ch_intensities_7imgs.tsv"
NUM_PROCESSERS = 7


def get_images_path(image_dir):
    images_path = glob.glob(f"{image_dir}/*.nd2")
    return images_path


def voxel_threshold(image, threshold):

    thresholding_idx = np.where(image[:, 0] >= threshold) # you threshold only the first channel received as the input

    thresholded_voxel_intensities = image[thresholding_idx[0], :, thresholding_idx[1], thresholding_idx[2]]
    
    return thresholded_voxel_intensities


def voxel_analysis(args):
    image_path, channel_interest_1, channel_interest_1_thresholding, channel_interest_2 = args
    
    image = nd2.imread(image_path)

    channels_interest = [channel_interest_1 - 1, channel_interest_2 - 1]

    img_channels_interest = image[:,channels_interest,:,:]

    thresholded_voxel_intenstity = voxel_threshold(img_channels_interest, channel_interest_1_thresholding)

    voxel = Voxel(thresholded_voxel_intenstity)

    return voxel


def main(image_dir, channel_interest_1, channel_interest_1_thresholding, channel_interest_2, voxel_report_fname, num_processers: int = 1):

    odirname = Path(os.path.dirname(voxel_report_fname))
    odirname.mkdir(parents= True, exist_ok=True)

    images_path = get_images_path(image_dir)

    with open(voxel_report_fname, "w") as voxel_out:
            
            voxel_out.write("Channel_1\tChannel_2\n") # Write header
            
            with Pool(processes=num_processers) as pool:

                args_iter = zip(
                    images_path,
                    repeat(channel_interest_1),
                    repeat(channel_interest_1_thresholding),
                    repeat(channel_interest_2)
                )

                for voxel in tqdm(pool.imap_unordered(voxel_analysis, args_iter), total=len(images_path)):

                    if voxel.channel_intensity.size == 0:
                        continue

                    voxel_report = "\n".join(["\t".join(intensities.astype(str)) for intensities in voxel.channel_intensity])

                    voxel_out.write(voxel_report + "\n")


if __name__ == "__main__":
    main(IMAGE_DIR, CHANNEL_OF_INTEREST_1, CHANNEL_OF_INTEREST_1_THRESHOLDING, CHANNEL_OF_INTEREST_2, VOXEL_REPORT_FNAME, NUM_PROCESSERS)


