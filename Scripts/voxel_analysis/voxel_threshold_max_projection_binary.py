import glob
from itertools import repeat
from multiprocessing import Pool
import os
from pathlib import Path
import re

import nd2
import numpy as np
from tqdm import tqdm
import tifffile as tiff


Z_STACK_IMAGE_DIR = "Joao_repository/Data/img_mTB_zstack_voxel_analysis"
MAX_PROJECTION_IMAGE_DIR = "U:/RG Bumann/ScientificData/Group/DataTansfer/New_mTB_Data_4Channel/MAX"
ODIRNAME = "Joao_repository/Results/Voxel_Analysis/Images"
THRESHOLD_CONDITIONS = "2>=1000,3<=500 2>=1000,3<=250"
NUM_PROCESSERS = 10


def get_images_path(image_dir):
    images_path = glob.glob(f"{image_dir}/*.nd2")
    return images_path


def slice_threshold(threshold_condition):
    channel, operator, threshold = re.split(r'(<=|>=|<|>)', threshold_condition)
    return int(channel), operator, int(threshold)


def voxel_channel_intensity_threshold(image, channel, threshold, operator):

    channel = channel - 1 # 1-based -> 0-based

    if operator == ">":
        voxel_coi_intensity_mask = image[:, channel] > threshold
    elif operator == ">=":
        voxel_coi_intensity_mask = image[:, channel] >= threshold
    elif operator == "<":
        voxel_coi_intensity_mask = image[:, channel] < threshold
    elif operator == "<=":
        voxel_coi_intensity_mask = image[:, channel] <= threshold

    return voxel_coi_intensity_mask


def multi_channel_threshold_mask(image, thresholds):
    image_masks = []

    for channel_threshold in thresholds.split(","):
        channel, operator, threshold = slice_threshold(channel_threshold)

        channel_mask = voxel_channel_intensity_threshold(
            image,
            channel,
            threshold,
            operator
        )

        image_masks.append(channel_mask)

    final_mask = np.logical_and.reduce(image_masks)

    return final_mask


def binary_max_projection(final_mask):

    max_projection = final_mask.any(axis=0)

    max_projection_binary = np.where(max_projection, 65535, 0)

    return max_projection_binary.astype(np.uint16)


def all_binary_max_projections(image, threshold_conditions):
    binary_max_projections = []

    for thresholds in threshold_conditions:

        final_mask = multi_channel_threshold_mask(image, thresholds)
        binary_max_proj = binary_max_projection(final_mask)
        binary_max_projections.append(binary_max_proj)

    return binary_max_projections


def get_max_projection_image(image_dir, img_basename):

    regexp = os.path.join(image_dir, f"*{img_basename}*")

    max_proj_img_path = glob.glob(regexp)

    max_proj_img = tiff.imread(max_proj_img_path[0])

    return max_proj_img


def combine_max_proj_and_binary(image, all_binary_max_projections):

    combined_img = np.vstack((image, all_binary_max_projections))

    return combined_img


def combine_and_save_max_proj_with_binary(args):
    image_path, max_projection_image_dir, channel_thresholds, odirname = args

    z_stack_image = nd2.imread(image_path)

    binary_max_projections = all_binary_max_projections(z_stack_image, channel_thresholds)

    base_img_name = os.path.basename(image_path).replace(".nd2", "")
    
    max_proj_image = get_max_projection_image(max_projection_image_dir, base_img_name)

    combined_image = combine_max_proj_and_binary(max_proj_image, binary_max_projections)

    out_img_name = f"{odirname}/{base_img_name}_with_binary_thresholded_channels.tif"

    tiff.imwrite(
        out_img_name,
        combined_image,
        imagej=True,
        metadata={"mode": "color", "axes": "CYX"},
    )
    

def main(z_stack_image_dir, max_projection_image_dir, threshold_conditions, odirname, num_processers: int = 1):

    odirname = Path(odirname)
    odirname.mkdir(parents= True, exist_ok=True)

    threshold_conditions = threshold_conditions.split()

    z_stack_images_path = get_images_path(z_stack_image_dir)

    with Pool(processes=num_processers) as pool:

        args_iter = zip(
            z_stack_images_path,
            repeat(max_projection_image_dir),
            repeat(threshold_conditions),
            repeat(odirname)
        )

        for _ in tqdm(pool.imap_unordered(combine_and_save_max_proj_with_binary, args_iter), total=len(z_stack_images_path)):
            continue

if __name__ == "__main__":

    main(Z_STACK_IMAGE_DIR,
        MAX_PROJECTION_IMAGE_DIR,
        THRESHOLD_CONDITIONS,
        ODIRNAME,
        NUM_PROCESSERS)
