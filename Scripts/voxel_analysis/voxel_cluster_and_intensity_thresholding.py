import glob
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
import re

import nd2
import numpy as np
from tqdm import tqdm
from skimage.measure import label, regionprops


Z_STACK_IMAGE_DIR = "M:/mTB Data/4CHANNELsTAINING_JoaoSamples" # "Joao_repository/Data/img_mTB_zstack_voxel_analysis"
ODIRNAME = "Joao_repository/Results/Voxel_Analysis/"
CHANNEL_THRESHOLD_CONDITIONS = "2>=1000,3<=250"
CLUSTER_SIZE_THRESHOLD = 5
NUM_PROCESSERS = 10


def get_images_path(image_dir):
    images_path = glob.glob(f"{image_dir}/*.nd2")
    return sorted(images_path)


def build_out_file_sufix(threshold_conditions):

    if ">=" in threshold_conditions:
        threshold_conditions = threshold_conditions.replace(">=", "_greater_equal_")
    if "<=" in threshold_conditions:
        threshold_conditions = threshold_conditions.replace("<=", "_less_equal_")
    if ">" in threshold_conditions:
        threshold_conditions = threshold_conditions.replace(">", "_greater_")
    if "<" in threshold_conditions:
        threshold_conditions = threshold_conditions.replace("<", "_less_")

    threshold_conditions = "channel" + threshold_conditions.replace(",", "_channel")

    return threshold_conditions


def slice_channel_threshold(threshold_conditions):

    channel_threshold_conditions = []
    
    for channel_threshold in threshold_conditions.split(","):

        try: channel, operator, threshold = re.split(r'(<=|>=|<|>)', channel_threshold)
        except: raise ValueError(f"The operator in \"{channel_threshold}\" is not valid.")
        
        channel_threshold_conditions.append((int(channel) - 1, operator, int(threshold))) # channel will go from 1- to 0-based

    return channel_threshold_conditions


def voxel_channel_intensity_threshold(image, channel, operator, threshold):

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

    for channel_threshold in thresholds:

        channel, operator, threshold = channel_threshold

        channel_mask = voxel_channel_intensity_threshold(
            image,
            channel,
            operator,
            threshold 
        )

        image_masks.append(channel_mask)

    final_mask = np.logical_and.reduce(image_masks)

    return final_mask


def find_voxel_clusters(intensity_thresholded_mask):

    mask_label = label(intensity_thresholded_mask, connectivity = 3)
    clusters_properties = regionprops(mask_label)

    return clusters_properties


def filter_size_clusters(all_clusters_props, cluster_size_threshold: int = 5):
    filtered_clusters = [
        cluster for cluster in all_clusters_props if cluster.area > cluster_size_threshold
    ]
    return filtered_clusters


def build_file_report(image_name, filtered_clusters_props):

    num_clusters = len(filtered_clusters_props)

    file_content = [0] * num_clusters 

    for idx in range(num_clusters):

        z, y, x = filtered_clusters_props[idx].centroid

        content = "\t".join([
            image_name,
            str(filtered_clusters_props[idx].area),
            str(x), 
            str(y),
            str(z),
        ])

        file_content[idx] = content
        
    return "\n".join(file_content) + "\n"


def build_voxel_clusters(args):

    image_path, channel_threshold_conditions, cluster_size_threshold = args

    img_name = image_path.split("\\")[-1].split(".")[0]

    image = nd2.imread(image_path)

    final_thresholded_mask = multi_channel_threshold_mask(image, channel_threshold_conditions)

    all_clusters_props = find_voxel_clusters(final_thresholded_mask)

    filtered_clusters_props = filter_size_clusters(all_clusters_props, cluster_size_threshold)

    if not filtered_clusters_props:
        return []

    voxel_cluster_info = build_file_report(img_name, filtered_clusters_props)

    return voxel_cluster_info


def main(z_stack_image_dir, channel_threshold_conditions, cluster_size_threshold, odirname, num_processers: int = 1):

    odirname = Path(odirname)
    odirname.mkdir(parents= True, exist_ok=True)

    z_stack_images_path = get_images_path(z_stack_image_dir)

    sliced_channel_thresholds = slice_channel_threshold(channel_threshold_conditions)

    renamed_threshold_conditions = build_out_file_sufix(channel_threshold_conditions)

    voxel_cluster_report_fname = f"{odirname}/voxel_clusters_size_over{cluster_size_threshold}_{renamed_threshold_conditions}.tsv"

    with open(voxel_cluster_report_fname, "w") as voxel_cluster_out:

        voxel_cluster_out.write("Image\tVolume\tX\tY\tZ\n")
        
        with Pool(processes=num_processers) as pool:

            args_iter = zip(
                z_stack_images_path,
                repeat(sliced_channel_thresholds),
                repeat(cluster_size_threshold)
            )

            for voxel_cluster_info in tqdm(pool.imap(build_voxel_clusters,
                            args_iter), total=len(z_stack_images_path)):
                
                if not voxel_cluster_info:
                    continue

                voxel_cluster_out.write(voxel_cluster_info)


if __name__ == "__main__":

    main(Z_STACK_IMAGE_DIR,
        CHANNEL_THRESHOLD_CONDITIONS,
        CLUSTER_SIZE_THRESHOLD,
        ODIRNAME,
        NUM_PROCESSERS)
