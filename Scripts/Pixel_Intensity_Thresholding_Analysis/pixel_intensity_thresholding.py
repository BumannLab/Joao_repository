# TODO: Add CLI, use click

import time
import glob
from itertools import repeat
from multiprocessing import Pool

import tifffile as tiff
import numpy as np
from tqdm import tqdm
from skimage import measure


IMG_FOLDER = "U:/RG Bumann/ScientificData/Group/DataTansfer/New_mTB_Data_4Channel/MAX"  # "Joao_repository/Images/Cropped_Images_test/"
CHANNEL_OF_INTEREST = 2  # 1-based & Channel of Interest == coi
PIXEL_INTENSITY_THRESHOLD = 1000
CLUSTER_SIZE_THRESHOLD = 5
PIXEL_REPORT_OUT_FNAME = (
    "Joao_repository/Results/Pixel_Intensity_Thresholding/pixels_intensity_over1000.tsv"
)
CLUSTER_REPORT_OUT_FNAME = "Joao_repository/Results/Pixel_Intensity_Thresholding/clusters_size5plus_intensity_over1000.tsv"
NUM_PROCESSERS = 10


def get_images_path(folder: str):
    """
    TODO
    """
    return glob.glob(f"{folder}/*.tif")


def build_cluster_report_header(num_channels: int):
    """
    TODO
    """
    ch_cols = [f"Channel_{i+1}_Summed_Intensity" for i in range(num_channels)]
    return "Image\tArea\tX_position\tY_position\t" + "\t".join(ch_cols) + "\n"


def build_pixel_report_header(num_channels: int, channel_of_interest: int):
    """
    TODO
    """
    ch_cols = [f"Channel_{i+1}" for i in range(num_channels)]
    ratio_cols = [
        f"Channel_{i+1}/Channel_{channel_of_interest+1}"
        for i in range(num_channels)
        if i != channel_of_interest
    ]
    return "Image\tX_position\tY_position\t" + "\t".join(ch_cols + ratio_cols) + "\n"


def find_pixel_clusters(
    image: np.ndarray, channel_of_interest: int, intensity_threshold: int = 1000
):
    """
    TODO
    """
    coi_image = image[channel_of_interest]
    mask = coi_image >= intensity_threshold
    labeled = measure.label(mask, connectivity=2)
    clusters_properties = measure.regionprops(labeled, intensity_image=coi_image)
    return clusters_properties


def filter_clusters(all_clusters, cluster_size: int):
    filtered_clusters = [
        cluster for cluster in all_clusters if cluster.area > cluster_size
    ]
    return filtered_clusters


def get_channel_intensity(image: np.ndarray, pixel_coords):
    """
    TODO
    """
    x_indices = pixel_coords[:, 0]
    y_indices = pixel_coords[:, 1]

    pixel_intensities = image[:, x_indices, y_indices].T
    return pixel_intensities


def get_summed_intensities(image, cluster_coords):
    intensities = get_channel_intensity(image, cluster_coords)

    sum_intensities = np.sum(intensities, axis=0)

    return sum_intensities.astype(str)


def build_cluster_report(image_name, image, cluster_report):
    file_content = [0] * len(cluster_report)

    for i in range(len(cluster_report)):
        summed_intensities = get_summed_intensities(image, cluster_report[i].coords)

        y, x = cluster_report[i].centroid

        row = [
            image_name,
            str(cluster_report[i].area),
            str(x),
            str(y),
            *summed_intensities,
        ]

        file_content[i] = "\t".join(row)

    return "\n".join(file_content) + "\n"


def get_all_pixel_positions(pixel_clusters):
    pixel_coords = np.vstack([cluster.coords for cluster in pixel_clusters])
    return pixel_coords


def normalize_by_coi_intensity(
    pixel_intensities: np.ndarray, num_channels: int, channel_of_interest: int
):
    """
    TODO
    """
    coi_intensity = pixel_intensities[:, [channel_of_interest]]

    non_coi_channels = [
        channel for channel in range(num_channels) if channel != channel_of_interest
    ]
    non_coi_channels_intensity = pixel_intensities[:, non_coi_channels]

    normalized_ratios = non_coi_channels_intensity / coi_intensity

    return normalized_ratios.astype(np.float32)


def build_pixel_report_tsv(
    image_name: str, pixel_coords, pixel_intensity, pixel_ratio
): 
    """
    TODO
    """
    file_content = [0] * len(pixel_coords)

    for i in range(len(pixel_coords)):

        pos_y = pixel_coords[:, 0][i]
        pos_x = pixel_coords[:, 1][i]

        raw_intensity = pixel_intensity[i].astype(str)
        normalized_intensity = pixel_ratio[i].astype(str)

        row = [
            image_name,
            str(pos_x),
            str(pos_y),
            *raw_intensity,
            *normalized_intensity,
        ]

        file_content[i] = "\t".join(row)

    return "\n".join(file_content) + "\n"


def analyse_clusters_and_pixels(args):
        image_path, channel_of_interest, pixel_intensity_threshold, cluster_size_threshold = args
        
        img = tiff.imread(image_path)
        img_name = image_path.split("\\")[-1].split(".")[0]

        num_channels = img.shape[0]

        clusters_high_intensity = find_pixel_clusters(
            img, channel_of_interest, pixel_intensity_threshold
        )
        filtered_clusters_high_intensity = filter_clusters(
            clusters_high_intensity, cluster_size_threshold
        )

        if not filtered_clusters_high_intensity:
            return []

        clusters_info = build_cluster_report(
            img_name, img, filtered_clusters_high_intensity
        )

        pixel_coords = get_all_pixel_positions(filtered_clusters_high_intensity)
        pixel_intensity = get_channel_intensity(img, pixel_coords)
        pixel_ratios = normalize_by_coi_intensity(
            pixel_intensity, num_channels, channel_of_interest
        )
        pixel_info = build_pixel_report_tsv(
            img_name, pixel_coords, pixel_intensity, pixel_ratios
        )

        return clusters_info, pixel_info, num_channels


def main(
    image_folder,
    channel_of_interest,
    pixel_intensity_threshold,
    cluster_size_threshold,
    pixel_report_fname,
    cluster_report_fname,
    num_processers
):
    has_header = False
    channel_of_interest = channel_of_interest - 1

    images_path = get_images_path(image_folder)

    if num_processers == None:
        num_processers = 1

    with open(pixel_report_fname, "w") as pixel_out, open(
        cluster_report_fname, "w"
    ) as cluster_out:
        
        with Pool(processes=num_processers) as pool:

            args_iter = zip(
                images_path,
                repeat(channel_of_interest),
                repeat(pixel_intensity_threshold),
                repeat(cluster_size_threshold)
            )

            for result in tqdm(pool.imap_unordered(analyse_clusters_and_pixels,
                            args_iter), total=len(images_path)):
                
                if not result:
                    continue
                
                clusters_info, pixels_info, num_channels = result

                if not has_header:

                    pixel_header = build_pixel_report_header(
                        num_channels, channel_of_interest
                    )
                    cluster_header = build_cluster_report_header(num_channels)

                    pixel_out.write(pixel_header)
                    cluster_out.write(cluster_header)

                    has_header = True

                cluster_out.write(clusters_info)
                pixel_out.write(pixels_info)


if __name__ == "__main__":
    start_time = time.time()

    main(
        IMG_FOLDER,
        CHANNEL_OF_INTEREST,
        PIXEL_INTENSITY_THRESHOLD,
        CLUSTER_SIZE_THRESHOLD,
        PIXEL_REPORT_OUT_FNAME,
        CLUSTER_REPORT_OUT_FNAME,
        NUM_PROCESSERS
    )

    print(f"Finished in {time.time() - start_time} seconds")
