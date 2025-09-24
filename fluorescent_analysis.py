import time
import tifffile as tiff
import glob
import numpy as np
from tqdm import tqdm # NOTE: Should I keep this?

IMG_FOLDER = "U:/RG Bumann/ScientificData/Group/DataTansfer/New_mTB_Data_4Channel/MAX"
CHANNEL_OF_INTEREST = 2 # 1-based & Channel of Interest == coi
THRESHOLD = 2000
OUT_FNAME = "Fluorescent_thresholding_over_2000_updated_normalized.tsv"


def get_images_path(folder: str):
    """
    TODO
    """
    return (glob.glob(f"{folder}/*.tif"))


def image_iter(images_path: list[str]):
    """
    TODO
    """
    for path in images_path:
        image = tiff.imread(path)
        yield image, path


def reorder_channels(image: np.ndarray, channel_of_interest: int):
    """
    TODO
    """
    order = [channel_of_interest - 1, *[i for i in range(image.shape[0]) if i != channel_of_interest - 1]]
    image_reordered = image[order, ...]
    return image_reordered, order


def get_thresholded_coi_indices(image: np.ndarray, intensity_threshold: int = 1000): # NOTE: 0-based basically channel 1 is 0 and channel 2 is 1
    """
    TODO
    """
    threshold_indices  = np.where(image[0] >= intensity_threshold)
    return threshold_indices 


def get_channel_intensity(image: np.ndarray, filtered_channel_idx: tuple[np.ndarray, np.ndarray]):
    """
    TODO
    """
    x_indices, y_indices = filtered_channel_idx
    pixel_intensities = image[:, x_indices, y_indices].T 
    return pixel_intensities


def normalize_by_coi_intensity(pixel_intensities: np.ndarray):
    """
    TODO
    """
    non_coi_channels = pixel_intensities[:, 1:]
    coi_intensity = pixel_intensities[:, [0]]
    normalized_ratios = non_coi_channels / coi_intensity
    return normalized_ratios


def build_header(channel_order: list[int], channel_of_interest: int): 
    """
    TODO
    """
    ch_cols = [f"Channel_{i+1}" for i in channel_order]
    ratio_cols = [f"Channel_{i+1}/Channel_{channel_of_interest}" for i in channel_order[1:]]
    return "Image\tX_position\tY_position\t" + "\t".join(ch_cols + ratio_cols) + "\n"


def build_pixel_report_tsv(image_name: str, raw_pixel_intensities: np.ndarray, pixel_coords: tuple[np.ndarray, np.ndarray], normalized_intensities: np.ndarray): # Tested was done in 33m
    """
    TODO
    """
    file_content = [""] * len(raw_pixel_intensities)

    for i in range(len(raw_pixel_intensities)):

        pos_x, pos_y = pixel_coords[1][i], pixel_coords[0][i]

        raw_intensity  = raw_pixel_intensities[i]
        normalized_intensity = normalized_intensities[i]

        row = [image_name, str(pos_x), str(pos_y), *raw_intensity, *normalized_intensity]

        file_content[i] = "\t".join(row)

    return "\n".join(file_content) + "\n"



def main(image_folder, channel_of_interest, threshold, out_fname):
    has_header = False

    images_path = get_images_path(image_folder)
    images = image_iter(images_path)

    with open(out_fname, "w") as f:

        for img, img_name in tqdm(images, total=len(images_path)):
            img_name = img_name.split("\\")[-1].split(".")[0]
            
            img_reordered, order = reorder_channels(img, channel_of_interest)

            if not has_header:
                header = build_header(order, channel_of_interest)
                f.write(header)
                has_header = True

            threshold_indices = get_thresholded_coi_indices(img_reordered, threshold)  # if this is empty will raise error, so make a condition
            pixel_intensities = get_channel_intensity(img_reordered, threshold_indices)
            pixel_ratios = normalize_by_coi_intensity(pixel_intensities)

            file_content = build_pixel_report_tsv(img_name, pixel_intensities, threshold_indices, pixel_ratios)
            f.write(file_content)


if __name__ == "__main__":
    start_time = time.time()
    main(IMG_FOLDER, CHANNEL_OF_INTEREST, THRESHOLD, OUT_FNAME)      
    print(f"Finished in {time.time() - start_time} seconds")


