import time

start_time = time.time()

import tifffile as tiff
import glob
import numpy as np
from tqdm import tqdm # NOTE: Should I keep this?


IMG_FOLDER = "U:/RG Bumann/ScientificData/Group/DataTansfer/New_mTB_Data_4Channel/MAX"
CHANNEL_OF_INTEREST = 2 # 1-based
THRESHOLD = 2000
OUT_FNAME = "test.tsv"


def get_images_path(folder: str):
    """
    Input: TODO
    Output: TODO
    Process: TODO
    """
    return (glob.glob(f"{folder}/*.tif"))


def image_iter(images_path: list[str]):
    """
    Input: TODO
    Output: TODO
    Process: TODO
    """
    for path in images_path:
        image = tiff.imread(path)
        yield image, path


def reorder_channels(image: np.ndarray, channel_of_interest: int):
    """
    Input: TODO
    Output: TODO
    Process: TODO
    """
    order = [channel_of_interest - 1, *[i for i in range(image.shape[0]) if i != channel_of_interest - 1]]
    image_reordered = image[order, ...]
    return image_reordered, order


def thresholding_channel_of_interest(image: np.ndarray, threshold: int = 100): # NOTE: 0-based basically channel 1 is 0 and channel 2 is 1
    """
    Input: TODO
    Output: TODO
    Process: TODO
    """
    filtered_channel_idx = np.where(image[0] >= threshold)
    return filtered_channel_idx


def pixel_intensity_across_channels(image: np.ndarray, filtered_channel_idx: tuple[np.ndarray, np.ndarray]):
    """
    Input: TODO
    Output: TODO
    Process: TODO
    """
    x_indices, y_indices = filtered_channel_idx
    pixel_intensities = image[:, x_indices, y_indices].T 
    return pixel_intensities


def normalize_intensity(pixel_intesities: np.ndarray):
    """
    Input: TODO
    Output: TODO
    Process: TODO
    """
    normalized_pixel_intesities = pixel_intesities[1:] / pixel_intesities[0]
    return normalized_pixel_intesities


def build_header(order: list[int]):
    """
    Input: TODO
    Output: TODO
    Process: TODO
    """
    ch_cols = [f"Channel_{i+1}" for i in order]
    ratio_cols = [f"Channel_1/Channel_{i+1}" for i in order[1:]]
    return "Image\tX_position\tY_position\t" + "\t".join(ch_cols + ratio_cols) + "\n"


def build_file_content(img_name: str, pixel_intensity_across_channels: np.ndarray, pixel_positions: tuple[np.ndarray, np.ndarray]):
    """
    Input: TODO
    Output: TODO
    Process: TODO
    """
    file_content = [""] * len(pixel_intensity_across_channels)
    for idx in range(len(pixel_intensity_across_channels)):
        pos_x, pos_y = pixel_positions[1][idx], pixel_positions[0][idx]
        pixel_intesities = [str(intensity) for intensity in pixel_intensity_across_channels[idx]]
        normalized_pixel_intensities = [str(intensity / pixel_intensity_across_channels[idx][0]) for intensity in pixel_intensity_across_channels[idx][1:]]
        content = str(img_name) + "\t" + str(pos_x) + "\t" + str(pos_y) + "\t" + "\t".join( pixel_intesities + normalized_pixel_intensities)
        file_content[idx] = content
    return "\n".join(file_content) + "\n"


def main():
    has_header = False

    images_path = get_images_path(IMG_FOLDER)
    images = image_iter(images_path)

    with open(OUT_FNAME, "w") as f:

        for img, img_name in tqdm(images, total=len(images_path)):
            img_name = img_name.split("/")[-1].split(".")[0]
            
            img_reordered, order = reorder_channels(img, CHANNEL_OF_INTEREST)

            if not has_header:
                header = build_header(order)
                f.write(header)
                has_header = True

            threshold_pixel_pos = thresholding_channel_of_interest(img_reordered, THRESHOLD)
            pixel_intesities = pixel_intensity_across_channels(img_reordered, threshold_pixel_pos)

            file_content = build_file_content(img_name, pixel_intesities, threshold_pixel_pos)
            f.write(file_content)


if __name__ == "__main__":
    main()      

print(f"Finished in {time.time() - start_time}")