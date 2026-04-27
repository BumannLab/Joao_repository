from collections import defaultdict
from pathlib import Path

import numpy as np

from utils import get_images_path, read_image


def read_input(input_path: Path, channels_to_load: list | None = None):

    _input = read_image(input_path)

    if channels_to_load:
        _input = np.take(_input, indices = channels_to_load, axis = -3)

    return _input


def read_mask(mask_path: Path):

    mask = read_image(mask_path).astype(np.uint8)

    mask[mask > 0] = 255

    mask = np.expand_dims(mask, axis=-3)

    return mask


def load_images(input_dir: Path, mask_dir: Path, channels_to_load: list | None = None):

    input_paths = get_images_path(input_dir, ".tif", ".tiff")
    
    mask_paths = get_images_path(mask_dir, ".tif", ".tiff")

    assert len(input_paths) == len(mask_paths)
    assert [input_paths[i].stem == mask_paths[i].stem for i in range(len(input_paths))]

    grouped_pairs = defaultdict(list)

    for input_path, mask_path in zip(input_paths, mask_paths):
        file_name = input_path.stem 
        label = 1 if "positive" in input_path.stem.lower() else 0

        input_image = read_input(input_path, channels_to_load)
        mask_image = read_mask(mask_path)

        grouped_pairs[label].append((file_name, input_image, mask_image))

    return grouped_pairs


if __name__ == "__main__":
    x_dir = "/data/joao/Materials/Joao_repository/NN_Dataset/DATASET_COMPARISON/3D_comparison/Dataset_threshold_800/inputs"
    y_dir = "/data/joao/Materials/Joao_repository/NN_Dataset/DATASET_COMPARISON/3D_comparison/Dataset_threshold_800/masks"

    print(load_images(x_dir, y_dir, [0, 3]))
