import glob
import numpy as np
from pathlib import Path

import tifffile as tiff
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from Unet import Unet

# TODO: implement class dataset method: Done
# TODO: Find a way to get the number of channels # how can we know the channel
# size in advance ? Should we load the first image and get the shape? Or use empty lists and then convert them to numpy arrays
# TODO: Add a way to handle if class is just 2 (background and label) in function merge_mask_label_images

IMG_DIR = "Joao_repository/Images/Cropped_Images_test"
MASK_IMG_DIR = "Joao_repository/Images/Masked_Images_test"
BATCH_SIZE = 0
LEARNING_RATE = 0
NUM_EPOCHS = 0


class BacteriaMaskPair(Dataset):
    def __init__(self, input_images: np.ndarray, label_images: np.ndarray, transform=None):
        self.input_images = input_images
        self.label_images = label_images
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx: int):
        input_image = self.input_images[idx]
        label_image = self.label_images[idx]

        input_tensor = torch.from_numpy(input_image / 65535)
        label_tensor = torch.from_numpy(label_image / 255)

        if self.transform:
            input_tensor = self.transform(input_tensor)
            label_tensor = self.transform(label_tensor)

        return input_tensor, label_tensor


def load_images(folder: Path):
    """
    Input: Path of the folder containing the images for training (Either
    to load bacteria or mask of the bacteria images)
    Output: Numpy array containing all the images
    Process: TODO
    """
    images_path = glob.glob(f"{folder}/*.tif")

    num_images = len(images_path)

    images = [0] * num_images # TODO: should i create empty lists and then convert to numpy array or read the first image to obtain the size in a function for example
    for img_idx in range(num_images):
        images[img_idx] = tiff.imread(images_path[img_idx])
    return images


def merge_mask_label_images(folder: Path, class_to_override_overlap: int = 1):
    mask_folders = glob.glob(f"{folder}/*")

    num_classes = len(mask_folders)

    mask_folder_images_path = [0] * num_classes # TODO: should i create empty lists and then convert to numpy array or read the first image to obtain the size in a function for example

    for class_num in range(num_classes):
        images = load_images(mask_folders[class_num])
        mask_folder_images_path[class_num] = images
    
    masked_images = np.stack(np.array(mask_folder_images_path))  # TODO: should i create empty lists and then convert to numpy array or read the first image to obtain the size in a function for example

    for i in range(num_classes):
        mask = masked_images[i]
        mask[mask == 255] = i + 1
        
    multi_class_mask = np.sum(masked_images, axis=0)

    multi_class_mask[multi_class_mask > num_classes] = class_to_override_overlap
    
    return multi_class_mask


def main():
    input_images = load_images(IMG_DIR)
    label_images = merge_mask_label_images(MASK_IMG_DIR)

    bacteria_dataset = BacteriaMaskPair(input_images, label_images)

    train, test = torch.utils.data.random_split(bacteria_dataset, [0.5, 0.5]) # this is just to practice on the small images, change the ratios to 80, 20 after

    dataloader = DataLoader(bacteria_dataset, batch_size=64, shuffle=True)

    unet_model = Unet(input_images) # Get shape and number of classes when getting the arrays return these variables ig

    optimizer = torch.optim.Adam()  # TODO: add parameters

    for batch_images, batch_labels in dataloader:
        print(batch_images)
        print(batch_labels)
        # print(f"Batch shape: {batch_images}, Labels: {batch_labels.shape}")

if __name__ == "__main__":
    main()