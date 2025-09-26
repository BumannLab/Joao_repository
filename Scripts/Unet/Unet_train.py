import glob
import tifffile as tiff
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# TODO: implement class dataset method: Done
# TODO: Find a way to get the number of channels # how can we know the channel
# size in advance ? Should we load the first image and get the shape?

IMG_DIR = "Unet/ImageS/Cropped_Images"
MASK_IMG_DIR = "Unet/Images/Mask_Images_test"

class BacteriaMaskPair(Dataset):
    def __init__(self, input_images: np.ndarray, label_images: np.ndarray, transform=None):
        self.input_images = self.load_images(input_images)      
        self.label_images = self.load_images(label_images)      
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

    def load_images(self, folder: str):
        """
        Input: Path of the folder containing the images for training (Either
        to load bacteria or mask of the bacteria images)
        Output: Numpy array containing all the images
        Process: TODO
        """
        images_path = (glob.glob(f"{folder}/*.tif"))
        # how can we know the channel size in advance ? Should we load the first image and get the shape? 
        images = np.empty((len(images_path), 4, 128, 128), dtype=np.float64)
        for img_idx in range(len(images_path)):
            images[img_idx] = tiff.imread(images_path[img_idx])
        return images

bacteria_dataset = BacteriaMaskPair(IMG_DIR, MASK_IMG_DIR)

dataloader = DataLoader(bacteria_dataset, batch_size=64, shuffle=True)

# for batch_images, batch_labels in dataloader:
#     print(f"Batch shape: {batch_images}, Labels: {batch_labels.shape}")
        