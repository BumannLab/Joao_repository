import torch

from NeuralNetworkSegmentation.data.io import load_images
from NeuralNetworkSegmentation.data.splits import ordered_split



class BacteriaMaskPair(torch.utils.data.Dataset):
    def __init__(self, samples: list[tuple], transform=None, generate_data=False, num_images=None):
        self.samples = samples
        self.transform = transform
        self.generate_data = generate_data
        self.num_images = num_images

    def __len__(self):
        if self.generate_data:
            return len(self.samples) * self.num_images
        
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        if self.generate_data:
            idx = idx // self.num_images

        sample = self.samples[idx]

        file_name, input_image, label_image = sample # not using file_name but can be helpful for future


        input_tensor = torch.from_numpy(input_image / 65535).to(torch.float32)
        label_tensor = torch.from_numpy(label_image / 255).to(torch.long)

        if self.transform:
            input_tensor, label_tensor = self.transform(input_tensor, label_tensor)

        if all([input_tensor.ndim == 4, label_tensor.ndim == 4]):
            input_tensor = input_tensor.permute(1, 0, 2, 3) 
            label_tensor = label_tensor.permute(1, 0, 2, 3) 
        
        return input_tensor, label_tensor


if __name__ == "__main__":
    x_dir = "/data/joao/Materials/Joao_repository/NN_Dataset/DATASET_COMPARISON/3D_comparison/Dataset_threshold_800/inputs"
    y_dir = "/data/joao/Materials/Joao_repository/NN_Dataset/DATASET_COMPARISON/3D_comparison/Dataset_threshold_800/masks"

    group_pairs = load_images(x_dir, y_dir, [0, 3])

    splits = ordered_split(group_pairs, 0.6, 0.2, 0.2)

    train_dataset = BacteriaMaskPair(splits["train"])
    test_dataset = BacteriaMaskPair(splits["test"])
    validation_dataset = BacteriaMaskPair(splits["val"])

    print(train_dataset[0])

