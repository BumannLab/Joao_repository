# from collections import defaultdict
# from pathlib import Path

# import numpy as np
# import torch
# from torch.utils.data import DataLoader, Dataset

# from ZOrganized_scripts.NeuralNetworkSegmentationConfig.transforms import build_transform_pipeline
# from Scripts.utils import get_images_path, read_image

# class BacteriaMaskDataset(Dataset):
#     def __init__(self,
#                 samples,
#                 transform=None,
#                 generate_data=False,
#                 num_images=None
#             ):
#         self.samples = samples
#         self.transform = transform
#         self.generate_data = generate_data
#         self.num_images = num_images

#     def __len__(self):
#         if self.generate_data:
#             return len(self.samples) * self.num_images
        
#         return len(self.samples)
    
#     def __getitem__(self, idx: int):
#         if self.generate_data:
#             idx = idx // self.num_images

#         sample = self.samples[idx]

#         label, file_name, input_image, label_image = sample # not using file_name neither label but can be helpful for future

#         input_tensor = torch.from_numpy(input_image / 65535).to(torch.float32)
#         label_tensor = torch.from_numpy(label_image / 255).to(torch.long)

#         if self.transform:
#             input_tensor, label_tensor = self.transform(input_tensor, label_tensor)

#         input_tensor = input_tensor.permute(1, 0, 2, 3) 
#         label_tensor = label_tensor.permute(1, 0, 2, 3) 
        
#         return input_tensor, label_tensor


# def read_input(input_path: Path, channels_to_load: list | None = None):

#     _input = read_image(input_path)

#     if channels_to_load:
#         _input = np.take(_input, indices = channels_to_load, axis = -3)
    
#     while _input.ndim != 4:
#         # If image shape is DxHxW -> this will break the code forward
#         _input = np.expand_dims(_input, axis=0)

#     return _input


# def read_mask(mask_path: Path):

#     mask = read_image(mask_path).astype(np.uint8)

#     mask[mask > 0] = 255

#     while mask.ndim != 4:
#         # If image shape is DxHxW -> this will break the code forward
#         mask = np.expand_dims(mask, axis=-3)

#     return mask


# def load_images(
#     input_path: str | Path,
#     mask_path: str | Path,
#     channels_to_load: list[int] | None = None,
# ) -> dict[int, list[tuple[str, np.ndarray, np.ndarray]]]:
    
#     input_paths = get_images_path(input_path)
#     mask_paths = get_images_path(mask_path)

#     if len(input_paths) != len(mask_paths):
#         raise ValueError(
#             f"Found {len(input_paths)} input images but {len(mask_paths)} masks."
#         )

#     input_stems = [path.stem for path in input_paths]
#     mask_stems = [path.stem for path in mask_paths]
#     if input_stems != mask_stems:
#         raise ValueError("Input and mask files must have matching sorted stems.")

#     grouped_pairs = defaultdict(list)
#     for input_file, mask_file in zip(input_paths, mask_paths):
#         file_name = input_file.stem
#         label = 1 if "positive" in file_name.lower() else 0

#         input_image = read_input(input_file, channels_to_load)
#         mask_image = read_mask(mask_file)
#         grouped_pairs[label].append((label, file_name, input_image, mask_image))

#     return grouped_pairs


# def ordered_split(
#     group_pairs: dict[dict[np.ndarray]],
#     train_frac: float,
#     val_frac: float,
#     test_frac: float,
# ):
#     assert abs(train_frac + val_frac + test_frac - 1) == 0

#     # Generate random permutation for each positive vs negative tile
#     p0 = np.random.permutation(len(group_pairs[0]))
#     p1 = np.random.permutation(len(group_pairs[1]))

#     # Extract from the dictionary of positive and negative tiles the name&input&mask by order of the permutations
#     group_pairs0 = [group_pairs[0][i] for i in p0]
#     group_pairs1 = [group_pairs[1][i] for i in p1]

#     n_neg = len(group_pairs0)
#     n_pos = len(group_pairs1)

#     # ---- Split each class independently (FIXED) ----
#     tr_neg = int(round(n_neg * train_frac))
#     va_neg = int(round(n_neg * val_frac))
#     te_neg = n_neg - tr_neg - va_neg

#     tr_pos = int(round(n_pos * train_frac))
#     va_pos = int(round(n_pos * val_frac))
#     te_pos = n_pos - tr_pos - va_pos

#     neg_ptr = 0
#     pos_ptr = 0

#     def take_split(n_neg_take, n_pos_take):
#         nonlocal neg_ptr, pos_ptr

#         group_pairs_neg = group_pairs0[neg_ptr:neg_ptr + n_neg_take]
#         group_pairs_pos = group_pairs1[pos_ptr:pos_ptr + n_pos_take]

#         neg_ptr += n_neg_take
#         pos_ptr += n_pos_take

#         group_pairs_neg.extend(group_pairs_pos)

#         return group_pairs_neg

#     splits = {
#         "train": take_split(tr_neg, tr_pos),
#         "val":   take_split(va_neg, va_pos),
#         "test":  take_split(te_neg, te_pos),
#     }

#     return splits


# def build_datasets(config_data, config_transforms):

#     grouped_samples = load_images(
#         config_data["raw_path"],
#         config_data["label_path"],
#         config_data.get("channels_to_load", [0]),
#     )

#     train_frac = config_data.get("split_train", 0.7) 
#     val_frac = config_data.get("split_validation", 0.15) 
#     test_frac = config_data.get("split_test", 0.15) 
    
#     splits = ordered_split(
#         grouped_samples,
#         train_frac,
#         val_frac,
#         test_frac
#     )

#     train_transforms_pipeline = build_transform_pipeline(config_transforms.get("train"))
#     val_transforms_pipeline = build_transform_pipeline(config_transforms.get("val"))
#     test_transforms_pipeline = build_transform_pipeline(config_transforms.get("test"))

#     generate_data = bool(config_data.get("generate_data", False))
#     num_images = int(config_data.get("num_images", 1))

#     return {
#         "train": BacteriaMaskDataset(
#             splits["train"],
#             transform=train_transforms_pipeline,
#             generate_data=generate_data,
#             num_images=num_images,
#         ),
#         "val": BacteriaMaskDataset(
#             splits["val"],
#             transform=val_transforms_pipeline,
#         ),
#         "test": BacteriaMaskDataset(
#             splits["test"],
#             transform=test_transforms_pipeline,
#         ),
#     }


# def build_dataloaders(
#     config_data: dict,
#     config_transforms: dict | None = None,
# ) -> tuple[DataLoader, DataLoader, DataLoader]:
#     datasets = build_datasets(config_data, config_transforms)
#     batch_size = int(config_data.get("batch_size", 1))
#     num_workers = int(config_data.get("num_workers", 0))

#     return {
#         "train": DataLoader(
#             datasets["train"],
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=num_workers,
#         ),
#         "val": DataLoader(
#             datasets["val"],
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=num_workers,
#         ),
#         "test": DataLoader(
#             datasets["test"],
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=num_workers,
#         ),
#     }




from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ZOrganized_scripts.NeuralNetworkSegmentationConfig.transforms import create_split_augmentations
from Scripts.utils import get_images_path, read_image

class BacteriaMaskDataset(Dataset):
    def __init__(self,
                samples,
                raw_transform=None,
                label_transform = None,
                generate_data=False,
                num_images=None
            ):
        self.samples = samples
        self.raw_transform = raw_transform
        self.label_transform = label_transform
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

        label, file_name, input_image, label_image = sample # not using file_name neither label but can be helpful for future
        input_tensor = torch.from_numpy(input_image / 65535).to(torch.float32)
        label_tensor = torch.from_numpy(label_image / 255).to(torch.long)

        if self.raw_transform:
            input_tensor = self.raw_transform(input_tensor)

        if self.label_transform:
            label_tensor = self.label_transform(label_tensor)

        input_tensor = input_tensor.permute(1, 0, 2, 3) 
        label_tensor = label_tensor.permute(1, 0, 2, 3) 
        
        return input_tensor, label_tensor


def read_input(input_path: Path, channels_to_load: list | None = None):

    _input = read_image(input_path)

    if channels_to_load:
        _input = np.take(_input, indices = channels_to_load, axis = -3)
    
    while _input.ndim != 4:
        # If image shape is DxHxW -> this will break the code forward
        _input = np.expand_dims(_input, axis=0)

    return _input


def read_mask(mask_path: Path):

    mask = read_image(mask_path).astype(np.uint8)

    mask[mask > 0] = 255

    while mask.ndim != 4:
        # If image shape is DxHxW -> this will break the code forward
        mask = np.expand_dims(mask, axis=-3)

    return mask


def load_images(
    input_path: str | Path,
    mask_path: str | Path,
    channels_to_load: list[int] | None = None,
) -> dict[int, list[tuple[str, np.ndarray, np.ndarray]]]:
    
    input_paths = get_images_path(input_path)
    mask_paths = get_images_path(mask_path)

    if len(input_paths) != len(mask_paths):
        raise ValueError(
            f"Found {len(input_paths)} input images but {len(mask_paths)} masks."
        )

    input_stems = [path.stem for path in input_paths]
    mask_stems = [path.stem for path in mask_paths]
    if input_stems != mask_stems:
        raise ValueError("Input and mask files must have matching sorted stems.")

    grouped_pairs = defaultdict(list)
    for input_file, mask_file in zip(input_paths, mask_paths):
        file_name = input_file.stem
        label = 1 if "positive" in file_name.lower() else 0

        input_image = read_input(input_file, channels_to_load)
        mask_image = read_mask(mask_file)
        grouped_pairs[label].append((label, file_name, input_image, mask_image))

    return grouped_pairs


def ordered_split(
    group_pairs: dict[dict[np.ndarray]],
    train_frac: float,
    val_frac: float,
    test_frac: float,
):
    assert abs(train_frac + val_frac + test_frac - 1) == 0

    # Generate random permutation for each positive vs negative tile
    p0 = np.random.permutation(len(group_pairs[0]))
    p1 = np.random.permutation(len(group_pairs[1]))

    # Extract from the dictionary of positive and negative tiles the name&input&mask by order of the permutations
    group_pairs0 = [group_pairs[0][i] for i in p0]
    group_pairs1 = [group_pairs[1][i] for i in p1]

    n_neg = len(group_pairs0)
    n_pos = len(group_pairs1)

    # ---- Split each class independently (FIXED) ----
    tr_neg = int(round(n_neg * train_frac))
    va_neg = int(round(n_neg * val_frac))
    te_neg = n_neg - tr_neg - va_neg

    tr_pos = int(round(n_pos * train_frac))
    va_pos = int(round(n_pos * val_frac))
    te_pos = n_pos - tr_pos - va_pos

    neg_ptr = 0
    pos_ptr = 0

    def take_split(n_neg_take, n_pos_take):
        nonlocal neg_ptr, pos_ptr

        group_pairs_neg = group_pairs0[neg_ptr:neg_ptr + n_neg_take]
        group_pairs_pos = group_pairs1[pos_ptr:pos_ptr + n_pos_take]

        neg_ptr += n_neg_take
        pos_ptr += n_pos_take

        group_pairs_neg.extend(group_pairs_pos)

        return group_pairs_neg

    splits = {
        "train": take_split(tr_neg, tr_pos),
        "val":   take_split(va_neg, va_pos),
        "test":  take_split(te_neg, te_pos),
    }

    return splits


def build_datasets(config_data, config_transforms):

    grouped_samples = load_images(
        config_data["raw_path"],
        config_data["label_path"],
        config_data.get("channels_to_load", [0]),
    )

    train_frac = config_data.get("split_train", 0.7) 
    val_frac = config_data.get("split_validation", 0.15) 
    test_frac = config_data.get("split_test", 0.15) 
    
    splits = ordered_split(
        grouped_samples,
        train_frac,
        val_frac,
        test_frac
    )

    train_raw_augmentations, train_label_augmentations = create_split_augmentations(config_transforms.get("train", {}))
    val_raw_augmentations, val_label_augmentations = create_split_augmentations(config_transforms.get("val", {}))
    test_raw_augmentations, test_label_augmentations = create_split_augmentations(config_transforms.get("test", {}))

    generate_data = bool(config_data.get("generate_data", False))
    num_images = int(config_data.get("num_images", 0))

    return  {
        "train": BacteriaMaskDataset(
            splits["train"],
            raw_transform=train_raw_augmentations,
            label_transform=train_label_augmentations,
            generate_data=generate_data,
            num_images=num_images,
        ),
        "val": BacteriaMaskDataset(
            splits["val"],
            raw_transform=val_raw_augmentations,
            label_transform=val_label_augmentations,
        ),
        "test": BacteriaMaskDataset(
            splits["test"],
            raw_transform=test_raw_augmentations,
            label_transform=test_label_augmentations,
        ),
    }


def build_dataloaders(
    config_data: dict,
    config_transforms: dict | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    
    datasets = build_datasets(config_data, config_transforms)
    batch_size = int(config_data.get("batch_size", 1))
    num_workers = int(config_data.get("num_workers", 0))

    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }