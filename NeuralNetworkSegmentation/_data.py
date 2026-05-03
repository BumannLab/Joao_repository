from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils import get_images_path, read_image

class BacteriaMaskPair(Dataset):
    def __init__(self,
                samples,
                transform=None,
                generate_data=False,
                num_images=None
            ):
        self.samples = samples
        self.transform_input = transform_input
        self.transform_mask = transform_mask
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

        label, file_name, input_image, label_image = sample # not using file_name but can be helpful for future

        input_tensor = torch.from_numpy(input_image / 65535).to(torch.float32)
        label_tensor = torch.from_numpy(label_image / 255).to(torch.long)

        if self.transform:
            input_tensor, label_tensor = self.transform(input_tensor, label_tensor)

        if all([input_tensor.ndim == 4, label_tensor.ndim == 4]):
            input_tensor = input_tensor.permute(1, 0, 2, 3) 
            label_tensor = label_tensor.permute(1, 0, 2, 3) 
        
        return input_tensor, label_tensor


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
        grouped_pairs[label].append((file_name, input_image, mask_image))

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


def build_transform_pipeline(config_transforms: dict | None):
    if not config_transforms:
        return None

    input_config = config_transforms.get("input", {}) or {}
    mask_config = config_transforms.get("mask", {}) or {}
    order = list(config_transforms.get("order", []))

    for transform_name in list(input_config) + list(mask_config):
        if transform_name not in order:
            order.append(transform_name)

    steps = []
    for transform_name in order:
        apply_to_input = transform_name in input_config
        apply_to_mask = transform_name in mask_config

        if not apply_to_input and not apply_to_mask:
            continue
        
        if transform_key not in TRANSFORM_REGISTRY:
            raise ValueError(f"Unsupported transform: {transform_name}")

        input_spec = _normalize_transform_spec(input_config.get(transform_name))
        mask_spec = _normalize_transform_spec(mask_config.get(transform_name))

        params = {}
        params.update(mask_spec)
        params.update(input_spec)

        transform = TRANSFORM_REGISTRY[transform_key](**params)
        steps.append((transform, apply_to_input, apply_to_mask))

    return PairedTransformPipeline(steps)


class PairedTransformPipeline:
    def __init__(self, steps: list[tuple[Transform, bool, bool]]):
        self.steps = steps

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        for transform, apply_to_input, apply_to_mask in self.steps:
            x_arg = x if apply_to_input else None
            y_arg = y if apply_to_mask else None

            transformed_x, transformed_y = transform(x_arg, y_arg)

            if apply_to_input:
                x = transformed_x
            if apply_to_mask:
                y = transformed_y

        return x, y


def build_datasets(config_data, config_transforms, num_blocks):

    grouped_samples = load_images(
        config_data["input_path"],
        config_data["mask_path"],
        config_data.get("channels_to_load", [0]),
    )

    fractions = config_data["fractions"]
    train_frac = fractions.get("train", 0.7) 
    val_frac = fractions.get("val", 0.15) 
    test_frac = fractions.get("test", 0.15) 
    
    splits = ordered_split(
        grouped_samples,
        train_frac,
        val_frac,
        test_frac

    )

    transforms_pipeline = 

def build_dataloaders(
        input_path,
        mask_path,
        channels_to_load,
        train_frac,
        val_frac,
        test_frac,
        batch_size,
        transform_input,
        transform_mask,
        generate_data,
        num_images
    ):

    

    samples = load_images(input_path, mask_path, channels_to_load)

    ordered_samples = ordered_split(samples, train_frac, val_frac, test_frac)

    train_dataset = BacteriaMaskPair(ordered_samples["train"], transform_input=transform_input, transform_mask=transform_mask, generate_data=generate_data, num_images = num_images)
    validation_dataset = BacteriaMaskPair(ordered_samples["val"], transform_input=transform_input, transform_mask=transform_mask)
    test_dataset  = BacteriaMaskPair(ordered_samples["test"])
     
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, validation_dataloader, test_dataloader
