from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from Scripts.utils import get_images_path, read_image
from Scripts.NeuralNetworkSegmentation.transforms import AugmentationTransformer


@dataclass(frozen=True)
class ImageMaskSample:
    label: int
    file_name: str
    input_path: Path
    mask_path: Path


class ConfigDataset(Dataset):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @classmethod
    def create_datasets(cls, dataset_config: dict, phase: str):
        raise NotImplementedError
    

class BaseMycobacteriumTuberculosisDataset(ConfigDataset):

    def __init__(
        self,
        samples: list[ImageMaskSample],
        phase: str,
        channels_to_load = None,
        transform=None,
        generate_data: bool = False,
        num_images: int = 1,
    ):
        if phase not in {"train", "val", "test"}:
            raise ValueError("phase must be one of: train, val, test.")

        self.samples = samples
        self.phase = phase
        self.channels_to_load = channels_to_load
        self.transform = transform
        self.generate_data = generate_data
        self.num_images = max(int(num_images), 1)

    def __len__(self):
        if self.generate_data:
            return len(self.samples) * self.num_images

        return len(self.samples)

    def __getitem__(self, idx: int):
        if self.generate_data:
            idx = idx // self.num_images

        # sample = self.samples[idx]
        input_image, mask_image = self.load_sample_arrays(idx)
        
        # NEW
        input_image = input_image / 65535
        mask_image = mask_image / 255
        
        input_image = input_image.astype(np.float32)
        mask_image = mask_image.astype(np.long)

        # OLD
        # input_tensor = torch.from_numpy(input_image / 65535).to(torch.float32)
        # mask_tensor = torch.from_numpy(mask_image / 255).to(torch.long)

        print(input_image.shape)
        print(mask_image.shape)

        if self.transform is not None:
            # NEW convert into albumentations desired shape D,H,W,C (shape before hand was D,C,H,W)
            input_tensor = np.transpose(input_image, (0, 2, 3, 1))
            mask_tensor = np.transpose(mask_image, (0, 2, 3, 1))


            transformed = self.transform(volume = input_tensor, mask3d = mask_tensor)
            input_tensor = transformed["volume"]
            mask_tensor = transformed["mask3d"]

        if input_tensor.ndim == 4 and mask_tensor.ndim == 4:
            # NEW conversion from D,H,W,C into C,D,H,W
            input_tensor = input_tensor.permute(3, 0, 1, 2)
            mask_tensor = mask_tensor.permute(3, 0, 1, 2)


            # OLD conversion from D,C,H,W to C,D,H,W
            # input_tensor = input_tensor.permute(1, 0, 2, 3)
            # mask_tensor = mask_tensor.permute(1, 0, 2, 3)

        print(self.samples[idx].file_name)
        return input_tensor, mask_tensor

    def load_sample_arrays(self, idx):
        raise NotImplementedError

    @classmethod # Initiates an instance of the class
    def create_dataset(cls, splits, config: dict, phase: str):

        loaders_config = config["loaders"]
        
        assert phase in loaders_config, f"{phase} not found in config"
        
        channels_to_load = loaders_config.get("channels_to_load", [])

        phase_config = loaders_config[phase]
        phase_transforms_config = phase_config.get("transforms", {})
        seed = config["seed"]

        transform_pipeline = AugmentationTransformer(phase_transforms_config, seed)._create_transform()
        
        generate_data = bool(loaders_config.get("generate_data", False))
        num_images = int(loaders_config.get("num_images", 1))

        return cls(
                splits[phase],
                phase=phase,
                channels_to_load = channels_to_load,
                transform=transform_pipeline,
                generate_data=generate_data if phase == "train" else False,
                num_images=num_images,
            )


class StandardMycobacteriumTuberculosisDataset(BaseMycobacteriumTuberculosisDataset):
    def __init__(
        self,
        samples: list[ImageMaskSample],
        phase: str,
        channels_to_load: list[int] = None,
        transform=None,
        generate_data: bool = False,
        num_images: int = 1,
    ):
        super().__init__(
            samples,
            phase,
            channels_to_load,
            transform=transform,
            generate_data=generate_data,
            num_images=num_images,
        )
        
        self.loaded_samples = [
            (read_input(sample.input_path, self.channels_to_load), read_mask(sample.mask_path))
            for sample in self.samples
        ]

    def load_sample_arrays(self, idx):
        return self.loaded_samples[idx]


class LazyMycobacteriumTuberculosisDataset(BaseMycobacteriumTuberculosisDataset):
    def __init__(
        self,
        samples: list[ImageMaskSample],
        phase: str,
        channels_to_load: list[int] = None,
        transform=None,
        generate_data: bool = False,
        num_images: int = 1,
    ):
    
        super().__init__(
            samples,
            phase,
            channels_to_load,
            transform=transform,
            generate_data=generate_data,
            num_images=num_images,
        )


    
    def load_sample_arrays(self, idx):
        sample = self.samples[idx]
        return read_input(sample.input_path, self.channels_to_load), read_mask(sample.mask_path)


def read_input(input_path: Path, channels_to_load: list | None = None):
    
    input_image = read_image(input_path)

    if channels_to_load:
        input_image = np.take(input_image, indices=channels_to_load, axis=-3)

    while input_image.ndim != 4:
        input_image = np.expand_dims(input_image, axis=0)

    return input_image


def read_mask(mask_path: Path):
    mask = read_image(mask_path).astype(np.uint8)

    mask[mask > 0] = 255

    while mask.ndim != 4:
        mask = np.expand_dims(mask, axis=-3)

    return mask


def collect_samples(
    input_path: str | Path,
    mask_path: str | Path,
):

    input_paths = get_images_path(input_path, ".tif", ".tiff")
    mask_paths = get_images_path(mask_path, ".tif", ".tiff")

    assert input_paths, "No input paths found"
    assert mask_paths, "No mask paths found"

    if len(input_paths) != len(mask_paths):  
        raise ValueError(
            f"Found {len(input_paths)} input images but {len(mask_paths)} masks."
        )

    input_stems = [path.stem for path in input_paths]
    mask_stems = [path.stem for path in mask_paths]
    if input_stems != mask_stems:
        raise ValueError("Input and mask files must have matching sorted stems.")

    grouped_samples = defaultdict(list)
    for input_file, mask_file in zip(input_paths, mask_paths):
        file_name = input_file.stem
        label = 1 if "positive" in file_name.lower() else 0
        grouped_samples[label].append(
            ImageMaskSample(
                label=label,
                file_name=file_name,
                input_path=Path(input_file),
                mask_path=Path(mask_file),
            )
        )

    return grouped_samples


def ordered_split(
    grouped_samples: dict[dict[np.ndarray]],
    train_frac: float,
    val_frac: float,
    test_frac: float | None = None,
):
    if test_frac is None:
        assert abs(train_frac + val_frac - 1) == 0

    assert abs(train_frac + val_frac + test_frac - 1) == 0

    n_negatives = len(grouped_samples[0])
    n_positives = len(grouped_samples[1])

    # Generate random permutation for each positive vs negative tile
    p0 = np.random.permutation(n_negatives)
    p1 = np.random.permutation(n_positives)

    # Extract from the dictionary of positive and negative tiles the name&input&mask by order of the permutations
    group_pairs0 = [grouped_samples[0][i] for i in p0]
    group_pairs1 = [grouped_samples[1][i] for i in p1]

    tr_neg = int(round(n_negatives * train_frac))
    va_neg = int(round(n_negatives * val_frac))

    tr_pos = int(round(n_positives * train_frac))
    va_pos = int(round(n_positives * val_frac))

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
    }

    if test_frac:
        te_neg = n_negatives - tr_neg - va_neg
        te_pos = n_positives - tr_pos - va_pos
        splits["test"] = take_split(te_neg, te_pos)

    return splits


def build_datasets(config):

    loaders_config = config["loaders"]

    grouped_sample_paths = collect_samples(
        loaders_config["raw_path"],
        loaders_config["label_path"],
    )

    train_frac = loaders_config.get("split_train", 0.6) 
    val_frac = loaders_config.get("split_val", 0.2)

    # TODO: Handle if test fraction is None or not present 
    # This will have wide different things, first the split will not happen for the test
    # Computing the final confusion matrix is done over the test set so this would perhaps change
    # Probably add smth like if current epoch == n_epochs, save model
    # If early stop is added the check would be different like, or just add like
    # a boolean to the current epoch == n epochs and not early_stop, where early stop is like
    # max_eval_score > eval_score: early_stop_count += 1; if early_stop_count == threhsold; early_stop == true
    # Seems complex but for now this is enough
    # TODO: Also handle in the case that the phase contains a path to a folder
    # would perhaps change a few things but i dont know for certain
    test_frac = loaders_config.get("split_test", 0.2)     

    splits = ordered_split(
        grouped_sample_paths,
        train_frac,
        val_frac,
        test_frac
    )

    datasets = dict()
    
    for phase in ("train", "val", "test"):
        datasets[phase] = get_dataset_class(loaders_config).create_dataset(splits, config, phase) #[0] What is this for???
    
    return datasets


def build_dataloaders(
    config: dict
):

    datasets = build_datasets(config)

    loaders_config = config["loaders"]

    batch_size = loaders_config.get("batch_size", 16)
    num_workers = loaders_config.get("num_workers", 1)

    return {
        
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            # generator=_torch_generator(seed),
            # worker_init_fn=_seed_worker(seed) if seed is not None else None,
            # pin_memory=bool(loader_config.get("pin_memory", False)),
            # drop_last=bool(loader_config.get("drop_last", False)),
        ),
        
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            # worker_init_fn=_seed_worker(seed) if seed is not None else None,
            # pin_memory=bool(loader_config.get("pin_memory", False)),
            # drop_last=bool(loader_config.get("drop_last", False)),
        ),
        
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            # worker_init_fn=_seed_worker(seed) if seed is not None else None,
            # pin_memory=bool(loader_config.get("pin_memory", False)),
            # drop_last=bool(loader_config.get("drop_last", False)),
        ),
    }


def get_dataset_class(loaders_config: dict):
    # Standard option when nothing is present in the file
    dataset = loaders_config.get("dataset", "standard")       
    
    if dataset == "lazy":
        return LazyMycobacteriumTuberculosisDataset
    
    elif dataset == "standard":
        return StandardMycobacteriumTuberculosisDataset
    
    raise ValueError(f"Unsupported dataset: {dataset}")


# def _torch_generator(seed: int | None):
#     if seed is None:
#         return None
#     generator = torch.Generator()
#     generator.manual_seed(seed)
#     return generator


# def _seed_worker(seed: int):
#     def seed_worker(worker_id: int):
#         worker_seed = seed + worker_id
#         random.seed(worker_seed)
#         np.random.seed(worker_seed)
#         torch.manual_seed(worker_seed)

#     return seed_worker


def show_split_pairs(batch): 

    import matplotlib.pyplot as plt

    raw, label = batch

    print(raw.shape)
    print(label.shape)

    for i in range(raw.shape[0]):

        _raw = raw[i]
        _label = label[i]

        print(_raw.shape, "in loop")
        print(_label.shape, "in loop")

        sybrgold = torch.amax(_raw, dim=1)[0]  # FIRST CHANNEL ONLY
        mask = torch.amax(_label, dim=1)[0]  # FIRST CHANNEL ONLY

        print(sybrgold.shape)
        print(mask.shape)

        sybrgold = sybrgold.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()

        fig, ax = plt.subplots(1, 3, figsize=(9, 3))

        ax[0].imshow(sybrgold, cmap="gray")
        ax[0].set_title(f"original sybrgold")
        ax[0].axis("off")

        ax[1].imshow(mask, cmap="gray")
        ax[1].set_title("mask")
        ax[1].axis("off")

        # overlay
        ax[2].imshow(sybrgold, cmap="gray")
        ax[2].imshow(mask, alpha=0.4)
        ax[2].set_title("overlay")
        ax[2].axis("off")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    
    from Scripts.NeuralNetworkSegmentation.config import load_config_yaml
    from Scripts.NeuralNetworkSegmentation.utils import seed_everything
    
    seed_everything(47)
    
    config = load_config_yaml("/data/joao/Materials/Joao_repository/Scripts/NeuralNetworkSegmentation/config.yml")

    print(config)

    dataloaders = build_dataloaders(config)

    val_loader = dataloaders["train"]
    print(val_loader)
    
    for batch in val_loader:
        show_split_pairs(batch)
        inp, lab = batch
        print(inp.shape, "main inp shape")
        print(lab.shape, "main lab shape")

