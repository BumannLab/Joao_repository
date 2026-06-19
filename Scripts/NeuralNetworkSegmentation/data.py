from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Dataset

from Scripts.utils import get_images_path, read_image
from Scripts.NeuralNetworkSegmentation.transforms import AugmentationTransformer
from Scripts.NeuralNetworkSegmentation.utils import convert_to_tensor

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

        input_image, mask_image = self.load_sample_arrays(idx)
        
        # NEW
        input_image = input_image / 65535
        mask_image = mask_image / 255
        
        input_image = input_image.astype(np.float32)
        mask_image = mask_image.astype(np.long)

        # OLD
        # input_tensor = torch.from_numpy(input_image / 65535).to(torch.float32)
        # mask_tensor = torch.from_numpy(mask_image / 255).to(torch.long)
        
        # NEW convert into albumentations desired shape D,H,W,C (shape before hand was D,C,H,W)
        input_image = np.transpose(input_image, (0, 2, 3, 1))
        mask_image = np.transpose(mask_image, (0, 2, 3, 1))
        
        if self.transform is not None:
            transformed = self.transform(volume = input_image, mask3d = mask_image)
            input_image = transformed["volume"]
            mask_image = transformed["mask3d"]

        input_tensor, mask_tensor = convert_to_tensor(input_image, mask_image)
        # TODO: Augmentation to convert to tensor if not convert it here add function for it
        # NEW conversion from D,H,W,C into C,D,H,W
        input_tensor = input_tensor.permute(3, 0, 1, 2)
        mask_tensor = mask_tensor.permute(3, 0, 1, 2)

        # OLD conversion from D,C,H,W to C,D,H,W
        # input_tensor = input_tensor.permute(1, 0, 2, 3)
        # mask_tensor = mask_tensor.permute(1, 0, 2, 3)

        return input_tensor, mask_tensor

    def load_sample_arrays(self, idx):
        raise NotImplementedError

    @classmethod # Initiates an instance of the class
    def create_dataset(cls, splits, loaders_config: dict, seed, phase: str):
        
        assert phase in loaders_config, f"{phase} not found in config"
        
        phase_config = loaders_config.get(phase) or {} # If key exists but its None it will return {} so it will run

        phase_transforms_config = phase_config.get("transforms", [])
        transform_pipeline = AugmentationTransformer(phase_transforms_config, seed)._create_transform()
        
        channels_to_load = loaders_config.get("channels_to_load", [])

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


def ordered_split(grouped_samples, train_frac, val_frac, test_frac=None):
    total_frac = train_frac + val_frac
    
    if test_frac is not None:
       total_frac += test_frac
    
    assert abs(total_frac - 1) == 0

    def shuffle_and_slice(samples):
        shuffled = [samples[i] for i in np.random.permutation(len(samples))]
        n = len(shuffled)
        tr = int(round(n * train_frac))
        va = int(round(n * val_frac))
        chunks = [shuffled[:tr], shuffled[tr:tr + va]]
        if test_frac is not None:
            chunks.append(shuffled[tr + va:])
        return chunks

    names = ["train", "val"] + (["test"] if test_frac is not None else [])
    neg_chunks = shuffle_and_slice(grouped_samples[0])
    pos_chunks = shuffle_and_slice(grouped_samples[1])

    return {name: neg + pos for name, neg, pos in zip(names, neg_chunks, pos_chunks)}


def build_datasets(loaders_config, seed):

    grouped_sample_paths = collect_samples(
        loaders_config["raw_path"],
        loaders_config["label_path"],
    )

    train_frac = loaders_config.get("split_train", 0.7) 
    val_frac = loaders_config.get("split_val", 0.15)

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
    test_frac = loaders_config.get("split_test", 0.15)     

    splits = ordered_split(
        grouped_sample_paths,
        train_frac,
        val_frac,
        test_frac
    )

    datasets = dict()
    
    for phase in ("train", "val", "test"):
        datasets[phase] = get_dataset_class(loaders_config).create_dataset(splits, loaders_config, seed, phase) #[0] What is this for???
    
    return datasets


def build_dataloaders(
    config: dict
    ):
    
    loaders_config = config["loaders"]
    seed = config["seed"]
    
    datasets = build_datasets(loaders_config, seed)

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
