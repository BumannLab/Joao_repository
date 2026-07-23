from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import collections
from typing import Any


import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from Scripts.utils import get_images_path, read_tiff_image
from Scripts.NeuralNetworkSegmentation.dataset.tiff import StandardTiffDataset, LazyTiffDataset 
from Scripts.NeuralNetworkSegmentation.utils import get_logger

logger = get_logger("Dataset")

@dataclass(frozen=True)
class PatientPatchSample:
    patient_id: str
    file_name: str
    input_path: Path
    mask_path: Path | None = None


def extract_patient_id(file_name: str) -> str:
    """patient1_x_y_z -> patient1"""
    return file_name.split("_")[0]


def collect_patient_samples(
    input_path: str | Path,
    mask_path: str | Path | None = None,
    group_by: str = "patient"
) -> dict[str, list[PatientPatchSample]]:

    input_paths = get_images_path(input_path, ".tif", ".tiff")
    assert input_paths, "No input paths found"

    if mask_path is not None:
        mask_paths = get_images_path(mask_path, ".tif", ".tiff")
        assert mask_paths, "No mask paths found"

        if len(input_paths) != len(mask_paths):
            raise ValueError(
                f"Found {len(input_paths)} input images but {len(mask_paths)} masks."
            )

        input_stems = [path.stem for path in input_paths]
        mask_stems = [path.stem for path in mask_paths]
        if input_stems != mask_stems:
            raise ValueError("Input and mask files must have matching sorted stems.")
    else:
        mask_paths = [None] * len(input_paths)

    samples = defaultdict(list)

    counter = 0
    for input_file, mask_file in zip(input_paths, mask_paths):
        file_name = input_file.stem
        patient_id = extract_patient_id(file_name)
        
        patient_raw_label = PatientPatchSample(
                patient_id=patient_id,
                file_name=file_name,
                input_path=Path(input_file),
                mask_path=Path(mask_file) if mask_file is not None else None,
        )

        # Will create a dataset with all samples
        if group_by == "all":            
            samples["all"].append(
                patient_raw_label
            )
        
        # Will create a dataset for each patient with all belonging to that patient 
        elif group_by == "patient": 
            samples[patient_id].append(
                patient_raw_label
            )
        
        # Will create a dataset for each image
        elif group_by == "image":
            samples[counter].append(
                patient_raw_label
            )
            counter += 1

        else:
            raise ValueError(f"Unknown group_by: {group_by}. Choose 'patient' or 'image'.")

    return samples


def read_input(input_path: Path):

    input_image = read_tiff_image(input_path)  # Shape is CxHxW or DxCxHxW

    # CxHxW -> 1xCxHxW
    while input_image.ndim != 4:

        input_image = np.expand_dims(input_image, axis=0)

    return np.transpose(input_image, (1, 0, 2, 3)).astype(np.float32)


def read_mask(mask_path: Path):

    mask = read_tiff_image(mask_path)  # Shape is HxW or DxHxW

    mask[mask > 0] = 1

    # HxW -> 1x1xHxW; DxHxW -> Dx1xHxW
    while mask.ndim != 4:

        mask = np.expand_dims(mask, axis=-3)


    return np.transpose(mask, (1, 0, 2, 3)).astype(np.long)


def calculate_stats(patches: list[np.ndarray] | None, skip: bool = False) -> dict[str, float | None]:
    """
    Computes percentile clip bounds, mean and std of the raw data.
    """

    if skip or not patches:
        return {"pmin": None, "pmax": None, "mean": None, "std": None}

    pooled = np.concatenate([patch.ravel() for patch in patches]).astype(np.float64)

    return {
        "pmin": float(np.percentile(pooled, 1)),
        "pmax": float(np.percentile(pooled, 99.6)),
        "mean": float(np.mean(pooled)),
        "std": float(np.std(pooled)),
    }


class ConfigDataset(Dataset):
    """
    Abstract class for datasets that are configured via a dictionary.
    """

    def __getitem__(self, index) -> np.ndarray:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    @classmethod
    def create_datasets(cls, dataset_config: dict, phase: str) -> list[Dataset]:
        """
        Factory method for creating a list of datasets based on the provided config.

        Args:
            dataset_config: dataset configuration
            phase: one of ['train', 'val', 'test']

        Returns:
            list of `Dataset` instances
        """
        raise NotImplementedError

    @classmethod
    def prediction_collate(cls, batch: list) -> Any:
        """Default collate_fn. Override in child class for non-standard datasets.

        Args:
            batch: List of samples from the dataset.

        Returns:
            Collated batch.
        """
        return default_prediction_collate(batch)


def _loader_classes(class_name):
    if class_name == "standard":
        return StandardTiffDataset
    elif class_name == "lazy":
        return LazyTiffDataset
    else:
        raise ValueError(f"Class {class_name} does not exist, choose standard or lazy")


def get_train_loaders(config: dict) -> dict[str, DataLoader]:
    """
    Returns dictionary containing the training and validation loaders (torch.utils.data.DataLoader).
    Args:
        config:  a top level configuration object containing the 'loaders' key
    Returns:
        dict {
            'train': <train_loader>
            'val': <val_loader>
        }
    """
    assert "loaders" in config, "Could not find data loaders configuration"
    loaders_config = config["loaders"]

    assert loaders_config["train"]["raw_path"] != loaders_config["val"]["raw_path"], (
        "Train and validation 'file_paths' overlap. One cannot use validation data for training!"
    )

    logger.info("Creating training and validation set loaders...")

    # get dataset class
    dataset_cls_str = loaders_config.get("dataset", None)
    if dataset_cls_str is None:
        dataset_cls_str = "standard"
        logger.warning(f"Cannot find dataset class in the config. Using default '{dataset_cls_str}'.")
    dataset_class = _loader_classes(dataset_cls_str)

    train_datasets = list(dataset_class.create_datasets(loaders_config, phase="train"))
    val_datasets = list(dataset_class.create_datasets(loaders_config, phase="val"))

    num_workers = loaders_config.get("num_workers", 1)
    logger.info(f"Number of workers for train/val dataloader: {num_workers}")

    batch_size = loaders_config.get("batch_size", 1)
    logger.info(f"Batch size for train/val loader: {batch_size}")

    return {
        "train": DataLoader(
            ConcatDataset(train_datasets),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        # don't shuffle during validation: useful when showing how predictions for a given batch get better over time
        "val": DataLoader(
            ConcatDataset(val_datasets),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }


def get_test_loaders(config: dict) -> collections.abc.Iterator[DataLoader]:
    """Returns test DataLoader.

    Args:
        config: A top level configuration object containing the 'loaders' key.

    Returns:
        Generator of DataLoader objects.
    """

    assert "loaders" in config, "Could not find data loaders configuration"
    loaders_config = config["loaders"]

    logger.info("Creating test set loaders...")

    # get dataset class
    dataset_cls_str = loaders_config.get("dataset", None)
    if dataset_cls_str is None:
        dataset_cls_str = "standard"
        logger.warning(f"Cannot find dataset class in the config. Using default '{dataset_cls_str}'.")
    dataset_class = _loader_classes(dataset_cls_str)

    test_datasets = dataset_class.create_datasets(loaders_config, phase="test")

    num_workers = loaders_config.get("num_workers", 1)
    logger.info(f"Number of workers for the dataloader: {num_workers}")

    batch_size = loaders_config.get("batch_size", 1)
    logger.info(f"Batch size for dataloader: {batch_size}")

    # use generator in order to create data loaders lazily one by one
    for test_dataset in test_datasets:
        collate_fn = test_dataset.prediction_collate

        yield DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            )


def default_prediction_collate(batch: list) -> Any:
    """Default collate_fn to form a mini-batch of Tensor(s) for Tiff based datasets.

    Args:
        batch: List of samples from the dataset.

    Returns:
        Collated batch.
    """
    error_msg = "batch must contain tensors or str; found {}"
    if isinstance(batch[0], str):
        return list(batch)
    elif isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], collections.abc.Sequence):
        transposed = zip(*batch, strict=True)
        return [default_prediction_collate(samples) for samples in transposed]

    raise TypeError(error_msg.format(type(batch[0])))
