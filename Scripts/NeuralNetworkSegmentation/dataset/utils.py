from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import collections
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

from Scripts.utils import get_images_path, read_image
from Scripts.NeuralNetworkSegmentation.utils import get_logger
from compute_direction_and_cut_overlap import (
    get_all_tiles_directions_to_remove_overlap,
)

logger = get_logger("Dataset")

@dataclass(frozen=True)
class PatientPatchSample:
    patient_id: str
    file_name: str
    input_path: Path
    mask_path: Path | None = None
    directions_remove_overlap: list[str] | None = None


def extract_patient_id(file_name: str) -> str:
    """patient1_x_y_z -> patient1; MAX_patient1_x_y_z -> patient1 (max-projection prefix)"""
    parts = file_name.split("_")
    if parts[0] == "MAX":
        return parts[1]
    return parts[0]


def collect_patient_samples(
    input_path: str | Path,
    mask_path: str | Path | None = None,
    group_by: str = "patient",
    overlap: int = 0,
) -> dict[str, list[PatientPatchSample]]:

    input_paths = get_images_path(input_path, ".tif", ".tiff", ".nd2")

    if 0 < overlap < 100:
        all_directions_remove_overlap = get_all_tiles_directions_to_remove_overlap(
            input_paths
        )
    else:
        all_directions_remove_overlap = [[""]] * len(input_paths)

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
    for input_file, mask_file, directions_remove_overlap in zip(input_paths, mask_paths, all_directions_remove_overlap):
        file_name = input_file.stem
        patient_id = extract_patient_id(file_name)
        
        patient_raw_label = PatientPatchSample(
                patient_id=patient_id,
                file_name=file_name,
                input_path=Path(input_file),
                mask_path=Path(mask_file) if mask_file is not None else None,
                directions_remove_overlap=directions_remove_overlap if directions_remove_overlap is not None else None,
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

    input_image = read_image(input_path)  # Shape is CxHxW or DxCxHxW

    # CxHxW -> 1xCxHxW
    while input_image.ndim != 4:

        input_image = np.expand_dims(input_image, axis=0)

    return np.transpose(input_image, (1, 0, 2, 3)).astype(np.float32)


def read_mask(mask_path: Path):

    mask = read_image(mask_path)  # Shape is HxW or DxHxW

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
    # Imported here, not at module level, to avoid a circular import: tiff.py imports
    # from this module too (read_input, read_mask, ConfigDataset, ...).
    from Scripts.NeuralNetworkSegmentation.dataset.tiff import StandardTiffDataset, LazyTiffDataset

    if class_name == "standard":
        return StandardTiffDataset
    elif class_name == "lazy":
        return LazyTiffDataset
    else:
        raise ValueError(f"Class {class_name} does not exist, choose standard or lazy")


def _is_positive_sample(dataset, idx: int) -> bool:
    # generate_data repeats each physical patch num_images times (idx 0,1,2 all point at the
    # same underlying file when num_images=3), so divide it back out to find the real patch
    # before resolving which file this index actually comes from.
    raw_idx = idx // dataset.num_images if dataset.generate_data else idx
    # samples_slices[raw_idx].sample_idx is the index into dataset.samples for this patch;
    # its file_name still carries the original "_positive_"/"_negative_" marker set when the
    # dataset was generated, so no need to touch pixel data to know which pool it belongs to.
    sample_idx = dataset.samples_slices[raw_idx].sample_idx
    return "_positive_" in dataset.samples[sample_idx].file_name


def _build_positive_negative_indices(concat_dataset: ConcatDataset) -> tuple[list[int], list[int]]:
    """Splits a ConcatDataset's global indices into positive (contains foreground) and
    negative (empty mask) pools, using the '_positive_'/'_negative_' marker already present
    in each patch's file name.
    """
    positive, negative = [], []
    offset = 0
    # ConcatDataset stitches several per-group datasets end to end; a "global" index i into
    # the concatenated whole is just (offset of that group's dataset) + (i within that
    # dataset), so walk each underlying dataset in turn and shift its local indices by how
    # many samples came before it.
    for dataset in concat_dataset.datasets:
        for i in range(len(dataset)):
            (positive if _is_positive_sample(dataset, i) else negative).append(offset + i)
        offset += len(dataset)
    return positive, negative


class MinPositiveBatchSampler(Sampler):
    """Yields batches of `batch_size`, each guaranteed at least `min_positive` positive
    (foreground-containing) indices. Most of the dataset is empty-mask background patches, so
    plain random shuffling occasionally draws an all-negative batch, which the Dice loss handles
    poorly (near-zero denominator), producing an outsized, destabilizing gradient. Guaranteeing a
    positive floor per batch removes that failure case at the source instead of dampening its
    effect downstream (loss smoothing / gradient clipping).
    """

    def __init__(self, positive: list[int], negative: list[int], batch_size: int, min_positive: int, generator=None):
        assert min_positive <= batch_size
        self.positive = positive
        self.negative = negative
        self.batch_size = batch_size
        self.min_positive = min_positive
        self.generator = generator  # optional torch.Generator; None -> uses torch's global RNG,
                                     # the same one seed_everything(seed) seeds at run start.

        # How many full-size batches fit, and how many samples are left over once they do.
        # E.g. total=1746, batch_size=9 -> 194 full batches, remainder=0 (divides evenly).
        # A non-divisible total (e.g. total=100) would give num_full_batches=11, remainder=1.
        total = len(positive) + len(negative)
        self.num_full_batches = total // batch_size
        self.remainder = total % batch_size

        # If there's a leftover (partial) batch, decide how many positives it gets guaranteed.
        # Capped three ways: never more than min_positive itself, never more than the partial
        # batch's own size (self.remainder), and never more than what's actually left in the
        # positive pool after every full batch has already taken its min_positive share
        # (available_for_partial). That third cap matters because Python slicing doesn't error
        # on an out-of-range end -- without it, an over-ambitious reservation would silently
        # truncate in __iter__ below, producing an undersized, wrongly-composed partial batch.
        available_for_partial = max(0, len(positive) - self.num_full_batches * min_positive)
        self.partial_min_positive = min(min_positive, self.remainder, available_for_partial) if self.remainder else 0

    def __iter__(self):
        # --- Step 1: reserve the guaranteed-positive "floor" pool. ---
        # Shuffle every positive index once, then slice off the front: exactly enough for
        # min_positive per full batch, plus partial_min_positive for the trailing batch (if
        # any). This slice (floor_pool) is spoken for -- nothing else will touch it. Whatever
        # positives didn't make the cut (leftover_positive) still get used, just not guaranteed.
        pos_shuffled = [self.positive[i] for i in torch.randperm(len(self.positive), generator=self.generator).tolist()]
        floor_needed = self.num_full_batches * self.min_positive + self.partial_min_positive
        floor_pool, leftover_positive = pos_shuffled[:floor_needed], pos_shuffled[floor_needed:]

        # --- Step 2: pool everything else and shuffle it once too. ---
        # leftover_positive (positives that missed the floor cut) + every negative, combined
        # and reshuffled. This is what fills the rest of each batch after its floor is taken --
        # since it still contains positives, a batch can end up with *more* than the guaranteed
        # floor by chance (the floor is a minimum, not a cap).
        remaining_pool = leftover_positive + self.negative
        remaining_shuffled = [remaining_pool[i] for i in torch.randperm(len(remaining_pool), generator=self.generator).tolist()]

        # --- Step 3: hand out both pools batch by batch, front to back, no replacement. ---
        # floor_i and remaining_i each only ever move forward across the whole epoch (never
        # reset per batch, never wrap around) -- that's what guarantees every sample is used at
        # most once. Each batch takes its fixed floor share, then tops up to batch_size from
        # the remaining pool.
        floor_i = remaining_i = 0
        total_yielded = 0
        for _ in range(self.num_full_batches):
            batch = floor_pool[floor_i:floor_i + self.min_positive]
            floor_i += self.min_positive
            fill = self.batch_size - self.min_positive
            batch += remaining_shuffled[remaining_i:remaining_i + fill]
            remaining_i += fill
            total_yielded += len(batch)
            yield batch

        # --- Step 4: the trailing partial batch, if the dataset didn't divide evenly. ---
        # Same mechanism as above, just sized down to self.remainder instead of batch_size, and
        # using partial_min_positive instead of min_positive as its floor. This is what replaced
        # the original drop_last-style behaviour: instead of discarding whatever didn't fit into
        # a full batch, it ships as one smaller batch that still gets a (capped) positive floor.
        if self.remainder:
            batch = floor_pool[floor_i:floor_i + self.partial_min_positive]
            floor_i += self.partial_min_positive
            fill = self.remainder - self.partial_min_positive
            batch += remaining_shuffled[remaining_i:remaining_i + fill]
            remaining_i += fill
            total_yielded += len(batch)
            yield batch

        # Safety sign: this should always equal len(positive)+len(negative) -- if it doesn't,
        # something upstream (e.g. a future edit to the reservation math) is dropping or
        # duplicating samples again.
        expected_total = len(self.positive) + len(self.negative)
        logger.info(
            f"MinPositiveBatchSampler epoch complete: {total_yielded} samples yielded across "
            f"{self.num_full_batches + (1 if self.remainder else 0)} batches "
            f"(expected {expected_total}, {'OK' if total_yielded == expected_total else 'MISMATCH'})."
        )

    def __len__(self):
        return self.num_full_batches + (1 if self.remainder else 0)


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

    train_concat = ConcatDataset(train_datasets)

    # use_balanced_sampler toggles MinPositiveBatchSampler on/off from the config, e.g. for an
    # ablation run comparing against plain random shuffling. Defaults to True so existing
    # configs that don't set it keep the current (balanced) behaviour.
    use_balanced_sampler = loaders_config.get("use_balanced_sampler", True)

    if use_balanced_sampler:
        positive, negative = _build_positive_negative_indices(train_concat)
        num_batches = len(train_concat) // batch_size
        # floor(positive / num_batches): the largest per-batch floor for which the sampler can draw
        # its guaranteed positives from the shuffled positive pool without reusing one twice in the
        # same epoch (any higher and the floor-filling step would have to wrap around early).
        min_positive = max(1, min(batch_size, len(positive) // max(1, num_batches)))
        logger.info(
            f"Train set: {len(positive)} positive / {len(negative)} negative patches. "
            f"Guaranteeing >= {min_positive} positive patch(es) per batch."
        )
        train_loader = DataLoader(
            train_concat,
            batch_sampler=MinPositiveBatchSampler(positive, negative, batch_size, min_positive),
            num_workers=num_workers,
        )
    else:
        logger.info("use_balanced_sampler=False: using plain random shuffling for the train loader.")
        train_loader = DataLoader(
            train_concat,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    return {
        "train": train_loader,
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
    error_msg = "batch must contain tensors, str, lists, or None; found {}"
    if isinstance(batch[0], (str, list, type(None))):
        return list(batch)
    elif isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], np.ndarray):
        return np.stack(batch, 0)
    elif isinstance(batch[0], collections.abc.Sequence):
        transposed = zip(*batch, strict=True)
        return [default_prediction_collate(samples) for samples in transposed]

    raise TypeError(error_msg.format(type(batch[0])))
