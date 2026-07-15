from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader

from Scripts.utils import get_images_path, read_image
from Scripts.NeuralNetworkSegmentation.transforms import AugmentationTransformer

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

    samples_by_patient: dict[str, list[PatientPatchSample]] = defaultdict(list)

    for input_file, mask_file in zip(input_paths, mask_paths):
        file_name = input_file.stem
        patient_id = extract_patient_id(file_name)

        samples_by_patient[patient_id].append(
            PatientPatchSample(
                patient_id=patient_id,
                file_name=file_name,
                input_path=Path(input_file),
                mask_path=Path(mask_file) if mask_file is not None else None,
            )
        )

    return dict(samples_by_patient)


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
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @classmethod
    def create_datasets(cls, dataset_config: dict, phase: str):
        raise NotImplementedError
    

class AbstractMycobacteriaTuberculosisDataset(ConfigDataset):
    """One instance per patient, iterating over that patient's patches.

    Args:
        patient_id: patient identifier parsed from the patch file names.
        samples: every patch belonging to this patient.
        phase: 'train' for training, 'val' for validation, 'test' for testing/inference.
        transform_config: this patient's phase transform config, forwarded to
            AugmentationTransformer once self.stats is available.
        seed: seed forwarded to AugmentationTransformer.
        global_normalization: if True, pmin/pmax/mean/std are calculated over every patch
            belonging to this patient and exposed via `self.stats`.

    Whether labels are read is driven by whether a mask_path is present on the samples and if the phase is test,
    : a 'test' phase with label_path configured evaluates against ground truth (returns
    (raw, label) like train/val, so it plugs into ModelTrainer.test()); a 'test' phase without
    label_path is real-data inference and returns (raw, file_name) instead.
    """

    def __init__(
        self,
        patient_id: str,
        samples: list[PatientPatchSample],
        phase: str,
        transform_config: list | None = None,
        seed: int = 47,
        global_normalization: bool = False,
        generate_data: bool = False,
        num_images: int = 1,
    ):
        assert phase in ("train", "val", "test"), "phase must be one of: train, val, test."

        self.patient_id = patient_id
        self.samples = samples
        self.phase = phase
        self.has_mask = all(sample.mask_path is not None for sample in samples)

        if global_normalization:
            # divided by 65535 here so these stats are on the same [0, 1] scale _prepare_raw
            # puts the volume on before the transform pipeline (and Normalize) ever sees it.
            patches = [
                read_input(sample.input_path) / 65535
                for sample in self.samples
            ]
            self.stats = calculate_stats(patches)
        else:
            self.stats = calculate_stats(None, skip=True)

        # stats must be computed before the transform pipeline is built: this patient's own
        # pmin/pmax/mean/std need to already exist so they can be handed to
        # AugmentationTransformer for this patient's normalization.
        self.transform = AugmentationTransformer(transform_config, seed, data_stats=self.stats)._create_transform()
        
        self.generate_data = generate_data
        self.num_images = num_images

    @abstractmethod
    def get_raw_patch(self, idx: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_label_patch(self, idx: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def is_lazy(self) -> bool:
        raise NotImplementedError

    def __len__(self) -> int:
        
        if self.generate_data:
            return len(self.samples) * self.num_images

        return len(self.samples)


    def __getitem__(self, idx: int):
        
        if self.generate_data:
            idx = idx // self.num_images

        raw = self._prepare_raw(self.get_raw_patch(idx))

        if not self.has_mask and self.phase == "test":
            # No ground truth for this patient (real-data inference, or a 'test' phase with no
            # label_path configured): the label dataset is never read.
            if self.transform is not None:
                raw = self.transform(volume=raw)["volume"]

            raw_tensor = raw

            # return the sample identifier alongside the patch so callers can place/save predictions
            return raw_tensor, self.samples[idx].file_name

        label = self._prepare_label(self.get_label_patch(idx))

        if self.transform is not None:
            transformed = self.transform(volume=raw, mask3d=label)
            raw = transformed["volume"]
            label = transformed["mask3d"]

        raw_tensor = raw
        label_tensor = label

        return raw_tensor, label_tensor

    @staticmethod
    def _prepare_raw(raw: np.ndarray) -> np.ndarray:
        # raw = (raw / 65535).astype(np.float32)
        
        # D,C,H,W -> D,H,W,C (albumentations volume layout)
        return np.transpose(raw, (0, 2, 3, 1))

    @staticmethod
    def _prepare_label(label: np.ndarray) -> np.ndarray:
        # label = (label / 255).astype(np.long)
        return np.transpose(label, (0, 2, 3, 1))

    @classmethod
    def create_datasets(cls, loaders_config: dict, phase: str) -> Iterable["AbstractMycobacteriaTuberculosisDataset"]:
        phase_config = loaders_config[phase]

        transform_config = phase_config.get("transforms", [])
        seed = loaders_config.get("seed", 47)

        global_normalization = loaders_config.get("global_normalization", False)

        raw_path = phase_config["raw_path"]
        # label_path is optional for the test phase; whether it's configured decides if labels get read, see __getitem__.
        label_path = phase_config.get("label_path")

        samples_by_patient = collect_patient_samples(raw_path, label_path)

        generate_data = bool(loaders_config.get("generate_data", False))
        num_images = int(loaders_config.get("num_images", 1))

        for patient_id, samples in samples_by_patient.items():
            # transform_config/seed are forwarded (not a pre-built transform): each patient
            # builds its own pipeline in __init__, after computing its own stats.
            yield cls(
                patient_id=patient_id,
                samples=samples,
                phase=phase,
                transform_config=transform_config,
                seed=seed,
                global_normalization=global_normalization,
                generate_data=generate_data if phase == "train" else False,
                num_images=num_images,
            )


class StandardMycobacteriaTuberculosisDataset(AbstractMycobacteriaTuberculosisDataset):
    """Loads a patient's patches into memory on first access. Fast but memory heavy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._raw = None
        self._label = None

    def _load_raw(self) -> None:
        if self._raw is not None:
            return

        self._raw = [
            read_input(sample.input_path)
            for sample in self.samples
        ]

    def get_raw_patch(self, idx: int) -> np.ndarray:
        self._load_raw()
        return self._raw[idx]

    def get_label_patch(self, idx: int) -> np.ndarray:
        if self._label is None:
            self._label = [read_mask(sample.mask_path) for sample in self.samples]
        return self._label[idx]

    def is_lazy(self) -> bool:
        return False


class LazyMycobacteriaTuberculosisDataset(AbstractMycobacteriaTuberculosisDataset):
    """Reads each patch from disk on every access. Slower, but has a low memory footprint."""

    def get_raw_patch(self, idx: int) -> np.ndarray:
        return read_input(self.samples[idx].input_path)

    def get_label_patch(self, idx: int) -> np.ndarray:
        return read_mask(self.samples[idx].mask_path)

    def is_lazy(self) -> bool:
        return True


def read_input(input_path: Path):

    input_image = read_image(input_path) # Shape is CxHxW or DxCxHxW

    # CxHxW -> 1xCxHxW
    while input_image.ndim != 4:

        input_image = np.expand_dims(input_image, axis=0)

    return input_image


def read_mask(mask_path: Path):

    mask = read_image(mask_path) # Shape is HxW or DxHxW

    mask[mask > 0] = 1

    # HxW -> 1x1xHxW; DxHxW -> Dx1xHxW
    while mask.ndim != 4: 
       
        mask = np.expand_dims(mask, axis=-3) 

    return mask.astype(np.long)


def _loader_classes(class_name: str):
    if class_name == "standard":
        return StandardMycobacteriaTuberculosisDataset
    elif class_name == "lazy":
        return LazyMycobacteriaTuberculosisDataset
    else:
        raise ValueError(f"Class {class_name} does not exist, choose standard or lazy")


def get_train_loaders(loaders_config: dict) -> dict[str, DataLoader]:
    """Returns a dict with only 'train' and 'val' DataLoaders, each backed by a ConcatDataset
    of all per-patient datasets for that phase.
    """
    loaders_config["seed"] = loaders_config.get("seed", 47)

    train_raw_path = loaders_config["train"]["raw_path"]
    val_raw_path = loaders_config["val"]["raw_path"]
    assert train_raw_path != val_raw_path, (
        "Train and validation 'raw_path' are identical. One cannot use validation data for training!"
    )

    dataset_cls_str = loaders_config.get("dataset", "standard")
    dataset_class = _loader_classes(dataset_cls_str)

    train_datasets = list(dataset_class.create_datasets(loaders_config, phase="train"))
    val_datasets = list(dataset_class.create_datasets(loaders_config, phase="val"))

    batch_size = loaders_config.get("batch_size", 16)
    num_workers = loaders_config.get("num_workers", 1)

    return {
        "train": DataLoader(
            ConcatDataset(train_datasets),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            # generator=_torch_generator(seed),
            # worker_init_fn=_seed_worker(seed) if seed is not None else None,
            # pin_memory=bool(loader_config.get("pin_memory", False)),
            # drop_last=bool(loader_config.get("drop_last", False)),
        ),
        "val": DataLoader(
            ConcatDataset(val_datasets),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            # worker_init_fn=_seed_worker(seed) if seed is not None else None,
            # pin_memory=bool(loader_config.get("pin_memory", False)),
            # drop_last=bool(loader_config.get("drop_last", False)),
        ),
    }


def get_test_loaders(loaders_config: dict) -> Iterable[DataLoader]:
    """Yields one DataLoader per patient found in the 'test' phase, so predictions/evaluation
    stay grouped per patient/volume instead of being pooled together.

    If loaders_config["test"]["label_path"] is set, each dataset has ground truth and every
    loader yields (raw, label) batches, e.g. for ModelTrainer.test() to compute loss/eval
    metrics/confusion matrix. If it's absent, this is real-data inference: every loader yields
    (raw, file_name) batches instead, with no label ever read. Check `loader.dataset.has_mask`
    to tell which one you got.
    """
    loaders_config["seed"] = loaders_config.get("seed", 47)

    dataset_cls_str = loaders_config.get("dataset", "standard")
    dataset_class = _loader_classes(dataset_cls_str)

    test_datasets = dataset_class.create_datasets(loaders_config, phase="test")

    batch_size = loaders_config.get("batch_size", 16)
    num_workers = loaders_config.get("num_workers", 1)

    for test_dataset in test_datasets:
        yield DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            # worker_init_fn=_seed_worker(seed) if seed is not None else None,
            # pin_memory=bool(loader_config.get("pin_memory", False)),
            # drop_last=bool(loader_config.get("drop_last", False)),
        )

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

