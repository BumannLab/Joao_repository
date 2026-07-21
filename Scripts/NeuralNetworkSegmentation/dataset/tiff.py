from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from monai.data.utils import dense_patch_slices
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from Scripts.utils import get_images_path, read_image
from Scripts.NeuralNetworkSegmentation.augmentation.transforms_monai import AugmentationTransformer
from Scripts.NeuralNetworkSegmentation.utils import get_logger

logger = get_logger("TiffDataset")


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


def read_input(input_path: Path):

    input_image = read_image(input_path)  # Shape is CxHxW or DxCxHxW

    # CxHxW -> 1xCxHxW
    while input_image.ndim != 4:

        input_image = np.expand_dims(input_image, axis=0)

    return input_image


def read_mask(mask_path: Path):

    mask = read_image(mask_path)  # Shape is HxW or DxHxW

    mask[mask > 0] = 1

    # HxW -> 1x1xHxW; DxHxW -> Dx1xHxW
    while mask.ndim != 4:

        mask = np.expand_dims(mask, axis=-3)

    return mask.astype(np.long)


def crop_window(volume_dchw: np.ndarray, window: tuple[slice, ...] | None) -> np.ndarray:
    """Crops a D,C,H,W (or D,1,H,W) array to a (slice_d, slice_h, slice_w) window, keeping
    every channel. A no-op when window is None (static mode: the whole file is the patch).
    """
    if window is None:
        return volume_dchw
    slice_d, slice_h, slice_w = window
    return volume_dchw[slice_d, :, slice_h, slice_w]


def mirror_pad(volume_dchw: np.ndarray, halo_shape: tuple[int, int, int]) -> np.ndarray:
    """Pads a D,C,H,W (or D,1,H,W) array with a mirror reflection of itself along D,H,W by
    halo_shape on each side, leaving the channel axis untouched. A no-op when halo_shape is
    all zeros. Mirrors pytorch-3dunet's datasets.utils.mirror_pad, adapted to this module's
    D,C,H,W layout (pytorch-3dunet's is C,D,H,W).
    """
    if not any(halo_shape):
        return volume_dchw
    pad_d, pad_h, pad_w = halo_shape
    pad_width = ((pad_d, pad_d), (0, 0), (pad_h, pad_h), (pad_w, pad_w))
    return np.pad(volume_dchw, pad_width, mode="reflect")


def _pad_window(window: tuple[slice, slice, slice], halo_shape: tuple[int, int, int]) -> tuple[slice, slice, slice]:
    """Re-expresses a (slice_d, slice_h, slice_w) window so it indexes the matching region of
    a volume already mirror_pad-ed by halo_shape: each slice's start is unchanged and its stop
    is pushed out by 2 * halo, since the padded volume's own origin has shifted by halo.
    Mirrors pytorch-3dunet's datasets.hdf5._create_padded_indexes.
    """
    if not any(halo_shape):
        return window
    return tuple(slice(s.start, s.stop + 2 * halo) for s, halo in zip(window, halo_shape, strict=True))


@dataclass(frozen=True)
class _PatchLocation:
    """One entry in a dataset's flat patch index.

    sample_idx: index into self.samples identifying which file this patch comes from.
    window: None in "static" mode (the whole file is the patch). In "window" mode, the
        (slice_d, slice_h, slice_w) tuple (as produced by
        monai.data.utils.dense_patch_slices(..., return_slice=True)) identifying where inside
        the file's volume this patch sits -- multiple _PatchLocation entries share one
        sample_idx when a file is split into several windows.
    """

    sample_idx: int
    window: tuple[slice, ...] | None = None


class AbstractTiffDataset(ConfigDataset):
    """One instance per group -- a patient, or a single image/patch file, depending on
    group_by -- iterating over that group's patches via self._index.

    Args:
        group_id: identifier of this dataset's group: a patient_id when group_by="patient",
            or a file_name when group_by="image". self.patient_id (below) is always the true
            owning patient regardless of group_by.
        samples: every patch belonging to this group.
        phase: "train" for training, "val" for validation, "test" for testing/inference.
        transform_config: {"raw": [...], "label": [...], "full_raw": [...]} forwarded to
            transforms_monai.AugmentationTransformer once self.stats is available. "full_raw"
            only matters for label-less test (real inference) -- see module docstring.
        seed: seed forwarded to AugmentationTransformer.
        global_normalization: if True, pmin/pmax/mean/std are calculated over every sample in
            this group (after channel selection, see _read_stats_patch) and exposed via
            self.stats.
        channels_to_load: channel indices to keep, applied before the transform pipeline in
            every phase (so augmentation only ever touches channels that reach the model) --
            see module docstring for how label-less test (real inference) still gets the full,
            un-sliced raw image back out despite this.
        patch_mode: "static" (one sample is one patch, the whole file) or "window" (each
            sample's (D, H, W) volume is tiled into one or more windows via
            monai.data.utils.dense_patch_slices).
        patch_config: forwarded to the "window" index builder -- {"patch_size": (D, H, W),
            "scan_interval": (D, H, W), "halo_shape": (D, H, W)}. "scan_interval" (the window
            stride) defaults to "patch_size" (non-overlapping tiling) when omitted. Unused in
            "static" mode. Windowed inference stitching is handled by
            monai.inferers.SlidingWindowInferer(mode='gaussian'), see inference.py.
            "halo_shape" (default (0, 0, 0)) only applies to the raw patch, and only in
            "window" mode during label-less "test" (real inference): each window is read with
            halo_shape extra context mirror-padded onto every side (see mirror_pad/
            _pad_window), same trick as pytorch-3dunet's datasets.hdf5 halo_shape, so the model
            gets border context that a naive per-window crop would otherwise lack. The window
            recorded in _PatchLocation stays the exact, non-padded patch position; whatever
            stitches predictions back together is responsible for stripping the halo back off
            (pytorch-3dunet's predictor.py does this via remove_padding -- no counterpart is
            wired up here yet).

    Whether labels are read is driven by has_mask and phase: has_mask is required for
    train/val (asserted in __init__, since a silent crash inside read_mask(None) is worse than
    failing fast); a "test" phase with has_mask evaluates against ground truth (returns
    (raw, label) like train/val); a "test" phase without a mask is real-data inference and
    returns (raw_input, raw_full, file_name) instead.
    """

    def __init__(
        self,
        group_id: str,
        samples: list[PatientPatchSample],
        phase: str,
        transform_config: dict | None = None,
        seed: int = 47,
        global_normalization: bool = False,
        generate_data: bool = False,
        num_images: int = 1,
        channels_to_load: list[int] | None = None,
        patch_mode: str = "static",
        patch_config: dict | None = None,
    ):
        assert phase in ("train", "val", "test"), "phase must be one of: train, val, test."
        assert samples, "samples must be non-empty"

        self.group_id = group_id
        self.patient_id = samples[0].patient_id
        self.samples = samples
        self.phase = phase
        self.has_mask = all(sample.mask_path is not None for sample in samples)

        if phase != "test":
            assert self.has_mask, (
                f"phase={phase!r} requires every sample to have a mask_path; a label-less "
                "group is only valid for phase='test' (real-data inference)."
            )

        self.channels_to_load = list(channels_to_load) if channels_to_load is not None else None

        self.patch_mode = patch_mode
        self.patch_config = dict(patch_config) if patch_config else {}
        self.halo_shape = tuple(self.patch_config.get("halo_shape", (0, 0, 0)))
        self._index: list[_PatchLocation] = self._build_index()

        if self.patch_mode == "window" and phase == "test" and not self.has_mask and not any(self.halo_shape):
            logger.warning(
                "Found halo_shape to be (0, 0, 0) for a label-less 'window' mode test group. "
                "This might lead to checkerboard artifacts in the prediction. Consider using a "
                "non-zero halo shape, e.g. 'halo_shape: [8, 8, 8]' in patch_config."
            )

        if global_normalization:
            # stats are local to this group's samples, not a whole-corpus scan (same semantics
            # as data.py's global_normalization).
            patches = [self._read_stats_patch(sample) for sample in self.samples]
            self.stats = calculate_stats(patches)
        else:
            self.stats = calculate_stats(None, skip=True)

        # stats must be computed before the transform pipeline is built: this group's own
        # pmin/pmax/mean/std need to already exist so they can be handed to
        # AugmentationTransformer for this group's normalization.
        transform_config = transform_config or {"raw": [], "label": [], "full_raw": []}
        self.transform = AugmentationTransformer(transform_config, seed, data_stats=self.stats)._create_transform()

        self.generate_data = generate_data
        self.num_images = num_images

    def _read_stats_patch(self, sample: PatientPatchSample) -> np.ndarray:
        # divided by 65535 here so these stats are on the same [0, 1]-ish scale _prepare_raw
        # puts the volume on (via its float32 cast) before the transform pipeline ever sees it.
        raw = read_input(sample.input_path).astype(np.float64) / 65535  # D,C,H,W
        return self._select_channels(raw)

    def _build_index(self) -> list[_PatchLocation]:
        if self.patch_mode == "static":
            return [_PatchLocation(sample_idx=i) for i in range(len(self.samples))]

        if self.patch_mode == "window":
            patch_size = self.patch_config.get("patch_size")
            assert patch_size is not None, (
                "patch_config['patch_size'] is required for patch_mode='window'."
            )
            scan_interval = self.patch_config.get("scan_interval", patch_size)

            index: list[_PatchLocation] = []
            for sample_idx, sample in enumerate(self.samples):
                raw_dchw = read_input(sample.input_path)  # D,C,H,W
                image_size = (raw_dchw.shape[0], raw_dchw.shape[2], raw_dchw.shape[3])  # D,H,W
                windows = dense_patch_slices(image_size, patch_size, scan_interval)
                index.extend(
                    _PatchLocation(sample_idx=sample_idx, window=window) for window in windows
                )
            return index

        raise ValueError(f"Unknown patch_mode: {self.patch_mode!r}. Choose 'static' or 'window'.")

    def _select_channels(self, array: np.ndarray) -> np.ndarray:
        # Always called on a D,C,H,W array (before the C,D,H,W transpose _prepare_raw does),
        # so the channel axis is always 1.
        if self.channels_to_load is None:
            return array
        return np.take(array, indices=self.channels_to_load, axis=1)

    @abstractmethod
    def get_raw_patch(self, location: _PatchLocation) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_label_patch(self, location: _PatchLocation) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def is_lazy(self) -> bool:
        raise NotImplementedError

    def __len__(self) -> int:
        base = len(self._index)
        return base * self.num_images if self.generate_data else base

    def __getitem__(self, idx: int):
        if self.generate_data:
            idx = idx // self.num_images

        location = self._index[idx]
        raw_dchw = self.get_raw_patch(location)  # D,C,H,W, all channels, native dtype

        if not self.has_mask and self.phase == "test":
            return self._getitem_inference(location, raw_dchw)

        return self._getitem_labeled(location, raw_dchw)

    def _getitem_labeled(self, location: _PatchLocation, raw_dchw: np.ndarray):
        # channels_to_load applied before the transform pipeline: augmentation only ever
        # touches channels that actually reach the model.
        raw_dchw = self._select_channels(raw_dchw)
        raw_cdhw = self._prepare_raw(raw_dchw)

        label_dchw = self.get_label_patch(location)
        label_cdhw = self._prepare_label(label_dchw)

        if self.transform is not None:
            transformed = self.transform(volume=raw_cdhw, mask3d=label_cdhw)
            raw_cdhw = transformed["volume"]
            label_cdhw = transformed["mask3d"]

        return raw_cdhw, label_cdhw

    def _getitem_inference(self, location: _PatchLocation, raw_dchw: np.ndarray):
        # raw_input ("volume") and raw_full ("full_raw") are two independent dict-transform
        # targets fed into the SAME Compose call, so a geometric transform config entry
        # repeated (byte-identical) under both the "raw" and "full_raw" transform_config lists
        # applies to both with one shared random draw -- keeping them spatially aligned -- while
        # an entry listed only under "raw" (e.g. NormalizeIntensityd) only ever touches
        # "volume": monai.transforms.MapTransform silently skips any key it wasn't configured
        # for. So raw_full only picks up intensity-altering transforms if you explicitly repeat
        # them into "full_raw" too -- by default it stays at its original (geometric-transform-
        # only) scale. See transforms_monai.AugmentationTransformer's docstring.
        raw_input_dchw = self._select_channels(raw_dchw)
        raw_input_cdhw = self._prepare_raw(raw_input_dchw)
        raw_full_cdhw = self._prepare_raw(raw_dchw)

        if self.transform is not None:
            transformed = self.transform(volume=raw_input_cdhw, full_raw=raw_full_cdhw)
            raw_input_cdhw = transformed["volume"]
            raw_full_cdhw = transformed["full_raw"]

        file_name = self.samples[location.sample_idx].file_name
        return raw_input_cdhw, raw_full_cdhw, file_name

    @staticmethod
    def _prepare_raw(raw_dchw: np.ndarray) -> np.ndarray:
        # D,C,H,W -> C,D,H,W (monai channel-first layout). float32 for MONAI intensity/spatial
        # transforms; no baked-in /65535 -- scaling is left to a config-driven transform step
        # (e.g. NormalizeIntensityd), same philosophy as data.py.
        return np.transpose(raw_dchw, (1, 0, 2, 3)).astype(np.float32)

    @staticmethod
    def _prepare_label(label_dchw: np.ndarray) -> np.ndarray:
        # D,1,H,W -> 1,D,H,W. float32 so spatial/interpolating transforms (RandAffined,
        # RandRotated, ...) don't choke on an integer dtype -- any transform that spatially
        # resamples "mask3d" must use a nearest-neighbor mode for that key so label values stay
        # exact class indices, not fractional.
        return np.transpose(label_dchw, (1, 0, 2, 3)).astype(np.float32)

    @classmethod
    def create_datasets(cls, loaders_config: dict, phase: str) -> Iterable["AbstractTiffDataset"]:
        phase_config = loaders_config[phase]

        transform_config = phase_config.get("transforms", {"raw": [], "label": [], "full_raw": []})
        seed = loaders_config.get("seed", 47)

        global_normalization = loaders_config.get("global_normalization", False)
        channels_to_load = loaders_config.get("channels_to_load")
        patch_mode = loaders_config.get("patch_mode", "static")
        patch_config = loaders_config.get("patch_config")
        group_by = loaders_config.get("group_by", "patient")

        raw_path = phase_config["raw_path"]
        # label_path is optional for the test phase; whether it's configured decides if labels
        # get read, see __getitem__.
        label_path = phase_config.get("label_path")

        samples_by_patient = collect_patient_samples(raw_path, label_path)

        generate_data = bool(loaders_config.get("generate_data", False))
        num_images = int(loaders_config.get("num_images", 1))

        for group_id, samples in cls._regroup(samples_by_patient, group_by):
            # transform_config/seed are forwarded (not a pre-built transform): each group
            # builds its own pipeline in __init__, after computing its own stats.
            yield cls(
                group_id=group_id,
                samples=samples,
                phase=phase,
                transform_config=transform_config,
                seed=seed,
                global_normalization=global_normalization,
                generate_data=generate_data if phase == "train" else False,
                num_images=num_images,
                channels_to_load=channels_to_load,
                patch_mode=patch_mode,
                patch_config=patch_config,
            )

    @staticmethod
    def _regroup(
        samples_by_patient: dict[str, list[PatientPatchSample]],
        group_by: str,
    ) -> Iterable[tuple[str, list[PatientPatchSample]]]:
        if group_by == "patient":
            yield from samples_by_patient.items()
        elif group_by == "image":
            for samples in samples_by_patient.values():
                for sample in samples:
                    yield sample.file_name, [sample]
        elif group_by == "all":
            total_samples = list()
            for samples in samples_by_patient.values():
                total_samples.extend(samples)
            yield "all", total_samples 
        else:
            raise ValueError(f"Unknown group_by: {group_by!r}. Choose 'patient' or 'image'.")


class StandardTiffDataset(AbstractTiffDataset):
    """Loads each file into memory on first access, cached for the dataset's lifetime. Fast
    but memory heavy.

    The cache is keyed by sample_idx (not by flat patch index): in "window" mode, several
    _PatchLocation entries share one sample_idx, and keying by sample_idx means the underlying
    file is only ever read/cached once no matter how many windows are carved out of it. In
    "static" mode sample_idx and idx coincide 1:1.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._raw_cache: dict[int, np.ndarray] = {}
        self._label_cache: dict[int, np.ndarray] = {}

    def get_raw_patch(self, location: _PatchLocation) -> np.ndarray:
        sample_idx = location.sample_idx
        if sample_idx not in self._raw_cache:
            self._raw_cache[sample_idx] = read_input(self.samples[sample_idx].input_path)
        return crop_window(self._raw_cache[sample_idx], location.window)

    def get_label_patch(self, location: _PatchLocation) -> np.ndarray:
        sample_idx = location.sample_idx
        if sample_idx not in self._label_cache:
            self._label_cache[sample_idx] = read_mask(self.samples[sample_idx].mask_path)
        return crop_window(self._label_cache[sample_idx], location.window)

    def is_lazy(self) -> bool:
        return False


class LazyTiffDataset(AbstractTiffDataset):
    """Reads each file from disk on every access. Slower, but has a low memory footprint."""

    def get_raw_patch(self, location: _PatchLocation) -> np.ndarray:
        raw = read_input(self.samples[location.sample_idx].input_path)
        return crop_window(raw, location.window)

    def get_label_patch(self, location: _PatchLocation) -> np.ndarray:
        label = read_mask(self.samples[location.sample_idx].mask_path)
        return crop_window(label, location.window)

    def is_lazy(self) -> bool:
        return True


def _dataset_classes(class_name: str):
    if class_name == "standard":
        return StandardTiffDataset
    elif class_name == "lazy":
        return LazyTiffDataset
    else:
        raise ValueError(f"Class {class_name} does not exist, choose standard or lazy")


def get_train_loaders(loaders_config: dict) -> dict[str, DataLoader]:
    """Returns a dict with only 'train' and 'val' DataLoaders, each backed by a ConcatDataset
    of all per-group datasets for that phase.
    """
    loaders_config["seed"] = loaders_config.get("seed", 47)

    train_raw_path = loaders_config["train"]["raw_path"]
    val_raw_path = loaders_config["val"]["raw_path"]
    assert train_raw_path != val_raw_path, (
        "Train and validation 'raw_path' are identical. One cannot use validation data for training!"
    )

    dataset_class = _dataset_classes(loaders_config.get("dataset", "standard"))

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
        ),
        "val": DataLoader(
            ConcatDataset(val_datasets),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }


def get_test_loaders(loaders_config: dict) -> Iterable[DataLoader]:
    """Yields one DataLoader per group found in the 'test' phase (per patient, or per image if
    group_by="image"), so predictions/evaluation stay grouped instead of being pooled together.

    If loaders_config["test"]["label_path"] is set, each dataset has ground truth and every
    loader yields (raw, label) batches. If it's absent, this is real-data inference: every
    loader yields (raw_input, raw_full, file_name) batches instead, with no label ever read.
    Check `loader.dataset.has_mask` to tell which one you got.
    """
    loaders_config["seed"] = loaders_config.get("seed", 47)

    dataset_class = _dataset_classes(loaders_config.get("dataset", "standard"))

    test_datasets = dataset_class.create_datasets(loaders_config, phase="test")

    batch_size = loaders_config.get("batch_size", 16)
    num_workers = loaders_config.get("num_workers", 1)

    for test_dataset in test_datasets:
        yield DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
