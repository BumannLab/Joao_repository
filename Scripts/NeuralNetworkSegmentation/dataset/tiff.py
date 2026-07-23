from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from monai.data.utils import dense_patch_slices
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from Scripts.utils import get_images_path, read_tiff_image
from Scripts.NeuralNetworkSegmentation.augmentation.transforms_monai import AugmentationTransformer
from Scripts.NeuralNetworkSegmentation.utils import get_logger
from Scripts.NeuralNetworkSegmentation.dataset.utils import (
    ConfigDataset,
    PatientPatchSample,
    read_input,
    read_mask,
    calculate_stats,
    collect_patient_samples,
    )

logger = get_logger("TiffDataset")


def crop_window(volume_dchw: np.ndarray, window: tuple[slice, ...] | None) -> np.ndarray:
    """Crops a C,D,H,W (or 1,D,H,W) array to a (slice_d, slice_h, slice_w) window, keeping
    every channel.
    """
    slice_d, slice_h, slice_w = window
    return volume_dchw[:, slice_d, slice_h, slice_w]

@dataclass(frozen=True)
class PatchLocation:
    """One entry in a dataset's flat patch index.

    sample_idx: index into self.samples identifying which file this patch comes from.
    window: (slice_d, slice_h, slice_w) identifying where inside
        the file's volume this patch sits
    """

    sample_idx: int
    window: tuple[slice, ...] | None = None

class AbstractTiffDataset(ConfigDataset):
    """torch Dataset over tiff files, iterating patch by patch. Abstract base for the standard
    (in-memory) and lazy (read-from-disk) implementations.

    Args:
        samples: patches to draw from, one PatientPatchSample per input (and optional mask) file.
        phase: 'train', 'val', or 'test'.
        slice_builder_config: {"window_size": (D, H, W), "window_stride": (D, H, W)} controlling
            how each file is tiled into patches. Defaults to using the whole file as one patch.
        transformer_config: augmentation config forwarded to AugmentationTransformer.
        seed: seed for the augmentation pipeline's random state.
        generate_data: if True, yields each patch num_images times.
        num_images: number of times to repeat each patch when generate_data is True.
        channels_to_load: channel indices to keep; None keeps every channel.
        global_normalization: if True, mean/std/percentiles are computed over every sample in
            this group and used to normalize the data.
    """

    def __init__(
        self,
        samples: list[PatientPatchSample],
        phase: str,
        slice_builder_config: dict,
        transformer_config: dict,
        seed: int,
        generate_data: bool = False,
        num_images: int = 1,
        channels_to_load: list[int] | None = None,
        global_normalization: bool = False,
    ):
        assert phase in ["train", "val", "test"]
        logger.info(f"Creating {self.__class__.__name__} for {phase} phase from {samples}")
        self.phase = phase
        self.samples = samples
        self.generate_data = generate_data
        self.num_images = num_images
        self.channels_to_load = list(channels_to_load) if channels_to_load is not None else None
        self.has_mask = all(sample.mask_path is not None for sample in samples)

        if global_normalization:
            logger.info("Calculating mean and std of the raw data...")
            raw_samples = [self._select_channels(read_input(sample.input_path).astype(np.float64) / 65535) for sample in self.samples]
            stats = calculate_stats(raw_samples)
        else:
            stats = calculate_stats(None, True)

        random_state = np.random.RandomState(seed)
        self.transform = AugmentationTransformer(transformer_config, random_state, stats)._create_transform()
        
        if phase != "test":
            assert self.has_mask, (
                f"phase={phase!r} requires every sample to have a mask_path; a label-less "
                "group is only valid for phase='test' (real-data inference)."
            )

        self.volume_shape = read_input(self.samples[0].input_path).shape[1:]

        if self.has_mask:
            label = read_mask(self.samples[0].mask_path)
            # check that raw and label shapes match
            assert label.shape[1:] == self.volume_shape, "Raw and label shapes do not match"

        logger.info(f"Volume shape: {self.volume_shape}. Creating slices...")

        # build slice indices for raw and label data sets
        self.samples_slices = self.get_samples_slices(slice_builder_config)

        self.slices_count = len(self.samples_slices)

    def get_samples_slices(self, slice_builder_config):
        window_size = tuple(slice_builder_config.get("window_size", (0, 0, 0)))
        
        if window_size == (0, 0, 0):
            logger.info(f"Using whole tile for all samples")

        window_stride = slice_builder_config.get("window_stride", window_size)

        if window_stride == window_size:
            logger.info(f"Using non overlapping windows")

        slices = []
        for sample_idx, _ in enumerate(self.samples):
            windows = dense_patch_slices(self.volume_shape, window_size, window_stride)
            slices.extend(
                PatchLocation(sample_idx=sample_idx, window=window) for window in windows
            )
        return slices

    def _select_channels(self, array: np.ndarray) -> np.ndarray:
        # Always called on a C,D,H,W array,
        # so the channel axis is always 0.
        if self.channels_to_load is None:
            return array
        return np.take(array, indices=self.channels_to_load, axis=0)

    @abstractmethod
    def get_raw_patch(self, idx: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_label_patch(self, idx) -> np.ndarray:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple:
        if idx >= len(self):
            raise IndexError

        if self.generate_data:
            idx = idx // self.num_images

        location = self.samples_slices[idx]
        full_raw = self.get_raw_patch(location)
        raw_input = self._select_channels(full_raw)

        if self.phase == "test" and not self.has_mask:

            if self.transform is not None:
                # TODO: CHANGE VOLUME TO IMAGE, FULL_RAW TO FULL_IMAGE Change it in transforms as well
                transformed = self.transform(volume=raw_input, full_raw=full_raw) 
                raw_input = transformed["volume"].to(torch.float32)
                full_raw = transformed["full_raw"]

            file_name = self.samples[location.sample_idx].file_name
            return raw_input, full_raw, file_name

        else:
            label = self.get_label_patch(location)

            if self.transform is not None:
                # TODO: CHANGE VOLUME TO IMAGE, MASK3D TO MASK Change it in transforms as well
                transformed = self.transform(volume=raw_input, mask3d=label)
                raw_input = transformed["volume"].to(torch.float32)
                label = transformed["mask3d"].to(torch.long)

            # return the transformed raw and label patches
            return raw_input, label

    def __len__(self) -> int:
        return self.slices_count * self.num_images if self.generate_data else self.slices_count

    @classmethod
    def create_datasets(cls, loaders_config: dict, phase: str) -> Iterable["AbstractTiffDataset"]:
        phase_config = loaders_config[phase]

        # load data augmentation configuration
        transformer_config = phase_config["transforms"]
        seed = loaders_config["seed"]

        raw_path = phase_config["raw_path"]

        # label_path is optional for the test phase
        label_path = phase_config.get("label_path", None)

        # load slice builder config
        slice_builder_config = phase_config.get("slice_builder", {})

        # load files to process
        group_by = loaders_config.get("group_by", "patient")
        grouped_samples = collect_patient_samples(raw_path, label_path, group_by)

        for _, samples in grouped_samples.items():
            yield cls(
                samples = samples,
                phase = phase,
                transformer_config = transformer_config,
                seed = seed,
                slice_builder_config = slice_builder_config,
                generate_data = loaders_config.get("generate_data", False) if phase == "train" else False,
                num_images = loaders_config.get("num_images", 1) if phase == "train" else 1,
                channels_to_load = loaders_config.get("channels_to_load", None),
                global_normalization = loaders_config.get("global_normalization", False),
            )


class StandardTiffDataset(AbstractTiffDataset):
    """Loads each file into memory on first access, cached for the dataset's lifetime. Fast
    but memory heavy.
    """

    def __init__(
        self,
        samples: list[PatientPatchSample],
        phase: str,
        slice_builder_config: dict,
        transformer_config: dict,
        seed: int,
        generate_data: bool = False,
        num_images: int = 1,
        channels_to_load: list[int] | None = None,
        global_normalization: bool = False,
    ):
        super().__init__(
            samples=samples,
            phase=phase,
            slice_builder_config=slice_builder_config,
            transformer_config=transformer_config,
            seed=seed,
            generate_data=generate_data,
            num_images=num_images,
            channels_to_load=channels_to_load,
            global_normalization=global_normalization,
        )
        self._raw_cache: dict[int, np.ndarray] = {}
        self._label_cache: dict[int, np.ndarray] = {}

    def get_raw_patch(self, location: PatchLocation) -> np.ndarray:
        sample_idx = location.sample_idx
        if sample_idx not in self._raw_cache:
            self._raw_cache[sample_idx] = read_input(self.samples[sample_idx].input_path)
        return crop_window(self._raw_cache[sample_idx], location.window)

    def get_label_patch(self, location: PatchLocation) -> np.ndarray:
        sample_idx = location.sample_idx
        if sample_idx not in self._label_cache:
            self._label_cache[sample_idx] = read_mask(self.samples[sample_idx].mask_path)
        return crop_window(self._label_cache[sample_idx], location.window)


class LazyTiffDataset(AbstractTiffDataset):
    """Reads each file from disk on every access. Slower, but has a low memory footprint."""

    def __init__(
        self,
        samples: list[PatientPatchSample],
        phase: str,
        slice_builder_config: dict,
        transformer_config: dict,
        seed: int,
        generate_data: bool = False,
        num_images: int = 1,
        channels_to_load: list[int] | None = None,
        global_normalization: bool = False,
    ):
        super().__init__(
            samples=samples,
            phase=phase,
            slice_builder_config=slice_builder_config,
            transformer_config=transformer_config,
            seed=seed,
            generate_data=generate_data,
            num_images=num_images,
            channels_to_load=channels_to_load,
            global_normalization=global_normalization,
        )

    def get_raw_patch(self, location: PatchLocation) -> np.ndarray:
        raw = read_input(self.samples[location.sample_idx].input_path)
        return crop_window(raw, location.window)

    def get_label_patch(self, location: PatchLocation) -> np.ndarray:
        label = read_mask(self.samples[location.sample_idx].mask_path)
        return crop_window(label, location.window)
