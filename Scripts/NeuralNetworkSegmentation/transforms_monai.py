import numpy as np
import torch
from scipy.ndimage import uniform_filter, median_filter

from monai.transforms import Compose, MapTransform, OneOf, RandomizableTransform

from Scripts.NeuralNetworkSegmentation.utils import get_class

# monai.transforms works dictionary-first: every entry is a *d-suffixed MapTransform that
# declares which keys ("volume", "mask3d", ...) it touches, rather than auto-detecting which
# registered targets it supports the way albumentations does. AugmentationTransformer below
# derives that "keys" argument for every transform from where it's placed in the "raw"/"label"
# config lists, so config entries themselves don't need to name "volume"/"mask3d" directly.
#
# Arrays are expected in monai's native channel-first layout, i.e. (C, D, H, W), rather than
# the (D, H, W, C) layout used by transforms.py's albumentations pipeline.


class SliceReplication3DFilterd(RandomizableTransform, MapTransform):
    """Dict-transform mirroring transforms.SliceReplication3DFilter: increases depth
    resolution by repeating slices along the depth axis, then smooths the result with a 3D
    filter. "mask3d" is always smoothed with a median filter regardless of filter_type,
    matching the per-target behavior of the albumentations version.

    Expects channel-first arrays, (C, D, H, W), with D (depth) as axis=1.

    Args:
        keys (KeysCollection): keys to apply the transform to. Default: ("volume", "mask3d").
        num_slice_replications (int): number of times to copy all slices. Default: 1.
        filter_type (str): 3D filter applied to "volume" after slice replication. One of
            "mean", "median". Default: "mean".
        kernel_size (int): kernel size in 3D; (1, k, k, k) so the channel axis is never
            filtered across. Default: 3.
        prob (float): probability of applying the transform. Default: 0.5.
        allow_missing_keys (bool): don't raise if a key (e.g. "mask3d" during label-less
            inference) is missing. Default: True.
    """

    def __init__(
        self,
        keys=("volume", "mask3d"),
        num_slice_replications=1,
        filter_type="mean",
        kernel_size=3,
        prob=0.5,
        allow_missing_keys=True,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.num_slice_replications = num_slice_replications
        self.kernel_size = kernel_size

        if filter_type == "mean":
            self.filter_func = uniform_filter
        elif filter_type == "median":
            self.filter_func = median_filter
        else:
            raise ValueError(f"Unknown filter_type: {filter_type}")

    def __call__(self, data):

        d = dict(data)

        self.randomize(None)
        if not self._do_transform:
            return d

        repeats = self.num_slice_replications + 1
        size = (1, self.kernel_size, self.kernel_size, self.kernel_size)

        for key in self.key_iterator(d):
            array = np.repeat(_to_numpy(d[key]), repeats=repeats, axis=1)

            filter_func = median_filter if key == "mask3d" else self.filter_func
            d[key] = filter_func(array, size=size)

        return d


class OneHotd(MapTransform):
    """Dict-transform mirroring transforms.OneHot: turns an integer label mask into a
    one-hot encoded array along the channel axis. Always applied, there is no probability.

    Expects a channel-first mask, (1, D, H, W); output is (num_channels, D, H, W).

    Args:
        keys (KeysCollection): keys to one-hot encode. Default: ("mask3d",).
        num_channels (int): size of the output channel dimension. If None, inferred as
            max(mask) + 1. If given, must be >= the number of classes actually present.
            Default: None.
        allow_missing_keys (bool): don't raise if a key is missing. Default: True.
    """

    def __init__(self, keys=("mask3d",), num_channels=None, allow_missing_keys=True):
        super().__init__(keys, allow_missing_keys)
        self.num_channels = num_channels

    def __call__(self, data):

        d = dict(data)

        for key in self.key_iterator(d):
            mask = _to_numpy(d[key])

            n_channels = self.num_channels
            if n_channels is None:
                n_channels = int(mask.max()) + 1

            assert n_channels >= int(mask.max()) + 1, (
                f"n_channels={n_channels} is smaller or equal than the number of classes"
                f"found in mask ({int(mask.max()) + 1})"
            )

            one_hot = np.eye(n_channels, dtype=np.float32)[mask[0].astype(np.int64)]  # D, H, W, n_channels

            d[key] = np.transpose(one_hot, (3, 0, 1, 2))  # n_channels, D, H, W

        return d


def _to_numpy(array):
    return array.detach().cpu().numpy() if torch.is_tensor(array) else np.asarray(array)


class _MonaiTransformPipeline:
    """Callable wrapper around a monai.transforms.Compose so it can be invoked with the same
    volume=/mask3d= keyword interface as the albumentations pipeline built by
    transforms.AugmentationTransformer, keeping the two backends swappable without touching
    call sites such as dataset/data.py.
    """

    def __init__(self, compose):
        self._compose = compose

    def __call__(self, volume, mask3d=None):

        data = {"volume": volume}
        if mask3d is not None:
            data["mask3d"] = mask3d

        transformed = self._compose(data)

        return {key: _to_numpy(transformed[key]) for key in data}


def _assign_keys(augmentation, keys):
    """Injects "keys" into a raw/label config entry so _create_augmentation knows which
    dict-transform target(s) to build it for. "OneOf" entries don't take "keys" themselves --
    monai.transforms.OneOf just groups other transforms, it isn't a MapTransform -- so the key
    set is instead pushed down into each of its nested "transforms" entries."""

    augmentation = dict(augmentation)

    if augmentation.get("name") == "OneOf":
        augmentation["transforms"] = [
            _assign_keys(nested, keys) for nested in augmentation.get("transforms", [])
        ]
    else:
        augmentation.setdefault("keys", list(keys))

    return augmentation


class AugmentationTransformer:
    def __init__(self, phase_transform_config, seed, data_stats=None):

        """Factory class for creating data augmentation pipelines, backed by monai.transforms
        instead of albumentations.

        Unlike transforms.AugmentationTransformer's single flat list, phase_transform_config
        here is a dict with "raw" and "label" lists of {"name": <transform class name>,
        **kwargs} dicts, matching the raw/label naming dataset/data.py already uses:

            {
                "raw": [{"name": "RandGaussianNoised", "prob": 0.5, "std": 0.1}],
                "label": [],
            }

        A "raw" entry applies only to "volume"; a "label" entry only to "mask3d". If the same
        "name" appears in both lists, the two entries are merged into a single transform
        applied to both "volume" and "mask3d" (so e.g. a crop or flip lands on the same
        location/orientation in both) -- but only once their parameters are checked to be
        identical, since a mismatch there would silently desync image and mask. Repeat the
        exact same entry under both "raw" and "label" to opt into that behavior.

        "OneOf" grouping is supported the same way as transforms.AugmentationTransformer,
        though monai's OneOf picks via "weights" rather than a top-level "p". Which keys a
        OneOf's branches run on is decided by where the OneOf itself was matched (raw-only,
        label-only, or merged), and that key set is propagated into every one of its nested
        "transforms" entries.
        """

        self.phase_transform_config = phase_transform_config
        self.seed = seed
        self.data_stats = data_stats or {}

    def _create_transform(self):

        merged_config = self._merge_raw_label(
            self.phase_transform_config.get("raw", []),
            self.phase_transform_config.get("label", []),
        )

        compose = Compose(
            [self._create_augmentation(augmentation) for augmentation in merged_config],
        )
        compose.set_random_state(seed=self.seed)

        return _MonaiTransformPipeline(compose)

    @staticmethod
    def _merge_raw_label(raw_config, label_config):

        label_config = list(label_config)
        consumed = [False] * len(label_config)
        merged = []

        for raw_entry in raw_config:
            name = raw_entry.get("name")

            match_idx = next(
                (
                    i for i, label_entry in enumerate(label_config)
                    if not consumed[i] and label_entry.get("name") == name
                ),
                None,
            )

            if match_idx is None:
                merged.append(_assign_keys(raw_entry, ["volume"]))
                continue

            label_entry = label_config[match_idx]
            consumed[match_idx] = True

            assert raw_entry == label_entry, (
                f"'{name}' appears in both raw and label transforms with different "
                f"parameters, raw={raw_entry} label={label_entry}. Give it identical "
                f"parameters in both entries so the same draw is applied to volume and mask3d."
            )

            merged.append(_assign_keys(raw_entry, ["volume", "mask3d"]))

        merged.extend(
            _assign_keys(label_entry, ["mask3d"])
            for i, label_entry in enumerate(label_config)
            if not consumed[i]
        )

        return merged

    def _create_augmentation(self, augmentation):

        augmentation = augmentation.copy()

        assert "name" in augmentation, f"No name found in {augmentation}"

        augmentation_name = augmentation.pop("name")

        if augmentation_name == "NormalizeIntensityd":

            if all(self.data_stats.get(stat) is not None for stat in ["mean", "std"]):

                # setdefault: an explicit subtrahend/divisor in the YAML entry still wins.
                augmentation.setdefault("subtrahend", self.data_stats["mean"])
                augmentation.setdefault("divisor", self.data_stats["std"])

        if augmentation_name == "OneOf":
            assert "transforms" in augmentation, f"No transforms found in {augmentation_name}: {augmentation}"

            return self._create_one_of(
                augmentation.pop("transforms"),
                **augmentation,
            )

        _transform = get_class(augmentation_name, modules=["monai.transforms", "Scripts.NeuralNetworkSegmentation.transforms_monai"])
        augmentation.setdefault("allow_missing_keys", True)
        _params = augmentation

        return _transform(**_params)

    def _create_one_of(self, one_of_config, **params):

        one_of_transforms = [
            self._create_augmentation(one_of_augmentation)
            for one_of_augmentation in one_of_config
        ]

        return OneOf(one_of_transforms, **params)


if __name__ == "__main__":
    slice_replication = {
        "name": "SliceReplication3DFilterd",
    }
    one_of = {
        "name": "OneOf",
        "transforms": [
            {
                "name": "CenterSpatialCropd",
                "roi_size": [-1, 128, 128],
            },
            {
                # per-key "mode" list ([volume, mask3d]) lives inside the shared entry itself,
                # so raw's and label's copies of this dict stay equal.
                "name": "RandRotated",
                "mode": ["bilinear", "nearest"],
                "range_x": 3.14159,
                "prob": 1.0,
            },
        ],
        "weights": [0.5, 0.5],
    }

    config = {
        # shared with "label" below -> SliceReplication3DFilterd and the OneOf group both run
        # on volume and mask3d with the same random draw.
        "raw": [slice_replication, one_of],
        "label": [slice_replication, one_of],
    }

    transformer_pipeline = AugmentationTransformer(config, seed=47)._create_transform()

    print(transformer_pipeline._compose)
