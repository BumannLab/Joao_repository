import numpy as np
from scipy.ndimage import uniform_filter, median_filter, zoom

from monai.transforms import Compose, MapTransform, OneOf, RandomizableTransform
from monai.utils import convert_to_numpy, ensure_tuple_rep

from Scripts.NeuralNetworkSegmentation.utils import get_class


class OneHotd(MapTransform):
    """Turns an integer label mask into a one-hot encoded array along the channel axis. 
    Always applied, there is no probability.

    Expects a channel-first mask, (1, D, H, W); output is (num_channels, D, H, W).

    Args:
        keys (KeysCollection): keys to one-hot encode. Default: ("mask",).
        num_channels (int): size of the output channel dimension. If None, inferred as
            max(mask) + 1. If given, must be >= the number of classes actually present.
            Default: None.
        allow_missing_keys (bool): don't raise if a key is missing. Default: True.
    """

    def __init__(self, keys=("mask",), num_channels=None, allow_missing_keys=True):
        super().__init__(keys, allow_missing_keys)
        self.num_channels = num_channels

    def __call__(self, data):

        d = dict(data)

        for key in self.key_iterator(d):
            mask = d[key]

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
    

class SliceReplication3DFilterd(RandomizableTransform, MapTransform):
    """Increases depth resolution by repeating slices along the depth axis, 
    then smooths the result with a 3D filter. "mask" is always smoothed with a median filter
    regardless of filter_type.

    Expects channel-first arrays, (C, D, H, W), with D (depth) as axis=1.

    Args:
        keys (KeysCollection): keys to apply the transform to. Default: ("img", "mask").
        num_slice_replications (int): number of times to copy all slices. Default: 1.
        filter_type (str): 3D filter applied to "volume" after slice replication. One of
            "mean", "median". Default: "mean".
        kernel_size (int): kernel size in 3D; (1, k, k, k) so the channel axis is never
            filtered across. Default: 3.
        prob (float): probability of applying the transform. Default: 0.5.
        allow_missing_keys (bool): don't raise if a key (e.g. "mask" during label-less
            inference) is missing. Default: True.
    """


    def __init__(
        self,
        keys=("img", "mask"),
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
            array = np.repeat(convert_to_numpy(d[key]), repeats=repeats, axis=1)

            filter_func = median_filter if key == "mask" else self.filter_func
            d[key] = filter_func(array, size=size)

        return d


class RandSpatialShiftd(RandomizableTransform, MapTransform):
    """Translates the array along spatial axes by a random whole-pixel offset.

    The border exposed by the shift (the side the content moves away from) is
    filled according to `mode`, using the same semantics as `numpy.pad`.

    Args:
        keys (KeysCollection): keys to apply the transform to. Default: ("img", "mask").
        shift_range (Sequence[int]): per spatial-axis maximum absolute shift, in
            pixels/voxels. The offset for each axis is drawn uniformly from
            [-shift_range[i], shift_range[i]]; 0 disables shifting along that axis.
            Default: (0, 20, 20).
        mode (str): `numpy.pad` mode used to fill the border exposed by the shift,
            e.g. "constant", "edge" (replicate the edge value), "reflect" (mirror
            without repeating the edge), "symmetric" (mirror repeating the edge),
            "wrap" (circular). Note "reflect"/"symmetric" require the shift along
            an axis to not exceed that axis's size. Default: "constant".
        pad_value (float): fill value used when mode="constant". Default: 0.
        prob (float): probability of applying the transform. Default: 0.5.
        allow_missing_keys (bool): don't raise if a key (e.g. "mask" during label-less
            inference) is missing. Default: True.
    """

    def __init__(
        self,
        keys=("img", "mask"),
        shift_range=(0, 20, 20),
        mode="constant",
        pad_value=0,
        prob=0.5,
        allow_missing_keys=True,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.shift_range = shift_range
        self.mode = mode
        self.pad_value = pad_value
        self._shift = None

    def randomize(self, data=None):
        super().randomize(data)
        self._shift = tuple(
            self.R.randint(-r, r + 1) if r else 0
            for r in self.shift_range
        )

    def __call__(self, data):

        d = dict(data)

        self.randomize(None)
        if not self._do_transform:
            return d

        for key in self.key_iterator(d):
            array = convert_to_numpy(d[key])
            d[key] = self._shift_array(array)

        return d

    def _shift_array(self, array):

        shift = (0, *self._shift[: array.ndim - 1])

        pad_width, crop_slices = [], []
        for dim, s in zip(array.shape, shift):
            if s >= 0:
                pad_width.append((s, 0))
                crop_slices.append(slice(0, dim))
            else:
                pad_width.append((0, -s))
                crop_slices.append(slice(-s, dim - s))

        pad_kwargs = {"constant_values": self.pad_value} if self.mode == "constant" else {}
        padded = np.pad(array, pad_width, mode=self.mode, **pad_kwargs)

        return padded[tuple(crop_slices)]


class ToIsotropicd(MapTransform):
    """Resamples channel-first spatial data to isotropic voxel spacing.

    Given each spatial axis's known physical voxel spacing, resizes so every axis ends
    up at the same target spacing. 

    A coarser axis (larger physical spacing) gets more voxels added via interpolation
    to reach the target spacing; a finer axis gets voxels removed. Not randomized --
    the same fixed resampling is applied every call.

    Expects channel-first arrays, (C, ...spatial_dims), with `spacing` covering just
    the spatial dims (channel axis is left untouched).

    Args:
        keys (KeysCollection): keys to apply the transform to. Default: ("img", "mask").
        spacing (Sequence[float]): current physical voxel spacing per spatial axis,
            e.g. (z, y, x). Required.
        target_spacing (float | None): desired isotropic spacing. Defaults to
            min(spacing), i.e. resamples every axis to the finest spacing already
            present so no axis loses resolution.
        mode (SequenceStr): per-key interpolation mode, one of "nearest" (order=0,
            exact/no interpolation -- use for "mask") or "bilinear" (order=1). A single
            value is broadcast to every key. Default: ("bilinear", "nearest").
        allow_missing_keys (bool): don't raise if a key (e.g. "mask" during label-less
            inference) is missing. Default: True.
    """

    _MODE_TO_ORDER = {"nearest": 0, "bilinear": 1}

    def __init__(
        self,
        keys=("img", "mask"),
        spacing=None,
        target_spacing=None,
        mode=("bilinear", "nearest"),
        allow_missing_keys=True,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)

        assert spacing is not None, "spacing must be provided, e.g. (z, y, x) physical voxel size"

        self.spacing = tuple(spacing)
        self.target_spacing = target_spacing if target_spacing is not None else min(self.spacing)

        mode = ensure_tuple_rep(mode, len(self.keys))
        self.mode = dict(zip(self.keys, mode))

    def __call__(self, data):

        d = dict(data)

        zoom_factors = tuple(s / self.target_spacing for s in self.spacing)

        for key in self.key_iterator(d):
            array = convert_to_numpy(d[key])

            assert array.ndim - 1 == len(self.spacing), (
                f"spacing has {len(self.spacing)} axes but {key} has "
                f"{array.ndim - 1} spatial dims (shape {array.shape})"
            )

            order = self._MODE_TO_ORDER[self.mode[key]]
            full_zoom = (1.0, *zoom_factors)

            resampled = zoom(array.astype(np.float32), zoom=full_zoom, order=order)
            d[key] = resampled.astype(array.dtype)
    
        return d


class DivisiblePadCropDepthd(MapTransform):
    """Pads or crops the depth axis so its size becomes a multiple of `k`,
    rounding to the *nearest* multiple rather than always padding up to the
    next one (e.g. with k=8: 21 -> pads to 24, 25 -> crops to 24).

    Expects channel-first arrays, (C, D, H, W), with D (depth) as axis=1.

    Whatever voxels need adding/removing on the depth axis are split unevenly
    between the two ends: fewer at the top (index 0 side), more at the bottom,
    e.g. a difference of 3 becomes top=1, bottom=2.

    Args:
        keys (KeysCollection): keys to apply the transform to. Default: ("img", "mask").
        k (int): depth is rounded to the nearest multiple of k, e.g. k=8 for a
            3-level encoder (2*2*2 depth downsampling). Default: 8.
        pad_value (float): fill value used when padding. Default: 0.
        allow_missing_keys (bool): don't raise if a key (e.g. "mask" during label-less
            inference) is missing. Default: True.
    """

    def __init__(
        self,
        keys=("img", "mask"),
        k=8,
        pad_value=0,
        allow_missing_keys=True,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)

        self.k = k
        self.pad_value = pad_value

    def __call__(self, data):

        d = dict(data)

        depth = next(
            (convert_to_numpy(d[key]).shape[1] for key in self.key_iterator(d)), None
        )

        if depth is None:
            return d

        target = max(self.k, round(depth / self.k) * self.k)
        diff = target - depth

        if diff == 0:
            return d

        low = abs(diff) // 2
        high = abs(diff) - low

        for key in self.key_iterator(d):
            array = convert_to_numpy(d[key])

            if diff > 0:
                pad_width = [(0, 0)] * array.ndim
                pad_width[1] = (low, high)
                d[key] = np.pad(array, pad_width, mode="constant", constant_values=self.pad_value)
            else:
                d[key] = array[:, low:depth - high]

        return d


class AugmentationTransformer:

    def __init__(self, phase_transform_config, randomstate, data_stats=None):
        """Factory class for creating data augmentation pipelines
        """

        self.phase_transform_config = phase_transform_config
        self.seed = randomstate.randint(10_000_000)
        self.data_stats = data_stats or {}

    def _create_transform(self):

        compose = Compose(
            [self._create_augmentation(augmentation) for augmentation in self.phase_transform_config],
        )
        compose.set_random_state(seed=self.seed)

        return compose

    def _create_augmentation(self, augmentation):

        augmentation = augmentation.copy()

        assert "name" in augmentation, f"No name found in {augmentation}"

        augmentation_name = augmentation.pop("name")

        if augmentation_name == "NormalizeIntensityd":

            if all(self.data_stats.get(stat) is not None for stat in ["mean", "std"]):

                # setdefault: an explicit subtrahend/divisor in the YAML entry still wins.
                augmentation.setdefault("subtrahend", self.data_stats["mean"])
                augmentation.setdefault("divisor", self.data_stats["std"])

        if augmentation_name == "ScaleIntensityRanged":

            if all(self.data_stats.get(stat) is not None for stat in ["pmin", "pmax"]):

                # setdefault: an explicit percentilemin/percentilemax in the YAML entry still wins.
                augmentation.setdefault("a_min", self.data_stats["pmin"])
                augmentation.setdefault("a_max", self.data_stats["pmax"])

        if augmentation_name == "OneOf":
            assert "transforms" in augmentation, f"No transforms found in {augmentation_name}: {augmentation}"

            return self._create_one_of(
                augmentation.pop("transforms"),
                **augmentation,
            )

        _transform = get_class(augmentation_name, modules=["monai.transforms", "Scripts.NeuralNetworkSegmentation.augmentation.transforms"])
        augmentation.setdefault("allow_missing_keys", True)
        _params = augmentation

        return _transform(**_params)

    def _create_one_of(self, one_of_config, **params):

        one_of_transforms = [
            self._create_augmentation(one_of_augmentation)
            for one_of_augmentation in one_of_config
        ]

        return OneOf(one_of_transforms, **params)