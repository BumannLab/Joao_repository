import numpy as np
from scipy.ndimage import uniform_filter, median_filter

from monai.transforms import Compose, MapTransform, OneOf, RandomizableTransform
from monai.utils import convert_to_numpy

from Scripts.NeuralNetworkSegmentation.utils import get_class


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