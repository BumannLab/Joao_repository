import random

import torch
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F
import numpy as np
from scipy.ndimage import uniform_filter, median_filter

from Scripts.NeuralNetworkSegmentation.utils import get_class

import albumentations as A


class SliceReplication3DFilter(A.Transform3D):
    """Increase depth resolution by repeating slices, then applying a 3D filter."""

    def __init__(self, num_slice_replications=1, filter_type="mean", kernel_size=3, p=0.5):
        super().__init__(p=p)

        self.num_slice_replications = num_slice_replications
        self.filter_type = filter_type
        self.kernel_size = kernel_size

    def get_params_dependent_on_data(self, params, data):

        repeats = self.num_slice_replications + 1
        size = (self.kernel_size, self.kernel_size, self.kernel_size, 1)

        if self.filter_type == "mean":
            filter_func = uniform_filter
        elif self.filter_type == "median":
            filter_func = median_filter
        else:
            raise ValueError(f"Unknown filter_type: {self.filter_type}")

        return {"repeats": repeats, "filter_func": filter_func, "size": size}

    def apply_to_volume(self, volume, repeats, filter_func, size, *args, **params):

        volume = np.repeat(volume, repeats=repeats, axis=0)

        return filter_func(volume, size=size)

    def apply_to_mask3d(self, mask3d, repeats, filter_func, size, *args, **params):

        mask3d = np.repeat(mask3d, repeats=repeats, axis=0)

        return median_filter(mask3d, size=size)
    
    
class AugmentationTransformer:
    def __init__(self, phase_transform_config, seed):
    
        """Factory class for creating data augmentation pipelines."""

        self.phase_transform_config = phase_transform_config
        self.seed = seed

    def _create_transform(self):

        return A.Compose(
            [self._create_augmentation(augmentation) for augmentation in self.phase_transform_config],
            seed = self.seed,
        )
    
    def _create_augmentation(self, augmentation):

        augmentation = augmentation.copy()

        assert "name" in augmentation, f"No name found in {augmentation}"

        augmentation_name = augmentation.pop("name")

        if augmentation_name == "OneOf":
            assert "transforms" in augmentation, f"No transforms found in {augmentation_name}: {augmentation}"

            return self._create_one_of(
                augmentation.pop("transforms"),
                **augmentation,
            )

        _transform = getattr(A, augmentation_name)
        _params =  augmentation
    
        return _transform(**_params)

    def _create_one_of(self, one_of_config, **params):

        one_of_transforms = [
            self._create_augmentation(one_of_augmentation)
            for one_of_augmentation in one_of_config
        ]

        return A.OneOf(one_of_transforms, **params)
    

if __name__ == "__main__":
    config = [
        {
            "name": "CenterCrop",
            "height": 128,
            "width": 128,
        },
        {
            "name": "OneOf",
            "transforms": [
                {
                    "name": "CenterCrop",
                    "height": 128,
                    "width": 128,
                },
                {
                    "name": "Rotate",
                    "angle_range": [-180, 180],
                    "p": 1.0,
                },
            ],
            "p": 1.0,
        },
    ]

    transformer_pipeline = AugmentationTransformer(config, seed = 47)._create_transform()

    import numpy as np

    raw = np.random.randint(1, 65536, (8, 4, 256, 256)).astype(np.uint16)
    mask = np.random.randint(1, 256, (8, 1, 256, 256)).astype(np.uint8)

    raw_transposed = np.transpose(raw, (0, 2, 3, 1))
    mask_transposed = np.transpose(mask, (0, 2, 3, 1))

    augmented = transformer_pipeline(volume = raw_transposed, mask3d = mask_transposed)
