import random

import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
import numpy as np
from scipy.ndimage import uniform_filter, median_filter

from Scripts.NeuralNetworkSegmentation.utils import get_class

import albumentations as A  

# D,H,W,C Albumentations input and mask shape

class SliceReplication3DFilter(A.Transform3D):
    """Increase depth resolution by repeating slices, then applying a 3D filter. For mask3d it will
    only perform mean 3D filter.

    Args:
        num_slice_replications (int): Number of times to copy all slides. Default: 1.
        filter_type (str): 3D Filter to apply to the full volume after slice replication. 
            Should be one of: mean, median. Default: mean.
        kernel_size (int): Determines kernel size in 3D; (k, k, k, 1) where k is kernel size.
            Default: 3.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        volume, mask3d

    Image types:
        float32
    """

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
    

class OneHot(A.DualTransform):
    """Transforms mask into a onehot encoded matrix. There is no probability.

    Args:
        num_channels (int): Transforms mask into a onehot were the channel dimension will
            have the same shape as num_channels. If not provided this will be infered from the data.
            If num_channels is smaller or equal to the maximum number of targets of the input it will
            throw an assertion error
            Default: None.
       
    Targets:
        mask3d

    Image types:
        float32
    """

    def __init__(self, num_channels = None):
        super().__init__(p=1.0) 
        self.num_channels = num_channels        

    def get_params_dependent_on_data(self, params, data):

        return {"n_channels": self.num_channels}
    
    def apply_to_volume(self, volume, *args, **params):
        
        return volume

    def apply_to_mask3d(self, mask3d, n_channels, *args, **params):
        
        if n_channels is None:
            n_channels = int(np.max(mask3d)) + 1

        assert n_channels >= int(np.max(mask3d)) + 1, (
            f"n_channels={n_channels} is smaller or equal than the number of classes"
            f"found in mask ({int(np.max(mask3d)) + 1})"
        )

        one_hot = np.eye(n_channels, dtype=np.float32)[mask3d.squeeze(-1)]  # (D, H, W, n_channels)

        return one_hot

    
class AugmentationTransformer:
    def __init__(self, phase_transform_config, seed, data_stats=None):

        """Factory class for creating data augmentation pipelines."""

        self.phase_transform_config = phase_transform_config
        self.seed = seed
        self.data_stats = data_stats or {}

    def _create_transform(self):

        return A.Compose(
            [self._create_augmentation(augmentation) for augmentation in self.phase_transform_config],
            seed = self.seed,
        )
    
    def _create_augmentation(self, augmentation):

        augmentation = augmentation.copy()

        assert "name" in augmentation, f"No name found in {augmentation}"

        augmentation_name = augmentation.pop("name")

        if augmentation_name == "Normalize":

            if all(self.data_stats.get(stat) is not None for stat in ["mean", "std"]):
               
                # setdefault: an explicit mean/std/max_pixel_value in the YAML entry still wins.
                augmentation.setdefault("mean", self.data_stats["mean"])
                augmentation.setdefault("std", self.data_stats["std"])
                # NOTE: First Augmentation should be ToFloat
                augmentation.setdefault("max_pixel_value", 1.0)

        if augmentation_name == "OneOf":
            assert "transforms" in augmentation, f"No transforms found in {augmentation_name}: {augmentation}"

            return self._create_one_of(
                augmentation.pop("transforms"),
                **augmentation,
            )

        _transform = get_class(augmentation_name, modules=["albumentations", "Scripts.NeuralNetworkSegmentation.transforms"])
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
            "name": "SliceReplication3DFilter"
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
    
    print(transformer_pipeline)
    