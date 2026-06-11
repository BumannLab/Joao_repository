import random

import torch
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F
import numpy as np

from Scripts.NeuralNetworkSegmentation.utils import get_class

import albumentations as A

ORDER_TO_INTERPOLATION = {
    0: InterpolationMode.NEAREST,
    1: InterpolationMode.BILINEAR,
}


def _as_pair(value):
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value, value)


def get_interpolation_from_order(order):
    return ORDER_TO_INTERPOLATION.get(order, InterpolationMode.BILINEAR)


class AdjustBrightness:
    def __init__(self, brightness = (0.8, 1.2), same_channels_brightness = False, **kwargs):
        self.brightness = _as_pair(brightness)
        self.same_channels_brightness = same_channels_brightness

    def __call__(self, img):

        brightness_factors = torch.empty(
            img.size(-3), device=img.device
        )
        
        if self.same_channels_brightness:
            value = random.uniform(self.brightness[0], self.brightness[1])
            brightness_factors.fill_(value)
        else:
            # change this to random call not torch
            brightness_factors.uniform_(
                self.brightness[0], self.brightness[1])
        
        shape = [1] * img.ndim
        shape[-3] = -1

        img = img * brightness_factors.view(*shape)

        img = torch.clamp(img, 0.0, 1.0)

        return img
    

class RandomAffine:
    def __init__(self, 
                degrees = 90,
                translate = 0.2,
                scale = 2,
                shear = 2,  
                order = 1,
                **kwargs):

        self.degrees = degrees
        self.translate = _as_pair(translate)
        self.scale = _as_pair(scale)
        self.shear = _as_pair(shear) 
        self.order = order

    def __call__(self, img):
        angle = random.uniform(-self.degrees, self.degrees)
        
        max_dx = self.translate[0] * img.shape[-1]
        max_dy = self.translate[1] * img.shape[-2]
        
        translations = (
            int(random.uniform(-max_dx, max_dx)),
            int(random.uniform(-max_dy, max_dy)),
        )
       
        scale_factor = random.uniform(*self.scale)
        shear_factor = random.uniform(*self.shear)

        return F.affine(
            img,
            angle,
            translations,
            scale_factor,
            shear_factor,
            interpolation=get_interpolation_from_order(self.order),
        )
    

class GaussianBlur:
    def __init__(self, kernel = 3, sigma = (1, 4), blur_probability = 0.5, **kwargs): # TODO change default sigma value
        self.kernel = kernel
        self.sigma = tuple(sigma)
        self.blur_probability = blur_probability

    def __call__(self, img):
        
        if random.random() < self.blur_probability:
            img = F.gaussian_blur(img, self.kernel, self.sigma)

        return img
    

class RandomHorizontalFlip:
    def __init__(self, flip_probability=0.5, **kwargs):
        self.flip_probability = flip_probability

    def __call__(self, img):
        if random.random() < self.flip_probability:
            img = torch.flip(img, dims=(-1,))

        return img
    

class RandomVerticalFlip:
    def __init__(self, flip_probability=0.5, **kwargs):
        self.flip_probability = flip_probability

    def __call__(self, img):
        if random.random() < self.flip_probability:
            img = torch.flip(img, dims=(-2,))

        return img
    

class Resize:
    def __init__(self, size = 128, order = 0, **kwargs):
        self.size = size
        self.order = order

    def __call__(self, img):

        return F.resize(img, (self.size, self.size), get_interpolation_from_order(self.order))
    

class CenterCrop:
    def __init__(self, size = 128, random_state=None, **kwargs):
        self.size = size
        self.random_state = random_state

    def __call__(self, img):

        return F.center_crop(img, (self.size, self.size))
    

class DepthPadding:
    def __init__(self, padding = 0, **kwargs):
        self.padding = _as_pair(padding)
        # TODO: ADD PADDING MODE

    def __call__(self, img):

        pad_up, pad_down = self.padding

        up_pad = img.new_zeros((pad_up, *img.shape[-3:]))
        down_pad = img.new_zeros((pad_down, *img.shape[-3:]))

        img = torch.concatenate((up_pad, img, down_pad), dim = 0)

        return img
    
    
class AugmentationTransformer:
    def __init__(self, phase_transform_config, seed):
    
        """Factory class for creating data augmentation pipelines."""

        self.phase_transform_config = phase_transform_config
        self.seed = seed

    def _create_transform(self):

        return A.Compose([self._create_augmentation(augmentation) for augmentation in self.phase_transform_config], seed = self.seed)
    
    def _create_augmentation(self, augmentation):

        assert "name" in augmentation, f"No name found in {augmentation}"

        augmentation_name = augmentation.pop("name")

        _transform = getattr(A, augmentation_name)
        _params =  augmentation
    
        return _transform(**_params)
    

if __name__ == "__main__":
    config = [{'name': 'CenterCrop', 'height': 128, 'width': 128}, {'name': 'Rotate', 'angle_range': (-180, 180), 'p': 1.0}]

    transformer_pipeline = AugmentationTransformer(config, seed = 47)._create_transform()

    import numpy as np

    raw = np.random.randint(1, 65536, (8, 4, 256, 256)).astype(np.uint16)
    mask = np.random.randint(1, 256, (8, 1, 256, 256)).astype(np.uint8)

    raw_transposed = np.transpose(raw, (0, 2, 3, 1))
    mask_transposed = np.transpose(mask, (0, 2, 3, 1))

    augmented = transformer_pipeline(volume = raw_transposed, mask3d = mask_transposed)
    
    transforms = A.Compose(
        [
            # A.Rotate(angle_range=(-45, 45), p=1.0),
            A.SaltAndPepper((0.1, 0.2)),
        ]
    )
    
    print(augmented.keys())
    
    augmented_raw = augmented["volume"]
    augmented_mask = augmented["mask3d"]
    print(augmented_raw)
    print(augmented_raw.shape)
    print(augmented_raw.dtype)

    print(augmented_mask.shape)
    print(augmented_mask.dtype)

