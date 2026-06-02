import random
from collections import defaultdict

import torch
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F
import tifffile as tiff
import matplotlib.pyplot as plt

from ZOrganized_scripts.NeuralNetworkSegmentationConfig.utils import get_class

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
    

class PairedTransformPipeline:
    def __init__(self, steps):
        self.steps = steps

    def _apply_transform(self, transform, img):
            return transform(img)

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        
        for transform, apply_to_input, apply_to_mask in self.steps:
            
            if apply_to_input:
                x = self._apply_transform(transform, x)
            if apply_to_mask:
                y = self._apply_transform(transform, y)

        return x, y


def build_transform_pipeline(config_transforms):

    # TODO: handle cases where raw or label has nothing on the config file"
    raw_transforms = config_transforms["raw"]
    label_transforms = config_transforms["label"]

    steps = list()

    for transform in raw_transforms:
        
        apply_input, apply_mask = False, False
        
        if transform in raw_transforms:
            apply_input = True
        if transform in label_transforms:
            apply_mask = True

        _transform = get_class(transform, ["ZOrganized_scripts.NeuralNetworkSegmentationConfig.transforms"])
        _params = dict(raw_transforms.get(transform, {})) # TODO: ASSERT PARAMS ARE EQUAL FOR BOTH RAW AND LABEL

        steps.append((_transform(**_params), apply_input, apply_mask))

    return PairedTransformPipeline(steps)


class Compose:
    
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img: torch.Tensor):
        
        for transform in self.transforms:
            
            img = transform(img)

        return img
    

def create_split_augmentations(seed, split_config):

    raw_transforms = create_transforms(seed, split_config.get("raw", {}))
    label_transforms = create_transforms(seed, split_config.get("label", {}))

    return raw_transforms, label_transforms


def create_transforms(seed, phase_config):

    transforms = list()

    for transform in phase_config:

        augmentation = create_augmentation(seed, phase_config, transform)
        transforms.append(augmentation)

    return Compose(transforms)


def create_augmentation(seed, phase_config, transform):

    _transform = get_class(transform, ["ZOrganized_scripts.NeuralNetworkSegmentationConfig.transforms"])
    _params =  phase_config.get(transform, {})
    _params["random_state"] = seed

    return _transform(**_params)
