import random

import torch
import torchvision.transforms.v2 as v2
import torchvision.transforms.functional as F
import tifffile as tiff
import matplotlib.pyplot as plt

from NeuralNetworkSegmentation._utils import get_class


def _match_return_type(x, y):
    if y is None:
        return x
    return x, y


class AdjustBrightness:
    def __init__(self, brightness, same_channels_brightness = False):
        if not isinstance(brightness, list):
            brightness = (brightness, brightness)
        self.brightness = tuple(brightness)
        self.same_channels_brightness = same_channels_brightness

    def __call__(self, x, y=None):

        brightness_factors = torch.empty(
            x.size(-3), device=x.device
        )
        
        if self.same_channels_brightness:
            value = random.uniform(1 - self.brightness[0], 1 + self.brightness[1])
            brightness_factors.fill_(value)
            print()
        else:
            brightness_factors.uniform_(
                1 - self.brightness[0], 1 + self.brightness[1])
        
        shape = [1] * x.ndim
        shape[-3] = -1

        x = x * brightness_factors.view(*shape)

        x = torch.clamp(x, 0.0, 1.0)

        return _match_return_type(x, y)
    

class RandomAffine:
    def __init__(self, 
                degrees,
                translate,
                scale,
                shear,  
                interpolation = (v2.InterpolationMode.BILINEAR, v2.InterpolationMode.NEAREST)):
        """
        interpolation: if tuple is given, idx 0 will be used for x and 
        idx 1 will be used for y.
        """
        self.degrees = degrees
        self.translate = tuple(translate)
        self.scale = tuple(scale)
        self.shear = tuple(shear) 
        if not isinstance(interpolation, tuple):
            interpolation = (interpolation, interpolation)
        self.interpolation = interpolation

    def __call__(self, x, y=None):
        
        angle = random.uniform(-self.degrees, self.degrees)
        
        max_dx = self.translate[0] * x.shape[-1]
        max_dy = self.translate[1] * x.shape[-2]
        translations = (
        int(random.uniform(-max_dx, max_dx)),
        int(random.uniform(-max_dy, max_dy)),
        )
        
        scale_factor = random.uniform(*self.scale)
        
        shear_factor = random.uniform(*self.shear)

        x = F.affine(x, angle, translations, scale_factor, shear_factor, interpolation=self.interpolation[0])
        if y is not None:
            y = F.affine(y, angle, translations, scale_factor, shear_factor, interpolation=self.interpolation[1])

        return _match_return_type(x, y)
    

class GaussianBlur:
    def __init__(self, kernel, sigma, blur_probability = 0.5):
        self.kernel = kernel
        self.sigma = tuple(sigma)
        self.blur_probability = blur_probability

    def __call__(self, x, y=None):
        
        if random.random() < self.blur_probability:
            x = F.gaussian_blur(x, self.kernel, self.sigma)

        return _match_return_type(x, y)
    

class Resize:
    def __init__(self, size, interpolation = (v2.InterpolationMode.BILINEAR, v2.InterpolationMode.NEAREST)):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, x, y=None):

        x = F.resize(x, (self.size, self.size), self.interpolation[0])
        if y is not None:
            y = F.resize(y, (self.size, self.size), self.interpolation[1])

        return _match_return_type(x, y)
    

class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, x, y=None):

        x = F.center_crop(x, (self.size, self.size))
        if y is not None:
            y = F.center_crop(y, (self.size, self.size))
        
        return _match_return_type(x, y)
    

class DepthPadding:
    def __init__(self, padding):
        if not isinstance(padding, list):
            padding = (padding, padding)
        self.padding = tuple(padding)

    def __call__(self, x, y = None):

        pad_up, pad_down = self.padding

        up_padx = torch.zeros((pad_up, *x.shape[-3:]))
        down_padx = torch.zeros((pad_down, *x.shape[-3:]))

        x = torch.concatenate((up_padx, x, down_padx), dim = 0)

        if y is not None:
            
            up_pady = torch.zeros((pad_up, *y.shape[-3:]), dtype=torch.long)
            down_pady = torch.zeros((pad_down, *y.shape[-3:]), dtype=torch.long)
            
            y = torch.concatenate((up_pady, y, down_pady), dim = 0)
        
        return _match_return_type(x, y)
    

class PairedTransformPipeline:
    def __init__(self, steps):
        self.steps = steps

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        for transform, apply_to_input, apply_to_mask in self.steps:
            x_arg = x if apply_to_input else None
            y_arg = y if apply_to_mask else None

            transformed_x, transformed_y = transform(x_arg, y_arg)

            if apply_to_input:
                x = transformed_x
            if apply_to_mask:
                y = transformed_y

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

        _transform = get_class(transform, ["NeuralNetworkSegmentation.transforms"])
        _params =  raw_transforms.get(transform, {}) # TODO: ASSERT PARAMS ARE EQUAL FOR BOTH RAW AND LABEL

        steps.append((_transform(**_params), apply_input, apply_mask))

    return PairedTransformPipeline(steps)


        


