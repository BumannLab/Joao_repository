import random

import torch
import torchvision.transforms.v2 as v2
import torchvision.transforms.functional as F
import tifffile as tiff
import matplotlib.pyplot as plt

from utils import get_images_path


def _match_return_type(x, y):
    if y is None:
        return x
    return x, y


class AdjustBrightness:
    def __init__(self, brightness, same_brightness_all_channels = False):
        if not isinstance(brightness, tuple):
            brightness = (brightness, brightness)
        self.brightness = brightness
        self.same_brightness_all_channels = same_brightness_all_channels

    def __call__(self, x, y=None):

        brightness_factors = torch.empty(
            x.size(-3), device=x.device
        )
        
        if self.same_brightness_all_channels:
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
        self.translate = translate
        self.scale = scale
        self.shear = shear 
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
    def __init__(self, kernel, sigma, blur_prob = 0.5):
        self.kernel = kernel
        self.sigma = sigma
        self.blur_prob = blur_prob

    def __call__(self, x, y=None):
        
        if random.random() < self.blur_prob:
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
        if not isinstance(padding, tuple):
            padding = (padding, padding)
        self.padding = padding

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
    

class ComposePair:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, y=None):
        for transform in self.transforms:
            result = transform(x, y)
            if y is None:
                x = result
            else:
                x, y = result
        return _match_return_type(x, y)


from NeuralNetworkSegmentation._utils import get_class

x = get_class("DepthPadding", ["NeuralNetworkSegmentation.transforms"])