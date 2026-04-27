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


def plot_random_augmentation(
    augmentation,
    input_tensor,
    mask_tensor
    ):
    def _as_stack(tensor):
        if tensor.ndim == 2:
            return tensor.unsqueeze(0).unsqueeze(0), False
        if tensor.ndim == 3:
            return tensor.unsqueeze(0), False
        if tensor.ndim == 4:
            return tensor, True
        raise ValueError(
            "Expected tensor with shape [H, W], [C, H, W], or [Z, C, H, W], "
            f"got {tuple(tensor.shape)}"
        )
    
    def _original_slice_for_display(slice_tensor):
        return slice_tensor.detach().cpu().float().mean(dim=0)
    
    x_augmented, y_augmented = augmentation(input_tensor, mask_tensor)
    input_stack, is_3d = _as_stack(input_tensor)
    x_augmented_stack, _ = _as_stack(x_augmented)
    y_augmented_stack, _ = _as_stack(y_augmented)

    augmentation_name = augmentation.__class__.__name__

    num_slices = x_augmented_stack.shape[0]
    num_channels = x_augmented_stack.shape[1]

    fig, axes = plt.subplots(
        num_channels + 2,
        num_slices,
        figsize=(2.5 * num_slices, 2.5 * (num_channels + 2)),
        squeeze=False,
    )

    for slice_idx in range(num_slices):
        original_ax = axes[0, slice_idx]
        original_ax.imshow(_original_slice_for_display(input_stack[slice_idx]), cmap="gray")
        original_ax.axis("off")
        if slice_idx == 0:
            original_ax.set_ylabel("Original", rotation=90, size="large")
        original_ax.set_title(f"z={slice_idx}" if is_3d else "Image")

        for channel_idx in range(num_channels):
            input_ax = axes[channel_idx + 1, slice_idx]
            input_ax.imshow(x_augmented_stack[slice_idx, channel_idx].detach().cpu(), cmap="gray")
            input_ax.axis("off")
            if slice_idx == 0:
                input_ax.set_ylabel(f"Ch {channel_idx}", rotation=90, size="large")

        mask_ax = axes[-1, slice_idx]
        mask_ax.imshow(_original_slice_for_display(y_augmented_stack[slice_idx]), cmap="gray")
        mask_ax.axis("off")
        if slice_idx == 0:
            mask_ax.set_ylabel("Mask", rotation=90, size="large")

    fig.suptitle(f"{augmentation_name}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show(block=True)
    plt.close()


if __name__ == "__main__":

    mask = torch.from_numpy(tiff.imread("/data/joao/Materials/Joao_repository/NN_Dataset/3D_NEW/mask/positive_1121_AS41_1.tif") / 255)
    mask = mask.unsqueeze(-3)

    input = torch.from_numpy(tiff.imread("/data/joao/Materials/Joao_repository/NN_Dataset/3D_NEW/input/positive_1121_AS41_1.tif") / 65535)
    
    x, y = DepthPadding((2, 1))(input, mask)
    print(x.shape, y.shape)
    # augmentations = [
    #     AdjustBrightness((0.5, 1.0)),
    #     RandomAffine(180, (0.09, 0.09), (0.85, 1.15) , (-4, 4)),
    #     GaussianBlur(3, (0.1, 1.5), 0.5),
    #     Resize(64),
    #     CenterCrop(64)
    # ]
    # for augmentation in augmentations:
    #     plot_random_augmentation(augmentation, input, mask)

    # mask = torch.from_numpy(tiff.imread("/data/joao/Materials/Joao_repository/NN_Dataset/2D_NEW/mask/positive_1121_AS41_1.tif") / 255)
    # mask = mask.unsqueeze(-3)

    # input = torch.from_numpy(tiff.imread("/data/joao/Materials/Joao_repository/NN_Dataset/2D_NEW/input/positive_1121_AS41_1.tif") / 65535)

    # for augmentation in augmentations:
    #     plot_random_augmentation(augmentation, input, mask)

    
