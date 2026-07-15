from pathlib import Path

import matplotlib.pyplot as plt
import torch
import numpy as np
import tifffile as tiff

from Scripts.NeuralNetworkSegmentation.utils import convert_to_numpy


class TensorboardFormatter:
    """
    Converts a (C, D, H, W) volume into a single 2D image suitable for tensorboard's add_image.

    - Volumes with depth > 1 (e.g. 128x128x24) are collapsed via a maximum-intensity projection
      along depth; volumes with depth == 1 (e.g. 128x128x1) are just squeezed, no projection needed.
    - Each channel is independently rescaled from its native intensity range (e.g. 16-bit) to 8-bit
      so it's actually visible.
    - A single channel is returned as a grayscale (H, W) image.
    - Multiple channels are composited into an (H, W, 3) RGB image the way Fiji/ImageJ displays
      composite hyperstacks: each channel is tinted with an assigned color and the tinted channels
      are additively blended.
    """

    # Fiji's default per-channel LUT colors, cycled if there are more channels than colors.
    CHANNEL_COLORS = [
        (255, 255, 0),  # yellow
        (0, 255, 255),  # cyan
        (255, 0, 0),    # red
        (0, 255, 0),    # green
        (0, 0, 255),    # blue
        (255, 0, 255),  # magenta
        (255, 255, 255),  # gray
    ]

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        """
        Args:
            volume: ndarray of shape (C, D, H, W)

        Returns:
            uint8 ndarray of shape (H, W) if single-channel, or (H, W, 3) if multi-channel.
        """
        channels = [self._to_8bit(self._project(channel)) for channel in volume]

        if len(channels) == 1:
            return channels[0]

        return self._composite(channels)

    @staticmethod
    def _project(channel: np.ndarray) -> np.ndarray:
        """channel: (D, H, W) -> (H, W). Max-intensity projection if D > 1, plain squeeze if D == 1."""
        if channel.shape[0] > 1:
            return np.max(channel, axis=0)
        return channel[0]

    @staticmethod
    def _to_8bit(image: np.ndarray) -> np.ndarray:
        """Min-max normalizes a single-channel image (e.g. native 16-bit) into 8-bit."""
        image = image.astype(np.float32)
        mn, mx = image.min(), image.max()
        if mx > mn:
            image = (image - mn) / (mx - mn)
        else:
            image = np.zeros_like(image)
        return (image * 255).astype(np.uint8)

    @classmethod
    def _composite(cls, channels: list[np.ndarray]) -> np.ndarray:
        """Additively blends 8-bit channels tinted by CHANNEL_COLORS, Fiji composite-style."""
        h, w = channels[0].shape
        composite = np.zeros((h, w, 3), dtype=np.float32)

        for idx, channel in enumerate(channels):
            color = np.array(cls.CHANNEL_COLORS[idx % len(cls.CHANNEL_COLORS)], dtype=np.float32) / 255.0
            composite += channel[..., None].astype(np.float32) * color

        return np.clip(composite, 0, 255).astype(np.uint8)
