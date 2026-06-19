from pathlib import Path

import matplotlib.pyplot as plt
import torch
import numpy as np
import tifffile as tiff

from Scripts.NeuralNetworkSegmentation.utils import convert_to_numpy

def model_test_preview(tensorboard_writer, input, mask, pred, i):
    
    def _normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            return (arr - mn) / (mx - mn)
        return np.zeros_like(arr, dtype=np.float32)

    for channels in zip(input, mask, pred):

        c_input, c_mask, c_pred = channels
        c_pred = torch.argmax(c_pred, dim=0, keepdim=True)

        c_input, c_mask, c_pred = convert_to_numpy(c_input, c_mask, c_pred)

        combined = np.concatenate((_normalize(c_input), _normalize(c_mask), _normalize(c_pred)), axis=0)

        combined = np.transpose(combined, axes=(1, 0, 2, 3))  # [D, C, H, W]

        combined = np.max(combined, axis = 0)

        tensorboard_writer.add_image(f"ModelPreview/test/{i}Input", combined[0], dataformats='HW')
        tensorboard_writer.add_image(f"ModelPreview/test/{i}Mask", combined[1], dataformats='HW')
        tensorboard_writer.add_image(f"ModelPreview/test/{i}Pred", combined[2], dataformats='HW')

        i += 1
