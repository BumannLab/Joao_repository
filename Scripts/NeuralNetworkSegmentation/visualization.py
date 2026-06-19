from pathlib import Path

import matplotlib.pyplot as plt
import torch
import numpy as np
import tifffile as tiff

from Scripts.NeuralNetworkSegmentation.utils import convert_to_numpy

def model_test_preview(tensorboard_writer, input, mask, pred, i):

    for channels in zip(input, mask, pred):

        c_input, c_mask, c_pred = channels
        c_pred = torch.argmax(c_pred, dim=0, keepdim=True)

        c_input, c_mask, c_pred = convert_to_numpy(c_input, c_mask, c_pred)

        combined = np.concatenate((c_input, c_mask, c_pred), axis=0)

        combined = np.transpose(combined, axes=(1, 0, 2, 3))  # [D, C, H, W]

        combined = np.max(combined, axis = 0)

        tensorboard_writer.add_image(f"Test/Input/{i}", combined[0], dataformats='HW')
        tensorboard_writer.add_image(f"Test/Mask/{i}", combined[1], dataformats='HW')
        tensorboard_writer.add_image(f"Test/Pred/{i}", combined[2], dataformats='HW')

        i += 1
