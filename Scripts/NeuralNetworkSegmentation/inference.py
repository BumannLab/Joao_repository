from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

import torch
from tqdm import tqdm
import tifffile as tiff
import numpy as np
from skimage.measure import label, regionprops
from monai.inferers import SlidingWindowInferer
from torch.nn import ReflectionPad3d
from torch.utils.tensorboard import SummaryWriter

from utils import get_images_path, read_image, AXES
from Scripts.NeuralNetworkSegmentation.utils import load_checkpoint, convert_to_numpy
from Scripts.NeuralNetworkSegmentation.models.unet import is_model_2d
from Scripts.NeuralNetworkSegmentation.transforms import AugmentationTransformer

IMAGE_DIR = "/data/joao/mTBData/20251127/full111/1"
CHANNEL_MARKERS = "SYBRGold,CD68AL555,LAMAL647,DAPI"
CHANNELS_TO_USE = [0]
CLUSTER_SIZE_THRESHOLD = 5

WINDOW_SIZE = (21, 144, 144)
WINDOW_STRIDE = (21, 72, 72)

TORCH_MODEL_PATH = "/data/joao/Materials/Joao_repository/mTB_segmentation/center_crop_no_translation/model.pt"
NUM_BLOCKS = 3

SAVE_MASKS = True

ODIRNAME = "NeuralNetworkSegmentation/inference/test_inference"

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


def remove_halo(halo_shape, output):
    left, right, top, bottom, front, back = halo_shape
    d, h, w = output.shape[-3:]
    return output[
        ...,
        front : d - back if back else d,
        top : h - bottom if bottom else h,
        left : w - right if right else w,
    ]


class Inferer:
    def __init__(self, 
                model, 
                loader,
                batch_size,
                window_shape,
                overlap,
                seed,
                device,
                save_dir,
                mode = "constant",
                sigma_scale = 0.125,
                halo_shape = (0, 0, 0, 0, 0, 0),
                full_raw_transform_config = [],
                metrics = None,
            ):
        
        self.model = model
        self.loader = loader
        self.halo_shape = halo_shape 
        self.inferer = SlidingWindowInferer(window_shape, batch_size, overlap, mode, sigma_scale, device = device)
        self.raw_full_transform = AugmentationTransformer(full_raw_transform_config, seed, {})._create_transform()

        self.metrics = metrics
        self.resulting_metrics = defaultdict(list)
        self.device = device

        self.save_dir = save_dir

    def __call__(self):
        i = 0

        has_mask = self.loader.dataset.has_mask

        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(self.loader):
                input = batch[0]

                input = input.to(self.device)

                if sum(self.halo_shape) > 0:
                    input = ReflectionPad3d(self.halo_shape)(input)

                if is_model_2d(self.model):

                    # remove the singleton z-dimension from the input
                    input = torch.squeeze(input, dim=-3)

                    # forward pass
                    output = self.inferer(input, self.model, return_logits = False)

                    # add the singleton z-dimension to the output and back to the input
                    output = torch.unsqueeze(output, dim=-3)
                    input = torch.unsqueeze(input, dim=-3)

                else:
                    # forward pass
                    output = self.inferer(input, self.model, return_logits = False)

                # Not being used so far
                if sum(self.halo_shape) > 0:
                    # strip the halo back off both, so what's yielded stays spatially aligned
                    # for combining channels / running connected components downstream.
                    input = remove_halo(self.halo_shape, input)
                    output = remove_halo(self.halo_shape, output)

                if self.metrics:
                    assert has_mask is True, "No mask was found during the test evaluation"
                    target = batch[1]
                    
                    for i, criterion in enumerate(self.metrics):

                        output, target = convert_to_numpy(output, target)
                        result = criterion(output, target) 
                        self.resulting_metrics[i] = result
                    input = input.cpu().numpy().astype(np.uint16)

                    output[output > 0.5] = 1
                    output = output.cpu().numpy().astype(np.uint16)
                    output[output > 0] = 65535
        
                    target = target.cpu().numpy().astype(np.uint16)

                    save_prediction(input, output, f"test{i}", self.save_dir, target)
                    
                    i += 1

                elif has_mask is False:
                    raw_full, tile_name = batch[1], batch[2]
                    
                    # Add transforms for the raw full to match output
                    if self.raw_full_transform:
                        raw_full = np.transpose(raw_full, (0, 2, 3, 1))
                        raw_full = self.raw_full_transform(volume = raw_full)["volume"]
                        # Use to tensor augmentation to convert back to tensor
                        raw_full = convert_to_numpy(raw_full)
                    # Sometimes output will be padded or contain a single slice that got removed, mimic that behavior to match.
                    # If slice got removed in top, add it to top of the output. And so on
                    # Usually pad (if pad is three use 2 for the last layers) or crop after the last stacks since usually 
                    # those layers are already out of focus, and out of the specimen while if the sample is uneven the 
                    # early layers might contain traces of bacteria, whereas the last will mostly contain out of focus light,
                    # with no structure
                    output[output > 0.5] = 1
                    output = convert_to_numpy(output)[0].astype(np.uint16)
                    output[output > 0] = 65535

                    if 65535 in output: # TODO: If it has a single positive value use np to search for 65535
                        has_bacteria = "positive"
                    else:
                        has_bacteria = "negative"

                    save_prediction(raw_full, output, f"{has_bacteria}_{tile_name}.tif", self.save_dir)

            return self.resulting_metrics

def save_prediction(input: np.ndarray, pred: np.ndarray, name: str, odirname: Path, target = None) -> None:

    odirname = odirname
    odirname.mkdir(parents=True, exist_ok=True)

    concat = np.concatenate((input, pred), axis=1)

    if target:
        concat = np.concatenate((concat, target), axis=1)

    axes = AXES[-concat.ndim:]

    tiff.imwrite(
        f"{odirname}/{name}",
        concat,
        imagej=True,
        metadata={"mode": "color", "axes": axes},
    )
