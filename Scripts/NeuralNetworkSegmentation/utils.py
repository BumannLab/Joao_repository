import importlib
import random
from datetime import datetime

import torch
import numpy as np


def get_class(class_name: str, modules: list[str]) -> type:
    """Helper function which searches for a class in the given list of modules and returns it."""
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f"Unsupported dataset class: {class_name}")


def convert_to_numpy(*inputs):
    """
    Coverts input tensors to numpy ndarrays

    Args:
        inputs (iteable of torch.Tensor): torch tensor

    Returns:
        tuple of ndarrays
    """

    def _to_numpy(i):
        assert isinstance(i, torch.Tensor), "Expected input to be torch.Tensor"
        return i.detach().cpu().numpy()

    return (_to_numpy(i) for i in inputs)


def show_positive_transformed_samples(dataset, max_samples=5, channel=0, z_index=None):
    """Show transformed input/mask pairs for samples whose dataset label is 1."""

    import matplotlib.pyplot as plt

    def _to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)

    def _select_plane(array):
        array = _to_numpy(array)
        if array.ndim == 5:
            array = array[0]
        if array.ndim == 4:
            channel_idx = min(channel, array.shape[0] - 1)
            z_idx = array.shape[1] // 2 if z_index is None else min(z_index, array.shape[1] - 1)
            return array[channel_idx, z_idx]
        if array.ndim == 3:
            z_idx = array.shape[0] // 2 if z_index is None else min(z_index, array.shape[0] - 1)
            return array[z_idx]
        return array

    def _get_positive_items():
        if hasattr(dataset, "samples"):
            for label, file_name, input_image, label_image in dataset.samples:
                if label != 1:
                    continue

                input_tensor = torch.from_numpy(input_image / 65535).to(torch.float32)
                label_tensor = torch.from_numpy(label_image / 255).to(torch.long)

                if dataset.raw_transform:
                    input_tensor = dataset.raw_transform(input_tensor)
                if dataset.label_transform:
                    label_tensor = dataset.label_transform(label_tensor)

                yield file_name, input_tensor.permute(1, 0, 2, 3), label_tensor.permute(1, 0, 2, 3)
        else:
            for idx in range(len(dataset)):
                input_tensor, label_tensor = dataset[idx]
                if torch.any(label_tensor > 0):
                    yield str(idx), input_tensor, label_tensor

    shown = 0
    for file_name, input_tensor, label_tensor in _get_positive_items():
        input_plane = _select_plane(input_tensor)
        label_plane = _select_plane(label_tensor)

        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        fig.suptitle(str(file_name))

        ax[0].imshow(input_plane, cmap="gray")
        ax[0].set_title("input")
        ax[0].axis("off")

        ax[1].imshow(label_plane, cmap="gray")
        ax[1].set_title("label")
        ax[1].axis("off")

        ax[2].imshow(input_plane, cmap="gray")
        ax[2].imshow(label_plane, alpha=0.4)
        ax[2].set_title("overlay")
        ax[2].axis("off")

        plt.tight_layout()
        plt.show()

        shown += 1
        if shown >= max_samples:
            break


def build_seed_from_time():
    return int(datetime.now().timestamp())


def seed_everything(seed):
    """ Seeds all the packages into one seed
    

    Args:
        seed (_type_): _description_
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_save_dir_model_name(config):
    # Make sure it exists or else we train a model that will not be saved   
    assert config.get("save_point"), "Make sure to add a save point that contains the directory to write the model output and the name of the model name"
    save_point_config =  config["save_point"]
    
    save_dir = save_point_config["out_dir"]
    save_model_name = save_point_config["out_model_name"]

    return save_dir, save_model_name
