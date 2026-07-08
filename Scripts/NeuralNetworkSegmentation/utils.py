import importlib
import random
from pathlib import Path
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


def convert_to_tensor(*inputs):
    """
    Coverts input numpy ndarrays to tensors 

    Args:
        inputs (iteable of np.ndarray): numpy array

    Returns:
        tuple of tensors
    """

    def _to_tensor(i):
        assert isinstance(i, np.ndarray), "Expected input to be torch.Tensor"
        return torch.from_numpy()

    return (_to_tensor(i) for i in inputs)


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
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    save_model_name = save_point_config["out_model_name"]

    return save_dir, save_model_name


