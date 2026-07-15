import importlib
import random
import os
import shutil
import logging
import sys

import torch
from torch import nn
import numpy as np
from torch.optim import Optimizer


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


def save_checkpoint(state: dict, is_best: bool, checkpoint_dir: str):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.

    Args:
        state: contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best: if True state contains the best model seen so far
        checkpoint_dir: directory where the checkpoint are to be saved
    """

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, "last_checkpoint.pytorch")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, "best_checkpoint.pytorch")
        shutil.copyfile(last_file_path, best_file_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optimizer = None,
    model_key: str = "model_state_dict",
    optimizer_key: str = "optimizer_state_dict",
) -> dict:
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.

    Args:
        checkpoint_path: path to the checkpoint to be loaded
        model: model into which the parameters are to be copied
        optimizer: optimizer instance into which the parameters are to be copied
        model_key: key corresponding to the model state_dict in the checkpoint
        optimizer_key: key corresponding to the optimizer state_dict in the checkpoint

    Returns:
        state dict stored in the checkpoint
    """
    if not os.path.exists(checkpoint_path):
        raise OSError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state[model_key])

    if optimizer is not None:
        optimizer.load_state_dict(state[optimizer_key])

    return state


loggers = {}


def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Initializes and returns a logger with the given name."""

    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s")
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        loggers[name] = logger

        return logger