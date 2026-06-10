import importlib

import torch
from sklearn.metrics import confusion_matrix


def get_class(class_name: str, modules: list[str]) -> type:
    """Helper function which searches for a class in the given list of modules and returns it."""
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f"Unsupported dataset class: {class_name}")


def convert_tensor_to_numpy(tensor):

    return tensor.cpu().numpy()