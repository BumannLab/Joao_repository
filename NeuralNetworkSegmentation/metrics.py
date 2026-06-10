from monai.metrics import (
    DiceMetric,
    MeanIoU,
    get_confusion_matrix
)
import torch.nn.functional as F

from NeuralNetworkSegmentation._utils import convert_tensor_to_numpy


def get_metrics(metrics_config: dict):

    metrics = {}

    for metric_name, metric_params in metrics_config.items():

        if metric_name == "Dice":
            include_bg = metric_params.get("include_background", True)
            metrics[metric_name] = DiceMetric(include_background=include_bg)

        elif metric_name == "MeanIoU":
            include_bg = metric_params.get("include_background", True)
            metrics[metric_name] = MeanIoU(include_background=include_bg)

        else:
            raise ValueError(f"Unsupported metric: {metric_name}")

    return metrics


def confusion_matrix(output, target):

    n_channels = output.shape[1]

    output_flat = F.one_hot(output.argmax(dim=1), num_classes=output.shape[1]).permute(0, 3, 1, 2).contiguous()
    
    target_flat = F.one_hot(y.squeeze(dim=1), num_classes=y.shape[1]).permute(0, 3, 1, 2).contiguous()

    conf_mat = get_confusion_matrix(output, y)

    return _confusion_matrix