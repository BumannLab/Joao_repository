from monai.metrics import (
    DiceMetric,
    MeanIoU,
    get_confusion_matrix
)
import torch.nn.functional as F


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


def confusion_matrix(y_pred, y):

    y_pred = F.one_hot(y_pred.argmax(dim=1), num_classes=y_pred.shape[1]).permute(0, 3, 1, 2).contiguous()
    
    y = F.one_hot(y.squeeze(dim=1), num_classes=y.shape[1]).permute(0, 3, 1, 2).contiguous()

    _confusion_matrix = get_confusion_matrix(y_pred, y)

    return _confusion_matrix