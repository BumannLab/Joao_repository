import warnings

import numpy as np
import torch
import torch.nn.functional as F
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, MeanIoU, get_confusion_matrix
from monai.metrics.metric import CumulativeIterationMetric
from skimage import measure
from monai.utils import MetricReduction, LossReduction 
from monai.networks.utils import one_hot


from Scripts.NeuralNetworkSegmentation.utils import convert_to_numpy


class HardDice(DiceMetric):
    def __init__(
        self, 
        threshold_value = 0.5, 
        include_background = True, 
        reduction = MetricReduction.MEAN,
        ignore_empty = True,
        **kwargs,
    ):
        super().__init__(
            include_background=include_background,
            reduction=reduction,
            get_not_nans=False,
            ignore_empty=ignore_empty,
        )

        self.threshold_value = threshold_value

    def __call__(self, y_pred, y):

        y_pred = (y_pred > self.threshold_value).float()

        y = y.float()

        return super().__call__(y_pred, y)
    

class SoftDice(CumulativeIterationMetric):
    def __init__(
        self,
        mode="sigmoid",
        include_background=True,
        to_onehot_y=None,
        squared_pred=False,
        jaccard=False,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
        weight=None,
        **kwargs,
    ):
        super().__init__()

        if mode not in ("sigmoid", "softmax"):
            raise ValueError("mode must be 'sigmoid' or 'softmax'")

        # softmax mode expects class-index targets → convert to one-hot by default
        # sigmoid mode expects per-channel binary targets → no conversion needed
        if to_onehot_y is None:
            to_onehot_y = mode == "softmax"

        self.dice_loss = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=False,   # input is already post-activation probabilities
            softmax=False,   # input is already post-activation probabilities
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=LossReduction.MEAN,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=False,
            weight=weight,
        )

    def _compute_tensor(self, y_pred, y):
        scores = []
        for i in range(y_pred.shape[0]):
            loss = self.dice_loss(y_pred[i:i+1], y[i:i+1])
            scores.append(1.0 - loss.item())
        return torch.tensor(scores, dtype=y_pred.dtype, device=y_pred.device)

    def aggregate(self, reduction=MetricReduction.MEAN):
        data = self.get_buffer()
        if data is None:
            raise RuntimeError("No data accumulated.")
        if reduction == MetricReduction.MEAN:
            return data.mean()
        if reduction == MetricReduction.SUM:
            return data.sum()
        return data


class BlobsAveragePrecision(CumulativeIterationMetric):
    def __init__(
        self,
        input_channel=0,
        threshold=None,
        iou_threshold=0.5,
        min_instance_size=None,
        metric="precision",
        **kwargs,
    ):
        super().__init__()

        if threshold is None:
            threshold = 0.5

        if metric not in {"precision", "accuracy"}:
            raise ValueError("metric must be 'precision' or 'accuracy'")

        self.threshold = threshold
        self.input_channel = input_channel
        self.iou_threshold = iou_threshold
        self.min_instance_size = min_instance_size
        self.metric = metric

    def _compute_tensor(self, input, target):
        input_np, target_np = convert_to_numpy(input, target)

        batch_scores = []
        for prediction, target_mask in zip(input_np, target_np, strict=True):
            target_seg = self.target_to_seg(target_mask)
            score = self._score_segmentation(self.input_to_seg(prediction, self.threshold), target_seg)
            batch_scores.append(score)

        dtype = input.dtype if input.is_floating_point() else torch.float32
        return torch.tensor(batch_scores, dtype=dtype)

    def aggregate(self, reduction=MetricReduction.MEAN):
        data = self.get_buffer()
        if reduction == MetricReduction.MEAN:
            return data.mean()
        if reduction == MetricReduction.SUM:
            return data.sum()
        return data

    def input_to_seg(self, prediction, threshold):
        # The model predicts background/foreground channels; we segment only the
        # configured foreground probability map.
        foreground = prediction[self.input_channel]
        binary_mask = foreground > threshold
        return self._label_mask(binary_mask)

    def target_to_seg(self, target):
        target = np.asarray(target)

        if target.ndim == 4:
            # Training targets are stored as Cx... with a singleton class axis.
            target = target[0]

        return self._label_mask(target > 0)


    def _label_mask(self, mask):
        seg = measure.label(mask.astype(np.uint8), background=0, connectivity=2)
        return self._filter_instances(seg)

    def _filter_instances(self, seg):
        if self.min_instance_size is None:
            return seg

        seg = seg.copy()
        labels, counts = np.unique(seg, return_counts=True)
        for label, count in zip(labels, counts, strict=True):
            if label != 0 and count < self.min_instance_size:
                seg[seg == label] = 0

        # Relabel after removing small components so instance ids stay compact.
        return measure.label(seg > 0, background=0, connectivity=2)
    

    def _score_segmentation(self, pred_seg, target_seg):
        pred_labels = [label for label in np.unique(pred_seg) if label != 0]
        target_labels = [label for label in np.unique(target_seg) if label != 0]

        if not pred_labels and not target_labels:
            return 1.0
        if not pred_labels:
            return 0.0
        if not target_labels:
            return 0.0

        pred_matches = 0
        target_matches = 0

        for pred_label in pred_labels:
            pred_mask = pred_seg == pred_label
            max_iou = 0.0

            for target_label in target_labels:
                target_mask = target_seg == target_label
                intersection = np.logical_and(pred_mask, target_mask).sum()
                if intersection == 0:
                    continue

                union = np.logical_or(pred_mask, target_mask).sum()
                max_iou = max(max_iou, intersection / union)

            if max_iou >= self.iou_threshold:
                pred_matches += 1

        for target_label in target_labels:
            target_mask = target_seg == target_label
            max_iou = 0.0

            for pred_label in pred_labels:
                pred_mask = pred_seg == pred_label
                intersection = np.logical_and(pred_mask, target_mask).sum()
                if intersection == 0:
                    continue

                union = np.logical_or(pred_mask, target_mask).sum()
                max_iou = max(max_iou, intersection / union)

            if max_iou >= self.iou_threshold:
                target_matches += 1

        false_positives = len(pred_labels) - pred_matches
        false_negatives = len(target_labels) - target_matches

        if self.metric == "precision":
            denominator = pred_matches + false_positives
        else:
            denominator = pred_matches + false_positives + false_negatives

        numerator = pred_matches
        return numerator / denominator if denominator > 0 else 0.0


def get_eval_criterion(metrics_config: dict):

        metric_name = metrics_config["name"]

        if metric_name == "SoftDice":
            return SoftDice(**metrics_config)
        
        if metric_name == "HardDice":
            return HardDice(**metrics_config)

        if metric_name == "MeanIoU":
            warnings.warn("Still not fully implemented")
            return MeanIoU(**metrics_config)

        if metric_name == "BlobsAveragePrecision":
            return BlobsAveragePrecision(**metrics_config)

        raise ValueError(f"Unsupported metric: {metric_name}")


def compute_confusion_matrix(output, target, threshold=0.5):

    output = (output > threshold).float()

    confusion_matrix, = convert_to_numpy(get_confusion_matrix(output.contiguous(), target.contiguous()))
    
    return np.sum(confusion_matrix, axis=0)  # TP, FP, TN, FN