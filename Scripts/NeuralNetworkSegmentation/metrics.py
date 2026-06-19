import numpy as np
import torch
import torch.nn.functional as F
from monai.metrics import LossMetric, DiceMetric, MeanIoU, get_confusion_matrix
from skimage import measure

from Scripts.NeuralNetworkSegmentation.utils import convert_to_numpy


class BlobsAveragePrecision:
    def __init__(
        self,
        input_channel=1,
        threshold=None,
        iou_threshold=0.5,
        min_instance_size=None,
        metric="precision",
        **kwargs,
    ):
        if threshold is None:
            threshold = 0.5

        if metric not in {"precision", "accuracy"}:
            raise ValueError("metric must be 'precision' or 'accuracy'")

        self.threshold = threshold
        self.input_channel = input_channel
        self.iou_threshold = iou_threshold
        self.min_instance_size = min_instance_size
        self.metric = metric

    def __call__(self, input, target):
        input_np, target_np = convert_to_numpy(input, target)

        batch_scores = []
        for prediction, target_mask in zip(input_np, target_np, strict=True):
            # Convert each target sample to an instance-labeled segmentation once,
            # then score the prediction across several foreground thresholds.
            target_seg = self.target_to_seg(target_mask)

            score = self._score_segmentation(self.input_to_seg(prediction, self.threshold), target_seg)

            batch_scores.append(score)

        return torch.tensor(batch_scores, dtype=input.dtype if input.is_floating_point() else torch.float32).mean()

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

        if metric_name == "Dice":
            include_bg = metrics_config.get("include_background", True)
            return DiceMetric(include_background=include_bg)

        if metric_name == "MeanIoU":
            include_bg = metrics_config.get("include_background", True)
            return MeanIoU(include_background=include_bg)

        if metric_name == "BlobsAveragePrecision":
            return LossMetric(loss_fn=BlobsAveragePrecision(**metrics_config))

        raise ValueError(f"Unsupported metric: {metric_name}")


def compute_confusion_matrix(output, target, threshold = 0.5): # TODO: ADD THRESHOLD INSTEAD OF ARGMAX AFTER
    n_channels = output.shape[1]

    output_flat = F.one_hot(output.argmax(dim=1), num_classes=n_channels).movedim(-1, 1).contiguous()
    target_flat = F.one_hot(target.squeeze(dim=1), num_classes=n_channels).movedim(-1, 1).contiguous()

    confusion_matrix, = convert_to_numpy(get_confusion_matrix(output_flat, target_flat))
    # MONAI returns shape B x C x 4 here: one row per batch item, one row per
    # class/channel, and the 4 values are TP, FP, TN, FN. It is not a single
    # prediction channel; summing over axis 0 below collapses the batch only.

    return np.sum(confusion_matrix, axis = 0)
