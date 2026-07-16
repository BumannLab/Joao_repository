from pathlib import Path

import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

from Scripts.utils import read_image


IMAGE_FILE_2D = "/data/joao/Materials/Joao_repository/mTB_segmentation/3Dvs2D/2D/5th_model/model_test_performance.tif"
IMAGE_FILE_3D = "/data/joao/Materials/Joao_repository/mTB_segmentation/3Dvs2D/3D/5th_model_v2/model_test_performance.tif"



def get_objects(mask):
    """
    Convert a binary segmentation mask into connected components.

    Each connected foreground region is treated as an individual
    object/instance using connected-component labeling.

    Implementation details
    ----------------------
    - Uses 8-connectivity in 2D (connectivity=2).
    - Each connected component is returned as an independent object.
    - The returned object dictionary contains:
        * label : connected-component integer label
        * mask  : binary mask for the object
        * area  : object area in pixels

    Parameters
    ----------
    mask : ndarray
        Binary segmentation mask.

    Returns
    -------
    list of dict
        List containing object dictionaries.
    """

    # Label connected foreground regions
    labeled = label(mask > 0, connectivity=2)

    # Extract region properties for each connected component
    props = regionprops(labeled)

    # Store extracted objects
    objects = []

    # Convert each connected component into a dictionary
    for p in props:

        objects.append({
            "label": p.label,
            "mask": labeled == p.label,
            "area": p.area
        })

    return objects


def dice_score(a, b):
    """
    Compute Dice similarity coefficient between two binary masks.

    Dice coefficient measures spatial overlap between masks.

    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Implementation details
    ----------------------
    - Uses binary logical intersection.
    - Returns 1.0 if both masks are empty.

    Parameters
    ----------
    a : ndarray
        Binary mask.

    b : ndarray
        Binary mask.

    Returns
    -------
    float
        Dice overlap score in [0, 1].
    """

    intersection = np.logical_and(a, b).sum()

    denom = a.sum() + b.sum()

    # Handle empty-mask edge case
    if denom == 0:
        return 1.0

    # Compute Dice overlap
    return 2 * intersection / denom


def detect_objects(gt_objects, pred_objects, dice_threshold=0.2):
    """
    Perform GT-centric object detection evaluation.

    Each GT object independently searches for the prediction
    with maximum Dice overlap.

    Implementation details
    ----------------------
    - Matching is GT-centric rather than one-to-one.
    - Multiple GT objects ARE allowed to match the SAME prediction.
      This intentionally allows merged predictions to count as
      multiple detections.
    - Predicted objects that fail to sufficiently overlap ANY GT
      are counted as false positives.
    - This metric prioritizes lesion/object sensitivity rather
      than strict instance-segmentation quality.

    Parameters
    ----------
    gt_objects : list
        Ground-truth objects from get_objects().

    pred_objects : list
        Predicted objects from get_objects().

    dice_threshold : float
        Minimum Dice overlap required to count a GT object
        as detected.

    Returns
    -------
    TP : int
        Number of detected GT objects.

    FN : int
        Number of missed GT objects.

    FP : int
        Number of predicted objects not matching any GT object.

    dice_scores : list
        Best Dice score for each detected GT object.
    """

    # GT objects successfully detected
    matched_gt = set()

    # Predictions matched to at least one GT object
    matched_pred = set()

    # Dice scores for detected GT objects
    dice_scores = []

    # Evaluate each GT object independently
    for gt_idx, gt_obj in enumerate(gt_objects):

        # Best Dice overlap for current GT object
        best_dice = 0

        # Index of prediction achieving best Dice
        best_pred_idx = None

        # Search across all predicted objects
        for pred_idx, pred_obj in enumerate(pred_objects):

            # Compute overlap between GT and prediction
            d = dice_score(gt_obj["mask"], pred_obj["mask"])

            # Keep best overlap
            if d > best_dice:
                best_dice = d
                best_pred_idx = pred_idx

        # GT object considered detected
        # if overlap exceeds threshold
        if best_dice >= dice_threshold:

            matched_gt.add(gt_idx)

            # Mark prediction as used
            matched_pred.add(best_pred_idx)

            # Store overlap quality
            dice_scores.append(best_dice)

    # Number of detected GT objects
    TP = len(matched_gt)

    # Number of missed GT objects
    FN = len(gt_objects) - TP

    # Predictions unused by any GT object
    FP = len(pred_objects) - len(matched_pred)

    return TP, FN, FP, dice_scores


def evaluate_dataset(gt_list, pred_list, dice_threshold=0.2):
    """
    Evaluate segmentation predictions across a dataset.

    Evaluation is performed at two levels:

    1. Image-level evaluation
       Applied only to images without GT objects.

       - TN_images:
         Empty GT image correctly predicted empty.

       - FP_images:
         Empty GT image containing predicted objects.

    2. Object-level evaluation
       Applied to images containing GT objects.

       - TP_objects:
         GT objects successfully detected.

       - FN_objects:
         Missed GT objects.

       - FP_objects:
         Predicted objects not matching any GT object.

    Implementation details
    ----------------------
    - Uses GT-centric object matching.
    - Multiple GT objects may match the same prediction.
      This intentionally allows merged predictions to count
      as multiple detections.
    - Hallucinated predicted objects are penalized through
      FP object counting.
    - The metric prioritizes sensitivity/object detection
      rather than strict instance separation quality.

    Parameters
    ----------
    gt_list : list
        List of GT masks.

    pred_list : list
        List of predicted masks.

    dice_threshold : float
        Minimum Dice overlap required for detection.

    Returns
    -------
    dict
        Dictionary containing image-level and object-level metrics.
    """

    # True-negative empty images
    TN_images = 0

    # False-positive empty images
    FP_images = 0

    # Correctly detected GT objects
    TP_objects = 0

    # Missed GT objects
    FN_objects = 0

    # Predicted objects without GT correspondence
    FP_objects = 0

    # Dice scores for detected GT objects
    all_dice = []

    # Process dataset image-by-image
    for i, (gt, pred) in enumerate(zip(gt_list, pred_list)):

        # Determine whether GT contains foreground objects
        gt_has_objects = np.any(gt > 0)

        # Determine whether prediction contains foreground objects
        pred_has_objects = np.any(pred > 0)

        # Handle empty GT images
        if not gt_has_objects:

            # Prediction exists in an empty image
            if pred_has_objects:

                # Image-level false positive
                FP_images += 1

                # Extract predicted connected components
                pred_objects = get_objects(pred)

                # All predicted objects are false positives
                FP_objects += len(pred_objects)

            else:
                # Correctly predicted empty image
                TN_images += 1

            continue

        # Extract GT connected components
        gt_objects = get_objects(gt)

        # Extract predicted connected components
        pred_objects = get_objects(pred)

        # Perform object-level evaluation
        TP, FN, FP, dice_vals = detect_objects(
            gt_objects,
            pred_objects,
            dice_threshold=dice_threshold
        )

        # Accumulate TP count
        TP_objects += TP

        # Accumulate FN count
        FN_objects += FN

        # Accumulate FP count
        FP_objects += FP

        # Store Dice scores
        all_dice.extend(dice_vals)

    # Compute recall/sensitivity
    recall = (
        TP_objects / (TP_objects + FN_objects)
        if (TP_objects + FN_objects) > 0 else 0
    )

    # Compute precision
    precision = (
        TP_objects / (TP_objects + FP_objects)
        if (TP_objects + FP_objects) > 0 else 0
    )

    # Compute image-level specificity
    specificity = (
        TN_images / (TN_images + FP_images)
        if (TN_images + FP_images) > 0 else 0
    )

    # Mean Dice across detected GT objects
    mean_dice = np.mean(all_dice) if len(all_dice) > 0 else 0

    # Return evaluation metrics
    return {
        "TP_objects": TP_objects,
        "FN_objects": FN_objects,
        "FP_objects": FP_objects,

        "TN_images": TN_images,
        "FP_images": FP_images,

        "recall": recall,
        "precision": precision,
        "specificity": specificity,
        "mean_dice": mean_dice
    }


def main(image_file_2d, image_file_3d):
    
    images_2d = read_image(Path(image_file_2d))
    gt_images = images_2d[:, 1]
    pred_2d = images_2d[:, -1]

    images_3d = read_image(Path(image_file_3d))
    pred_3d_proj = np.max(images_3d, axis = 1)[:, -1]

    results_2d = evaluate_dataset(gt_images, pred_2d, dice_threshold=0.2)
    results_3d = evaluate_dataset(gt_images, pred_3d_proj, dice_threshold=0.2)

    print("-2D-\n", results_2d)
    print("-3D-\n", results_3d)


if __name__ == "__main__":
    main(IMAGE_FILE_2D, IMAGE_FILE_3D)