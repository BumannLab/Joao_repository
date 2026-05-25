from pathlib import Path

import numpy as np
from skimage.measure import label, regionprops
from scipy.spatial.distance import cdist

from utils import read_image

IMAGE_FILE_2D = "mTB_segmentation/3Dvs2D/2D/2nd_model/model_test_performance.tif"
IMAGE_FILE_3D = "mTB_segmentation/3Dvs2D/3D/2nd_model/model_test_performance.tif"

# -----------------------------
# Dice between two binary masks
# -----------------------------
def dice(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    inter = np.logical_and(mask1, mask2).sum()
    size = mask1.sum() + mask2.sum()

    if size == 0:
        return 1.0

    return 2 * inter / size


# -----------------------------
# Extract objects from mask
# -----------------------------
def get_objects(mask):
    labeled = label(mask > 0)
    props = regionprops(labeled)

    objects = []
    for p in props:
        objects.append({
            "label": p.label,
            "coords": p.coords,
            "centroid": p.centroid,
            "area": p.area
        })

    return labeled, objects


# -----------------------------
# Match objects (simple centroid matching)
# -----------------------------
def match_objects(gt_objs, pred_objs, dist_thresh=10):

    if len(gt_objs) == 0 or len(pred_objs) == 0:
        return [], set(), set()

    gt_centroids = np.array([o["centroid"] for o in gt_objs])
    pred_centroids = np.array([o["centroid"] for o in pred_objs])

    dist_matrix = cdist(gt_centroids, pred_centroids)

    matches = []
    matched_gt = set()
    matched_pred = set()

    while True:
        i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        min_dist = dist_matrix[i, j]

        if min_dist > dist_thresh:
            break

        if i in matched_gt or j in matched_pred:
            dist_matrix[i, j] = np.inf
            continue

        matches.append((i, j))
        matched_gt.add(i)
        matched_pred.add(j)

        dist_matrix[i, :] = np.inf
        dist_matrix[:, j] = np.inf

    return matches, matched_gt, matched_pred


# -----------------------------
# Main evaluation function
# -----------------------------
def evaluate_objects(gt_mask, pred_mask, dist_thresh=10):

    gt_labeled, gt_objs = get_objects(gt_mask)
    pred_labeled, pred_objs = get_objects(pred_mask)

    matches, matched_gt, matched_pred = match_objects(
        gt_objs, pred_objs, dist_thresh
    )

    dice_scores = []

    # compute Dice for matched objects
    for gi, pi in matches:
        gt_obj = gt_labeled == gt_objs[gi]["label"]
        pred_obj = pred_labeled == pred_objs[pi]["label"]

        d = dice(gt_obj, pred_obj)
        dice_scores.append(d)

    # missed and extra objects
    fn = len(gt_objs) - len(matched_gt)
    fp = len(pred_objs) - len(matched_pred)

    results = {
        "mean_object_dice": np.mean(dice_scores) if dice_scores else 0.0,
        "median_object_dice": np.median(dice_scores) if dice_scores else 0.0,
        "num_gt_objects": len(gt_objs),
        "num_pred_objects": len(pred_objs),
        "tp_objects": len(matches),
        "fp_objects": fp,
        "fn_objects": fn,
        "dice_per_object": dice_scores
    }

    return results


def evaluate_dataset(gt_list, pred_list, dist_thresh=10):
    """
    gt_list: list of GT masks (2D arrays)
    pred_list: list of prediction masks (2D arrays)
    """

    assert len(gt_list) == len(pred_list)

    all_dice = []
    all_tp = []
    all_fp = []
    all_fn = []

    for gt, pred in zip(gt_list, pred_list):

        results = evaluate_objects(
            gt_mask=gt,
            pred_mask=pred,
            dist_thresh=dist_thresh
        )

        # collect per-image results
        all_dice.append(results["dice_per_object"])
        all_tp.append(results["tp_objects"])
        all_fp.append(results["fp_objects"])
        all_fn.append(results["fn_objects"])

    # flatten Dice across ALL objects in dataset
    all_dice_flat = np.concatenate([d for d in all_dice if len(d) > 0])

    return {
        "mean_object_dice": np.mean(all_dice_flat) if len(all_dice_flat) > 0 else 0,
        "median_object_dice": np.median(all_dice_flat) if len(all_dice_flat) > 0 else 0,
        "mean_tp_objects": np.mean(all_tp),
        "mean_fp_objects": np.mean(all_fp),
        "mean_fn_objects": np.mean(all_fn),
        "num_images": len(gt_list)
    }


def main(image_file_2d, image_file_3d):

    image_2d = read_image(Path(image_file_2d))

    gt = image_2d[:, 1]
    pred_2d = image_2d[:, -1]
    print(pred_2d.shape)

    image_3d = read_image(Path(image_file_3d))
    pred_3d_proj = np.max(image_3d, axis = 1)[:, -1]
    print(pred_3d_proj.shape)

    results_2d = evaluate_dataset(gt, pred_2d)
    results_3d = evaluate_dataset(gt, pred_3d_proj)

    print(results_2d)
    print(results_3d)

if __name__=="__main__":
    main(IMAGE_FILE_2D,
         IMAGE_FILE_3D)

# -----------------------------
# Example usage
# -----------------------------

# gt = your ground truth mask (2D)
# pred_2d = 2D model output
# pred_3d_proj = max(z) projection of 3D output

# results_2d = evaluate_objects(gt, pred_2d)
# results_3d = evaluate_objects(gt, pred_3d_proj)

# print(results_2d)
# print(results_3d)