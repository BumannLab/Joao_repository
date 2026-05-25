import numpy as np
from skimage.measure import label, regionprops
from scipy.spatial.distance import cdist

from utils import read_image

IMAGE_FILE_2D = ""
IMAGE_FILE_3D = ""

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


def main(image_file):

    image_2D = read_image(image_file)

    gt = image[1]
    pred_2d = image[2]

    image_3d = read_image(image_file)
    pred_3d = np.max(image_3d, axis = 0)[-1]

    results_2d = evaluate_objects(gt, pred_2d)
    results_3d = evaluate_objects(gt, pred_3d_proj)

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