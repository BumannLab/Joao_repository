from pathlib import Path

import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

from Scripts.utils import read_image


IMAGE_FILE_2D = ""
IMAGE_FILE_3D = ""


def get_objects(mask):
    """
    Convert binary mask into connected components (objects).
    """

    labeled = label(mask > 0, connectivity=2)
    props = regionprops(labeled)

    objects = []

    for p in props:

        obj_mask = labeled == p.label

        objects.append({
            "label": p.label,
            "mask": obj_mask,
            "area": p.area
        })

    return labeled, objects


def dice_score(mask1, mask2):
    """
    Compute Dice overlap between two binary masks.
    """

    intersection = np.logical_and(mask1, mask2).sum()

    size1 = mask1.sum()
    size2 = mask2.sum()

    if size1 + size2 == 0:
        return 1.0

    return (2 * intersection) / (size1 + size2)


def match_objects(gt_objects, pred_objects, dice_threshold=0.3):
    """
    Match GT objects to predicted objects using Dice overlap.
    """

    matches = []
    matched_gt = set()
    matched_pred = set()

    for gt_idx, gt_obj in enumerate(gt_objects):

        best_pred = None
        best_dice = 0

        for pred_idx, pred_obj in enumerate(pred_objects):

            d = dice_score(gt_obj["mask"], pred_obj["mask"])

            if d > dice_threshold and d > best_dice:
                best_dice = d
                best_pred = pred_idx

        if best_pred is not None:
            matches.append((gt_idx, best_pred, best_dice))
            matched_gt.add(gt_idx)
            matched_pred.add(best_pred)

    return matches, matched_gt, matched_pred


def visualize(gt, pred, idx=None):
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(gt, cmap="gray")
    plt.title(f"GT {idx}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(pred, cmap="gray")
    plt.title(f"Pred {idx}")
    plt.axis("off")

    plt.show()


def evaluate_dataset(gt_list, pred_list, dice_threshold=0.3, visualize=False):

    TP_objects = 0
    FN_objects = 0
    FP_objects = 0

    TN_images = 0

    all_dice = []

    for i, (gt, pred) in enumerate(zip(gt_list, pred_list)):

        gt_has_objects = np.any(gt > 0)
        pred_has_objects = np.any(pred > 0)

        if visualize:
            visualize(gt, pred, i)

        # NEGATIVE IMAGES
        if not gt_has_objects:

            if pred_has_objects:

                # count ALL predicted objects as FP
                _, pred_objects = get_objects(pred)
                FP_objects += len(pred_objects)

            else:
                TN_images += 1

            continue


        # POSITIVE IMAGES
        _, gt_objects = get_objects(gt)
        _, pred_objects = get_objects(pred)

        matches, matched_gt, matched_pred = match_objects(
            gt_objects,
            pred_objects,
            dice_threshold=dice_threshold
        )

        # TP objects
        TP_objects += len(matches)

        # FN objects
        FN_objects += len(gt_objects) - len(matched_gt)

        # store Dice
        for _, _, d in matches:
            all_dice.append(d)

    # metrics
    recall = TP_objects / (TP_objects + FN_objects) if (TP_objects + FN_objects) > 0 else 0

    specificity = TN_images / (TN_images + (len(gt_list) - TN_images)) if len(gt_list) > 0 else 0

    mean_dice = np.mean(all_dice) if len(all_dice) > 0 else 0

    precision = TP_objects / (TP_objects + FP_objects) if (TP_objects + FP_objects) > 0 else 0

    return {
        "TP_objects": TP_objects,
        "FN_objects": FN_objects,
        "FP_objects": FP_objects,
        "TN_images": TN_images,
        "recall": recall,
        "precision": precision,
        "specificity": specificity,
        "mean_object_dice": mean_dice
    }

def main(image_file_2d, image_file_3d):
    
    images_2d = read_image(Path(image_file_2d))
    gt_images = images_2d[:, 1]
    pred_2d = images_2d[:, -1]

    images_3d = read_image(Path(image_file_3d))
    pred_3d_proj = np.max(images_3d, axis = 1)[:, -1]

    print("-2D-\n", evaluate_dataset(gt_images, pred_2d, dice_threshold=0.2))
    print("-3D-\n", evaluate_dataset(gt_images, pred_3d_proj, dice_threshold=0.2))

