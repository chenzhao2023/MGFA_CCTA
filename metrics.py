import torch
import numpy as np
from scipy.spatial.distance import cdist

def compute_dice_coefficientnp(seg_true, seg_pred, smooth=1e-5):
    intersection = np.sum(seg_true * seg_pred)
    union = np.sum(seg_true) + np.sum(seg_pred)
    dice_coeff = (2.0 * intersection + smooth) / (union + smooth)
    return dice_coeff

def compute_dice_coefficient(seg_true, seg_pred, smooth=1e-5):
    intersection = torch.sum(seg_true * seg_pred)
    union = torch.sum(seg_true) + torch.sum(seg_pred)

    dice_coeff = (2.0 * intersection + smooth) / (union + smooth)
    return dice_coeff


def compute_sensitivity(seg_true, seg_pred):
    true_positive = np.sum(seg_true * seg_pred)
    false_negative = np.sum(seg_true * (1 - seg_pred))
    sensitivity = true_positive / (true_positive + false_negative)
    return sensitivity


def compute_specificity(seg_true, seg_pred):
    true_negative = np.sum((1 - seg_true) * (1 - seg_pred))
    false_positive = np.sum((1 - seg_true) * seg_pred)
    specificity = true_negative / (true_negative + false_positive)
    return specificity



def calculate_precision_recall(pred_mask, true_mask):

    TP = np.sum((pred_mask == 1) & (true_mask == 1))
    FP = np.sum((pred_mask == 1) & (true_mask == 0))
    FN = np.sum((pred_mask == 0) & (true_mask == 1))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return precision, recall


def calculate_hd95(prediction, ground_truth, pixel_spacing=(1.0, 1.0, 1.0)):

    def surface_points(mask):

        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(mask)
        mask = mask.astype(bool)
        surface = mask & ~eroded

        return np.argwhere(surface)


    pred_surface = surface_points(prediction) * np.array(pixel_spacing).reshape(1, 3)
    gt_surface = surface_points(ground_truth) * np.array(pixel_spacing).reshape(1, 3)

    if pred_surface.size == 0 or gt_surface.size == 0:
        raise ValueError("null error")

    distances = cdist(pred_surface, gt_surface)

    pred_to_gt_min = np.min(distances, axis=1)

    gt_to_pred_min = np.min(distances, axis=0)

    all_distances = np.concatenate([pred_to_gt_min, gt_to_pred_min])
    hd95 = np.percentile(all_distances, 95)

    return hd95