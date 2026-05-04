"""
Metrics computation for semantic segmentation evaluation.

This module provides utilities for computing Intersection over Union (IoU) metrics
using confusion matrices, supporting both per-class and mean IoU calculations.
"""

import torch


@torch.no_grad()
def update_confusion_matrix(
    confmat: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    """
    preds:   [B, H, W]
    targets: [B, H, W]
    """

    preds = preds.detach()
    targets = targets.detach()

    # Filter out ignored pixels
    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]

    # Filter out invalid class indices
    valid = (targets >= 0) & (targets < num_classes)
    preds = preds[valid]
    targets = targets[valid]

    # Compute confusion matrix using bincount for efficiency
    inds = targets * num_classes + preds
    cm = torch.bincount(
        inds,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    return confmat + cm.to(confmat.device)


def compute_iou_from_confmat(
    confmat: torch.Tensor,
    eps: float = 1e-10,
):
    """Compute IoU metrics from confusion matrix.

    Returns:
        tuple: (per_class_iou, mean_iou)
            - per_class_iou: IoU for each class [num_classes]
            - mean_iou: Mean IoU across valid classes
    """
    confmat = confmat.float()

    # True positives, false positives, false negatives
    tp = torch.diag(confmat)
    fp = confmat.sum(dim=0) - tp
    fn = confmat.sum(dim=1) - tp

    # IoU = TP / (TP + FP + FN)
    denom = tp + fp + fn
    iou = tp / (denom + eps)

    # Mean IoU over classes that appear in the data
    valid = denom > 0
    miou = iou[valid].mean().item() if valid.any() else 0.0

    return iou.cpu(), miou


class MeanIoU:
    """Mean Intersection over Union metric for semantic segmentation.

    Accumulates predictions and targets into a confusion matrix,
    then computes per-class IoU and mean IoU on demand.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255, device: str = "cpu"):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device
        self.reset()

    def reset(self):
        """Reset confusion matrix to zeros."""
        self.confmat = torch.zeros(
            self.num_classes,
            self.num_classes,
            dtype=torch.int64,
            device=self.device,
        )

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.confmat = update_confusion_matrix(
            confmat=self.confmat,
            preds=preds,
            targets=targets,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

    def compute(self):
        per_class_iou, miou = compute_iou_from_confmat(self.confmat)

        return {
            "per_class_iou": per_class_iou,
            "miou": miou,
        }