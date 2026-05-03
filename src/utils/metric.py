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

    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]

    valid = (targets >= 0) & (targets < num_classes)
    preds = preds[valid]
    targets = targets[valid]

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
    confmat = confmat.float()

    tp = torch.diag(confmat)
    fp = confmat.sum(dim=0) - tp
    fn = confmat.sum(dim=1) - tp

    denom = tp + fp + fn
    iou = tp / (denom + eps)

    valid = denom > 0
    miou = iou[valid].mean().item() if valid.any() else 0.0

    return iou.cpu(), miou


class MeanIoU:
    def __init__(self, num_classes: int, ignore_index: int = 255, device: str = "cpu"):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device
        self.reset()

    def reset(self):
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