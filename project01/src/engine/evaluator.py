import torch
from tqdm import tqdm


@torch.no_grad()
def evaluate(
    model,
    val_loader,
    criterion,
    metric,
    device,
    cfg=None,
    desc="Validation",
):
    model.eval()

    total_loss = 0.0
    num_batches = 0

    if hasattr(metric, "reset"):
        metric.reset()

    pbar = tqdm(val_loader, desc=desc, leave=False)

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        use_amp = bool(getattr(cfg.training, "amp", True)) and device.type == "cuda"

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        total_loss += loss.item()
        num_batches += 1

        preds = torch.argmax(outputs, dim=1)

        if hasattr(metric, "update"):
            metric.update(preds, targets)

        avg_loss = total_loss / max(num_batches, 1)
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

    avg_loss = total_loss / max(num_batches, 1)

    if hasattr(metric, "compute"):
        metric_result = metric.compute()
    else:
        metric_result = {}

    if isinstance(metric_result, dict):
        miou = metric_result.get("miou", metric_result.get("mIoU", 0.0))
    else:
        miou = float(metric_result)

    return {
        "loss": avg_loss,
        "miou": float(miou),
        "metric": metric_result,
    }