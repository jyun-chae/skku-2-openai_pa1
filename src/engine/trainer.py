"""
Training engine for semantic segmentation models.

This module provides training utilities including single epoch training and
full training loop with validation, checkpointing, and logging.
"""

import os
import torch
from tqdm import tqdm

from src.engine.evaluator import evaluate
from src.engine.checkpoint import save_checkpoint

from src.utils.logger import log_metrics_to_wandb


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch,
    cfg,
    scaler=None,
    scheduler=None,
):
    """Train model for one epoch.

    Returns dict: Training results containing average loss.
    """
    model.train()

    total_loss = 0.0
    num_batches = 0

    # Check if using automatic mixed precision
    use_amp = bool(getattr(cfg.training, "amp", True)) and device.type == "cuda"

    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)

    for step, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            # Mixed precision training
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision training
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Step scheduler if provided (e.g., for OneCycleLR)
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        pbar.set_postfix(loss=f"{avg_loss:.4f}")

    return {
        "loss": total_loss / max(num_batches, 1),
    }


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    metric,
    device,
    cfg,
    scheduler=None,
    scaler=None,
    start_epoch=0,
    best_val_miou=0.0,
    history=None,
):
    """Run full training loop with validation and checkpointing.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training dataset.
        val_loader: DataLoader for validation dataset (optional).
        criterion: Loss function.
        optimizer: Optimizer for updating model parameters.
        metric: Metric object for validation evaluation.
        device: Device to run training on.
        cfg: Configuration object.
        scheduler: Learning rate scheduler (optional).
        scaler: Gradient scaler for mixed precision (optional).
        start_epoch: Epoch to start training from (for resuming).
        best_val_miou: Best validation mIoU achieved so far.
        history: Training history dictionary.

    Returns:
        dict: Training results containing best validation mIoU and full history.
    """
    if history is None:
        history = {}

    # Initialize history tracking
    history.setdefault("train_loss", [])
    history.setdefault("val_loss", [])
    history.setdefault("val_miou", [])

    epochs = int(cfg.training.epochs)

    eval_interval = int(getattr(cfg.training, "eval_interval", 1))

    checkpoint_dir = getattr(cfg.checkpoint, "save_dir", "checkpoints")

    last_path = os.path.join(checkpoint_dir, "last.pth")
    best_path = os.path.join(checkpoint_dir, "best.pth")

    for epoch in range(start_epoch, epochs):
        # Train for one epoch
        train_result = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            cfg=cfg,
            scaler=scaler,
            scheduler=scheduler,
        )

        train_loss = train_result["loss"]
        history["train_loss"].append(train_loss)

        print(f"[Epoch {epoch}] train_loss: {train_loss:.4f}")

        # Determine if validation should be performed
        do_eval = (
            val_loader is not None
            and ((epoch + 1) % eval_interval == 0 or epoch == epochs - 1)
        )

        if do_eval:
            # Run validation
            val_result = evaluate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                metric=metric,
                device=device,
                cfg=cfg,
                desc=f"Validation Epoch {epoch}",
            )

            val_loss = val_result["loss"]
            val_miou = val_result["miou"]

            history["val_loss"].append(val_loss)
            history["val_miou"].append(val_miou)

            print(
                f"[Epoch {epoch}] "
                f"val_loss: {val_loss:.4f}, "
                f"val_mIoU: {val_miou:.4f}"
            )

            # Save best checkpoint if improved
            if val_miou > best_val_miou:
                best_val_miou = val_miou

                save_checkpoint(
                    save_path=best_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_val_miou=best_val_miou,
                    history=history,
                )

                print(f"Best checkpoint saved: {best_path}")

            # Log metrics to Weights & Biases
            log_metrics_to_wandb(
                epoch=epoch,
                train_metrics=train_result,
                val_metrics={
                    "loss": val_loss,
                    "miou": val_miou,
                    "per_class_iou": val_result["metric"].get("per_class_iou"),
                },
                optimizer=optimizer,
            )

        else:
            # Log only training metrics
            log_metrics_to_wandb(
                epoch=epoch,
                train_metrics=train_result,
                optimizer=optimizer,
            )

        # Save latest checkpoint
        save_checkpoint(
            save_path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_val_miou=best_val_miou,
            history=history,
        )

        print(f"Last checkpoint saved: {last_path}")

    return {
        "best_val_miou": best_val_miou,
        "history": history,
    }