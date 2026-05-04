"""
Training entrypoint for semantic segmentation.

This script sets up training for a segmentation model, including data loaders,
model creation, loss and metric definitions, optimizer and scheduler configuration,
and optional checkpoint resume.
"""

import torch
import torch.nn as nn

from src.data.build import build_dataloaders
from src.models.build import build_model
from src.engine.trainer import fit
from src.engine.checkpoint import load_checkpoint

from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.utils.metric import MeanIoU


def main(cfg):
    # -------------------------
    # 1. Runtime setup
    # -------------------------
    set_seed(cfg.runtime.seed)

    device = torch.device(
        cfg.runtime.device if torch.cuda.is_available() else "cpu"
    )

    logger = get_logger()
    logger.info(f"Using device: {device}")

    use_amp = bool(getattr(cfg.training, "amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # -------------------------
    # 2. Dataset / Dataloader
    # -------------------------
    train_loader, val_loader = build_dataloaders(cfg)

    # -------------------------
    # 3. Model
    # -------------------------
    model = build_model(cfg)
    model = model.to(device)

    # -------------------------
    # 4. Loss / Metric
    # -------------------------
    criterion = nn.CrossEntropyLoss(
        ignore_index=cfg.data.ignore_index
    )

    metric = MeanIoU(
        num_classes=cfg.model.num_classes,
        ignore_index=cfg.data.ignore_index,
        device=device,
    )

    # -------------------------
    # 5. Optimizer
    # -------------------------
    # optimizer = torch.optim.AdamW(
    #     [
    #         {"params": model.backbone.parameters(), "lr": cfg.training.backbone_lr},
    #         {"params": model.lateral_c2.parameters(), "lr": cfg.training.head_lr},
    #         {"params": model.lateral_c3.parameters(), "lr": cfg.training.head_lr},
    #         {"params": model.lateral_c4.parameters(), "lr": cfg.training.head_lr},
    #         {"params": model.lateral_c5.parameters(), "lr": cfg.training.head_lr},
    #         {"params": model.smooth_c2.parameters(), "lr": cfg.training.head_lr},
    #         {"params": model.smooth_c3.parameters(), "lr": cfg.training.head_lr},
    #         {"params": model.smooth_c4.parameters(), "lr": cfg.training.head_lr},
    #         {"params": model.head.parameters(), "lr": cfg.training.head_lr},
    #     ],
    #     weight_decay=cfg.training.weight_decay,
    # )
    optimizer = torch.optim.SGD(
        [
            {"params": model.backbone.parameters(), "lr": cfg.training.backbone_lr},
            {"params": model.lateral_c2.parameters(), "lr": cfg.training.head_lr},
            {"params": model.lateral_c3.parameters(), "lr": cfg.training.head_lr},
            {"params": model.lateral_c4.parameters(), "lr": cfg.training.head_lr},
            {"params": model.lateral_c5.parameters(), "lr": cfg.training.head_lr},
            {"params": model.smooth_c2.parameters(), "lr": cfg.training.head_lr},
            {"params": model.smooth_c3.parameters(), "lr": cfg.training.head_lr},
            {"params": model.smooth_c4.parameters(), "lr": cfg.training.head_lr},
            {"params": model.head.parameters(), "lr": cfg.training.head_lr},
        ],
        momentum=0.9,
        weight_decay=cfg.training.weight_decay,
        nesterov=True,
    )

    # -------------------------
    # 6. Scheduler
    # -------------------------
    # Optionally configure a learning rate scheduler.
    scheduler = None

    scheduler_name = getattr(cfg.training, "scheduler", "none").lower()

    if scheduler_name == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cfg.training.lr_decay,
        )
    elif scheduler_name in ["none", ""]:
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {cfg.training.scheduler}")

    # -------------------------
    # 7. Resume
    # -------------------------
    # Initialize resume state in case a checkpoint is loaded.
    start_epoch = 0
    best_val_miou = 0.0
    history = None

    resume_path = getattr(cfg.checkpoint, "resume_path", "")

    if resume_path:
        logger.info(f"Resuming from checkpoint: {resume_path}")

        start_epoch, best_val_miou, history, _ = load_checkpoint(
            load_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )

        logger.info(
            f"Resume success. "
            f"start_epoch={start_epoch}, "
            f"best_val_miou={best_val_miou:.4f}"
        )

    # -------------------------
    # 8. Training
    # -------------------------
    result = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        metric=metric,
        device=device,
        cfg=cfg,
        scheduler=scheduler,
        scaler=scaler,
        start_epoch=start_epoch,
        best_val_miou=best_val_miou,
        history=history,
    )

    logger.info(
        f"Training finished. "
        f"Best val mIoU: {result['best_val_miou']:.4f}"
    )

    return result


if __name__ == "__main__":
    from src.config.config import load_config

    cfg = load_config("src/config/default.yaml")
    main(cfg)