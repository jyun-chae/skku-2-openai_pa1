import os
import torch
import torch.nn as nn

from src.data.build import build_dataloaders
from src.models.build import build_model
from src.engine.trainer import train_one_epoch
from src.engine.evaluator import evaluate
from src.engine.checkpoint import save_checkpoint, load_checkpoint

from src.utils.logger import get_logger, log_metrics_to_wandb
from src.utils.seed import set_seed
from src.utils.metric import MeanIoU


def main(cfg):
    # -------------------------
    # 1. 기본 설정
    # -------------------------
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    logger = get_logger()
    logger.info(f"Using device: {device}")

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
    # 4. Loss / Optimizer
    # -------------------------
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.training.ignore_index)

    metric = MeanIoU(
        num_classes=cfg.data.num_classes,
        ignore_index=cfg.training.ignore_index,
    )

    optimizer = torch.optim.AdamW(
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
        weight_decay=cfg.training.weight_decay,
    )
    # -------------------------
    # 5. Resume
    # -------------------------
    start_epoch = 0
    best_miou = 0

    if cfg.checkpoint.resume_path:
        start_epoch, best_miou, history, _ = load_checkpoint(
            cfg.checkpoint.resume_path,
            model,
            optimizer
        )

    # -------------------------
    # 6. Training Loop
    # -------------------------
    for epoch in range(start_epoch, cfg.training.epochs):
        logger.info(f"Epoch [{epoch+1}/{cfg.training.epochs}]")

        train_result = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            cfg=cfg,
            scaler=scaler
        )

        train_loss = train_result["loss"]

        logger.info(f"Train Loss: {train_loss:.4f}")
        
        log_metrics_to_wandb(
            epoch=epoch + 1,
            train_metrics=train_result,
            optimizer=optimizer,
        )

        # -------------------------
        # 7. Validation
        # -------------------------
        if (epoch + 1) % cfg.training.eval_interval == 0:
            val_result = evaluate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                metric=metric,
                device=device,
                cfg=cfg,
                desc=f"Validation Epoch {epoch + 1}",
            )

            miou = val_result["miou"]
            val_loss = val_result["loss"]

            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Val mIoU: {miou:.4f}")

            log_metrics_to_wandb(
                epoch=epoch + 1,
                train_metrics=train_result,
                val_metrics={
                    "loss": val_loss,
                    "miou": miou,
                    "best_miou": max(best_miou, miou),
                },
                optimizer=optimizer,
            )

            # best 저장
            if miou > best_miou:
                best_path = os.path.join(cfg.checkpoint.save_dir, "best.pth")
                last_path = os.path.join(cfg.checkpoint.save_dir, "last.pth")
                best_miou = miou
                save_checkpoint(
                    save_path=best_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_val_miou=best_miou,
                )
                save_checkpoint(
                    save_path=last_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_val_miou=best_miou,
                )

        # last 저장
        last_path = os.path.join(cfg.checkpoint.save_dir, "last.pth")
        save_checkpoint(
            save_path=last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_miou=best_miou,
        )

    logger.info("Training Finished")


if __name__ == "__main__":
    from src.config import load_config
    cfg = load_config()
    main(cfg)