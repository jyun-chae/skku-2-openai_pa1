# src/eval.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.build import build_val_dataset
from src.models.build import build_model
from src.engine.evaluator import evaluate

from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.utils.metric import MeanIoU


def load_model_checkpoint(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model


def main(cfg):
    set_seed(cfg.seed)

    logger = get_logger()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # -------------------------
    # 1. Validation Dataset
    # -------------------------
    val_dataset = build_val_dataset(cfg)

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=False,
    )

    logger.info(f"Validation dataset size: {len(val_dataset)}")

    # -------------------------
    # 2. Model
    # -------------------------
    model = build_model(cfg)
    model = model.to(device)

    # -------------------------
    # 3. Load Checkpoint
    # -------------------------
    ckpt_path = cfg.checkpoint.resume_path

    if ckpt_path is None or ckpt_path == "":
        ckpt_path = os.path.join(cfg.checkpoint.save_dir, "best.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger.info(f"Loading checkpoint: {ckpt_path}")
    model = load_model_checkpoint(model, ckpt_path, device)

    # -------------------------
    # 4. Loss / Metric
    # -------------------------
    criterion = nn.CrossEntropyLoss(
        ignore_index=cfg.training.ignore_index
    )

    metric = MeanIoU(
        num_classes=cfg.data.num_classes,
        ignore_index=cfg.training.ignore_index,
    )

    # -------------------------
    # 5. Evaluation
    # -------------------------
    result = evaluate(
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        metric=metric,
        device=device,
        cfg=cfg,
        desc="Evaluation",
    )

    logger.info(f"Validation Loss: {result['loss']:.4f}")
    logger.info(f"Validation mIoU: {result['miou']:.4f}")

    return result


if __name__ == "__main__":
    from src.config.config import load_config

    cfg = load_config("src/config/default.yaml")
    main(cfg)