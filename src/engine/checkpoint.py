"""
Checkpoint management utilities for saving and loading model states.

This module provides functions to save training checkpoints including model weights,
optimizer states, scheduler states, and training history. It also handles loading
checkpoints to resume training from a specific point.
"""

import os
import torch


def save_checkpoint(
    save_path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    epoch=0,
    best_val_miou=0.0,
    history=None,
    extra=None,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "best_val_miou": best_val_miou,
        "history": history if history is not None else {},
    }

    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()

    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()

    if extra is not None:
        ckpt["extra"] = extra

    torch.save(ckpt, save_path)


def load_checkpoint(
    load_path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location="cpu",
):
    """Load model checkpoint from file.
    
    Returns:
        tuple: (start_epoch, best_val_miou, history, full_checkpoint_dict)
            - start_epoch: Next epoch to start training from (current_epoch + 1)
            - best_val_miou: Best validation mIoU from checkpoint
            - history: Training history dictionary
            - full_checkpoint_dict: Complete loaded checkpoint dictionary
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    ckpt = torch.load(load_path, map_location=map_location)

    model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = ckpt.get("epoch", 0) + 1
    best_val_miou = ckpt.get("best_val_miou", 0.0)
    history = ckpt.get("history", {})

    return start_epoch, best_val_miou, history, ckpt