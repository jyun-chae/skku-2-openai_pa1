import logging

def cfg_to_dict(cfg):
    if isinstance(cfg, dict):
        return {k: cfg_to_dict(v) for k, v in cfg.items()}

    if isinstance(cfg, list):
        return [cfg_to_dict(v) for v in cfg]

    if hasattr(cfg, "__dict__"):
        return {k: cfg_to_dict(v) for k, v in vars(cfg).items()}

    return cfg


def init_wandb(cfg):
    use_wandb = getattr(getattr(cfg, "wandb", None), "use", True)

    if not use_wandb:
        print("[WandB] disabled")
        return None

    try:
        import wandb

        wandb_cfg = getattr(cfg, "wandb", None)

        run = wandb.init(
            entity=getattr(wandb_cfg, "entity", None),
            project=getattr(wandb_cfg, "project", "project01"),
            name=getattr(wandb_cfg, "run_name", None),
            config=cfg_to_dict(cfg),
        )
        return run

    except Exception as e:
        print(f"[WandB] init failed: {e}")
        print("[WandB] continue without logging")
        return None


def log_metrics_to_wandb(
    epoch,
    train_metrics=None,
    val_metrics=None,
    optimizer=None,
    class_names=None,
):
    try:
        import wandb

        if wandb.run is None:
            return

        log_dict = {"epoch": epoch}

        if train_metrics is not None:
            log_dict["train/loss"] = train_metrics.get("loss")
            log_dict["train/miou"] = train_metrics.get("miou")

        if val_metrics is not None:
            log_dict["val/loss"] = val_metrics.get("loss")
            log_dict["val/miou"] = val_metrics.get("miou")

            if class_names is not None and "per_class_iou" in val_metrics:
                for i, name in enumerate(class_names):
                    log_dict[f"val_iou/{name}"] = float(
                        val_metrics["per_class_iou"][i].item()
                    )

        if optimizer is not None:
            log_dict["lr"] = optimizer.param_groups[0]["lr"]

        wandb.log(log_dict)

    except Exception as e:
        print(f"[WandB] log failed: {e}")
        
import logging


def get_logger(name: str = "project01"):
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.propagate = False

    return logger