# src/infer.py

"""
Semantic segmentation inference script.

This version is fully driven by cfg loaded from default.yaml.
It expects config.py to convert YAML dictionaries into objects that support
attribute access, for example cfg.data.input_size and cfg.submit.img_dir.

Main cfg sections used:
  - cfg.model.num_classes
  - cfg.data.input_size / mean / std / num_workers / pin_memory
  - cfg.training.amp
  - cfg.checkpoint.resume_path / save_dir
  - cfg.tta.use / hflip / vflip / scales / average_mode
  - cfg.runtime.device / seed
  - cfg.submit.img_dir / pred_dir
"""

import argparse
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.models.build import build_model
from src.utils.logger import get_logger
from src.utils.seed import set_seed


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


# -----------------------------------------------------------------------------
# cfg helpers
# -----------------------------------------------------------------------------

def cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    """Safely read nested values from SimpleNamespace or dict configs.

    This helper allows config objects to be accessed with dot-separated paths.
    It returns the default if any intermediate key is missing or if the config
    node becomes None.

    Examples:
        cfg_get(cfg, "data.input_size", 512)
        cfg_get(cfg, "tta.scales", [1.0])
    """
    cur = cfg
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
    return cur


def cfg_has(cfg: Any, path: str) -> bool:
    marker = object()
    return cfg_get(cfg, path, marker) is not marker


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def as_float_list(values: Optional[Iterable[Any]], default: Sequence[float]) -> List[float]:
    if values is None:
        values = default
    return [float(v) for v in values]


# -----------------------------------------------------------------------------
# dataset / save / checkpoint
# -----------------------------------------------------------------------------

class SubmitImageDataset(Dataset):
    """Dataset for submission images.

    It only returns image path and file name. Actual resize, normalization,
    TTA, and original-size restoration are handled in predict_single().
    """

    def __init__(self, img_dir: str | os.PathLike[str]):
        self.img_dir = Path(img_dir)

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Submit image directory not found: {self.img_dir}")

        image_paths: List[Path] = []
        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(self.img_dir.glob(ext))

        self.image_paths = sorted(set(image_paths))

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {self.img_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        path = self.image_paths[idx]
        return str(path), path.name


def save_prediction(pred: np.ndarray, save_path: str | os.PathLike[str]) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred.astype(np.uint8)).save(save_path)


def load_model_checkpoint(model: torch.nn.Module, ckpt_path: str | os.PathLike[str], device: torch.device) -> torch.nn.Module:
    """Load checkpoint saved in one of the common formats.

    Supported keys:
      - checkpoint["model"]
      - checkpoint["model_state_dict"]
      - raw state_dict
    """
    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return model


def resolve_checkpoint_path(cfg: Any) -> Path:
    """Only works if /content/project01/checkpoints/model.pth exists, otherwise raises error.
    """
    ckpt_path = Path("/content/project01/checkpoints/model.pth")
    if ckpt_path.exists():
        return ckpt_path
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")


# -----------------------------------------------------------------------------
# TTA / preprocessing
# -----------------------------------------------------------------------------

def get_tta_config(cfg: Any) -> Dict[str, Any]:
    """Read TTA options from cfg.tta with safe defaults."""
    return {
        "use": as_bool(cfg_get(cfg, "tta.use", False), default=False),
        "hflip": as_bool(cfg_get(cfg, "tta.hflip", False), default=False),
        "vflip": as_bool(cfg_get(cfg, "tta.vflip", False), default=False),
        "scales": as_float_list(cfg_get(cfg, "tta.scales", [1.0]), default=[1.0]),
        "average_mode": str(cfg_get(cfg, "tta.average_mode", "prob") or "prob").lower(),
    }


def make_size_divisible(size: int, divisor: int = 32, min_size: int = 32) -> int:
    size = int(round(size / divisor) * divisor)
    return max(min_size, size)


def pil_to_tensor(image: Image.Image, size: int, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    image = TF.resize(image, [size, size], interpolation=TF.InterpolationMode.BILINEAR)
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, mean=list(mean), std=list(std))
    return tensor.unsqueeze(0)


def invert_flip(logits: torch.Tensor, hflip: bool = False, vflip: bool = False) -> torch.Tensor:
    if hflip:
        logits = torch.flip(logits, dims=[-1])
    if vflip:
        logits = torch.flip(logits, dims=[-2])
    return logits


def autocast_context(device: torch.device, use_amp: bool):
    if device.type == "cuda" and use_amp:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return nullcontext()


@torch.no_grad()
def predict_single(
    model: torch.nn.Module,
    image_path: str | os.PathLike[str],
    device: torch.device,
    cfg: Any,
    tta_cfg: Dict[str, Any],
) -> np.ndarray:
    """Predict one image and return uint8 class-index mask in original size."""
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    # Read image normalization and size config values.
    mean = cfg_get(cfg, "data.mean", [0.485, 0.456, 0.406])
    std = cfg_get(cfg, "data.std", [0.229, 0.224, 0.225])
    base_size = int(cfg_get(cfg, "data.input_size", 512))
    num_classes = int(cfg_get(cfg, "model.num_classes", 21))
    use_amp = as_bool(cfg_get(cfg, "training.amp", True), default=True)

    scales = tta_cfg["scales"] if tta_cfg["use"] else [1.0]

    # Build the list of TTA views to evaluate: original, horizontal flip,
    # vertical flip, and both flips if configured.
    flip_settings: List[Tuple[bool, bool]] = [(False, False)]
    if tta_cfg["use"] and tta_cfg["hflip"]:
        flip_settings.append((True, False))
    if tta_cfg["use"] and tta_cfg["vflip"]:
        flip_settings.append((False, True))
        if tta_cfg["hflip"]:
            flip_settings.append((True, True))

    accumulator: Optional[torch.Tensor] = None
    num_views = 0

    for scale in scales:
        view_size = make_size_divisible(int(round(base_size * float(scale))), divisor=32)

        for hflip, vflip in flip_settings:
            view = image
            if hflip:
                view = TF.hflip(view)
            if vflip:
                view = TF.vflip(view)

            x = pil_to_tensor(view, view_size, mean, std).to(device, non_blocking=True)

            # Run the model inside AMP autocast if enabled and supported.
            with autocast_context(device, use_amp):
                logits = model(x)

            logits = invert_flip(logits, hflip=hflip, vflip=vflip)
            logits = F.interpolate(
                logits,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )

            if tta_cfg["average_mode"] == "prob":
                score = torch.softmax(logits, dim=1)
            elif tta_cfg["average_mode"] == "logit":
                score = logits
            else:
                raise ValueError(
                    f"Invalid cfg.tta.average_mode={tta_cfg['average_mode']!r}. "
                    "Use 'prob' or 'logit'."
                )

            if accumulator is None:
                accumulator = torch.zeros_like(score)

            accumulator += score
            num_views += 1

    if accumulator is None or num_views == 0:
        raise RuntimeError("No inference views were generated. Check cfg.tta.scales.")

    # Average the collected scores across all TTA views and take the argmax.
    pred = (accumulator / num_views).argmax(dim=1)[0]
    pred = pred.clamp(0, num_classes - 1).cpu().numpy().astype(np.uint8)
    return pred


# -----------------------------------------------------------------------------
# main inference flow
# -----------------------------------------------------------------------------

@torch.no_grad()
def inference(model: torch.nn.Module, data_loader: DataLoader, device: torch.device, cfg: Any) -> None:
    logger = get_logger()
    model.eval()

    # Create the prediction output directory if necessary.
    pred_dir = Path(cfg_get(cfg, "submit.pred_dir", "/content/project01/submit/pred"))
    pred_dir.mkdir(parents=True, exist_ok=True)

    tta_cfg = get_tta_config(cfg)
    logger.info(
        "TTA config: "
        f"use={tta_cfg['use']}, "
        f"scales={tta_cfg['scales']}, "
        f"hflip={tta_cfg['hflip']}, "
        f"vflip={tta_cfg['vflip']}, "
        f"average_mode={tta_cfg['average_mode']}"
    )

    total_saved = 0
    for batch_idx, (image_paths, image_names) in enumerate(data_loader):
        for image_path, image_name in zip(image_paths, image_names):
            pred = predict_single(
                model=model,
                image_path=image_path,
                device=device,
                cfg=cfg,
                tta_cfg=tta_cfg,
            )

            # Save each prediction with the same root filename but PNG extension.
            base_name = Path(image_name).stem
            save_path = pred_dir / f"{base_name}.png"
            save_prediction(pred, save_path)
            total_saved += 1

            if total_saved == 1 or total_saved % 20 == 0:
                logger.info(f"[{total_saved}] saved: {save_path}")

    logger.info(f"Done. Saved {total_saved} predictions to: {pred_dir}")


def main(cfg: Any) -> None:
    # Set random seeds for reproducible inference behavior.
    seed = int(cfg_get(cfg, "runtime.seed", 42))
    set_seed(seed)

    logger = get_logger()

    requested_device = str(cfg_get(cfg, "runtime.device", "cuda"))
    if requested_device == "cuda" and not torch.cuda.is_available():
        logger.warning("cfg.runtime.device is 'cuda', but CUDA is unavailable. Falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)
    logger.info(f"Using device: {device}")

    img_dir = Path(cfg_get(cfg, "submit.img_dir", "/content/project01/submit/img"))
    pred_dir = Path(cfg_get(cfg, "submit.pred_dir", "/content/project01/submit/pred"))
    pred_dir.mkdir(parents=True, exist_ok=True)

    infer_dataset = SubmitImageDataset(img_dir=img_dir)

    # TTA performs multiple forwards per image, so batch_size=1 is the safest default.
    batch_size = int(cfg_get(cfg, "inference.batch_size", 1))
    num_workers = int(cfg_get(cfg, "data.num_workers", 2))
    pin_memory = as_bool(cfg_get(cfg, "data.pin_memory", True), default=True) and device.type == "cuda"

    infer_loader = DataLoader(
        infer_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    logger.info(f"Submit image dir: {img_dir}")
    logger.info(f"Prediction dir: {pred_dir}")
    logger.info(f"Inference dataset size: {len(infer_dataset)}")

    model = build_model(cfg).to(device)

    # Load weights from the selected checkpoint and restore the model state.
    ckpt_path = resolve_checkpoint_path(cfg)
    logger.info(f"Loading checkpoint: {ckpt_path}")
    model = load_model_checkpoint(model, ckpt_path, device)

    inference(model, infer_loader, device, cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run semantic segmentation inference.")
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/default.yaml",
        help="Path to yaml config file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    from src.config.config import load_config

    args = parse_args()
    cfg = load_config(args.config)
    main(cfg)
