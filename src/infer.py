# src/infer.py

"""
Inference script for semantic segmentation model.

Supports strong TTA:
  - multi-scale resize
  - horizontal flip
  - probability averaging
  - original-size restoration
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from src.models.build import build_model
from src.utils.logger import get_logger
from src.utils.seed import set_seed


class SubmitImageDataset(Dataset):
    """
    Dataset for inference on submission images.

    This dataset returns image paths only.
    Actual image loading / resizing / TTA is handled in inference code.
    """

    def __init__(self, img_dir):
        self.img_dir = Path(img_dir)

        self.image_paths = sorted(
            list(self.img_dir.glob("*.jpg"))
            + list(self.img_dir.glob("*.jpeg"))
            + list(self.img_dir.glob("*.png"))
        )

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {self.img_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        return str(path), path.name


def save_prediction(pred, save_path):
    pred = pred.astype(np.uint8)
    Image.fromarray(pred).save(save_path)


def load_model_checkpoint(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model


def _get_nested_attr(obj, name, default=None):
    """
    Safe cfg access.
    Example:
        _get_nested_attr(cfg, "tta", None)
    """
    return getattr(obj, name, default) if obj is not None else default


def get_tta_config(cfg):
    """
    Read TTA config from cfg.tta if available.
    If cfg.tta does not exist, use aggressive default TTA.
    """

    tta_cfg = _get_nested_attr(cfg, "tta", None)

    use_tta = bool(getattr(tta_cfg, "use", True)) if tta_cfg is not None else True

    # Strong but still reasonable multi-scale TTA.
    # Forward count = len(scales) * (1 + hflip)
    scales = getattr(
        tta_cfg,
        "scales",
        [0.50, 0.75, 1.00, 1.25, 1.50, 1.75],
    ) if tta_cfg is not None else [0.50, 0.75, 1.00, 1.25, 1.50, 1.75]

    hflip = bool(getattr(tta_cfg, "hflip", True)) if tta_cfg is not None else True

    # Vertical flip / rotation can hurt natural image segmentation,
    # so keep them disabled by default even for strong TTA.
    vflip = bool(getattr(tta_cfg, "vflip", False)) if tta_cfg is not None else False

    # probability averaging is usually safer than averaging argmax masks.
    average_mode = getattr(tta_cfg, "average_mode", "prob") if tta_cfg is not None else "prob"

    return {
        "use": use_tta,
        "scales": [float(s) for s in scales],
        "hflip": hflip,
        "vflip": vflip,
        "average_mode": average_mode,
    }


def pil_to_tensor(image, size, mean, std):
    """
    Resize PIL image to square size and normalize.
    """
    image = TF.resize(
        image,
        [size, size],
        interpolation=TF.InterpolationMode.BILINEAR,
    )
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, mean=mean, std=std)
    return tensor.unsqueeze(0)


def _invert_flip(logits, hflip=False, vflip=False):
    """
    Restore flipped logits back to original orientation.
    logits: [1, C, H, W]
    """
    if hflip:
        logits = torch.flip(logits, dims=[-1])
    if vflip:
        logits = torch.flip(logits, dims=[-2])
    return logits


@torch.no_grad()
def predict_single_with_tta(
    model,
    image_path,
    device,
    cfg,
    tta_cfg,
):
    """
    Run strong TTA for one image and return final class-index prediction.

    Flow:
      PIL image
      -> multiple scale / flip views
      -> model forward
      -> invert flip
      -> resize logits/probabilities to original image size
      -> average
      -> argmax
    """
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    mean = list(getattr(cfg.data, "mean", [0.485, 0.456, 0.406]))
    std = list(getattr(cfg.data, "std", [0.229, 0.224, 0.225]))
    base_size = int(cfg.data.input_size)

    num_classes = int(cfg.model.num_classes)
    use_amp = bool(getattr(cfg.training, "amp", True)) and device.type == "cuda"

    if not tta_cfg["use"]:
        x = pil_to_tensor(image, base_size, mean, std).to(device)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(x)

        logits = F.interpolate(
            logits,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )

        pred = logits.argmax(dim=1)[0]
        return pred.clamp(0, num_classes - 1).cpu().numpy()

    accumulator = None
    num_views = 0

    flip_settings = [(False, False)]

    if tta_cfg["hflip"]:
        flip_settings.append((True, False))

    if tta_cfg["vflip"]:
        flip_settings.append((False, True))
        flip_settings.append((True, True))

    for scale in tta_cfg["scales"]:
        view_size = max(32, int(round(base_size * scale)))

        # EfficientNet/FPN is usually safer with size divisible by 32.
        view_size = int(round(view_size / 32) * 32)
        view_size = max(32, view_size)

        for hflip, vflip in flip_settings:
            view = image

            if hflip:
                view = TF.hflip(view)
            if vflip:
                view = TF.vflip(view)

            x = pil_to_tensor(view, view_size, mean, std).to(device)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(x)

            logits = _invert_flip(logits, hflip=hflip, vflip=vflip)

            logits = F.interpolate(
                logits,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )

            if tta_cfg["average_mode"] == "prob":
                score = torch.softmax(logits, dim=1)
            else:
                score = logits

            if accumulator is None:
                accumulator = torch.zeros_like(score)

            accumulator += score
            num_views += 1

    accumulator = accumulator / max(num_views, 1)

    pred = accumulator.argmax(dim=1)[0]
    pred = pred.clamp(0, num_classes - 1).cpu().numpy()

    return pred


@torch.no_grad()
def inference(model, data_loader, device, cfg):
    logger = get_logger()

    model.eval()

    pred_dir = Path(cfg.submit.pred_dir)
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

    for batch_idx, (image_paths, image_names) in enumerate(data_loader):
        for image_path, name in zip(image_paths, image_names):
            pred = predict_single_with_tta(
                model=model,
                image_path=image_path,
                device=device,
                cfg=cfg,
                tta_cfg=tta_cfg,
            )

            base_name = os.path.splitext(name)[0]
            save_path = pred_dir / f"{base_name}.png"
            save_prediction(pred, save_path)

            logger.info(f"[{batch_idx}] saved: {save_path}")

    logger.info(f"Predictions saved to: {pred_dir}")


def main(cfg):
    set_seed(cfg.runtime.seed)

    logger = get_logger()

    device = torch.device(cfg.runtime.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    img_dir = cfg.submit.img_dir
    pred_dir = cfg.submit.pred_dir

    os.makedirs(pred_dir, exist_ok=True)

    infer_dataset = SubmitImageDataset(
        img_dir=img_dir,
    )

    # TTA는 image별로 여러 번 forward하므로 batch_size=1이 가장 안전함.
    infer_loader = DataLoader(
        infer_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=False,
    )

    logger.info(f"Inference dataset size: {len(infer_dataset)}")

    model = build_model(cfg)
    model = model.to(device)

    ckpt_path = cfg.checkpoint.resume_path

    if ckpt_path is None or ckpt_path == "":
        model_path = os.path.join(cfg.checkpoint.save_dir, "model.pth")
        best_path = os.path.join(cfg.checkpoint.save_dir, "best.pth")
        last_path = os.path.join(cfg.checkpoint.save_dir, "last.pth")

        if os.path.exists(model_path):
            ckpt_path = model_path
        elif os.path.exists(best_path):
            ckpt_path = best_path
        else:
            ckpt_path = last_path

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger.info(f"Loading checkpoint: {ckpt_path}")
    model = load_model_checkpoint(model, ckpt_path, device)

    inference(model, infer_loader, device, cfg)


if __name__ == "__main__":
    from src.config.config import load_config

    cfg = load_config("src/config/default.yaml")
    main(cfg)