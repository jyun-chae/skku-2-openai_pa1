# src/infer.py

"""
Inference script for semantic segmentation model.

This script loads a trained model and performs inference on a directory of images,
saving the predicted segmentation masks to disk.
"""

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np

from src.models.build import build_model
from src.utils.logger import get_logger
from src.utils.seed import set_seed


class SubmitImageDataset(Dataset):
    """Dataset for inference on submission images.

    Loads images from a directory, resizes them to input size, and applies normalization.
    Supports .jpg, .jpeg, and .png formats.
    """

    def __init__(self, img_dir, input_size=512, mean=None, std=None):
        self.img_dir = Path(img_dir)
        self.input_size = input_size
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

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

        image = Image.open(path).convert("RGB")
        original_size = image.size  # (W, H)

        image = TF.resize(
            image,
            [self.input_size, self.input_size],
            interpolation=TF.InterpolationMode.BILINEAR,
        )

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.mean, std=self.std)

        return image, path.name, original_size


def save_prediction(pred, save_path):
    pred = pred.astype(np.uint8)
    Image.fromarray(pred).save(save_path)


def load_model_checkpoint(model, ckpt_path, device):
    """Load model checkpoint from file.

    Supports different checkpoint formats with keys 'model', 'model_state_dict', or direct state_dict.

    Returns Model with loaded state dict
    """
    checkpoint = torch.load(ckpt_path, map_location=device)

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model


@torch.no_grad()
def inference(model, data_loader, device, cfg):
    logger = get_logger()

    model.eval()

    pred_dir = Path(cfg.submit.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, (images, image_names, original_sizes) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)

        outputs = model(images)

        for i in range(outputs.shape[0]):
            name = image_names[i]

            # original_sizes may be batched as [widths, heights] from DataLoader
            if isinstance(original_sizes, (list, tuple)) and len(original_sizes) == 2:
                orig_w = int(original_sizes[0][i])
                orig_h = int(original_sizes[1][i])
            else:
                orig_w, orig_h = original_sizes[i]

            logit = outputs[i:i+1]

            logit = F.interpolate(
                logit,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )

            pred = torch.argmax(logit, dim=1)[0]
            pred = pred.clamp(0, 20).cpu().numpy()

            base_name = os.path.splitext(name)[0]
            save_path = pred_dir / f"{base_name}.png"
            save_prediction(pred, save_path)

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
        input_size=cfg.data.input_size,
        mean=cfg.data.mean,
        std=cfg.data.std,
    )

    infer_loader = DataLoader(
        infer_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    logger.info(f"Inference dataset size: {len(infer_dataset)}")

    model = build_model(cfg)
    model = model.to(device)

    ckpt_path = cfg.checkpoint.resume_path

    if ckpt_path is None or ckpt_path == "":
        ckpt_path = os.path.join(cfg.checkpoint.save_dir, "best.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger.info(f"Loading checkpoint: {ckpt_path}")
    model = load_model_checkpoint(model, ckpt_path, device)

    inference(model, infer_loader, device, cfg)


if __name__ == "__main__":
    from src.config.config import load_config

    cfg = load_config("src/config/default.yaml")
    main(cfg)