"""
Visualization utilities for semantic segmentation.

This module provides functions for displaying images, masks, predictions,
and creating comparison panels for data augmentation visualization.
"""

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image


VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


VOC_PALETTE = np.array([
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
], dtype=np.uint8)


def denormalize(
    image: torch.Tensor,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    image = image.detach().cpu().clone()

    for c in range(3):
        image[c] = image[c] * std[c] + mean[c]

    return image.clamp(0, 1)


def tensor_to_image(image: torch.Tensor, mean=None, std=None):
    if mean is not None and std is not None:
        image = denormalize(image, mean, std)
    else:
        image = image.detach().cpu().clamp(0, 1)

    return image.permute(1, 2, 0).numpy()


def mask_to_color(mask, palette=VOC_PALETTE, ignore_index: int = 255):
    """Convert class index mask to RGB color image."""
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()

    mask = mask.astype(np.int64)
    h, w = mask.shape

    color = np.zeros((h, w, 3), dtype=np.uint8)

    valid = (mask >= 0) & (mask < len(palette))
    color[valid] = palette[mask[valid]]

    ignore = mask == ignore_index
    color[ignore] = np.array([255, 255, 255], dtype=np.uint8)

    return color


def show_image_mask(
    image,
    mask,
    title: str = "sample",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    ignore_index: int = 255,
):
    img_np = tensor_to_image(image, mean, std)
    mask_color = mask_to_color(mask, ignore_index=ignore_index)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title("image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask_color)
    plt.title("mask")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def show_augmented_samples(
    dataset,
    indices=None,
    num_samples: int = 4,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    ignore_index: int = 255,
):
    """Display multiple samples from dataset to check augmentations.

    If dataset has transforms applied, each call shows different augmentations.
    """
    if indices is None:
        indices = list(range(num_samples))

    for idx in indices[:num_samples]:
        sample = dataset[idx]

        if isinstance(sample, dict):
            image = sample["image"]
            mask = sample["mask"]
        else:
            image, mask = sample[:2]

        show_image_mask(
            image=image,
            mask=mask,
            title=f"dataset[{idx}]",
            mean=mean,
            std=std,
            ignore_index=ignore_index,
        )


@torch.no_grad()
def show_prediction(
    model,
    sample,
    device,
    cfg,
    class_names=None,
):
    """Display model prediction alongside input image and ground truth."""
    model.eval()

    if isinstance(sample, dict):
        image = sample["image"]
        mask = sample.get("mask", None)
    else:
        image = sample[0]
        mask = sample[1] if len(sample) > 1 else None

    x = image.unsqueeze(0).to(device)
    logits = model(x)

    pred = logits.argmax(dim=1)[0].cpu()

    img_np = tensor_to_image(
        image,
        mean=getattr(cfg, "mean", (0.485, 0.456, 0.406)),
        std=getattr(cfg, "std", (0.229, 0.224, 0.225)),
    )

    ncols = 3 if mask is not None else 2
    plt.figure(figsize=(5 * ncols, 4))

    plt.subplot(1, ncols, 1)
    plt.imshow(img_np)
    plt.title("image")
    plt.axis("off")

    if mask is not None:
        plt.subplot(1, ncols, 2)
        plt.imshow(mask_to_color(mask, ignore_index=getattr(cfg, "ignore_index", 255)))
        plt.title("gt")
        plt.axis("off")

        plt.subplot(1, ncols, 3)
        plt.imshow(mask_to_color(pred, ignore_index=getattr(cfg, "ignore_index", 255)))
        plt.title("pred")
        plt.axis("off")
    else:
        plt.subplot(1, ncols, 2)
        plt.imshow(mask_to_color(pred, ignore_index=getattr(cfg, "ignore_index", 255)))
        plt.title("pred")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def pil_image_to_numpy(image: Image.Image):
    return np.array(image.convert("RGB"))


def pil_mask_to_numpy(mask: Image.Image):
    return np.array(mask, dtype=np.int64)


def overlay_mask_on_image(
    image,
    mask,
    alpha: float = 0.45,
    ignore_index: int = 255,
):
    """Overlay segmentation mask on image with transparency.

    Supports PIL images, numpy arrays, and torch tensors for both image and mask.
    """
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert("RGB"), dtype=np.float32)
    else:
        image_np = np.array(image, dtype=np.float32)

    if torch.is_tensor(mask):
        mask_np = mask.detach().cpu().numpy()
    elif isinstance(mask, Image.Image):
        mask_np = np.array(mask, dtype=np.int64)
    else:
        mask_np = np.array(mask, dtype=np.int64)

    color = mask_to_color(mask_np, ignore_index=ignore_index).astype(np.float32)

    # Only overlay foreground classes (exclude background=0 and ignore=255)
    fg = (mask_np != 0) & (mask_np != ignore_index)

    out = image_np.copy()
    out[fg] = (1.0 - alpha) * image_np[fg] + alpha * color[fg]

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def make_panel(
    items,
    cell_w: int = 320,
    title_h: int = 24,
    bg=(255, 255, 255),
):
    #Create a horizontal panel of images with titles.
    from PIL import ImageDraw

    processed = []

    for title, img in items:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))
        elif torch.is_tensor(img):
            img = Image.fromarray((tensor_to_image(img) * 255).astype(np.uint8))
        else:
            img = img.convert("RGB")

        w, h = img.size
        scale = cell_w / max(w, 1)
        new_h = max(1, int(h * scale))
        img = img.resize((cell_w, new_h), Image.Resampling.BILINEAR)
        processed.append((title, img))

    cell_h = max(img.size[1] for _, img in processed) + title_h
    panel = Image.new("RGB", (cell_w * len(processed), cell_h), bg)
    draw = ImageDraw.Draw(panel)

    for i, (title, img) in enumerate(processed):
        x = i * cell_w
        panel.paste(img, (x, title_h))
        draw.text((x + 6, 4), title, fill=(0, 0, 0))

    return panel


def save_aug_comparison(
    save_path,
    original_image: Image.Image,
    original_mask: Image.Image,
    aug_image: torch.Tensor,
    aug_mask: torch.Tensor,
    title: str = "aug",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    ignore_index: int = 255,
):
    """Save comparison panel of original and augmented image/mask pairs.

    Creates a side-by-side comparison showing original image/mask and
    augmented versions after applying transforms.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    orig_mask_np = pil_mask_to_numpy(original_mask)
    aug_img_np = (tensor_to_image(aug_image, mean=mean, std=std) * 255).astype(np.uint8)
    aug_mask_np = aug_mask.detach().cpu().numpy().astype(np.int64)

    orig_overlay = overlay_mask_on_image(
        original_image,
        orig_mask_np,
        ignore_index=ignore_index,
    )

    aug_overlay = overlay_mask_on_image(
        Image.fromarray(aug_img_np),
        aug_mask_np,
        ignore_index=ignore_index,
    )

    panel = make_panel([
        ("orig image", original_image),
        ("orig mask", mask_to_color(orig_mask_np, ignore_index=ignore_index)),
        ("orig overlay", orig_overlay),
        (f"aug image: {title}", aug_img_np),
        ("aug mask", mask_to_color(aug_mask_np, ignore_index=ignore_index)),
        ("aug overlay", aug_overlay),
    ])

    panel.save(save_path)
    