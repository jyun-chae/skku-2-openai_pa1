"""
Script to visualize the results of applying SegmentationTransform from src/data/transforms.py.

Important notes:
  - This file does not define new augmentations.
  - It uses the actual build_transform() / SegmentationTransform used in training.
  - Visualization uses common functions from src/utils/visualization.py.

Example usage:
  python -m src.data.visualize_aug \
    --sample-root src/data/aug_samples \
    --out-dir src/data/aug_vis_results \
    --num-repeats 5

Sample folder structure:
  src/data/aug_samples/images/0001.jpg
  src/data/aug_samples/masks/0001.png
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.transforms import build_transform
from src.utils.visualization import VOC_PALETTE, save_aug_comparison


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_image_files(image_dir: Path) -> list[Path]:
    """List all image files in the given directory."""
    return sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def find_mask(mask_dir: Path, stem: str) -> Path:
    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        path = mask_dir / f"{stem}{ext}"
        if path.exists():
            return path

    raise FileNotFoundError(f"Mask not found for stem={stem} in {mask_dir}")


def rgb_mask_to_index(mask_rgb: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """Convert VOC RGB palette mask to class-index mask.

    This is a helper function for loading input masks, not an augmentation.
    """
    h, w, _ = mask_rgb.shape
    out = np.full((h, w), ignore_index, dtype=np.uint8)

    for cls_id, color in enumerate(VOC_PALETTE):
        matched = np.all(mask_rgb == color, axis=-1)
        out[matched] = cls_id

    return out


def load_pair(
    image_path: Path,
    mask_path: Path,
    ignore_index: int = 255,
) -> tuple[Image.Image, Image.Image]:
    """Load image/mask pair as PIL images.

    Converts mask to class-index L mode.
    """
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path)

    if mask.mode in {"P", "L", "I"}:
        mask_np = np.array(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]
        mask_np = mask_np.astype(np.uint8)
    else:
        mask_np = rgb_mask_to_index(
            np.array(mask.convert("RGB")),
            ignore_index=ignore_index,
        )

    return image, Image.fromarray(mask_np, mode="L")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample-root", type=str, default="src/data/aug_samples")
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--mask-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="src/data/aug_vis_results")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-repeats", type=int, default=5)

    # Arguments matching build_transform in transforms.py
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--ignore-index", type=int, default=255)

    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--random-scale-prob", type=float, default=1.0)
    parser.add_argument("--scale-min", type=float, default=0.5)
    parser.add_argument("--scale-max", type=float, default=2.0)
    parser.add_argument("--random-crop-prob", type=float, default=1.0)

    parser.add_argument("--copy-paste-prob", type=float, default=0.15)
    parser.add_argument("--stitching-prob", type=float, default=0.05)
    parser.add_argument("--paste-scale-min", type=float, default=0.7)
    parser.add_argument("--paste-scale-max", type=float, default=1.3)

    parser.add_argument("--color-jitter-prob", type=float, default=0.4)

    return parser.parse_args()


def main() -> None:
    """Main function to run the augmentation visualization."""
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sample_root = Path(args.sample_root)
    image_dir = Path(args.image_dir) if args.image_dir else sample_root / "images"
    mask_dir = Path(args.mask_dir) if args.mask_dir else sample_root / "masks"
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"Sample folders not found.\n"
            f"images: {image_dir}\n"
            f"masks : {mask_dir}"
        )

    image_paths = list_image_files(image_dir)
    if len(image_paths) == 0:
        raise RuntimeError(f"No image files found in {image_dir}")

    pairs = []

    for image_path in image_paths:
        mask_path = find_mask(mask_dir, image_path.stem)
        image, mask = load_pair(
            image_path=image_path,
            mask_path=mask_path,
            ignore_index=args.ignore_index,
        )
        pairs.append((image_path.stem, image, mask))

    if len(pairs) == 0:
        raise RuntimeError("No valid image/mask pairs found.")

    transform = build_transform(
        input_size=args.input_size,
        is_train=True,
        hflip_prob=args.hflip_prob,
        random_scale_prob=args.random_scale_prob,
        scale_range=(args.scale_min, args.scale_max),
        random_crop_prob=args.random_crop_prob,
        copy_paste_prob=args.copy_paste_prob,
        stitching_prob=args.stitching_prob,
        paste_scale_range=(args.paste_scale_min, args.paste_scale_max),
        color_jitter_prob=args.color_jitter_prob,
        ignore_index=args.ignore_index,
    )

    def sample_getter() -> tuple[Image.Image, Image.Image]:
        """Sample getter function for Copy-Paste/Stitching augmentations.

        Returns original image/mask without any augmentation.
        """
        _, src_img, src_mask = random.choice(pairs)
        return src_img.copy(), src_mask.copy()

    print("[INFO] Visualizing actual src.data.transforms.SegmentationTransform")
    print(f"[INFO] num pairs: {len(pairs)}")
    print(f"[INFO] output dir: {out_dir.resolve()}")

    for repeat in range(args.num_repeats):
        for stem, image, mask in pairs:
            aug_image, aug_mask = transform(
                image=image.copy(),
                mask=mask.copy(),
                sample_getter=sample_getter,
            )

            save_path = out_dir / f"{stem}_r{repeat:02d}.png"

            save_aug_comparison(
                save_path=save_path,
                original_image=image,
                original_mask=mask,
                aug_image=aug_image,
                aug_mask=aug_mask,
                title="train_transform",
                ignore_index=args.ignore_index,
            )

    print(f"[OK] Saved augmentation visualization panels to: {out_dir.resolve()}")
    print("Check:")
    print("  1) Whether image and mask transform together consistently")
    print("  2) Whether mask class colors remain intact")
    print("  3) Whether copy-paste/stitching apply according to transforms.py settings")
    print("  4) Whether ignore_index=255 remains white")


if __name__ == "__main__":
    main()