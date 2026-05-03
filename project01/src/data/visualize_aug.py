"""
src/data/visualize_aug.py

Sample image/mask 몇 개를 넣어두고 segmentation augmentation이 제대로 적용되는지
직접 눈으로 확인하기 위한 standalone script.

권장 위치:
  project01/src/data/visualize_aug.py

권장 샘플 폴더:
  project01/src/data/aug_samples/images/0001.jpg
  project01/src/data/aug_samples/masks/0001.png

실행 예시:
  # project01 폴더에서 실행
  python -m src.data.visualize_aug

  # 또는 파일 직접 실행
  python src/data/visualize_aug.py --sample-root src/data/aug_samples --out-dir src/data/aug_vis_results

출력:
  src/data/aug_vis_results/*.png

주의:
  - image와 mask는 파일 stem이 같아야 함. 예: images/cat.jpg, masks/cat.png
  - mask는 class-index PNG가 가장 안전함. VOC palette PNG도 지원.
  - image에는 bilinear, mask에는 nearest interpolation을 사용함.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance


# Pascal VOC 21 classes palette. ignore_index=255는 흰색으로 표시.
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

VOC_PALETTE = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
    [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128],
], dtype=np.uint8)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BILINEAR = Image.Resampling.BILINEAR
NEAREST = Image.Resampling.NEAREST


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_image_files(image_dir: Path) -> list[Path]:
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS])


def find_mask(mask_dir: Path, stem: str) -> Path:
    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        p = mask_dir / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"mask not found for stem={stem} in {mask_dir}")


def rgb_mask_to_index(mask_rgb: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """VOC RGB palette mask를 class-index mask로 변환."""
    h, w, _ = mask_rgb.shape
    out = np.full((h, w), ignore_index, dtype=np.uint8)
    for cls_id, color in enumerate(VOC_PALETTE):
        matched = np.all(mask_rgb == color, axis=-1)
        out[matched] = cls_id
    return out


def load_pair(image_path: Path, mask_path: Path, ignore_index: int = 255) -> tuple[Image.Image, Image.Image]:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path)

    # mode P/L인 경우 np.array(mask)는 class index로 읽힘.
    if mask.mode in {"P", "L", "I"}:
        mask_np = np.array(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]
        mask_np = mask_np.astype(np.uint8)
    else:
        # RGB mask라면 VOC palette 기준 class index로 변환 시도.
        mask_np = rgb_mask_to_index(np.array(mask.convert("RGB")), ignore_index=ignore_index)

    return image, Image.fromarray(mask_np, mode="L")


def mask_to_color(mask: Image.Image | np.ndarray, ignore_index: int = 255) -> np.ndarray:
    mask_np = np.array(mask, dtype=np.int64) if isinstance(mask, Image.Image) else mask.astype(np.int64)
    h, w = mask_np.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    valid = (0 <= mask_np) & (mask_np < len(VOC_PALETTE))
    color[valid] = VOC_PALETTE[mask_np[valid]]
    color[mask_np == ignore_index] = np.array([255, 255, 255], dtype=np.uint8)
    return color


def overlay_mask(image: Image.Image, mask: Image.Image, alpha: float = 0.45) -> Image.Image:
    image_np = np.array(image.convert("RGB"), dtype=np.float32)
    color = mask_to_color(mask).astype(np.float32)
    mask_np = np.array(mask)
    fg = (mask_np != 0) & (mask_np != 255)

    out = image_np.copy()
    out[fg] = (1.0 - alpha) * image_np[fg] + alpha * color[fg]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def make_panel(
    items: Sequence[tuple[str, Image.Image]],
    cell_w: int = 320,
    font_h: int = 24,
    bg=(255, 255, 255),
) -> Image.Image:
    """[(title, PIL image), ...]를 가로 패널 하나로 저장."""
    resized: list[tuple[str, Image.Image]] = []
    for title, img in items:
        img = img.convert("RGB")
        w, h = img.size
        scale = cell_w / max(w, 1)
        new_h = max(1, int(h * scale))
        resized.append((title, img.resize((cell_w, new_h), BILINEAR)))

    cell_h = max(img.size[1] for _, img in resized) + font_h
    panel = Image.new("RGB", (cell_w * len(resized), cell_h), bg)
    draw = ImageDraw.Draw(panel)

    for i, (title, img) in enumerate(resized):
        x = i * cell_w
        panel.paste(img, (x, font_h))
        draw.text((x + 6, 4), title, fill=(0, 0, 0))

    return panel


class SegAugVisualizer:
    """시각화용 augmentation 구현체. 학습 transform에 옮기기 전에 여기서 먼저 확인."""

    def __init__(self, seed: int = 42, ignore_index: int = 255):
        self.rng = random.Random(seed)
        self.ignore_index = ignore_index

    # ---------- helpers ----------
    def _rand_uniform(self, a: float, b: float) -> float:
        return self.rng.uniform(a, b)

    def _rand_int(self, a: int, b: int) -> int:
        return self.rng.randint(a, b)

    def resize_pair(self, img: Image.Image, mask: Image.Image, size: tuple[int, int]):
        return img.resize(size, BILINEAR), mask.resize(size, NEAREST)

    def random_scale(self, img: Image.Image, mask: Image.Image, scale_range=(0.5, 2.0)):
        scale = self._rand_uniform(*scale_range)
        w, h = img.size
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        return self.resize_pair(img, mask, (nw, nh))

    def random_crop_or_pad(self, img: Image.Image, mask: Image.Image, crop_size: int = 512):
        """작으면 padding, 크면 random crop. image/mask에 같은 좌표 적용."""
        w, h = img.size
        pad_w = max(0, crop_size - w)
        pad_h = max(0, crop_size - h)

        if pad_w > 0 or pad_h > 0:
            new_w, new_h = w + pad_w, h + pad_h
            padded_img = Image.new("RGB", (new_w, new_h), (0, 0, 0))
            padded_mask = Image.new("L", (new_w, new_h), 0)
            padded_img.paste(img, (0, 0))
            padded_mask.paste(mask, (0, 0))
            img, mask = padded_img, padded_mask
            w, h = img.size

        left = self._rand_int(0, w - crop_size) if w > crop_size else 0
        top = self._rand_int(0, h - crop_size) if h > crop_size else 0
        box = (left, top, left + crop_size, top + crop_size)
        return img.crop(box), mask.crop(box)

    def hflip(self, img: Image.Image, mask: Image.Image):
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT), mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    def rotate(self, img: Image.Image, mask: Image.Image, max_deg: float = 10.0):
        angle = self._rand_uniform(-max_deg, max_deg)
        return (
            img.rotate(angle, resample=BILINEAR, fillcolor=(0, 0, 0)),
            mask.rotate(angle, resample=NEAREST, fillcolor=0),
        )

    def color_jitter(self, img: Image.Image, mask: Image.Image):
        # image only augmentation. mask는 변경하지 않음.
        brightness = self._rand_uniform(0.7, 1.3)
        contrast = self._rand_uniform(0.7, 1.3)
        saturation = self._rand_uniform(0.7, 1.3)
        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Color(img).enhance(saturation)
        return img, mask

    def gaussian_blur(self, img: Image.Image, mask: Image.Image, radius_range=(0.5, 2.0)):
        radius = self._rand_uniform(*radius_range)
        return img.filter(ImageFilter.GaussianBlur(radius=radius)), mask

    def cutmix_rect(self, img_a: Image.Image, mask_a: Image.Image, img_b: Image.Image, mask_b: Image.Image):
        """사각형 영역을 B에서 A로 가져옴. mask도 같은 영역 교체."""
        img_b, mask_b = self.resize_pair(img_b, mask_b, img_a.size)
        w, h = img_a.size
        rw = self._rand_int(max(1, w // 5), max(1, w // 2))
        rh = self._rand_int(max(1, h // 5), max(1, h // 2))
        x1 = self._rand_int(0, max(0, w - rw))
        y1 = self._rand_int(0, max(0, h - rh))

        out_img = img_a.copy()
        out_mask = mask_a.copy()
        box = (x1, y1, x1 + rw, y1 + rh)
        out_img.paste(img_b.crop(box), (x1, y1))
        out_mask.paste(mask_b.crop(box), (x1, y1))
        return out_img, out_mask

    def classmix(self, img_a: Image.Image, mask_a: Image.Image, img_b: Image.Image, mask_b: Image.Image):
        """A mask의 일부 class 영역은 A, 나머지는 B로 구성."""
        img_b, mask_b = self.resize_pair(img_b, mask_b, img_a.size)

        a_img = np.array(img_a)
        a_mask = np.array(mask_a)
        b_img = np.array(img_b)
        b_mask = np.array(mask_b)

        classes = np.unique(a_mask)
        classes = [int(c) for c in classes if c not in (0, self.ignore_index)]
        if len(classes) == 0:
            return img_a, mask_a

        self.rng.shuffle(classes)
        selected = set(classes[: max(1, len(classes) // 2)])
        keep_a = np.isin(a_mask, list(selected))

        out_img = b_img.copy()
        out_mask = b_mask.copy()
        out_img[keep_a] = a_img[keep_a]
        out_mask[keep_a] = a_mask[keep_a]
        return Image.fromarray(out_img), Image.fromarray(out_mask.astype(np.uint8), mode="L")

    def copy_paste(self, target_img: Image.Image, target_mask: Image.Image, source_img: Image.Image, source_mask: Image.Image):
        """source의 non-background class 하나를 target 위에 붙임."""
        t_img = np.array(target_img.convert("RGB")).copy()
        t_mask = np.array(target_mask).copy()
        s_img = np.array(source_img.convert("RGB"))
        s_mask = np.array(source_mask)

        classes = np.unique(s_mask)
        classes = [int(c) for c in classes if c not in (0, self.ignore_index)]
        if len(classes) == 0:
            return target_img, target_mask

        cls = self.rng.choice(classes)
        ys, xs = np.where(s_mask == cls)
        if len(xs) == 0:
            return target_img, target_mask

        x1, x2 = xs.min(), xs.max() + 1
        y1, y2 = ys.min(), ys.max() + 1
        crop_img = s_img[y1:y2, x1:x2]
        crop_mask = s_mask[y1:y2, x1:x2]
        obj = crop_mask == cls

        th, tw = t_mask.shape
        ch, cw = crop_mask.shape

        # object crop이 너무 크면 축소.
        max_ratio = 0.65
        if ch > th * max_ratio or cw > tw * max_ratio:
            scale = min((th * max_ratio) / max(ch, 1), (tw * max_ratio) / max(cw, 1))
            nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
            crop_img = np.array(Image.fromarray(crop_img).resize((nw, nh), BILINEAR))
            crop_mask = np.array(Image.fromarray(crop_mask.astype(np.uint8), mode="L").resize((nw, nh), NEAREST))
            obj = crop_mask == cls
            ch, cw = crop_mask.shape

        if ch > th or cw > tw:
            return target_img, target_mask

        px = self._rand_int(0, tw - cw)
        py = self._rand_int(0, th - ch)

        patch_img = t_img[py:py + ch, px:px + cw]
        patch_mask = t_mask[py:py + ch, px:px + cw]
        patch_img[obj] = crop_img[obj]
        patch_mask[obj] = crop_mask[obj]
        t_img[py:py + ch, px:px + cw] = patch_img
        t_mask[py:py + ch, px:px + cw] = patch_mask

        return Image.fromarray(t_img), Image.fromarray(t_mask.astype(np.uint8), mode="L")

    def stitch_horizontal(self, img_a: Image.Image, mask_a: Image.Image, img_b: Image.Image, mask_b: Image.Image):
        """Image stitching: A | B 형태로 붙인 뒤 높이를 맞춤."""
        aw, ah = img_a.size
        bw, bh = img_b.size
        scale = ah / max(bh, 1)
        new_bw = max(1, int(bw * scale))
        img_b, mask_b = self.resize_pair(img_b, mask_b, (new_bw, ah))

        out_img = Image.new("RGB", (aw + new_bw, ah), (0, 0, 0))
        out_mask = Image.new("L", (aw + new_bw, ah), 0)
        out_img.paste(img_a, (0, 0))
        out_img.paste(img_b, (aw, 0))
        out_mask.paste(mask_a, (0, 0))
        out_mask.paste(mask_b, (aw, 0))
        return out_img, out_mask

    def mosaic4(self, pairs: Sequence[tuple[Image.Image, Image.Image]], out_size: int = 512):
        """4-image stitching. 2x2 mosaic."""
        assert len(pairs) == 4
        half = out_size // 2
        out_img = Image.new("RGB", (out_size, out_size), (0, 0, 0))
        out_mask = Image.new("L", (out_size, out_size), 0)
        positions = [(0, 0), (half, 0), (0, half), (half, half)]
        for (img, mask), pos in zip(pairs, positions):
            img, mask = self.resize_pair(img, mask, (half, half))
            out_img.paste(img, pos)
            out_mask.paste(mask, pos)
        return out_img, out_mask

    def basic_train_aug(self, img: Image.Image, mask: Image.Image, crop_size: int = 512):
        """추천 기본 학습 aug: scale -> crop/pad -> hflip -> color jitter/blur."""
        names = []
        img, mask = self.random_scale(img, mask, (0.5, 2.0)); names.append("scale")
        img, mask = self.random_crop_or_pad(img, mask, crop_size); names.append("crop")
        if self.rng.random() < 0.5:
            img, mask = self.hflip(img, mask); names.append("hflip")
        if self.rng.random() < 0.5:
            img, mask = self.color_jitter(img, mask); names.append("jitter")
        if self.rng.random() < 0.15:
            img, mask = self.gaussian_blur(img, mask); names.append("blur")
        return img, mask, "+".join(names)


def save_result(
    out_dir: Path,
    stem: str,
    op_name: str,
    original_img: Image.Image,
    original_mask: Image.Image,
    aug_img: Image.Image,
    aug_mask: Image.Image,
) -> None:
    panel = make_panel([
        ("orig image", original_img),
        ("orig mask", Image.fromarray(mask_to_color(original_mask))),
        ("orig overlay", overlay_mask(original_img, original_mask)),
        (f"aug image: {op_name}", aug_img),
        ("aug mask", Image.fromarray(mask_to_color(aug_mask))),
        ("aug overlay", overlay_mask(aug_img, aug_mask)),
    ])
    panel.save(out_dir / f"{stem}__{op_name}.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-root", type=str, default="src/data/aug_samples")
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--mask-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="src/data/aug_vis_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--num-repeats", type=int, default=3)
    parser.add_argument(
        "--ops",
        type=str,
        default="basic,classmix,copy_paste,stitch,mosaic,cutmix",
        help="comma separated: basic,classmix,copy_paste,stitch,mosaic,cutmix,rotate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_root = Path(args.sample_root)
    image_dir = Path(args.image_dir) if args.image_dir else sample_root / "images"
    mask_dir = Path(args.mask_dir) if args.mask_dir else sample_root / "masks"
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"sample folders not found. Expected:\n"
            f"  images: {image_dir}\n"
            f"  masks : {mask_dir}\n"
            f"Create them and put same-stem image/mask pairs."
        )

    image_paths = list_image_files(image_dir)
    if len(image_paths) == 0:
        raise RuntimeError(f"No image files found in {image_dir}")

    pairs = []
    for img_path in image_paths:
        mask_path = find_mask(mask_dir, img_path.stem)
        img, mask = load_pair(img_path, mask_path)
        pairs.append((img_path.stem, img, mask))

    ops = {x.strip() for x in args.ops.split(",") if x.strip()}
    aug = SegAugVisualizer(seed=args.seed)
    rng = random.Random(args.seed)

    for repeat in range(args.num_repeats):
        for i, (stem, img, mask) in enumerate(pairs):
            suffix = f"{stem}_r{repeat:02d}"

            if "basic" in ops:
                a_img, a_mask, name = aug.basic_train_aug(img, mask, crop_size=args.crop_size)
                save_result(out_dir, suffix, name, img, mask, a_img, a_mask)

            if "rotate" in ops:
                a_img, a_mask = aug.rotate(img, mask)
                save_result(out_dir, suffix, "rotate", img, mask, a_img, a_mask)

            # two-image ops: partner 필요
            if len(pairs) >= 2:
                j = i
                while j == i:
                    j = rng.randrange(len(pairs))
                _, img_b, mask_b = pairs[j]

                if "classmix" in ops:
                    a_img, a_mask = aug.classmix(img, mask, img_b, mask_b)
                    save_result(out_dir, suffix, "classmix", img, mask, a_img, a_mask)

                if "copy_paste" in ops:
                    a_img, a_mask = aug.copy_paste(img, mask, img_b, mask_b)
                    save_result(out_dir, suffix, "copy_paste", img, mask, a_img, a_mask)

                if "stitch" in ops:
                    a_img, a_mask = aug.stitch_horizontal(img, mask, img_b, mask_b)
                    # stitch는 크기가 길어질 수 있으니 최종 crop/pad도 같이 확인
                    a_img, a_mask = aug.random_crop_or_pad(a_img, a_mask, args.crop_size)
                    save_result(out_dir, suffix, "stitch_crop", img, mask, a_img, a_mask)

                if "cutmix" in ops:
                    a_img, a_mask = aug.cutmix_rect(img, mask, img_b, mask_b)
                    save_result(out_dir, suffix, "cutmix_rect", img, mask, a_img, a_mask)

            if "mosaic" in ops and len(pairs) >= 4:
                idxs = [i] + rng.sample([k for k in range(len(pairs)) if k != i], 3)
                mosaic_pairs = [(pairs[k][1], pairs[k][2]) for k in idxs]
                a_img, a_mask = aug.mosaic4(mosaic_pairs, out_size=args.crop_size)
                save_result(out_dir, suffix, "mosaic4", img, mask, a_img, a_mask)

    print(f"[OK] saved visualization panels to: {out_dir.resolve()}")
    print("Check these points:")
    print("  1) image와 mask가 같은 위치/방향으로 변했는지")
    print("  2) mask class 색이 이상하게 섞이지 않았는지")
    print("  3) copy_paste/classmix 후 object 영역 label이 정확히 붙었는지")
    print("  4) ignore_index=255가 흰색으로 유지되는지")


if __name__ == "__main__":
    main()
