from __future__ import annotations

import random
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode

ImageMask = Tuple[Image.Image, Image.Image]
SampleGetter = Callable[[], ImageMask]


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------

def _to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def _mask_np(mask: Image.Image) -> np.ndarray:
    return np.array(mask, dtype=np.uint8)


def _np_mask(mask_np: np.ndarray) -> Image.Image:
    return Image.fromarray(mask_np.astype(np.uint8), mode="L")


def _safe_get_extra_samples(
    extra_samples: Optional[Sequence[ImageMask]],
    sample_getter: Optional[SampleGetter],
    num_samples: int,
    max_trials: int = 20,
) -> List[ImageMask]:
    """
    extra_samples가 직접 들어오면 그것을 우선 사용하고,
    없으면 sample_getter를 통해 transform 내부에서 random sample을 뽑는다.
    """
    samples: List[ImageMask] = []

    if extra_samples is not None:
        samples.extend(list(extra_samples))

    if len(samples) >= num_samples:
        random.shuffle(samples)
        return samples[:num_samples]

    if sample_getter is None:
        random.shuffle(samples)
        return samples[:num_samples]

    trials = 0
    while len(samples) < num_samples and trials < max_trials:
        trials += 1
        try:
            sample = sample_getter()
        except Exception:
            continue

        if sample is None:
            continue
        src_img, src_mask = sample
        if src_img is None or src_mask is None:
            continue
        samples.append((_to_rgb(src_img), src_mask))

    random.shuffle(samples)
    return samples[:num_samples]


# ---------------------------------------------------------------------
# Pair transforms
# ---------------------------------------------------------------------

def resize_pair(image: Image.Image, mask: Image.Image, size: int | Tuple[int, int]) -> ImageMask:
    """image/mask resize. mask는 class index 보존을 위해 반드시 NEAREST."""
    if isinstance(size, int):
        out_size = [size, size]
    else:
        # torchvision resize는 [height, width]
        out_size = [size[0], size[1]]

    image = F.resize(image, out_size, interpolation=InterpolationMode.BILINEAR)
    mask = F.resize(mask, out_size, interpolation=InterpolationMode.NEAREST)
    return image, mask


def random_horizontal_flip(image: Image.Image, mask: Image.Image, prob: float = 0.5) -> ImageMask:
    """image/mask를 같은 확률로 horizontal flip."""
    if random.random() < prob:
        image = F.hflip(image)
        mask = F.hflip(mask)
    return image, mask


def random_scale(
    image: Image.Image,
    mask: Image.Image,
    scale_range: Tuple[float, float] = (0.5, 2.0),
) -> ImageMask:
    """aspect ratio를 유지하면서 image/mask를 같은 비율로 random scale."""
    scale = random.uniform(scale_range[0], scale_range[1])
    w, h = image.size
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    image = F.resize(image, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)
    mask = F.resize(mask, [new_h, new_w], interpolation=InterpolationMode.NEAREST)
    return image, mask


def random_crop_or_pad(
    image: Image.Image,
    mask: Image.Image,
    crop_size: int,
    ignore_index: int = 255,
) -> ImageMask:
    """작으면 pad, 크면 random crop해서 crop_size x crop_size 보장."""
    w, h = image.size

    pad_w = max(crop_size - w, 0)
    pad_h = max(crop_size - h, 0)
    if pad_w > 0 or pad_h > 0:
        image = F.pad(image, [0, 0, pad_w, pad_h], fill=0)
        mask = F.pad(mask, [0, 0, pad_w, pad_h], fill=ignore_index)
        w, h = image.size

    left = 0 if w == crop_size else random.randint(0, w - crop_size)
    top = 0 if h == crop_size else random.randint(0, h - crop_size)

    image = F.crop(image, top, left, crop_size, crop_size)
    mask = F.crop(mask, top, left, crop_size, crop_size)
    return image, mask


def color_jitter_image_only(
    image: Image.Image,
    prob: float = 0.0,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.05,
) -> Image.Image:
    """mask에는 절대 적용하지 않는 image-only augmentation."""
    if random.random() >= prob:
        return image

    ops = []
    if brightness > 0:
        factor = random.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness)
        ops.append(lambda img, f=factor: F.adjust_brightness(img, f))
    if contrast > 0:
        factor = random.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast)
        ops.append(lambda img, f=factor: F.adjust_contrast(img, f))
    if saturation > 0:
        factor = random.uniform(max(0.0, 1.0 - saturation), 1.0 + saturation)
        ops.append(lambda img, f=factor: F.adjust_saturation(img, f))
    if hue > 0:
        factor = random.uniform(-hue, hue)
        ops.append(lambda img, f=factor: F.adjust_hue(img, f))

    random.shuffle(ops)
    for op in ops:
        image = op(image)
    return image


def to_tensor_and_normalize(
    image: Image.Image,
    mask: Image.Image,
    mean: Sequence[float],
    std: Sequence[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PIL image/mask를 학습용 tensor로 변환."""
    mask_np = np.array(mask, dtype=np.int64)
    valid_pixels = np.sum(mask_np != 255)

    if valid_pixels == 0:
        print("Transform produced all-ignore mask.")
    
    image = F.to_tensor(image)
    image = F.normalize(image, mean=list(mean), std=list(std))
    mask = torch.from_numpy(np.array(mask, dtype=np.int64)).long()
    return image, mask


# ---------------------------------------------------------------------
# Copy-Paste
# ---------------------------------------------------------------------

def _valid_foreground_classes(mask_np: np.ndarray, ignore_index: int = 255) -> List[int]:
    classes = np.unique(mask_np).tolist()
    return [int(c) for c in classes if int(c) not in (0, ignore_index)]


def _binary_bbox(binary: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(binary)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def copy_paste(
    image: Image.Image,
    mask: Image.Image,
    source_image: Image.Image,
    source_mask: Image.Image,
    ignore_index: int = 255,
    paste_scale_range: Tuple[float, float] = (0.7, 1.3),
    max_area_ratio: float = 0.6,
) -> ImageMask:
    """
    semantic mask 기반 simple Copy-Paste.

    source_mask에서 foreground class 하나를 골라 bbox crop 후 target에 붙인다.
    instance mask가 없는 VOC semantic mask에서도 동작하도록 class 영역 기반으로 구현했다.
    """
    image = _to_rgb(image)
    source_image = _to_rgb(source_image)

    tgt_mask_np = _mask_np(mask)
    src_mask_np = _mask_np(source_mask)

    valid_classes = _valid_foreground_classes(src_mask_np, ignore_index)
    if not valid_classes:
        return image, mask

    cls_id = random.choice(valid_classes)
    binary = src_mask_np == cls_id
    bbox = _binary_bbox(binary)
    if bbox is None:
        return image, mask

    x1, y1, x2, y2 = bbox
    obj_w, obj_h = x2 - x1, y2 - y1
    if obj_w <= 1 or obj_h <= 1:
        return image, mask

    tgt_w, tgt_h = image.size
    if obj_w * obj_h > tgt_w * tgt_h * max_area_ratio:
        return image, mask

    obj_img = source_image.crop((x1, y1, x2, y2))
    obj_bin = Image.fromarray((binary[y1:y2, x1:x2].astype(np.uint8) * 255), mode="L")

    scale = random.uniform(paste_scale_range[0], paste_scale_range[1])
    new_w = max(1, int(round(obj_w * scale)))
    new_h = max(1, int(round(obj_h * scale)))

    if new_w > tgt_w or new_h > tgt_h:
        shrink = min(tgt_w / new_w, tgt_h / new_h)
        new_w = max(1, int(round(new_w * shrink)))
        new_h = max(1, int(round(new_h * shrink)))

    obj_img = F.resize(obj_img, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)
    obj_bin = F.resize(obj_bin, [new_h, new_w], interpolation=InterpolationMode.NEAREST)

    px = random.randint(0, max(0, tgt_w - new_w))
    py = random.randint(0, max(0, tgt_h - new_h))

    out_img_np = np.array(image, dtype=np.uint8).copy()
    out_mask_np = tgt_mask_np.copy()
    obj_img_np = np.array(obj_img, dtype=np.uint8)
    obj_bin_np = np.array(obj_bin, dtype=np.uint8) > 0

    img_region = out_img_np[py:py + new_h, px:px + new_w]
    mask_region = out_mask_np[py:py + new_h, px:px + new_w]
    img_region[obj_bin_np] = obj_img_np[obj_bin_np]
    mask_region[obj_bin_np] = cls_id

    out_img_np[py:py + new_h, px:px + new_w] = img_region
    out_mask_np[py:py + new_h, px:px + new_w] = mask_region

    return Image.fromarray(out_img_np, mode="RGB"), _np_mask(out_mask_np)


# ---------------------------------------------------------------------
# Image Stitching / Mosaic
# ---------------------------------------------------------------------

def image_stitching(
    image: Image.Image,
    mask: Image.Image,
    source_samples: Sequence[ImageMask],
    output_size: int,
    mode: str = "random",
) -> ImageMask:
    """
    2-image stitching 또는 4-image mosaic.

    source_samples 1개 이상: horizontal/vertical stitching 가능
    source_samples 3개 이상: mosaic4 가능
    """
    if len(source_samples) == 0:
        return image, mask

    modes = ["horizontal", "vertical"]
    if len(source_samples) >= 3:
        modes.append("mosaic4")

    if mode == "random" or mode not in modes:
        mode = random.choice(modes)

    image = _to_rgb(image)

    if mode == "horizontal":
        src_img, src_mask = source_samples[0]
        src_img = _to_rgb(src_img)
        left_w = output_size // 2
        right_w = output_size - left_w

        img1, mask1 = resize_pair(image, mask, (output_size, left_w))
        img2, mask2 = resize_pair(src_img, src_mask, (output_size, right_w))

        out_img = Image.new("RGB", (output_size, output_size))
        out_mask = Image.new("L", (output_size, output_size), color=255)
        out_img.paste(img1, (0, 0))
        out_mask.paste(mask1, (0, 0))
        out_img.paste(img2, (left_w, 0))
        out_mask.paste(mask2, (left_w, 0))
        return out_img, out_mask

    if mode == "vertical":
        src_img, src_mask = source_samples[0]
        src_img = _to_rgb(src_img)
        top_h = output_size // 2
        bottom_h = output_size - top_h

        img1, mask1 = resize_pair(image, mask, (top_h, output_size))
        img2, mask2 = resize_pair(src_img, src_mask, (bottom_h, output_size))

        out_img = Image.new("RGB", (output_size, output_size))
        out_mask = Image.new("L", (output_size, output_size), color=255)
        out_img.paste(img1, (0, 0))
        out_mask.paste(mask1, (0, 0))
        out_img.paste(img2, (0, top_h))
        out_mask.paste(mask2, (0, top_h))
        return out_img, out_mask

    # mosaic4
    samples = [(image, mask)] + list(source_samples[:3])
    cell = output_size // 2
    positions = [(0, 0), (cell, 0), (0, cell), (cell, cell)]
    sizes = [
        (cell, cell),
        (cell, output_size - cell),
        (output_size - cell, cell),
        (output_size - cell, output_size - cell),
    ]

    out_img = Image.new("RGB", (output_size, output_size))
    out_mask = Image.new("L", (output_size, output_size), color=255)

    for (sample_img, sample_mask), (x, y), (h, w) in zip(samples, positions, sizes):
        sample_img = _to_rgb(sample_img)
        r_img, r_mask = resize_pair(sample_img, sample_mask, (h, w))
        out_img.paste(r_img, (x, y))
        out_mask.paste(r_mask, (x, y))

    return out_img, out_mask


# ---------------------------------------------------------------------
# Final transform class
# ---------------------------------------------------------------------

class SegmentationTransform:
    """
    Semantic segmentation용 transform.

    Copy-Paste / Stitching은 다른 image/mask가 필요하다.
    이제 transform 호출 시 sample_getter를 넘기면 transform 내부에서 random하게 가져온다.
    """

    def __init__(
        self,
        input_size: int = 512,
        is_train: bool = True,
        hflip_prob: float = 0.5,
        random_scale_prob: float = 1.0,
        scale_range: Tuple[float, float] = (0.75, 1.5),
        random_crop_prob: float = 1.0,
        copy_paste_prob: float = 0.1,
        stitching_prob: float = 0.05,
        paste_scale_range: Tuple[float, float] = (0.7, 1.3),
        color_jitter_prob: float = 0.3,
        ignore_index: int = 255,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ):
        self.input_size = input_size
        self.is_train = is_train
        self.hflip_prob = hflip_prob
        self.random_scale_prob = random_scale_prob
        self.scale_range = scale_range
        self.random_crop_prob = random_crop_prob
        self.copy_paste_prob = copy_paste_prob
        self.stitching_prob = stitching_prob
        self.paste_scale_range = paste_scale_range
        self.color_jitter_prob = color_jitter_prob
        self.ignore_index = ignore_index
        self.mean = list(mean)
        self.std = list(std)

    def _apply_random_mix_aug(
        self,
        image: Image.Image,
        mask: Image.Image,
        extra_samples: Optional[Sequence[ImageMask]] = None,
        sample_getter: Optional[SampleGetter] = None,
    ) -> ImageMask:
        """Copy-Paste / Stitching 중 random하게 최대 하나만 적용."""
        candidates = []
        if self.copy_paste_prob > 0 and random.random() < self.copy_paste_prob:
            candidates.append("copy_paste")
        if self.stitching_prob > 0 and random.random() < self.stitching_prob:
            candidates.append("stitching")

        if not candidates:
            return image, mask

        op = random.choice(candidates)
        need = 1 if op == "copy_paste" else 3
        samples = _safe_get_extra_samples(
            extra_samples=extra_samples,
            sample_getter=sample_getter,
            num_samples=need,
        )

        if op == "copy_paste":
            if len(samples) < 1:
                return image, mask
            src_img, src_mask = samples[0]
            return copy_paste(
                image=image,
                mask=mask,
                source_image=src_img,
                source_mask=src_mask,
                ignore_index=self.ignore_index,
                paste_scale_range=self.paste_scale_range,
            )

        if len(samples) < 1:
            return image, mask
        return image_stitching(
            image=image,
            mask=mask,
            source_samples=samples,
            output_size=self.input_size,
            mode="random",
        )

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
        extra_samples: Optional[Sequence[ImageMask]] = None,
        sample_getter: Optional[SampleGetter] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image = _to_rgb(image)

        if self.is_train:
            # 1. Mix augmentation.
            #    extra_samples가 있으면 그것을 쓰고, 없으면 sample_getter로 random source를 뽑는다.
            image, mask = self._apply_random_mix_aug(
                image=image,
                mask=mask,
                extra_samples=extra_samples,
                sample_getter=sample_getter,
            )

            # 2. RandomScale
            if random.random() < self.random_scale_prob:
                image, mask = random_scale(image, mask, self.scale_range)

            # 3. RandomCrop. 최종 input_size 보장.
            if random.random() < self.random_crop_prob:
                image, mask = random_crop_or_pad(
                    image,
                    mask,
                    crop_size=self.input_size,
                    ignore_index=self.ignore_index,
                )
            else:
                image, mask = resize_pair(image, mask, self.input_size)

            # 4. 기존 HorizontalFlip
            image, mask = random_horizontal_flip(image, mask, self.hflip_prob)

            # 5. image-only color jitter. 기본값 0이라 기존 학습에는 영향 없음.
            image = color_jitter_image_only(image, prob=self.color_jitter_prob)

        else:
            # val/test는 deterministic하게 resize만 수행
            image, mask = resize_pair(image, mask, self.input_size)

        return to_tensor_and_normalize(image, mask, mean=self.mean, std=self.std)


# ---------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------

def build_transform(
    input_size: int = 512,
    is_train: bool = True,
    hflip_prob: float = 0.5,
    random_scale_prob: float = 1.0,
    scale_range: Tuple[float, float] = (0.75, 1.5),
    random_crop_prob: float = 1.0,
    copy_paste_prob: float = 0.1,
    stitching_prob: float = 0.0,
    paste_scale_range: Tuple[float, float] = (0.7, 1.3),
    color_jitter_prob: float = 0.3,
    ignore_index: int = 255,
) -> SegmentationTransform:
    return SegmentationTransform(
        input_size=input_size,
        is_train=is_train,
        hflip_prob=hflip_prob,
        random_scale_prob=random_scale_prob,
        scale_range=scale_range,
        random_crop_prob=random_crop_prob,
        copy_paste_prob=copy_paste_prob,
        stitching_prob=stitching_prob,
        paste_scale_range=paste_scale_range,
        color_jitter_prob=color_jitter_prob,
        ignore_index=ignore_index,
    )
