import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection


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


COCO_TO_VOC_NAME = {
    "airplane": "aeroplane",
    "bicycle": "bicycle",
    "bird": "bird",
    "boat": "boat",
    "bottle": "bottle",
    "bus": "bus",
    "car": "car",
    "cat": "cat",
    "chair": "chair",
    "cow": "cow",
    "dining table": "diningtable",
    "dog": "dog",
    "horse": "horse",
    "motorcycle": "motorbike",
    "person": "person",
    "potted plant": "pottedplant",
    "sheep": "sheep",
    "couch": "sofa",
    "train": "train",
    "tv": "tvmonitor",
}


class COCOVOCSegDataset(Dataset):
    """
    COCO detection annotation을 VOC-style segmentation mask로 변환한다.

    핵심:
        - 처음 접근한 image의 mask는 annToMask로 생성 후 PNG cache 저장
        - 다음 epoch부터는 cached PNG를 바로 읽음

    mask label:
        0    = background
        1~20 = VOC class
        255  = ignore
    """

    def __init__(
        self,
        root: str | Path,
        ann_file: str | Path,
        transform: Optional[Callable] = None,
        ignore_index: int = 255,
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
    ):
        self.root = Path(root)
        self.ann_file = Path(ann_file)
        self.transform = transform
        self.ignore_index = ignore_index
        self.use_cache = use_cache

        if cache_dir is None:
            self.cache_dir = self.ann_file.parent.parent / "coco_voc_mask_cache"
        else:
            self.cache_dir = Path(cache_dir)

        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = CocoDetection(
            root=str(self.root),
            annFile=str(self.ann_file),
        )

        self.coco = self.dataset.coco

        self.voc_name_to_idx = {
            name: idx for idx, name in enumerate(VOC_CLASSES)
        }

        self.coco_cat_id_to_voc_idx = self._build_category_mapping()

        self.valid_indices = self._filter_valid_indices()

        print(
            f"[COCOVOCSegDataset] total images: {len(self.dataset)}, "
            f"valid images: {len(self.valid_indices)}, "
            f"use_cache: {self.use_cache}, "
            f"cache_dir: {self.cache_dir}"
        )

    def _build_category_mapping(self) -> dict[int, int]:
        mapping = {}

        for cat_id, cat in self.coco.cats.items():
            coco_name = cat["name"]

            if coco_name not in COCO_TO_VOC_NAME:
                continue

            voc_name = COCO_TO_VOC_NAME[coco_name]
            voc_idx = self.voc_name_to_idx[voc_name]
            mapping[cat_id] = voc_idx

        return mapping

    def _filter_valid_indices(self) -> list[int]:
        valid_indices = []

        for idx in range(len(self.dataset)):
            image_id = self.dataset.ids[idx]
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)

            has_valid_voc_object = any(
                ann["category_id"] in self.coco_cat_id_to_voc_idx
                and ann.get("iscrowd", 0) == 0
                and ann.get("area", 0) > 0
                for ann in anns
            )

            if has_valid_voc_object:
                valid_indices.append(idx)

        return valid_indices

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _get_cache_path(self, image_id: int) -> Path:
        return self.cache_dir / f"{image_id:012d}.png"

    def _load_or_build_mask(
        self,
        image: Image.Image,
        anns: list[dict],
        image_id: int,
    ) -> Image.Image:
        if self.use_cache:
            cache_path = self._get_cache_path(image_id)

            if cache_path.exists():
                return Image.open(cache_path)

        mask = self._build_mask(image, anns)

        if not self._has_foreground(mask):
            # 이론상 _filter_valid_indices를 고치면 거의 안 걸려야 정상
            # 디버깅 단계에서는 raise로 바로 잡는 게 좋음
            raise ValueError(f"COCO image_id={image_id} produced no foreground VOC pixels.")

        if self.use_cache:
            mask.save(cache_path)

        return mask
    
    def _load_raw_sample(self, index: int):
        real_index = self.valid_indices[index]
        image_id = self.dataset.ids[real_index]

        image, anns = self.dataset[real_index]
        image = image.convert("RGB")

        mask = self._load_or_build_mask(image, anns, image_id)

        return image, mask
    
    def _has_foreground(self, mask: Image.Image) -> bool:
        mask_np = np.array(mask, dtype=np.uint8)
        return np.any((mask_np >= 1) & (mask_np <= 20))
    
    def get_random_raw_sample(self):
        rand_index = random.randint(0, len(self) - 1)
        return self._load_raw_sample(rand_index)

    def __getitem__(self, index: int):
        image, mask = self._load_raw_sample(index)

        if self.transform is not None:
            if getattr(self.transform, "is_train", False):
                image, mask = self.transform(
                    image,
                    mask,
                    sample_getter=self.get_random_raw_sample,
                )
            else:
                image, mask = self.transform(image, mask)
        else:
            raise ValueError(
                "COCOVOCSegDataset requires transform. "
                "Use build_transform() from src.data.transforms."
            )

        return image, mask

    def _build_mask(self, image: Image.Image, anns: list[dict]) -> Image.Image:
        width, height = image.size

        mask = np.zeros((height, width), dtype=np.uint8)

        for ann in anns:
            cat_id = ann["category_id"]

            if cat_id not in self.coco_cat_id_to_voc_idx:
                continue

            voc_idx = self.coco_cat_id_to_voc_idx[cat_id]

            obj_mask = self.coco.annToMask(ann).astype(bool)

            if ann.get("iscrowd", 0) == 1:
                mask[obj_mask] = self.ignore_index
            else:
                mask[obj_mask] = voc_idx

        return Image.fromarray(mask)


def build_coco_voc_dataset(
    root: str | Path,
    ann_file: str | Path,
    transform: Callable,
    ignore_index: int = 255,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> COCOVOCSegDataset:
    return COCOVOCSegDataset(
        root=root,
        ann_file=ann_file,
        transform=transform,
        ignore_index=ignore_index,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )