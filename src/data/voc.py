"""
Pascal VOC semantic segmentation dataset implementation.

This module provides a wrapper around torchvision.datasets.VOCSegmentation
to apply segmentation transforms and return tensors in the expected format.
"""

from pathlib import Path
from typing import Callable, Optional

import random
from torch.utils.data import Dataset
from torchvision.datasets import VOCSegmentation


class VOCSegDataset(Dataset):
    """
    Pascal VOC semantic segmentation dataset wrapper.

    torchvision.datasets.VOCSegmentation returns image and target as PIL images.
    This wrapper applies segmentation transforms to return:
    - image: FloatTensor [3, H, W]
    - mask: LongTensor [H, W]
    """

    def __init__(
        self,
        root: str | Path,
        year: str = "2012",
        split: str = "train",
        transform: Optional[Callable] = None,
        download: bool = True,
    ):
        self.root = Path(root)
        self.year = str(year)
        self.split = split
        self.transform = transform

        self.dataset = VOCSegmentation(
            root=str(self.root),
            year=self.year,
            image_set=self.split,
            download=download,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def _load_raw_sample(self, index: int):
        image, mask = self.dataset[index]

        image = image.convert("RGB")
        # VOC mask may be palette PNG, so avoid convert("L") for safety.
        # transforms.py will handle class index conversion with np.array(mask).

        return image, mask

    def get_random_raw_sample(self):
        """Get a random raw sample for augmentations like copy-paste."""
        rand_index = random.randint(0, len(self.dataset) - 1)
        return self._load_raw_sample(rand_index)

    def __getitem__(self, index: int):
        """Get transformed sample at given index."""
        image, mask = self._load_raw_sample(index)

        if self.transform is not None:
            if getattr(self.transform, "is_train", False):
                # For training transforms that need sample_getter (e.g., copy-paste)
                image, mask = self.transform(
                    image,
                    mask,
                    sample_getter=self.get_random_raw_sample,
                )
            else:
                # For validation/test transforms
                image, mask = self.transform(image, mask)
        else:
            raise ValueError(
                "VOCSegDataset requires transform. "
                "Use build_transform() from src.data.transforms."
            )

        return image, mask


def build_voc_dataset(
    root: str | Path,
    year: str,
    split: str,
    transform: Callable,
    download: bool = True,
) -> VOCSegDataset:
    return VOCSegDataset(
        root=root,
        year=year,
        split=split,
        transform=transform,
        download=download,
    )