from pathlib import Path
from typing import Callable, Optional

import random
from torch.utils.data import Dataset
from torchvision.datasets import VOCSegmentation


class VOCSegDataset(Dataset):
    """
    Pascal VOC semantic segmentation dataset wrapper.

    torchvision.datasets.VOCSegmentation은 image, target을 PIL로 반환한다.
    여기서는 segmentation transform을 적용해서
    image: FloatTensor [3, H, W]
    mask : LongTensor  [H, W]
    형태로 반환한다.
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
        # VOC mask는 palette PNG일 수 있으므로 convert("L") 하지 않는 게 안전함.
        # transform.py 쪽에서 np.array(mask)로 class index를 읽으면 됨.

        return image, mask
    
    def get_random_raw_sample(self):
        rand_index = random.randint(0, len(self.dataset) - 1)
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