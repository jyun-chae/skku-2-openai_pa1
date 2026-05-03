# src/data/transforms.py

import random
from typing import Tuple
import numpy as np

import torch
from PIL import Image
import torchvision.transforms.functional as F


class SegmentationTransform:
    """
    Semantic segmentation용 transform.

    image: PIL RGB Image
    mask : PIL Image, 각 픽셀 값이 class index
    """

    def __init__(
        self,
        input_size: int = 512,
        is_train: bool = True,
        hflip_prob: float = 0.5,
    ):
        self.input_size = input_size
        self.is_train = is_train
        self.hflip_prob = hflip_prob

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        image = image.convert("RGB")

        # 1. resize
        image = F.resize(
            image,
            [self.input_size, self.input_size],
            interpolation=F.InterpolationMode.BILINEAR,
        )

        mask = F.resize(
            mask,
            [self.input_size, self.input_size],
            interpolation=F.InterpolationMode.NEAREST,
        )

        # 2. train augmentation
        if self.is_train:
            if random.random() < self.hflip_prob:
                image = F.hflip(image)
                mask = F.hflip(mask)

        # 3. PIL image -> Tensor
        image = F.to_tensor(image)
        image = F.normalize(image, mean=self.mean, std=self.std)

        # 4. mask -> LongTensor
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        return image, mask


def build_transform(input_size: int = 512, is_train: bool = True):
    return SegmentationTransform(
        input_size=input_size,
        is_train=is_train,
    )