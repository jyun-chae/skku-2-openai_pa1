from torch.utils.data import ConcatDataset, DataLoader

from src.data.transforms import build_transform
from src.data.voc import build_voc_dataset
from src.data.coco_voc import build_coco_voc_dataset

from types import SimpleNamespace

def build_dataset(dataset_cfg: SimpleNamespace, cfg: SimpleNamespace, is_train: bool):
    """
    dataset_cfg 예시:
        {"name": "voc", "year": "2012", "split": "train"}
        {"name": "coco_voc", "split": "train"}
    """

    name = dataset_cfg.name.lower()

    transform = build_transform(
        input_size=cfg.data.input_size,
        is_train=is_train,
        ignore_index=cfg.data.ignore_index,
        hflip_prob=cfg.transforms.hflip_prob if is_train else 0.0,
        random_scale_prob=cfg.transforms.random_scale_prob if is_train else 0.0,
        scale_range=tuple(cfg.transforms.scale_range),
        random_crop_prob=cfg.transforms.random_crop_prob if is_train else 0.0,
        copy_paste_prob=cfg.transforms.copy_paste_prob if is_train else 0.0,
        stitching_prob=cfg.transforms.stitching_prob if is_train else 0.0,
        paste_scale_range=tuple(cfg.transforms.paste_scale_range),
        color_jitter_prob=cfg.transforms.color_jitter_prob if is_train else 0.0,
    )

    if name == "voc":
        return build_voc_dataset(
            root=cfg.data.root,
            year=getattr(dataset_cfg, "year", "2012"),
            split=getattr(dataset_cfg, "split", "train"),
            transform=transform,
            download=getattr(cfg.data, "download", True),
        )

    if name == "coco_voc":
        return build_coco_voc_dataset(
            root=cfg.data.coco.image_dir,
            ann_file=cfg.data.coco.ann_file,
            transform=transform,
            ignore_index=cfg.data.ignore_index,
            cache_dir=cfg.data.coco.mask_cache_dir,
            use_cache=cfg.data.coco.use_mask_cache,
        )

    raise ValueError(f"Unknown dataset name: {name}")


def build_train_dataset(cfg: dict):
    datasets = []

    for dataset_cfg in cfg.data.train_datasets:
        dataset = build_dataset(
            dataset_cfg=dataset_cfg,
            cfg=cfg,
            is_train=True,
        )
        datasets.append(dataset)

    if len(datasets) == 0:
        raise ValueError("No training datasets are specified.")

    if len(datasets) == 1:
        return datasets[0]

    return ConcatDataset(datasets)


def build_val_dataset(cfg: dict):
    val_cfg = cfg.data.val_dataset

    return build_dataset(
        dataset_cfg=val_cfg,
        cfg=cfg,
        is_train=False,
    )


def build_train_loader(cfg: dict):
    train_dataset = build_train_dataset(cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
    )

    return train_loader


def build_val_loader(cfg: dict):
    val_dataset = build_val_dataset(cfg)

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=False,
    )

    return val_loader


def build_dataloaders(cfg: dict):
    train_loader = build_train_loader(cfg)
    val_loader = build_val_loader(cfg)

    return train_loader, val_loader