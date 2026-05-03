# src/config/config.py

from pathlib import Path
from types import SimpleNamespace
import yaml


def dict_to_namespace(obj):
    """
    dict를 cfg.data.input_size 같은 attribute 접근이 가능한 구조로 변환.
    list 안의 dict도 재귀적으로 변환.
    """
    if isinstance(obj, dict):
        return SimpleNamespace(
            **{key: dict_to_namespace(value) for key, value in obj.items()}
        )

    if isinstance(obj, list):
        return [dict_to_namespace(item) for item in obj]

    return obj


def load_config(config_path="src/config/default.yaml"):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    cfg = dict_to_namespace(cfg_dict)

    return cfg