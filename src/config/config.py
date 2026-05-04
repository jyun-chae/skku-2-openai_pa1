from pathlib import Path
from types import SimpleNamespace
import yaml

"""Config module.

This module loads YAML configuration files and converts the resulting
structure into a form that supports dot notation access
"""


def dict_to_namespace(obj):
    """Convert dictionaries and lists into namespace objects.

    Notes:
        - Nested dict values are recursively converted into SimpleNamespace.
        - Dicts inside lists are also recursively converted.
    """
    if isinstance(obj, dict):
        return SimpleNamespace(
            **{key: dict_to_namespace(value) for key, value in obj.items()}
        )

    if isinstance(obj, list):
        return [dict_to_namespace(item) for item in obj]

    return obj


def load_config(config_path="src/config/default.yaml"):
    """Load a YAML configuration file and convert it into a namespace.

    Args:
        config_path: Path to the configuration file. Defaults to src/config/default.yaml.

    Returns:
        The configuration object converted via dict_to_namespace.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    # Convert the dict structure into an object with attribute access.
    cfg = dict_to_namespace(cfg_dict)

    return cfg