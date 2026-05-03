from .seg_model import FPNLikeSegModel


def build_model(cfg):
    model_name = cfg.model.name.lower()

    if model_name == "fpn_efficientnet":
        return FPNLikeSegModel(cfg)

    raise ValueError(f"Unsupported model name: {cfg.model.name}")