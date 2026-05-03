from .seg_model import FPNLikeSegModel


def build_model(cfg):
    model_name = getattr(cfg.model, "name", "fpn_efficientnet").lower()

    if model_name in ["fpn_efficientnet", "efficientnet_b3", "fpn_like"]:
        return FPNLikeSegModel(cfg)

    raise ValueError(f"Unsupported model name: {cfg.model.name}")