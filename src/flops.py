import torch

from src.models.build import build_model
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def count_flops_with_thop(model, input_tensor):
    try:
        from thop import profile
    except ImportError:
        raise ImportError(
            "thop is not installed. Install it with:\n"
            "pip install thop"
        )

    flops, params = profile(
        model,
        inputs=(input_tensor,),
        verbose=False,
    )

    return flops, params


def format_number(num):
    if num >= 1e9:
        return f"{num / 1e9:.3f}G"
    elif num >= 1e6:
        return f"{num / 1e6:.3f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.3f}K"
    else:
        return str(num)


def main(cfg):
    set_seed(cfg.seed)

    logger = get_logger()

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # -------------------------
    # 1. Model
    # -------------------------
    model = build_model(cfg)
    model = model.to(device)
    model.eval()

    # -------------------------
    # 2. Dummy Input
    # -------------------------
    height = cfg.flops.height
    width = cfg.flops.width

    dummy_input = torch.randn(
        1, 3, height, width, device=device
    )

    # -------------------------
    # 3. FLOPs / Params
    # -------------------------
    flops, params = count_flops_with_thop(model, dummy_input)

    logger.info(f"FLOPs: {format_number(flops)}")
    logger.info(f"Params: {format_number(params)}")

    print("===================================")
    print(f"Input size : 1 x 3 x {height} x {width}")
    print(f"FLOPs      : {format_number(flops)}")
    print(f"Params     : {format_number(params)}")
    print("===================================")

    return {
        "flops": flops,
        "params": params,
        "flops_readable": format_number(flops),
        "params_readable": format_number(params),
    }


if __name__ == "__main__":
    from src.config import load_config

    cfg = load_config()
    main(cfg)