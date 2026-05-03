import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Colab T4/L4에서는 보통 이쪽이 빠름
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True