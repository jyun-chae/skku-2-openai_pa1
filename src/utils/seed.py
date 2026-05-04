import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    #Set random seed for reproducibility across libraries.
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Fully deterministic operations (slower)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Benchmark mode for better performance (usually faster on Colab T4/L4)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True