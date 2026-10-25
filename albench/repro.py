"""Seeding for reproducible training."""
import os
import random


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch and make cuDNN deterministic."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) if hasattr(torch, "cuda") else None
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
