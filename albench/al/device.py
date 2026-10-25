"""Pick the compute backend, preferring GPU then CPU."""
import os


def device() -> str:
    # ALBENCH_DEVICE pins one GPU per process. DDP would drop PPAL's callbacks.
    spec = os.environ.get("ALBENCH_DEVICE")
    if spec:
        return spec
    import torch
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
