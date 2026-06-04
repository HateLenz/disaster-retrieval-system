from __future__ import annotations
import torch

def get_device(prefer: str = "cuda") -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def setup_torch_runtime() -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True