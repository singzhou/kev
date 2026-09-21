"""Small cross-backend device helpers used by serving and evaluation."""
import importlib

import torch


def device_type(device):
    return str(device).split(":", 1)[0]


def _load_torch_npu(required=False):
    """Import TorchNPU explicitly so its PyTorch backend and torch.npu API are registered."""
    try:
        importlib.import_module("torch_npu")
    except Exception as exc:
        if required:
            raise RuntimeError("NPU requested, but torch_npu could not be imported") from exc
        return False
    return hasattr(torch, "npu")


def npu_available():
    return _load_torch_npu() and torch.npu.is_available()


def default_device():
    # Avoid importing torch_npu into a CUDA process: TorchNPU rejects two accelerator backends in one process.
    if torch.cuda.is_available():
        return "cuda"
    if npu_available():
        return "npu"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(device=None):
    if device in (None, "auto"):
        return default_device()
    kind = device_type(device)
    if kind == "npu":
        if not _load_torch_npu(required=True) or not torch.npu.is_available():
            raise RuntimeError("NPU requested, but torch.npu.is_available() is false")
    return device


def synchronize(device):
    kind = device_type(device)
    if kind == "mps":
        torch.mps.synchronize()
    elif kind == "cuda":
        torch.cuda.synchronize()
    elif kind == "npu":
        torch.npu.synchronize()


def empty_cache(device):
    kind = device_type(device)
    if kind == "mps":
        torch.mps.empty_cache()
    elif kind == "cuda":
        torch.cuda.empty_cache()
    elif kind == "npu":
        torch.npu.empty_cache()
