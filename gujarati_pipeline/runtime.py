from __future__ import annotations

import importlib.metadata
import sys
from typing import Any

from .types import StageError


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def import_torch():
    try:
        import torch

        return torch
    except Exception as exc:  # pragma: no cover - depends on Kaggle image
        raise StageError(
            "PyTorch is required. Install a CUDA-enabled torch>=2.7 build matching "
            "the NeMo 3.0 environment: https://pytorch.org/get-started/locally/"
        ) from exc


def resolve_device(preference: str, torch) -> str:
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise StageError("PIPELINE_DEVICE=cuda was requested but torch.cuda.is_available() is False.")
        return "cuda"
    if preference == "cpu":
        return "cpu"
    if preference != "auto":
        raise StageError(f"Unknown device preference: {preference!r} (use auto, cuda or cpu)")
    return "cuda" if torch.cuda.is_available() else "cpu"


def torch_dtype(torch, name: str):
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise StageError(f"Unsupported compute dtype: {name}")
    return mapping[name]


def gpu_info() -> dict[str, Any]:
    """GPU/torch facts for the progress log and processing report."""
    try:
        torch = import_torch()
    except StageError as exc:
        return {"torch_available": False, "error": str(exc)}
    info: dict[str, Any] = {
        "torch_available": True,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "python_version": sys.version.split()[0],
    }
    if torch.cuda.is_available():
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["device_count"] = torch.cuda.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)
    return info


def check_python_torch() -> list[str]:
    problems: list[str] = []
    if sys.version_info < (3, 12):
        problems.append(
            f"Python {sys.version_info.major}.{sys.version_info.minor} detected; "
            "NVIDIA NeMo Speech 3.0 documentation requires Python 3.12 or later."
        )
    info = gpu_info()
    if not info.get("torch_available"):
        problems.append(str(info.get("error")))
    elif not info.get("cuda_available"):
        problems.append("CUDA GPU not detected; Sortformer is GPU-optimised and CPU inference will be slow.")
    return problems
