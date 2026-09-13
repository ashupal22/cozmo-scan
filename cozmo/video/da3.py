"""Depth Anything 3 (ByteDance Seed, 2025) on Apple silicon or CPU.

Models (both Apache 2.0):
- DA3-BASE (0.12 B parameters): camera poses, intrinsics and depth from several views at once,
  in one shared frame with an arbitrary scale.
- DA3METRIC-LARGE (0.35 B): depth in metres from a single view, given the focal length; used only to
  set the scale.
DA3-LARGE (better poses) is CC BY-NC, so it is not used.

The upstream code assumes an NVIDIA GPU. Two adaptations, applied only while a model runs:
- Attention is computed in slices of queries. On Apple's GPU, PyTorch's attention builds the whole
  table at once: 16 frames asked for an 11.6 GB buffer and crashed the process. Slicing gives
  identical results with a fraction of the memory.
- Half-precision autocast is turned off on anything but CUDA; Apple's GPU runs these models in fp32.
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # the ray-to-camera solver uses QR, not on Apple's GPU yet

POSE_MODEL = os.environ.get("COZMO_DA3_POSE_MODEL", "depth-anything/DA3-BASE")
METRIC_MODEL = os.environ.get("COZMO_DA3_METRIC_MODEL", "depth-anything/DA3METRIC-LARGE")
QUERY_CHUNK = 1024


@dataclass
class ViewSet:
    """What DA3 returns for a set of views, all in its processed image size."""
    depth: np.ndarray        # (N, h, w) float32
    conf: np.ndarray         # (N, h, w) float32
    world_to_cam: np.ndarray  # (N, 3, 4) OpenCV convention
    K: np.ndarray            # (N, 3, 3)
    size: tuple[int, int]    # (w, h)


def device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@contextlib.contextmanager
def _apple_friendly():
    import torch
    import torch.nn.functional as F

    original_sdpa = F.scaled_dot_product_attention

    def sliced_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        n = q.shape[-2]
        if n <= QUERY_CHUNK or is_causal:
            return original_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)
        out = []
        for s in range(0, n, QUERY_CHUNK):
            mask = None
            if attn_mask is not None:
                mask = attn_mask[..., s:s + QUERY_CHUNK, :] if attn_mask.shape[-2] == n else attn_mask
            out.append(original_sdpa(q[..., s:s + QUERY_CHUNK, :], k, v, attn_mask=mask, dropout_p=dropout_p, **kwargs))
        return torch.cat(out, dim=-2)

    original_autocast = torch.autocast

    class _Autocast(original_autocast):
        def __init__(self, device_type, *args, **kwargs):
            kwargs["enabled"] = device_type == "cuda" and kwargs.get("enabled", True)
            super().__init__(device_type, *args, **kwargs)

    F.scaled_dot_product_attention = sliced_sdpa
    torch.autocast = _Autocast
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original_sdpa
        torch.autocast = original_autocast


def _stub_unused_imports():
    """DA3 imports pycolmap and evo at module level for features we do not use (COLMAP export, aligning to given
    poses). pycolmap's OpenMP clashes with PyTorch's on macOS, so neither is installed (scripts/install.sh); empty
    stand-ins let the import succeed."""
    import importlib.util
    import sys
    import types
    if importlib.util.find_spec("pycolmap") is None:
        sys.modules.setdefault("pycolmap", types.ModuleType("pycolmap"))
    if importlib.util.find_spec("evo") is None:
        for name in ("evo", "evo.core", "evo.core.trajectory"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["evo.core.trajectory"].PosePath3D = None


@lru_cache(maxsize=2)
def load(name: str):
    _stub_unused_imports()
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(name)
    return model.to(device()).eval()


def run_views(images: list, model_name: str = POSE_MODEL, process_res: int = 504, ray_pose: bool = False) -> ViewSet:
    """Poses, intrinsics and depth for a list of images (paths or HxWx3 uint8 arrays). `ray_pose` solves
    the cameras from the per-pixel ray output instead of the small camera head."""
    model = load(model_name)
    with _apple_friendly():
        pred = model.inference(images, process_res=process_res, use_ray_pose=ray_pose)
    h, w = pred.depth.shape[1:]
    return ViewSet(pred.depth.astype(np.float32), pred.conf.astype(np.float32),
                   pred.extrinsics.astype(np.float64), pred.intrinsics.astype(np.float64), (w, h))


def metric_depth(images: list, fx_over_width: float, process_res: int = 504) -> np.ndarray:
    """Depth in metres per image, (N, h, w), from the monocular metric model.

    The model predicts depth for a canonical camera; metres = output x focal_px / 300 (DA3 README). A
    picture of a room could be a wide lens up close or a narrow lens from further away, so the scale is
    only as good as the focal length passed in: 5% focal error = 5% size error. `fx_over_width` is the
    focal length in units of the image width, so it holds at any resolution."""
    model = load(METRIC_MODEL)
    out = []
    with _apple_friendly():
        for img in images:
            pred = model.inference([img], process_res=process_res)
            d = pred.depth[0].astype(np.float32)
            out.append(d * (fx_over_width * d.shape[1] / 300.0))
    return np.stack(out)
