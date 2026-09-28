# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Independent PyTorch reference for MiniMax-H3 video patch packing.

The public interface keeps the batch dimension. Diffusers flattens the first
two output dimensions after applying the same permutation.
"""

from __future__ import annotations

import torch

_CHANNELS = 24
_PATCH_T, _PATCH_H, _PATCH_W = 1, 2, 2
_WIDTH = _CHANNELS * _PATCH_T * _PATCH_H * _PATCH_W
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_pack(x: torch.Tensor) -> tuple[int, int, int, int]:
    if not isinstance(x, torch.Tensor):
        raise TypeError("video latents must be a torch.Tensor")
    if x.ndim != 5:
        raise ValueError(f"video latents must have shape (B,24,T,H,W), got {tuple(x.shape)}")
    b, c, t, h, w = x.shape
    if b < 1 or t < 1 or h < 1 or w < 1:
        raise ValueError("B, T, H and W must be positive")
    if c != _CHANNELS:
        raise ValueError(f"H3 video latents require {_CHANNELS} channels, got {c}")
    if h % _PATCH_H or w % _PATCH_W:
        raise ValueError("H and W must be divisible by the (2,2) spatial patch")
    if x.dtype not in _DTYPES:
        raise TypeError(f"unsupported video latent dtype: {x.dtype}")
    if not x.is_contiguous():
        raise ValueError("video latents must be contiguous")
    return b, t, h, w


def _validate_unpack(tokens: torch.Tensor, shape: tuple[int, int, int, int, int]) -> None:
    if not isinstance(tokens, torch.Tensor):
        raise TypeError("video tokens must be a torch.Tensor")
    if len(shape) != 5:
        raise ValueError("shape must be (B,24,T,H,W)")
    b, c, t, h, w = shape
    if min(b, t, h, w) < 1 or c != _CHANNELS or h % 2 or w % 2:
        raise ValueError("invalid H3 video latent shape")
    expected = (b, t * (h // 2) * (w // 2), _WIDTH)
    if tuple(tokens.shape) != expected:
        raise ValueError(f"video tokens must have shape {expected}, got {tuple(tokens.shape)}")
    if tokens.dtype not in _DTYPES:
        raise TypeError(f"unsupported video token dtype: {tokens.dtype}")
    if not tokens.is_contiguous():
        raise ValueError("video tokens must be contiguous")


def pack_h3_video_reference(x: torch.Tensor) -> torch.Tensor:
    """Return (B,S,96), with S ordered frame-major, then row-major."""
    b, t, h, w = _validate_pack(x)
    return (
        x.reshape(b, _CHANNELS, t, h // 2, 2, w // 2, 2)
        .permute(0, 2, 3, 5, 1, 4, 6)
        .reshape(b, t * (h // 2) * (w // 2), _WIDTH)
        .contiguous()
    )


def unpack_h3_video_reference(
    tokens: torch.Tensor, shape: tuple[int, int, int, int, int]
) -> torch.Tensor:
    """Invert the H3 patch permutation without arithmetic or dtype conversion."""
    _validate_unpack(tokens, shape)
    b, _, t, h, w = shape
    return (
        tokens.reshape(b, t, h // 2, w // 2, _CHANNELS, 1, 2, 2)
        .permute(0, 4, 1, 5, 2, 6, 3, 7)
        .reshape(shape)
        .contiguous()
    )


class NativeH3VideoPatchOp:
    backend_id = "pytorch-h3-video-patch-v1"

    def pack(self, x: torch.Tensor) -> torch.Tensor:
        return pack_h3_video_reference(x)

    def unpack(
        self, tokens: torch.Tensor, shape: tuple[int, int, int, int, int]
    ) -> torch.Tensor:
        return unpack_h3_video_reference(tokens, shape)
