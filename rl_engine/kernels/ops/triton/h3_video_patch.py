# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Portable Triton H3 video patch permutation for CUDA and ROCm."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from rl_engine.kernels.ops.pytorch.packing.h3_video_patch import (
    _validate_pack,
    _validate_unpack,
)

_BLOCK = 256


@triton.jit
def _copy_patch(src, dst, B: tl.constexpr, T: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                PACK: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    patches_per_frame = (H // 2) * (W // 2)
    patches_per_batch = T * patches_per_frame
    count = B * patches_per_batch * 96
    valid = offsets < count

    feature = offsets % 96
    patch = (offsets // 96) % patches_per_batch
    batch = offsets // (patches_per_batch * 96)
    channel = feature // 4
    patch_h = (feature % 4) // 2
    patch_w = feature % 2
    frame = patch // patches_per_frame
    row = (patch // (W // 2)) % (H // 2)
    col = patch % (W // 2)
    latent_offset = (
        (((batch * 24 + channel) * T + frame) * H + row * 2 + patch_h) * W
        + col * 2
        + patch_w
    )

    if PACK:
        value = tl.load(src + latent_offset, valid, other=0)
        tl.store(dst + offsets, value, valid)
    else:
        value = tl.load(src + offsets, valid, other=0)
        tl.store(dst + latent_offset, value, valid)


def _launch(src: torch.Tensor, shape: tuple[int, int, int, int, int], pack: bool) -> torch.Tensor:
    b, _, t, h, w = shape
    tokens_shape = (b, t * (h // 2) * (w // 2), 96)
    dst = torch.empty(tokens_shape if pack else shape, dtype=src.dtype, device=src.device)
    _copy_patch[(triton.cdiv(src.numel(), _BLOCK),)](
        src, dst, b, t, h, w, PACK=pack, BLOCK=_BLOCK
    )
    return dst


class _Pack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.shape = tuple(x.shape)
        return _launch(x, ctx.shape, True)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor]:
        return (_launch(grad.contiguous(), ctx.shape, False),)


class _Unpack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens: torch.Tensor, shape: tuple[int, int, int, int, int]) -> torch.Tensor:
        ctx.shape = shape
        return _launch(tokens, shape, False)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        return _launch(grad.contiguous(), ctx.shape, True), None


class TritonH3VideoPatchOp:
    backend_id = "triton-h3-video-patch-v1"

    def pack(self, x: torch.Tensor) -> torch.Tensor:
        _validate_pack(x)
        if x.device.type != "cuda":
            raise ValueError("Triton H3 video patch requires a CUDA or ROCm tensor")
        return _Pack.apply(x)

    def unpack(
        self, tokens: torch.Tensor, shape: tuple[int, int, int, int, int]
    ) -> torch.Tensor:
        _validate_unpack(tokens, shape)
        if tokens.device.type != "cuda":
            raise ValueError("Triton H3 video patch requires a CUDA or ROCm tensor")
        return _Unpack.apply(tokens, shape)
