# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""H3 1x2x2 video patch order, inverse, gradients, and batch invariance."""

import pytest
import torch

from rl_engine.kernels.ops.pytorch.packing.h3_video_patch import (
    NativeH3VideoPatchOp,
    pack_h3_video_reference,
    unpack_h3_video_reference,
)
from rl_engine.kernels.registry import KernelRegistry


def test_known_patch_order():
    x = torch.arange(24 * 2 * 4, dtype=torch.float32).reshape(1, 24, 1, 2, 4)
    y = pack_h3_video_reference(x)
    assert y.shape == (1, 2, 96)
    assert y[0, 0, :8].tolist() == [0, 1, 4, 5, 8, 9, 12, 13]
    assert y[0, 1, :8].tolist() == [2, 3, 6, 7, 10, 11, 14, 15]
    assert torch.equal(unpack_h3_video_reference(y, tuple(x.shape)), x)


@pytest.mark.parametrize("shape", [(1, 24, 1, 2, 2), (2, 24, 3, 6, 10), (1, 24, 2, 48, 84)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_reference_roundtrip_and_backward(shape, dtype):
    x = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
    x = x.to(dtype).requires_grad_()
    tokens = pack_h3_video_reference(x)
    assert torch.equal(unpack_h3_video_reference(tokens, shape), x)
    weights = torch.arange(tokens.numel(), dtype=torch.float32).reshape(tokens.shape).to(dtype)
    tokens.backward(weights)
    assert torch.equal(x.grad, unpack_h3_video_reference(weights, shape))


def test_reference_batch_invariance():
    x = torch.arange(24 * 3 * 6 * 10, dtype=torch.float32).reshape(1, 24, 3, 6, 10)
    alone = pack_h3_video_reference(x)
    others = torch.full_like(x, -100)
    assert torch.equal(pack_h3_video_reference(torch.cat([x, others]))[0], alone[0])
    assert torch.equal(pack_h3_video_reference(torch.cat([others, x]))[1], alone[0])


def test_invalid_inputs():
    x = torch.empty((1, 24, 1, 2, 4))
    with pytest.raises(ValueError, match="24"):
        pack_h3_video_reference(x[:, :23].contiguous())
    with pytest.raises(ValueError, match="divisible"):
        pack_h3_video_reference(torch.empty((1, 24, 1, 3, 4)))
    with pytest.raises(ValueError, match="contiguous"):
        pack_h3_video_reference(x.transpose(-1, -2))
    with pytest.raises(TypeError, match="dtype"):
        pack_h3_video_reference(x.to(torch.int32))
    with pytest.raises(ValueError, match="shape"):
        unpack_h3_video_reference(torch.empty((1, 3, 96)), tuple(x.shape))


def test_cpu_registry_is_reference():
    op = KernelRegistry().get_h3_video_patch_op("cpu")
    assert isinstance(op, NativeH3VideoPatchOp)
    assert op.backend_id == "pytorch-h3-video-patch-v1"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm GPU required")
def test_triton_matches_reference_and_gradient():
    pytest.importorskip("triton")
    op = KernelRegistry().get_h3_video_patch_op("cuda", strict=True)
    assert op.backend_id == "triton-h3-video-patch-v1"
    shape = (2, 24, 3, 48, 84)
    x = torch.randn(shape, device="cuda", dtype=torch.float32, requires_grad=True)
    y = op.pack(x)
    expected = pack_h3_video_reference(x)
    assert torch.equal(y, expected)
    assert torch.equal(op.unpack(y, shape), x)

    grad = torch.randn_like(y)
    y.backward(grad)
    assert torch.equal(x.grad, unpack_h3_video_reference(grad, shape))
    assert torch.equal(op.pack(torch.cat([x.detach(), x.detach() + 1]))[:2], y)
