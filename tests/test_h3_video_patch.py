# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""H3 1x2x2 order, raw bits, inverse, gradients, and batch invariance."""

import pytest
import torch

from rl_engine.kernels.ops.pytorch.packing.h3_video_patch import (
    NativeH3VideoPatchOp,
    pack_h3_video_reference,
    unpack_h3_video_reference,
)
from rl_engine.kernels.registry import KernelRegistry

DTYPES = (torch.float32, torch.float16, torch.bfloat16)
SHAPES = ((1, 24, 1, 2, 2), (2, 24, 3, 6, 10), (1, 24, 2, 48, 84))


def assert_bits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare all bytes, including signed zero and NaN payload bits."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.detach().contiguous().cpu().view(torch.uint8),
        expected.detach().contiguous().cpu().view(torch.uint8),
    )


def test_coordinate_order_across_frames():
    shape = (1, 24, 2, 2, 4)
    x = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
    packed = pack_h3_video_reference(x)
    assert packed.shape == (1, 4, 96)

    # Check against source coordinates, not another pack/unpack round trip.
    for frame in range(2):
        for patch_col in range(2):
            token_row = frame * 2 + patch_col
            for channel in range(24):
                for dy in range(2):
                    for dx in range(2):
                        feature = channel * 4 + dy * 2 + dx
                        assert packed[0, token_row, feature] == x[
                            0, channel, frame, dy, patch_col * 2 + dx
                        ]

    # An independently constructed token input checks unpack's direction.
    tokens = torch.arange(4 * 96, dtype=torch.float32).reshape(1, 4, 96)
    unpacked = unpack_h3_video_reference(tokens, shape)
    for frame in range(2):
        for patch_col in range(2):
            for channel in range(24):
                for dy in range(2):
                    for dx in range(2):
                        assert unpacked[0, channel, frame, dy, patch_col * 2 + dx] == tokens[
                            0, frame * 2 + patch_col, channel * 4 + dy * 2 + dx
                        ]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_reference_roundtrip_and_both_gradients(shape, dtype):
    generator = torch.Generator().manual_seed(420)
    x = torch.randn(shape, generator=generator).to(dtype).requires_grad_()
    packed = pack_h3_video_reference(x)
    assert_bits(unpack_h3_video_reference(packed, shape), x)

    grad_tokens = torch.randn(packed.shape, generator=generator).to(dtype)
    dx = torch.autograd.grad(packed, x, grad_tokens)[0]
    assert_bits(dx, unpack_h3_video_reference(grad_tokens, shape))

    tokens = torch.randn(packed.shape, generator=generator).to(dtype).requires_grad_()
    unpacked = unpack_h3_video_reference(tokens, shape)
    grad_latents = torch.randn(shape, generator=generator).to(dtype)
    dtokens = torch.autograd.grad(unpacked, tokens, grad_latents)[0]
    assert_bits(dtokens, pack_h3_video_reference(grad_latents))


def test_reference_batch_invariance():
    x = torch.arange(24 * 3 * 6 * 10, dtype=torch.float32).reshape(1, 24, 3, 6, 10)
    alone = pack_h3_video_reference(x)
    others = torch.full_like(x, -100)
    assert_bits(pack_h3_video_reference(torch.cat([x, others]))[0], alone[0])
    assert_bits(pack_h3_video_reference(torch.cat([others, x]))[1], alone[0])


def test_invalid_inputs():
    x = torch.empty((1, 24, 1, 2, 4))
    for bad in (torch.empty((0, 24, 1, 2, 4)), torch.empty((1, 24, 0, 2, 4))):
        with pytest.raises(ValueError, match="positive"):
            pack_h3_video_reference(bad)
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
    with pytest.raises(ValueError, match="shape"):
        unpack_h3_video_reference(torch.empty((1, 2, 96)), (1, 24, 1, 2))
    with pytest.raises(TypeError, match="integers"):
        unpack_h3_video_reference(torch.empty((1, 2, 96)), (1, 24, 1.0, 2, 4))
    with pytest.raises(ValueError, match="contiguous"):
        unpack_h3_video_reference(torch.empty((1, 96, 2)).transpose(1, 2), tuple(x.shape))


def test_cpu_registry_is_reference():
    op = KernelRegistry().get_h3_video_patch_op("cpu")
    assert isinstance(op, NativeH3VideoPatchOp)
    assert op.backend_id == "pytorch-h3-video-patch-v1"


@pytest.fixture
def gpu_op():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm GPU required")
    pytest.importorskip("triton")
    op = KernelRegistry().get_h3_video_patch_op("cuda", strict=True)
    assert op.backend_id == "triton-h3-video-patch-v1"
    return op


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_triton_matches_independent_reference(gpu_op, shape, dtype):
    generator = torch.Generator().manual_seed(420)
    cpu_x = torch.randn(shape, generator=generator).to(dtype).requires_grad_()
    gpu_x = cpu_x.detach().to("cuda").requires_grad_()
    expected = pack_h3_video_reference(cpu_x)
    actual = gpu_op.pack(gpu_x)
    assert_bits(actual, expected)
    assert_bits(gpu_op.unpack(actual, shape), cpu_x)

    # Independent unpack input prevents paired errors cancelling each other.
    cpu_tokens = torch.randn(expected.shape, generator=generator).to(dtype).requires_grad_()
    gpu_tokens = cpu_tokens.detach().to("cuda").requires_grad_()
    expected_unpacked = unpack_h3_video_reference(cpu_tokens, shape)
    actual_unpacked = gpu_op.unpack(gpu_tokens, shape)
    assert_bits(actual_unpacked, expected_unpacked)

    grad_tokens = torch.randn(expected.shape, generator=generator).to(dtype)
    grad_latents = torch.randn(shape, generator=generator).to(dtype)
    assert_bits(
        torch.autograd.grad(actual, gpu_x, grad_tokens.to("cuda"))[0],
        torch.autograd.grad(expected, cpu_x, grad_tokens)[0],
    )
    assert_bits(
        torch.autograd.grad(actual_unpacked, gpu_tokens, grad_latents.to("cuda"))[0],
        torch.autograd.grad(expected_unpacked, cpu_tokens, grad_latents)[0],
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_triton_preserves_special_bits(gpu_op, dtype):
    patterns = {
        torch.float32: (0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC01234, 0x7F801234, 1),
        torch.float16: (0, 0x8000, 0x7C00, 0xFC00, 0x7E55, 0x7C55, 1),
        torch.bfloat16: (0, 0x8000, 0x7F80, 0xFF80, 0x7FC5, 0x7F85, 1),
    }
    shape = (2, 24, 3, 2, 4)
    count = torch.tensor(shape).prod().item()
    bits = torch.uint32 if dtype == torch.float32 else torch.uint16
    repeated = torch.tensor(patterns[dtype], dtype=bits).repeat((count + 6) // 7)[:count]
    cpu_x = repeated.view(dtype).reshape(shape)
    gpu_x = cpu_x.to("cuda")
    assert_bits(gpu_op.pack(gpu_x), pack_h3_video_reference(cpu_x))
    assert_bits(gpu_op.unpack(gpu_op.pack(gpu_x), shape), cpu_x)

    token_count = 2 * 6 * 96
    token_bits = torch.tensor(patterns[dtype], dtype=bits).repeat((token_count + 6) // 7)
    cpu_tokens = token_bits[:token_count].view(dtype).reshape(2, 6, 96)
    assert_bits(
        gpu_op.unpack(cpu_tokens.to("cuda"), shape),
        unpack_h3_video_reference(cpu_tokens, shape),
    )


def test_triton_noncontiguous_grad_and_higher_order(gpu_op):
    shape = (2, 24, 3, 6, 10)
    x = torch.randn(shape, device="cuda", requires_grad=True)
    packed = gpu_op.pack(x)
    grad_tokens = torch.randn(2, 96, packed.shape[1], device="cuda").transpose(1, 2)
    grad_tokens.requires_grad_()
    dx = torch.autograd.grad(packed, x, grad_tokens, create_graph=True)[0]
    assert_bits(dx, unpack_h3_video_reference(grad_tokens, shape))
    probe = torch.randn_like(x)
    assert_bits(torch.autograd.grad(dx, grad_tokens, probe)[0], gpu_op.pack(probe))

    tokens = torch.randn(packed.shape, device="cuda", requires_grad=True)
    unpacked = gpu_op.unpack(tokens, shape)
    grad_latents = torch.randn((2, 24, 3, 10, 6), device="cuda").transpose(-1, -2)
    grad_latents.requires_grad_()
    dtokens = torch.autograd.grad(unpacked, tokens, grad_latents, create_graph=True)[0]
    assert_bits(dtokens, pack_h3_video_reference(grad_latents.contiguous()))
    token_probe = torch.randn_like(tokens)
    assert_bits(
        torch.autograd.grad(dtokens, grad_latents, token_probe)[0],
        gpu_op.unpack(token_probe, shape),
    )


def test_triton_batch_invariance_and_stream(gpu_op):
    shape = (1, 24, 3, 6, 10)
    x = torch.randn(shape, device="cuda")
    other = torch.randn_like(x)
    alone = gpu_op.pack(x)
    assert_bits(gpu_op.pack(torch.cat([x, other]))[:1], alone)
    assert_bits(gpu_op.pack(torch.cat([other, x]))[1:], alone)
    assert_bits(gpu_op.pack(x), alone)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        recovered = gpu_op.unpack(gpu_op.pack(x), shape)
    torch.cuda.current_stream().wait_stream(stream)
    assert_bits(recovered, x)
