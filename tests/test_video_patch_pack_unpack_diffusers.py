# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Compare H3 pack and its inverse with the pinned Diffusers implementation."""

import ast
from importlib.metadata import distribution
from pathlib import Path

import pytest
import torch

from rl_engine.kernels.ops.pytorch.video_patch_pack_unpack import NativeVideoPatchPackUnpackOp

DIFFUSERS_VERSION = "0.40.0"
SHAPES = ((2, 24, 3, 6, 10), (1, 24, 32, 48, 84))
DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def assert_bits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.detach().contiguous().cpu().view(torch.uint8),
        expected.detach().contiguous().cpu().view(torch.uint8),
    )


@pytest.fixture(scope="module")
def diffusers_pack():
    installed = distribution("diffusers")
    assert installed.version == DIFFUSERS_VERSION, (
        f"Expected diffusers=={DIFFUSERS_VERSION}; found {installed.version}"
    )
    source = Path(
        installed.locate_file("diffusers/modular_pipelines/minimax_h3/before_denoise.py")
    )
    functions = [
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "patchify_video_latents"
    ]
    assert len(functions) == 1, f"Expected one upstream patchify_video_latents in {source}"

    # Importing the full H3 module also imports unrelated pipeline/LoRA code.
    # Execute the installed upstream function definition itself, unchanged.
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["patchify_video_latents"]


@pytest.fixture(params=("pytorch", "triton", "cuda"))
def op(request):
    if request.param == "pytorch":
        return NativeVideoPatchPackUnpackOp(), "cpu"

    assert torch.cuda.is_available(), "GPU required for explicit Diffusers backend test"
    if request.param == "triton":
        from rl_engine.kernels.ops.triton.video_patch_pack_unpack import (
            TritonVideoPatchPackUnpackOp,
        )

        return TritonVideoPatchPackUnpackOp(), "cuda"

    assert torch.version.hip is None, "CUDA extension requires an NVIDIA GPU"
    from rl_engine.kernels.ops.cuda.video_patch_pack_unpack import CudaVideoPatchPackUnpackOp

    return CudaVideoPatchPackUnpackOp(), "cuda"


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_against_diffusers_pack_and_gradient(diffusers_pack, op, shape, dtype):
    kernel, device = op
    generator = torch.Generator().manual_seed(420)
    values = torch.randn(shape, generator=generator).to(device=device, dtype=dtype)
    upstream_x = values.detach().requires_grad_()
    kernel_x = values.detach().requires_grad_()

    upstream = diffusers_pack(upstream_x, (1, 2, 2))
    actual = kernel.pack(kernel_x)
    assert_bits(actual.reshape(-1, 96), upstream)
    assert_bits(kernel.unpack(upstream.reshape(actual.shape), shape), values)

    grad_rows = torch.randn(upstream.shape, generator=generator).to(device=device, dtype=dtype)
    upstream_dx = torch.autograd.grad(upstream, upstream_x, grad_rows)[0]
    kernel_dx = torch.autograd.grad(actual, kernel_x, grad_rows.reshape(actual.shape))[0]
    assert_bits(kernel_dx, upstream_dx)
