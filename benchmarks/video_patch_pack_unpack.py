# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Reproducible H3 video patch benchmark with explicit backend selection."""

from __future__ import annotations

import argparse
import json
import subprocess
import time

import torch

from rl_engine.kernels.ops.pytorch.video_patch_pack_unpack import (
    NativeVideoPatchPackUnpackOp,
)


def _time_ms(fn, *, warmup: int, repeat: int, device: str) -> float:
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / repeat
    start_cpu = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - start_cpu) * 1000 / repeat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--backend", choices=("pytorch", "triton", "cuda"))
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--width", type=int, default=84)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()
    if args.backend is None:
        args.backend = "pytorch" if args.device == "cpu" else "triton"
    if args.repeat < 1 or args.warmup < 0:
        parser.error("repeat must be positive and warmup nonnegative")
    if args.device == "cpu" and args.backend != "pytorch":
        parser.error("CPU benchmark requires --backend pytorch")
    if args.backend == "cuda" and torch.version.hip is not None:
        parser.error("CUDA backend requires an NVIDIA GPU")
    dtype = getattr(torch, args.dtype)
    shape = (args.batch, 24, args.frames, args.height, args.width)
    x = torch.randn(shape, device=args.device, dtype=dtype)
    reference = NativeVideoPatchPackUnpackOp()
    if args.backend == "cuda":
        from rl_engine.kernels.ops.cuda.video_patch_pack_unpack import (
            CudaVideoPatchPackUnpackOp,
        )

        op = CudaVideoPatchPackUnpackOp()
    elif args.backend == "triton":
        from rl_engine.kernels.ops.triton.video_patch_pack_unpack import (
            TritonVideoPatchPackUnpackOp,
        )

        op = TritonVideoPatchPackUnpackOp()
    else:
        op = reference
    kernel_ids = {
        "cuda": "video_patch_permute",
        "triton": "_copy_patch",
        "pytorch": "torch.reshape_permute",
    }
    tokens = op.pack(x)
    assert torch.equal(op.unpack(tokens, shape), x)
    assert torch.equal(tokens, reference.pack(x))
    timings = {
        "pack_ms": _time_ms(lambda: op.pack(x), warmup=args.warmup,
                            repeat=args.repeat, device=args.device),
        "unpack_ms": _time_ms(lambda: op.unpack(tokens, shape), warmup=args.warmup,
                              repeat=args.repeat, device=args.device),
        "reference_pack_ms": _time_ms(lambda: reference.pack(x), warmup=args.warmup,
                                       repeat=args.repeat, device=args.device),
        "reference_unpack_ms": _time_ms(lambda: reference.unpack(tokens, shape),
                                         warmup=args.warmup, repeat=args.repeat,
                                         device=args.device),
    }
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        sha = "unknown"
        dirty = None
    print(json.dumps({
        "schema_version": "rlkernel.video_patch_pack_unpack.benchmark.v1",
        "git_sha": sha,
        "git_dirty": dirty,
        "requested_backend": args.backend,
        "actual_backend": op.backend_id,
        "kernel_id": kernel_ids[args.backend],
        "device": args.device,
        "device_name": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "hip_version": torch.version.hip,
        "dtype": args.dtype,
        "shape": shape,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "timings": timings,
    }, indent=2))


if __name__ == "__main__":
    main()
