# H20 上的 video_patch_pack_unpack benchmark

## 环境与代码

- 源码提交：`04b1da8cc71c62b6aeb3073794e8be0b0e6efee3`，`git_dirty=false`。
- 分支：`walker-ai/RL-Kernel:feat/h3-video-patch-pack-unpack`。
- Merlin：单卡 NVIDIA H20，镜像 `hub.byted.org/base/verl-megatron:277872d57ad4cf72887d29e9f3b9ae81`。
- Driver `535.261.03`、CUDA `12.9`、PyTorch `2.9.1`、GPU capability `9.0`。
- 在开发机上确认远端分支指向上述提交。GitHub 克隆传输很慢，因此将本地该提交的 `git archive` 和 Git 元数据传入 Worker 的共享目录；归档 SHA-256 为 `3111b3b10ba7868e7ba7f596d7172619e302be138ad72042ea7de12aaf437dbb`。运行前核对 `git rev-parse HEAD`、干净工作树及 `git config --local` 的 `walker-ai` / `2398833647@qq.com`。

## 方法

编译 CUDA 扩展后，分别运行 CUDA、Triton、PyTorch 实现。每次测量先验证 `pack` 与独立 PyTorch 参考结果一致，且 `unpack(pack(x)) == x`。每种形状使用 BF16、20 次 warmup、100 次重复；脚本用 CUDA events 测量单次 `pack` / `unpack` 的平均耗时，包含 Python 调用与输出分配。不包含首次编译和 Triton JIT 时间。

```bash
MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=9.0 python setup.py build_ext --inplace
python benchmarks/video_patch_pack_unpack.py --device cuda --backend cuda --dtype bfloat16 --batch 1 --frames 32 --height 48 --width 84 --warmup 20 --repeat 100
```

对 `--backend triton`、`--backend pytorch` 及小形状 `--batch 2 --frames 3 --height 6 --width 10` 重复相同命令。每次输出的原始 JSON 与本报告同目录；其中含实际 backend、kernel ID、形状、版本和参考实现耗时。

## 单次测量结果

| 形状 `(B,C,T,H,W)` | Backend | `pack` (µs) | `unpack` (µs) |
| --- | --- | ---: | ---: |
| `(1,24,32,48,84)` | CUDA | 15.96 | 15.90 |
| `(1,24,32,48,84)` | Triton | 32.32 | 43.38 |
| `(1,24,32,48,84)` | PyTorch | 17.26 | 17.29 |
| `(2,24,3,6,10)` | CUDA | 11.64 | 12.77 |
| `(2,24,3,6,10)` | Triton | 33.19 | 43.59 |
| `(2,24,3,6,10)` | PyTorch | 11.91 | 12.36 |

真实形状下 CUDA 和 PyTorch 相差约 1–2 µs；这是单次顺序运行，没有跨轮方差，因此不据此宣称稳定加速。Triton 在这两个 H20 形状上较慢。ROCm 未测量，不能从 H20 结果推断 ROCm 性能。
