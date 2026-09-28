# MiniMax-H3 视频 patch pack/unpack

## 作用与边界

把视频 latent 的 `(B,24,T,H,W)` 重排为 `(B,S,96)`，其中
`S=T*(H/2)*(W/2)`。patch 大小固定为 `(1,2,2)`。`unpack` 做精确逆变换。
它不执行视频输入投影，也不把视频、文本、音频行组装成共同序列。

## 顺序契约

token 顺序是 batch、时间帧、空间 patch 行、空间 patch 列；每个 token
内部是通道、patch 内行、patch 内列。具体映射为：

```text
tokens[b, t*(H/2)*(W/2) + hi*(W/2) + wi, c*4 + dy*2 + dx]
    = latents[b, c, t, 2*hi + dy, 2*wi + dx]
```

上游 Diffusers 在相同重排后展平 batch 和 token 维，得到 `(B*S,96)`；
这里保留 batch 维，以符合 [#420](https://github.com/RL-Align/RL-Kernel/issues/420)
的工作项接口。与上游对比时使用 `tokens.reshape(B*S,96)`。

## 接口与后端

```python
op = KernelRegistry().get_h3_video_patch_op(x.device, strict=True)
tokens = op.pack(x)
recovered = op.unpack(tokens, tuple(x.shape))
```

CPU 使用独立 PyTorch 参考实现；CUDA/ROCm 的 strict 路径使用 Triton。
Triton 不可用时 strict 路径报错，不静默退回 PyTorch。`strict=False` 才允许
PyTorch fallback。记录 `op.backend_id` 可确认实际后端。普通 `get_op`
保留仓库现有的按优先级 fallback 行为；需要严格验证时应使用上面的显式入口。

## 数值与限制

每个输出元素只读取一个输入元素，不做加法、归约或 dtype 转换。
反向传播使用逆排列，因此前向、反向和往返结果应逐位一致。
支持 FP16、BF16、FP32 的连续张量；`B,T,H,W` 必须为正，
`H,W` 必须为偶数，通道数必须是 24。非法输入明确报错。
目前没有跨设备 tensor 参数，也没有索引参数。

## 验证

```bash
python -m pytest tests/test_h3_video_patch.py -q
mkdocs build --strict -f mkdocs.yaml
```

GPU 测试须在目标 CUDA/ROCm 硬件上运行。PR 中需附设备与运行时版本、
实际 backend、真实形状测量及机器可读结果；未验证的平台不得标记为已支持。
