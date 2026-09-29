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
op = KernelRegistry().get_op("video_patch_pack_unpack", device=x.device)
tokens = op.pack(x)
recovered = op.unpack(tokens, tuple(x.shape))
```

CPU 使用独立 PyTorch 参考实现；NVIDIA GPU 优先使用独立 CUDA 扩展，
扩展不可用时依次尝试 Triton 和 PyTorch；ROCm 优先使用 Triton，
不可用时回退 PyTorch。测试和 benchmark 直接创建指定的 CUDA 或
Triton 实现以核对各后端；报告记录实际 `backend_id`。

## 代码执行链路

1. PyTorch、Triton 与 CUDA 的 Python 包装层都调用相同的输入检查：
   `C=24`、`B/T/H/W>0`、`H/W` 为偶数、连续内存和允许的 dtype。
2. `pack` 的 Triton 网格第二维枚举 `(b,t)`，第一维枚举该帧的输出值。
   每个 lane 从 token 位置反推 `(c,h,w)`，读取
   `latents[b,c,t,h,w]`，写入连续的 token 地址。`unpack` 对调读写地址。
   输入是 `(B,C,T,H,W)`，所以输入帧与输出帧的内存起点在 `T>1` 时
   不同；不能直接套用 Qwen-Image 的单帧 `batch_base`。
3. CUDA 扩展沿用 #410 的 32×32 shared-memory pair transpose：将水平方向
   相邻的两个值作为一对，以连续读写和块内转置完成 `2×2` patch 排列。
   H3 的输入地址按 `(b,c,t,h,w)` 计算；输出仍按 `(b,t,patch,feature)` 排列。
4. 启动 Triton 内核前把 FP32 看成 `int32`、FP16/BF16 看成 `int16`；
   CUDA 扩展使用等宽整数 pair。GPU 只搬运
   整数位。它保留有符号零、无穷大和 NaN payload，不做浮点运算。
5. backward 再次通过 autograd 的 `apply` 执行逆排列。这样在
   `create_graph=True` 时，二阶梯度仍能沿排列关系传播。

该实现参考 #410 的块内顺序、原始位搬运和逆排列梯度，但不导入尚未
合并的 #410 代码。一个独立的 H3 PyTorch 参考实现用于核对两个 GPU 后端。

## 数值与限制

每个输出元素只读取一个输入元素，不做加法、归约或 dtype 转换。
反向传播使用逆排列，因此前向、反向和往返结果应逐位一致。
支持 FP16、BF16、FP32 的连续张量；`B,T,H,W` 必须为正，
`H,W` 必须为偶数，通道数必须是 24，`B*T<=65535` 以适配网格第二维。
非法输入明确报错。
目前没有跨设备 tensor 参数，也没有索引参数。

## 验证

```bash
python -m pytest tests/test_video_patch_pack_unpack.py -q
mkdocs build --strict -f mkdocs.yaml
python benchmarks/video_patch_pack_unpack.py --device cuda --backend cuda
python benchmarks/video_patch_pack_unpack.py --device cuda --backend triton
```

GPU 测试须在目标 CUDA/ROCm 硬件上运行。PR 中需附设备与运行时版本、
实际 backend、真实形状测量及机器可读结果；未验证的平台不得标记为已支持。
