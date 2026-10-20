---
title: "GEMM 优化：FlashInfer 与 CUTLASS"
date: 2026-10-20T13:10:00+08:00
lastmod: 2026-10-20T16:00:00+08:00
draft: false
description: "从 LLM 线性层的形状出发，结合 FlashInfer SegmentGEMM 和 CUTLASS 理解分块、流水线、小 M 与分组调度"
slug: "llm-gemm-flashinfer-cutlass"
tags: ["llm", "cuda", "flashinfer", "cutlass", "gemm"]
categories: ["LLM 算子优化"]
comments: true
math: true
---

# GEMM 优化：FlashInfer 与 CUTLASS

- [GEMM 优化：FlashInfer 与 CUTLASS](#gemm-优化flashinfer-与-cutlass)
  - [1. Linear 的矩阵形状](#1-linear-的矩阵形状)
  - [2. 分块与数据复用](#2-分块与数据复用)
  - [3. CUTLASS 的分层映射](#3-cutlass-的分层映射)
  - [4. 流水线](#4-流水线)
  - [5. FlashInfer SegmentGEMM 的语义](#5-flashinfer-segmentgemm-的语义)
  - [6. SM80 路径与配置](#6-sm80-路径与配置)
  - [7. 小 M、Split-K 与 grouped GEMM](#7-小-msplit-k-与-grouped-gemm)
  - [8. Epilogue 与低精度](#8-epilogue-与低精度)
  - [9. 调用与验证](#9-调用与验证)
  - [10. 实验变量](#10-实验变量)

从 Linear 的 M/N/K 出发，梳理分块复用、流水线与分组调度，并对照 FlashInfer SegmentGEMM 的 CUTLASS 实现。

源码基线：FlashInfer `v0.2.5`；CUTLASS `v3.8.0` 为补充阅读版本。

## 1. Linear 的矩阵形状

PyTorch 风格权重 `W[N,K]` 对应 `Y=XWᵀ`：

```text
X: [M,K]
W: [N,K]，连续存储时末维 K 连续
Y: [M,N]
```

数学中的右操作数 B 为 `[K,N]`。把连续的 `[N,K]` 权重按转置视图解释，可对应 B 的 column-major 存储，通常不必每次都物理转置整块权重。但下游 kernel 必须支持这种布局与 stride。

LLM 中 M 是本次 token 数，K 是输入通道，N 是输出通道。Prefill 往往有较大的 M；普通小 batch decode 中 M 很小。仅给出“4096×4096 权重”不足以描述一次 GEMM。

## 2. 分块与数据复用

一个线程计算一个输出时，需要遍历 K。相邻输出会反复使用 A 的同一行或 B 的同一列，因此先让一个线程块协同计算 `BM×BN` 的输出块。

每次沿 K 加载 `BK`：

```text
A tile: [BM,BK]
B tile: [BK,BN]
acc:    [BM,BN]
```

这一步完成约 `2*BM*BN*BK` 次浮点运算，输入元素数为 `BM*BK+BK*BN`。忽略输出与其他流量时，每加载一个输入元素对应的运算量为：

$$
\frac{2B_MB_N}{B_M+B_N}.
$$

增大 BM、BN 有助于复用，但也增加 accumulator 和共享内存占用。BK 影响每个 stage 的缓冲大小与循环次数；它不会直接提高上式中的输入复用比例。

## 3. CUTLASS 的分层映射

| 层次 | 职责 | 主要约束 |
|------|------|----------|
| Grid / threadblock | 划分输出矩阵 | tile 数量是否足够、尾块浪费 |
| Warp | 协同计算子块 | 数据布局、寄存器和共享内存访问 |
| MMA 指令 | 执行小矩阵乘加 | 指令支持的数据类型与形状 |
| Epilogue | 缩放、转换与写回 | 合并写入、向量化和融合操作 |

“寄存器里算一块矩阵”描述的是逻辑布局，Tensor Core 操作数分散在参与线程的寄存器中，不能理解为每个线程都存着完整 tile。基本结构可参照 [CUTLASS Efficient GEMM](https://github.com/NVIDIA/cutlass/blob/afa1772203677c5118fcd82537a9c8fefbcc7008/media/docs/efficient_gemm.md)。

全局内存需要尽量合并访问；共享内存布局还要适配 MMA 的取数方式，减少 bank conflict。数学上正确的转置，不代表物理加载就是高效的。

## 4. 流水线

朴素实现先加载、同步、计算、再加载下一块，计算单元可能等待数据。多 stage 流水线在计算当前块时，提前准备后续块。

```text
stage 0: 准备 tile 0
stage 1: 计算 tile 0，同时准备 tile 1
stage 2: 计算 tile 1，同时准备 tile 2
...
```

这里表示逻辑重叠，并非所有架构使用同样指令。Ampere 路径常围绕异步拷贝与 `mma.sync` 组织；Hopper 可以采用 TMA、WGMMA 和 producer/consumer warp group。增加 stage 会消耗更多共享内存，可能降低并发驻留数。

Hopper 的补充入口是 [CUTLASS example 48](https://github.com/NVIDIA/cutlass/blob/afa1772203677c5118fcd82537a9c8fefbcc7008/examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu)：关注 `CollectiveMainloop`、`CollectiveEpilogue`、`GemmUniversal` 和 `GemmUniversalAdapter`。它演示 CUTLASS 3.x 的组成方式，不等于下文 SM80 kernel 的实际调用路径。

## 5. FlashInfer SegmentGEMM 的语义

多个 segment 共用输入输出的大缓冲区，但每段行数可以不同：

```text
seg_indptr = [0, 16, 48, 96]
X          = [96,K]
weights    = [E,N,K]
weight_idx = [2,0,2]

Y[0:16]  = X[0:16]  @ weights[2].T
Y[16:48] = X[16:48] @ weights[0].T
Y[48:96] = X[48:96] @ weights[2].T
```

这里 `batch_size=3` 表示 segment 数，不是 token 数，也不必等于权重数量 E。`weight_column_major=True` 时存储权重形状为 `[E,N,K]`；设为 False 时要求 `[E,K,N]`。

实际接口见 [flashinfer/gemm.py](https://github.com/flashinfer-ai/flashinfer/blob/592b110a78725e034daf3a3c978053e7dae92e5c/flashinfer/gemm.py)。该 wrapper 准备每组问题的 `(M_i,N,K)`、指针和 leading dimension/stride，然后分派到 SM80 或 SM90 路径。

分组前提是输入已经按所需 segment 排好。MoE 的 top-k 路由、token 复制、排列和按权重归并不在这个接口内。

## 6. SM80 路径与配置

固定版本的调用链为：

```text
SegmentGEMMWrapper.run(backend="sm80")
  -> launch_compute_sm80_group_gemm_args
  -> get_gemm_module().cutlass_segment_gemm
  -> CutlassSegmentGEMMRun
  -> cutlass::gemm::device::GemmGrouped
```

设备配置见 [include/flashinfer/gemm/group_gemm.cuh](https://github.com/flashinfer-ai/flashinfer/blob/592b110a78725e034daf3a3c978053e7dae92e5c/include/flashinfer/gemm/group_gemm.cuh)。将模板参数整理出来：

| 参数 | 该路径配置 |
|------|------------|
| A / B / 输出 | A row-major；B 随权重布局分派；输出 row-major |
| 累加类型 | FP32 |
| 架构与指令类别 | SM80 / TensorOp |
| Threadblock shape | `128×128×32` |
| Warp shape | `64×64×32` |
| Instruction shape | `16×8×16` |
| A/B 对齐粒度 | 8 个元素 |
| Pipeline stages | 按每 SM 共享内存上限选择 4 或 2 |

这是该版本某条实现的配置，不是任意 dtype、尺寸和 GPU 的最佳参数。读取时尤其注意两个具体细节：

- `GemmGrouped::Arguments` 的 `threadblock_count` 在这里直接给了 4。不要把它当作已经按 GPU SM 数自动充分调优的调度；性能实验应检查实际 grid 和每个 block 的工作量。
- Epilogue 设置 `alpha=1,beta=1`，输入 C 与输出 D 都指向 y。Python wrapper 预先用 `torch.zeros` 创建 y，第一次运行因此得到纯矩阵乘结果。若修改底层接口并直接复用旧输出，要重新处理清零或 beta，否则会把旧值加进去。

这也说明只提取一段模板配置并不能复现完整行为，输出初始化和参数准备同样影响正确性及耗时。

## 7. 小 M、Split-K 与 grouped GEMM

普通输出分块的 tile 数约为：

$$
\left\lceil\frac{M}{B_M}\right\rceil
\left\lceil\frac{N}{B_N}\right\rceil.
$$

M 很小时，输出 tile 数可能不足以占满 GPU，大 BM 还会浪费行方向工作。可选策略包括缩小 tile、专用 GEMV、小 M GEMM、增加请求 batch 或将多个问题分组调度。

Split-K 沿 K 把同一个输出块拆给多个 block，随后归约部分和。它增加并行度，但增加中间结果与归约流量，并改变浮点累加次序，不能保证比原始方案快。

Grouped GEMM 合并多组独立问题的调度，不是 Split-K。它减少多次独立发射的开销并有机会改善负载均衡，但仍可能因 segment 极短、长度差异大或调度配置不佳而受限。

## 8. Epilogue 与低精度

`D=alpha*AB+beta*C` 是常见 epilogue。把 bias、激活或量化变换接在 accumulator 后，有机会减少中间张量的写回与重读；是否融合需要确认该具体接口支持什么。

FP16/BF16 输入加 FP32 累加仍可能在最终转换时损失精度。FP8/INT8 等路径还要指定 scale 的粒度和布局；INT4 权重可能是压缩存储后反量化计算，不能根据“四位”推断实际 Tensor Core 指令。

性能对比应报告量化、打包和重排是否计入。一次性权重预处理可以单独列出；随 token 重复执行的预处理需要计入逐 token 延迟。

## 9. 调用与验证

在已配置 FlashInfer v0.2.5 的 CUDA 环境中：

```python
import torch
from flashinfer import SegmentGEMMWrapper

workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
wrapper = SegmentGEMMWrapper(workspace, backend="sm80")
x = torch.randn(96, 128, device="cuda", dtype=torch.float16)
w = torch.randn(3, 256, 128, device="cuda", dtype=torch.float16)
indptr = torch.tensor([0, 16, 48, 96], device="cuda", dtype=torch.int64)
y = wrapper.run(x, w, 3, True, seg_indptr=indptr)
reference = torch.cat([
    x[:16].float() @ w[0].float().T,
    x[16:48].float() @ w[1].float().T,
    x[48:].float() @ w[2].float().T,
])
torch.testing.assert_close(y.float(), reference, rtol=1e-2, atol=2e-2)
```

该路径需要兼容 SM80 指令及相应 dtype/对齐要求的 GPU。示例显式传 `int64` 的 `seg_indptr`，不依赖隐式计算长度前缀；使用 `seg_lens` 时也应核对转换后的 dtype 和版本实现。

完整脚本见[配套示例](https://github.com/huluhuluu/llm-kernel-notes/blob/main/examples/verify_kernels.py)。它测量预热后的 API 调用，包含 wrapper 内的参数构造和输出分配，不冒充纯 GEMM kernel 时间。

## 10. 实验变量

| 维度 | 建议记录或扫描 |
|------|----------------|
| Shape | M=1/8/32/128/1024；实际模型的 N/K |
| 分组 | segment 数、总 token 数、每段长度分布 |
| 数值 | FP16/BF16、参考精度、最大绝对误差 |
| 布局 | 连续性、转置视图、对齐、打包耗时 |
| 调度 | backend、tile、stage、grid、workspace |
| 计时 | 初始化、预热、API 调用、kernel、端到端 |

对大矩阵报告 TFLOP/s，对小 M 还应报告微秒延迟。比较对象必须完成同样的分组、转置与输出语义；仅比较一次 `torch.matmul` 和一整套路由流程没有解释力。
