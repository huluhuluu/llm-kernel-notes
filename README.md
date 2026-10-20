# LLM 算子优化

结合 FlashInfer 与 CUTLASS，记录 Decoder 算子、GEMM 分块和 Attention 的实现。

## 目录

| 文章 | 说明 | 状态 |
|------|------|------|
| [LLM 算子梳理：从 Decoder 到 FlashInfer](blog/llm-operators-flashinfer/index.md) | 张量形状、prefill/decode、算子融合与源码入口 | ✅ 完成 |
| [GEMM 优化：FlashInfer 与 CUTLASS](blog/llm-gemm-flashinfer-cutlass/index.md) | 分块复用、流水线、SegmentGEMM 与小 M 调度 | 🚧 待实测 |
| [Attention 优化：FlashInfer 的 Prefill 与 Decode](blog/llm-attention-flashinfer/index.md) | Online Softmax、Split-KV、Paged KV 与验证方法 | 🚧 待实测 |

GEMM 与 Attention 的源码说明和验证脚本已整理，GPU 实测数据待补充。

## 阅读顺序

1. 算子梳理：明确 Decoder 各步的输入输出与主要开销。
2. GEMM：从矩阵形状读到 CUTLASS 分块、流水线和分组调度。
3. Attention：区分 prefill 与 decode，梳理状态归约和 KV Cache 布局。

## 源码版本

| 项目 | 版本 | 用途 |
|------|------|------|
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer/tree/592b110a78725e034daf3a3c978053e7dae92e5c) | v0.2.5 | 算子、SegmentGEMM 与 Attention 源码 |
| [CUTLASS](https://github.com/NVIDIA/cutlass/tree/afa1772203677c5118fcd82537a9c8fefbcc7008) | v3.8.0 | GEMM 分层结构与 Hopper 示例 |

版本用于固定源码阅读路径，FlashInfer 内部依赖以对应仓库为准。

## 示例运行

环境：Linux、CUDA、PyTorch、FlashInfer v0.2.5。GEMM 示例使用 SM80 路径。

```bash
python examples/verify_kernels.py --case attention
python examples/verify_kernels.py --case gemm
```

[verify_kernels.py](https://github.com/huluhuluu/llm-kernel-notes/blob/main/examples/verify_kernels.py) 输出设备、版本、误差和预热后的 API 耗时；计时包含接口内部的分配与元数据构造。

## 相关资料

- [LLM 核心算子手撕](https://github.com/huluhuluu/coding-interview-notes/blob/main/blog/llm-operator-implementations/index.md)
- [FlashInfer 文档](https://docs.flashinfer.ai/)
- [CUTLASS 仓库](https://github.com/NVIDIA/cutlass)
