---
title: "LLM 算子优化"
date: 2026-10-20T13:00:00+08:00
draft: false
description: "以 FlashInfer 为主线，结合 CUTLASS 理清 LLM 算子、GEMM 和 Attention 优化"
slug: "llm-kernel-notes"
tags: ["llm", "cuda", "flashinfer", "cutlass"]
categories: ["LLM 算子优化"]
build:
  list: never
comments: true
math: true
---

# LLM 算子优化

这一组从 Decoder 推理的一层计算出发，先确定张量形状和主要开销，再进入 GEMM 与 Attention 的实现。源码主线选用 FlashInfer：它同时包含服务场景的 Attention、分组 GEMM、归一化与激活，便于把单个 kernel 放回实际推理流程。

| 文章 | 内容 |
|------|------|
| [LLM 算子梳理：从 Decoder 到 FlashInfer](/p/llm-operators-flashinfer/) | 张量形状、prefill/decode、算子融合与源码入口 |
| [GEMM 优化：FlashInfer 与 CUTLASS](/p/llm-gemm-flashinfer-cutlass/) | 分块、数据复用、流水线、grouped GEMM 与小 M |
| [Attention 优化：FlashInfer 的 Prefill 与 Decode](/p/llm-attention-flashinfer/) | Online Softmax、Split-KV、Paged KV 与验证方法 |

源码解析固定到 FlashInfer `v0.2.5`（`592b110a78725e034daf3a3c978053e7dae92e5c`），CUTLASS 补充示例固定到 `v3.8.0`（`afa1772203677c5118fcd82537a9c8fefbcc7008`）。选择这些版本是为了固定可核对的教学路径，不代表最新发布版本；实际部署需要重新核对所用版本的 API 和硬件支持。

配套的 `examples/verify_kernels.py` 提供正确性检查与 CUDA Event 计时入口，需要 Linux / CUDA / PyTorch 及对应 FlashInfer 环境。本系列目前不提供实测加速比。
