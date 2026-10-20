---
title: "Attention 优化：FlashInfer 的 Prefill 与 Decode"
date: 2026-10-20T13:20:00+08:00
lastmod: 2026-10-20T16:00:00+08:00
draft: false
description: "从 Online Softmax 推导出发，结合 FlashInfer 源码解释 Prefill、Decode、Split-KV、Paged KV 与正确性验证"
slug: "llm-attention-flashinfer"
tags: ["llm", "cuda", "flashinfer", "attention", "kv-cache"]
categories: ["LLM 算子优化"]
comments: true
math: true
---

# Attention 优化：FlashInfer 的 Prefill 与 Decode

- [Attention 优化：FlashInfer 的 Prefill 与 Decode](#attention-优化flashinfer-的-prefill-与-decode)
  - [1. Shape、head 映射与 mask](#1-shapehead-映射与-mask)
  - [2. 朴素 Attention 的额外流量](#2-朴素-attention-的额外流量)
  - [3. Online Softmax 的状态](#3-online-softmax-的状态)
  - [4. Prefill：在分块矩阵乘中维护状态](#4-prefill在分块矩阵乘中维护状态)
  - [5. Decode：KV 流量与并行度](#5-decodekv-流量与并行度)
  - [6. Split-KV 的状态如何合并](#6-split-kv-的状态如何合并)
  - [7. Paged KV：逻辑连续，物理分散](#7-paged-kv逻辑连续物理分散)
  - [8. plan 与 run 的边界](#8-plan-与-run-的边界)
  - [9. 最小正确性验证](#9-最小正确性验证)
  - [10. 性能比较需要保持哪些条件一致](#10-性能比较需要保持哪些条件一致)

FlashAttention 减少注意力中间矩阵的读写，Paged KV 管理缓存页，Split-KV 沿 KV 长度拆分计算。下文结合 FlashInfer 区分 prefill 与 decode 的实现。

源码基线：FlashInfer `v0.2.5`；示例为无 dropout 的推理 Attention。

## 1. Shape、head 映射与 mask

采用 `NHD` 布局：

```text
Prefill Q: [Lq,Hq,d]
Decode  Q: [Hq,d]
K / V:     [Lkv,Hkv,d]
```

GQA 中 `group_size=Hq/Hkv`，连续一组 Query heads 共享一个 KV head。`Hq` 必须能被 `Hkv` 整除。

Prefill 初始 prompt 通常 `Lq=Lkv`；追加一段新 token 时可以 `Lq<Lkv`。对追加场景，下标 q 对应的绝对位置是 `Lkv-Lq+q`，因果条件为：

$$
k\le L_{kv}-L_q+q.
$$

因此非方阵 causal mask 需要特别确认对齐方式。普通 decode 的 KV 已包含可见历史和当前 token 时，最后一个 Query 可访问所有给定 K/V；通常不应再误套一个“第一行只看第一列”的 mask。

## 2. 朴素 Attention 的额外流量

先计算 `S=QKᵀ/√d`，再得到 `P=softmax(S)`，最后 `O=PV`。显式保存 S/P 时，中间张量规模为 `Hq*Lq*Lkv`。

例如 Hq=32、Lq=Lkv=4096、每元素 2 字节，一个完整分数矩阵就需要 1 GiB；S 和 P 若分别保存还会增加占用。因果 mask 减少有效元素，但普通稠密存储仍可能保留整个矩阵。

[FlashAttention](https://arxiv.org/abs/2205.14135) 的思路是分块计算并维护 Softmax 状态，避免把完整 S/P 写到 HBM。稠密 Attention 的算术复杂度仍随 `Lq*Lkv` 增长；节省中间读写不等于把精确全 Attention 变成线性时间。

## 3. Online Softmax 的状态

对某一行 logits，把已经处理的数据保存为三项：

$$
m=\max_i s_i,\qquad
\ell=\sum_i e^{s_i-m},\qquad
u=\sum_i e^{s_i-m}v_i.
$$

最终输出 `o=u/ℓ`。新块的 logits 为 t，取 `mb=max(t)`，更新：

$$
m'=\max(m,m_b),\quad \alpha=e^{m-m'},
$$
$$
\ell'=\alpha\ell+\sum_j e^{t_j-m'},\qquad
u'=\alpha u+\sum_j e^{t_j-m'}v_j.
$$

旧状态随最大值变化重新缩放，因此无需保存此前每个概率。初始状态为 `m=-∞,ℓ=0,u=0`；整个块全被 mask 时应跳过或显式处理，避免 `-∞-(-∞)` 产生 NaN。

下面是验证状态更新的 Python 参考，不是高性能 kernel：

```python
import math

def online_attention(scores, values, block_size):
    # scores 非空且有限；values 与 scores 等长，每行维度一致
    m, denom = -math.inf, 0.0
    numerator = [0.0] * len(values[0])
    for start in range(0, len(scores), block_size):
        stop = min(start + block_size, len(scores))
        new_m = max(m, max(scores[start:stop]))
        alpha = math.exp(m - new_m)
        denom *= alpha
        numerator = [x * alpha for x in numerator]
        for j in range(start, stop):
            p = math.exp(scores[j] - new_m)
            denom += p
            for c, value in enumerate(values[j]):
                numerator[c] += p * value
        m = new_m
    return [x / denom for x in numerator]
```

要求 `block_size>0`。真实 kernel 会使用向量化、warp 归约和寄存器分片，不能用这个 Python 循环的速度评价 FlashAttention。

## 4. Prefill：在分块矩阵乘中维护状态

一个 Query tile 与若干 KV tile 依次计算：

```text
加载 Q tile
对每个可见 KV tile：
    加载 K/V
    算 QKᵀ，并应用 scale / mask
    更新 m、ℓ，重新缩放已有 accumulator
    用当前概率块乘 V 并累加
最终归一化并写回 O
```

这两次矩阵乘的布局要求不同：QKᵀ 产生按行做 Softmax 的片段，概率块还要变成 PV 可消费的布局。融合的难点包括寄存器压力、共享内存空间、归约同步和数据搬运重叠。

固定版本从 [prefill.py](https://github.com/flashinfer-ai/flashinfer/blob/592b110a78725e034daf3a3c978053e7dae92e5c/flashinfer/prefill.py) 进入，`backend="fa2"` 对应的设备实现可从 [attention/prefill.cuh](https://github.com/flashinfer-ai/flashinfer/blob/592b110a78725e034daf3a3c978053e7dae92e5c/include/flashinfer/attention/prefill.cuh) 阅读。重点搜索 MMA、mask、Softmax 状态更新和输出片段，而不是只看 kernel 函数长度。

选择 `backend="auto"` 时还要追实际分派，不能把 FA2 文件里的结论直接套到 FA3 或其他版本的实现。

## 5. Decode：KV 流量与并行度

单请求普通 decode 只有一行 Q，每个 KV head 需要读取长度为 L 的 K/V。理想情况下，每层每个请求读取 KV 的字节量约为：

$$
\operatorname{bytes}_{KV}\approx 2LH_{kv}ds.
$$

2 对应 K 和 V，s 是每元素字节数。这个估计假定共享 KV 能被充分复用，实际还有重读、缓存命中、分页索引和量化 scale。

如果一个 block 负责很长的 KV 序列，但请求和 head 数较少，整个 GPU 可能只有少量 block 工作。Split-KV 将 KV 长度分段，让多个 block 分别产生局部状态，再合并。

固定版本的 [attention/decode.cuh](https://github.com/flashinfer-ai/flashinfer/blob/592b110a78725e034daf3a3c978053e7dae92e5c/include/flashinfer/attention/decode.cuh) 中可以查找 `partition_kv`、chunk 范围和临时输出；调度相关逻辑见 [scheduler.cuh](https://github.com/flashinfer-ai/flashinfer/blob/592b110a78725e034daf3a3c978053e7dae92e5c/include/flashinfer/attention/scheduler.cuh)。

分段增加了 workspace、状态写回和第二次归约。短序列或较大 batch 已有充足并行度时，继续细分可能更慢。

## 6. Split-KV 的状态如何合并

两个分段分别产生 `(ma,ℓa,ua)` 与 `(mb,ℓb,ub)`，取 `m=max(ma,mb)`：

$$
\ell=e^{m_a-m}\ell_a+e^{m_b-m}\ell_b,\qquad
u=e^{m_a-m}u_a+e^{m_b-m}u_b.
$$

再计算 `u/ℓ`。这说明每个分段的输出不能直接等权平均。

另一种表示是归一化输出 o 和 log-sum-exp L。若 L 使用自然对数，则可用 `exp(La-L)`、`exp(Lb-L)` 作为权重。实现可能使用以 2 为底的状态，此时应对应使用 `exp2`；不能混用。可结合 [Attention States 教程](https://docs.flashinfer.ai/tutorials/recursive_attention.html) 和固定版本 [cascade.cuh](https://github.com/flashinfer-ai/flashinfer/blob/592b110a78725e034daf3a3c978053e7dae92e5c/include/flashinfer/attention/cascade.cuh) 阅读。

## 7. Paged KV：逻辑连续，物理分散

Paged KV 把每个请求的序列按 page size P 切块，page table 保存逻辑页到物理页的映射。常见元数据包括：

| 字段 | 含义 |
|------|------|
| `indptr` | 每个请求在页索引数组中的起止位置，长度 B+1 |
| `indices` | 各请求按逻辑顺序使用的物理页编号 |
| `last_page_len` | 最后一页有效 token 数 |
| `page_size` | 一页能容纳的 token 数 |

例如 `P=16,indptr=[0,3],indices=[5,2,8],last_page_len=[7]`，这个请求的有效 KV 长度为 `2*16+7=39`。物理页 5、2、8 的顺序不必连续。

它改善动态分配、碎片管理与前缀复用，不会自动减少精确稠密 Attention 必须访问的历史 KV 数量。分页引入寻址开销，page size 还影响尾页浪费和加载粒度。布局细节参见 [KV-Cache Layout](https://docs.flashinfer.ai/tutorials/kv_layout.html)。

## 8. plan 与 run 的边界

批量接口可从 `BatchDecodeWithPagedKVCacheWrapper` 或 `BatchPrefillWithPagedKVCacheWrapper` 进入。通常先用 `plan` 提交元数据与配置，再由 `run` 传入当前 Q 和 KV 数据。

复用计划前要确认请求数、页表、长度、head 数、dtype、mask 等仍满足已规划的条件。不断增长的 decode 序列与动态 batch 可能要求更新元数据或重新 plan；不能假定一次 plan 永久有效。

CUDA Graph 对地址和执行结构有额外约束，应使用库提供的相应模式。将 plan 排除在 kernel benchmark 外可以帮助定位设备计算开销，但服务端到端测量仍要计入实际重复发生的规划成本。

## 9. 最小正确性验证

先从相同 head 数、连续 KV、无 RoPE 和无量化开始；再逐步加入 GQA、分页、mask 和位置编码。

```python
import math
import torch
import flashinfer

length, heads, dim = 257, 8, 64
q = torch.randn(heads, dim, device="cuda", dtype=torch.float16)
k = torch.randn(length, heads, dim, device="cuda", dtype=torch.float16)
v = torch.randn_like(k)
out = flashinfer.single_decode_with_kv_cache(q, k, v, kv_layout="NHD")
scores = torch.einsum("hd,lhd->hl", q.float(), k.float()) / math.sqrt(dim)
reference = torch.einsum("hl,lhd->hd", scores.softmax(-1), v.float())
torch.testing.assert_close(out.float(), reference, rtol=1e-2, atol=1e-2)
```

参考实现使用 FP32 中间结果。误差阈值需要结合输入尺度和 dtype，不要求浮点逐位一致。完整的 decode、causal prefill、GQA 检查和 CUDA Event 计时见[配套示例](https://github.com/huluhuluu/llm-kernel-notes/blob/main/examples/verify_kernels.py)。

## 10. 性能比较需要保持哪些条件一致

固定 GPU、版本、dtype、query/KV head 数、head dimension、KV 布局、mask 和 backend。Prefill 扫 query 长度，decode 扫 batch 与 KV 长度；变长请求还要保留长度分布。

首次 JIT、数据生成、plan 和 `run` 分别记录，预热后使用 CUDA Event 计时。对 decode 可估算有效 KV 带宽，但应明确它是按理想流量计算；若数据驻留在缓存，不能把超过 HBM 标称带宽的结果解释为物理 HBM 带宽。

最后回到服务指标：prefill 关注首 token 延迟与吞吐，decode 关注逐 token 延迟和 batch 吞吐。单个 kernel 的加速比例不能直接替代完整模型的收益。
