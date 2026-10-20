# LLM 算子优化

围绕 NVIDIA GPU 上的 Decoder 推理，结合 FlashInfer 和 CUTLASS 整理算子、GEMM 与 Attention 优化。

- `index.md`：系列目录与版本约定。
- `blog/llm-operators-flashinfer/index.md`：算子与张量形状。
- `blog/llm-gemm-flashinfer-cutlass/index.md`：GEMM 分块与分组 GEMM。
- `blog/llm-attention-flashinfer/index.md`：Attention 与 KV Cache。
- `examples/verify_kernels.py`：独立运行的 GPU 正确性与计时示例。

源码基线：FlashInfer v0.2.5（592b110a78725e034daf3a3c978053e7dae92e5c）。CUTLASS v3.8.0 是补充阅读基线，并不声明它是该 FlashInfer 版本内置依赖的版本。

## 运行示例

在已配置相应 CUDA / PyTorch / FlashInfer v0.2.5 的 Linux GPU 环境中运行：

```bash
python examples/verify_kernels.py --case attention
python examples/verify_kernels.py --case gemm
```

默认 FP16；GEMM 示例显式选择 SM80 路径，需要相应 Tensor Core 支持。脚本打印设备、版本、误差与调用时间；不包含安装、模型推理或性能保证。计时包含 Python API 内部的设备分配和元数据构造，不等于纯 kernel 时间。

父博客仅挂载 `blog/`，`examples/` 不参与内容渲染。文章中的链接使用固定 commit，文末保留原始资料入口。
