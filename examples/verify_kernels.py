"""FlashInfer v0.2.5 correctness examples and warmed CUDA API timing."""

import argparse
import importlib.metadata
import json
import math


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["attention", "gemm", "all"], default="all")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        parser.error("warmup and repeats must be positive")

    import torch
    import flashinfer

    version = importlib.metadata.version("flashinfer-python")
    if version.split("+")[0] != "0.2.5":
        raise RuntimeError("This example targets flashinfer-python==0.2.5, got " + version)
    if not torch.cuda.is_available():
        raise RuntimeError("A supported CUDA GPU is required")

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    print(json.dumps({
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flashinfer": version,
        "dtype": "float16",
        "timing_scope": "warmed API calls; includes internal GPU allocation/metadata work",
    }))

    def check_and_time(name, call, reference, **metadata):
        actual = call()
        error = (actual.float() - reference).abs().max().item()
        torch.testing.assert_close(actual.float(), reference, rtol=1e-2, atol=2e-2)
        for _ in range(args.warmup):
            call()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.repeats):
            call()
        end.record()
        end.synchronize()
        print(json.dumps({
            "case": name,
            "max_abs_error": error,
            "mean_api_us": start.elapsed_time(end) * 1000 / args.repeats,
            **metadata,
        }))

    with torch.inference_mode():
        if args.case in ("attention", "all"):
            heads, dim = 8, 64
            for kv_heads in (8, 2):
                kv_len = 257
                k = torch.randn(kv_len, kv_heads, dim, device="cuda", dtype=torch.float16)
                v = torch.randn_like(k)
                # Expand only in the reference; kernel receives shared KV heads.
                kr = k.float().repeat_interleave(heads // kv_heads, dim=1)
                vr = v.float().repeat_interleave(heads // kv_heads, dim=1)
                q = torch.randn(heads, dim, device="cuda", dtype=torch.float16)
                scores = torch.einsum("hd,lhd->hl", q.float(), kr) / math.sqrt(dim)
                ref = torch.einsum("hl,lhd->hd", scores.softmax(-1), vr)
                check_and_time(
                    "decode",
                    lambda: flashinfer.single_decode_with_kv_cache(q, k, v, kv_layout="NHD"),
                    ref,
                    q_shape=list(q.shape), kv_shape=list(k.shape),
                )

                for q_len in (257, 17):
                    q = torch.randn(q_len, heads, dim, device="cuda", dtype=torch.float16)
                    scores = torch.einsum("qhd,khd->hqk", q.float(), kr) / math.sqrt(dim)
                    q_pos = torch.arange(q_len, device="cuda") + kv_len - q_len
                    k_pos = torch.arange(kv_len, device="cuda")
                    mask = k_pos[None, :] <= q_pos[:, None]
                    scores.masked_fill_(~mask[None, :, :], -torch.inf)
                    ref = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vr)
                    check_and_time(
                        "causal_prefill",
                        lambda: flashinfer.single_prefill_with_kv_cache(
                            q, k, v, causal=True, kv_layout="NHD", backend="fa2"
                        ),
                        ref,
                        q_shape=list(q.shape), kv_shape=list(k.shape), backend="fa2",
                    )

        if args.case in ("gemm", "all"):
            from flashinfer import SegmentGEMMWrapper

            if torch.cuda.get_device_capability()[0] < 8:
                raise RuntimeError("The GEMM example requires SM80-compatible instructions")
            workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
            wrapper = SegmentGEMMWrapper(workspace, backend="sm80")
            boundaries = [0, 16, 48, 96]
            indptr = torch.tensor(boundaries, dtype=torch.int64, device="cuda")
            x = torch.randn(96, 128, dtype=torch.float16, device="cuda")
            weights = torch.randn(3, 256, 128, dtype=torch.float16, device="cuda")
            indices = torch.tensor([2, 0, 2], dtype=torch.int64, device="cuda")
            ref = torch.cat([
                x[boundaries[i]:boundaries[i + 1]].float() @ weights[j].float().T
                for i, j in enumerate((2, 0, 2))
            ])
            check_and_time(
                "segment_gemm",
                lambda: wrapper.run(
                    x, weights, 3, True, seg_indptr=indptr, weight_indices=indices
                ),
                ref,
                x_shape=list(x.shape), weights_shape=list(weights.shape),
                seg_indptr=boundaries, weight_indices=[2, 0, 2], backend="sm80",
            )


if __name__ == "__main__":
    main()
