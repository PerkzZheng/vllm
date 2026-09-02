#!/usr/bin/env python3
"""Report CUDA-kernel time inside one cold-L2 QSA attention case.

This diagnostic intentionally reuses the accepted standalone benchmark's
input construction and metadata. It profiles eager launches so PyTorch can
attribute CUDA time to the PrimTS main/reduction kernels and the Triton
baseline independently; accepted latency numbers still come from
``benchmark_qsa_prims_ts.py`` with CUDA graphs.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import benchmark_qsa_prims_ts as benchmark
import torch

from vllm.models.qwen4_exp.nvidia.ops.qsa import (
    qsa_build_page4_paged_metadata,
    qsa_prims_ts_paged_attention,
    qsa_prims_ts_prepare_paged_attention,
    qsa_prims_ts_run_prepared_attention,
    qsa_prims_ts_workspace_size,
    qsa_sparse_paged_attention,
)


def _profile(
    name: str,
    launch: Callable[[], None],
    *,
    iterations: int,
    cold_l2: bool,
) -> None:
    for _ in range(3):
        launch()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
    ) as profiler:
        for _ in range(iterations):
            if cold_l2:
                benchmark._evict_l2()
            with torch.profiler.record_function(name):
                launch()
        torch.cuda.synchronize()
    event = next(item for item in profiler.key_averages() if item.key == name)
    print(f"\n# {name}: iterations={iterations}, L2={'cold' if cold_l2 else 'warm'}")
    print(f"host launch: {event.cpu_time_total / iterations:.2f} us/iteration")
    print(
        profiler.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=20,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tail", type=int, choices=range(4), default=0)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--storage-page-size", type=int, default=256)
    parser.add_argument("--qkv-dtype", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warm-l2", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("QSA component profiling requires CUDA")
    benchmark._QKV_DTYPE = (
        torch.bfloat16 if args.qkv_dtype == "bf16" else torch.float8_e4m3fn
    )
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    num_query_heads, num_kv_heads = benchmark._tp_heads(args.tp_size)
    end_visible = (
        args.context_length
        - (args.context_length - args.tail) % benchmark._COMPRESS_RATIO
    )
    final_position = end_visible - 1
    workload = benchmark.Workload(
        phase="decode",
        batch_size=args.batch_size,
        seq_len_q=1,
        context_length=args.context_length,
        decode_end_tail=args.tail,
    )
    inputs = benchmark._make_qsa_inputs(
        workload,
        num_query_heads,
        num_kv_heads,
        args.storage_page_size,
        "topk",
        "independent",
    )
    if int(inputs.logical_positions[-1].item()) != final_position:
        raise AssertionError("decode-tail construction changed unexpectedly")

    page_capacity = benchmark._BLOCK_TOPK + 1
    paged_kv_indptr = torch.empty(workload.rows + 1, dtype=torch.int32, device="cuda")
    paged_kv_indices = torch.empty(
        workload.rows * page_capacity, dtype=torch.int32, device="cuda"
    )
    seq_lens = torch.empty(workload.rows, dtype=torch.int32, device="cuda")
    qsa_build_page4_paged_metadata(
        inputs.block_indices,
        inputs.block_table,
        inputs.token_to_req,
        inputs.logical_positions,
        args.storage_page_size,
        indices_are_blocks=True,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        seq_lens=seq_lens,
    )
    workspace = torch.zeros(
        qsa_prims_ts_workspace_size(
            inputs.q,
            inputs.k_cache,
            benchmark._QSA_MAX_SEQ_LEN,
            out_dtype=benchmark._output_dtype(),
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    prims_output = benchmark._empty_output_like(inputs.q)
    triton_output = benchmark._empty_output_like(inputs.q)
    triton_k_cache = inputs.k_cache.permute(0, 2, 1, 3)
    triton_v_cache = inputs.v_cache.permute(0, 2, 1, 3)
    prepared_plan = qsa_prims_ts_prepare_paged_attention(
        inputs.q,
        inputs.k_cache,
        inputs.v_cache,
        workspace,
        paged_kv_indptr,
        paged_kv_indices,
        seq_lens,
        benchmark._QSA_MAX_SEQ_LEN,
        prims_output,
    )

    def prims_one_shot() -> None:
        qsa_prims_ts_paged_attention(
            inputs.q,
            inputs.k_cache,
            inputs.v_cache,
            workspace,
            paged_kv_indptr,
            paged_kv_indices,
            seq_lens,
            benchmark._QSA_MAX_SEQ_LEN,
            prims_output,
        )

    def prims_prepared() -> None:
        qsa_prims_ts_run_prepared_attention(
            prepared_plan,
            inputs.q,
            prims_output,
            bmm1_scale=inputs.q.shape[-1] ** -0.5,
            bmm2_scale=1.0,
        )

    def triton() -> None:
        qsa_sparse_paged_attention(
            inputs.q,
            triton_k_cache,
            triton_v_cache,
            inputs.logical_indices,
            inputs.block_table,
            inputs.token_to_req,
            triton_output,
        )

    _profile(
        "prims_ts_attention_one_shot",
        prims_one_shot,
        iterations=args.iterations,
        cold_l2=not args.warm_l2,
    )
    _profile(
        "prims_ts_attention_prepared",
        prims_prepared,
        iterations=args.iterations,
        cold_l2=not args.warm_l2,
    )
    _profile(
        "triton_attention",
        triton,
        iterations=args.iterations,
        cold_l2=not args.warm_l2,
    )


if __name__ == "__main__":
    main()
