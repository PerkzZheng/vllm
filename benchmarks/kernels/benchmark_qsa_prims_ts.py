# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark flattened QSA routes against sparse Triton and contiguous PrimTS.

The default matrix models Qwen3.8-Flash-Next's two serving phases:

* prefill: batch 1, SQ=SKV=8192;
* decode/MTP: batch 1/8/64/256, SQ=1/4, and an 8192-token context.

Every query token is flattened into an independent SQ=1 PrimTS route. Synthetic
top-k blocks are sampled without replacement from the causally visible prefix.
They can be sampled independently per route or from one shared ranking per
request to bound the effect of cross-token cache reuse.
The contiguous performance target keeps the workload's full physical K/V
length (8192 tokens in the default matrix) in ordinary 128-token pages and
uses a causal 2048-token sliding window. Its cache scope is explicit:
``request`` shares physical pages across a request's query tokens, while
``route`` gives every flattened query token independent physical pages.
One or more contiguous real-route dump chunks can replace the synthetic
metadata while retaining the same correctness and baseline comparisons.
Consecutive captured rows can also be fused into exact packed-membership
unions over either the full prefill or only its saturated suffix.
The benchmark reports exact selected-token counts and their nominal K+V byte
demand for the selected BF16 or FP8-E4M3 Q/K/V dtype. Achieved HBM bandwidth
requires profiler DRAM counters because
repeated selections can be served from cache.

Timing defaults to CUDA-graph replay with an L2-sized eviction pass before
every sample. The eviction runs on the same stream but is outside the sample's
CUDA-event interval, so every backend starts cold without charging cache
clearing to its latency. Comparable backends are interleaved in a rotating,
periodically reversed order so each one occupies every position along the same
clock and thermal trajectory.

Run from the repository root with vLLM's managed environment:

    uv run python benchmarks/kernels/benchmark_qsa_prims_ts.py
"""

from __future__ import annotations

import argparse
import functools
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from vllm.models.qwen4_exp.nvidia.ops.qsa import (
    expand_qsa_block_indices_cuda,
    has_qsa_prims_ts_attention,
    qsa_build_page4_grouped_paged_metadata,
    qsa_build_page4_paged_metadata,
    qsa_prims_ts_build_page4_metadata,
    qsa_prims_ts_metadata_workspace_size,
    qsa_prims_ts_paged_attention,
    qsa_prims_ts_workspace_size,
    qsa_sparse_paged_attention,
)

_TOTAL_QUERY_HEADS = 24
_TOTAL_KV_HEADS = 2
_HEAD_DIM = 256
_QKV_DTYPE = torch.bfloat16
_TOKEN_TOPK = 2048
_COMPRESS_RATIO = 4
_BLOCK_TOPK = _TOKEN_TOPK // _COMPRESS_RATIO
_QSA_MAX_SEQ_LEN = _TOKEN_TOPK + _COMPRESS_RATIO - 1
_BASELINE_PAGE_SIZE = 128
_BASELINE_WINDOW_LEFT = _TOKEN_TOPK - 1
_TOPK_RANDOM_CHUNK_ROWS = 512
_QSA_PAGE_MEMBERSHIP_BITS = 4
_MIN_L2_FLUSH_BYTES = 256 * 1024 * 1024
_L2_FLUSH_MULTIPLIER = 2
_L2_FLUSH_BUFFERS: dict[int, torch.Tensor] = {}


def _randn_qkv(*shape: int) -> torch.Tensor:
    """Generate representable Q/K/V values for the active benchmark dtype."""

    values = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
    if torch.float8_e4m3fn == _QKV_DTYPE:
        # Match the scaled FP8 qualification domain used by FlashInfer. Unit
        # normal FP8 Q/K produces artificially sharp logits and magnifies the
        # expected P448 BMM2 quantization difference versus Triton's FP32 P.
        values.mul_(0.25)
    return values if torch.bfloat16 == _QKV_DTYPE else values.to(_QKV_DTYPE)


def _qkv_dtype_key() -> str:
    return "bfloat16" if torch.bfloat16 == _QKV_DTYPE else "float8_e4m3fn"


def _output_dtype() -> torch.dtype:
    # The Qwen3.8 model always consumes BF16 attention output. FP8 benchmark
    # rows therefore model the encoded-cache path rather than the older
    # standalone FP8->FP16 qualification surface.
    return torch.bfloat16


def _output_dtype_key() -> str:
    return "bfloat16"


def _grouped_swa_output_dtype() -> torch.dtype:
    """Return the supported two-byte output type for grouped SWA timing."""

    # The Q64/KV128 Keeps profile accepts FP8 Q/K/V with FP16 output. It does
    # not yet accept model-facing BF16 output. Both output types have identical
    # traffic, and this baseline is a timing projection rather than a numerical
    # reference for the sparse result.
    return torch.bfloat16 if torch.bfloat16 == _QKV_DTYPE else torch.float16


def _empty_output_like(query: torch.Tensor) -> torch.Tensor:
    return torch.empty(query.shape, dtype=_output_dtype(), device=query.device)


def _comparison_atol() -> float:
    return 0.02 if torch.bfloat16 == _QKV_DTYPE else 0.05


def _install_experimental_q4_keeps_selector(
    issuers: int | None,
    *,
    kv128_splits: int,
    kv256_d128: bool,
    kv256_d256: bool,
    kv256_load_warps: int,
    kv256_splits: int,
) -> None:
    """Force one unqualified Q64 Keeps profile for dense Q4 union timing."""

    import cutlass
    from flashinfer.attention.prims_ts import decode as decode_module
    from flashinfer.attention.prims_ts.kernels.fmha_decode import (
        fmha_decode_config as config_module,
    )

    original_resolver = decode_module._resolve_decode_launch_spec

    @functools.cache
    def resolve_with_experimental_q4(
        device_index: int,
        batch_size: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        max_kv_len: int,
        seq_len_q: int,
        q_dtype_key: str,
        kv_dtype_key: str,
        output_dtype_key: str,
        kv_layout: str,
        mask_type: str,
        use_packed_q: bool,
        window_left: int,
        storage_page_size: int | None = None,
    ):
        semantic_key = (
            device_index,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            max_kv_len,
            seq_len_q,
            q_dtype_key,
            kv_dtype_key,
            output_dtype_key,
            kv_layout,
            mask_type,
            use_packed_q,
            window_left,
            storage_page_size,
        )
        expected_head_dim = 128 if kv256_d128 else 256
        use_kv256 = kv256_d128 or kv256_d256
        requested_splits = kv256_splits if use_kv256 else kv128_splits
        is_q4_union = (
            num_qo_heads == 12 * num_kv_heads
            and num_kv_heads in (1, 2)
            and head_dim == expected_head_dim
            and page_size == _COMPRESS_RATIO
            and q_dtype_key == "bfloat16"
            and kv_dtype_key == "bfloat16"
            and output_dtype_key == "bfloat16"
            and kv_layout == "HND"
            and not use_packed_q
            and (
                (seq_len_q == 4 and mask_type in ("dense", "causal"))
                or (kv256_d128 and seq_len_q == 1 and mask_type == "causal")
            )
        )
        if not is_q4_union:
            return original_resolver(*semantic_key)

        original_validate = config_module._validate_profile_support
        config_module._validate_profile_support = lambda **_kwargs: None
        try:
            if use_kv256:
                config_args = {
                    # Preserve the qualified Q64/KV256 physical profile. D128
                    # uses two temporal instances; D256 selects the one-instance
                    # four-loader full-width PV schedule through config defaults.
                    "use_keeps_mma_ab": True,
                    "groups_tokens_heads_q": True,
                    "tile_size_q": 64,
                    "tile_size_kv": 256,
                    "use_persistent_scheduler": False,
                }
                if kv256_d128 and kv256_load_warps == 2:
                    config_args.update(
                        load_num_warps=2,
                        page_offsets_warp_idx=15,
                    )
                elif kv256_d128 and kv256_load_warps == 4:
                    config_args.update(
                        load_warp_idx=16,
                        load_num_warps=4,
                        page_offsets_warp_idx=13,
                    )
                elif kv256_d256 and kv256_load_warps == 8:
                    config_args.update(load_num_warps=8)
            else:
                assert issuers is not None
                config_args = {
                    "use_keeps_mma_ab": True,
                    "groups_tokens_heads_q": True,
                    "tile_size_q": 64,
                    "tile_size_kv": 128,
                    "head_dim_per_stage_kv": 128,
                    "num_insts_kv": 1,
                    "o_stages": 1,
                    "use_persistent_scheduler": False,
                    "correction_num_warps": 4,
                    "mma_warp_idx": 12,
                    "page_offsets_warp_idx": 13,
                    "load_warp_idx": 14 if issuers <= 2 else 16,
                    "load_num_warps": issuers,
                }
            cfg = config_module.make_decode_config(
                headdim=expected_head_dim,
                args=config_args,
                seq_len_q=seq_len_q,
                seq_len_kv=max_kv_len,
                batch_size=batch_size,
                num_heads_q=num_qo_heads,
                num_heads_kv=num_kv_heads,
                qkv_dtype=cutlass.BFloat16,
                o_dtype=cutlass.BFloat16,
                qkv_layout="pagedKv",
                num_tokens_per_page=page_size,
                storage_tokens_per_page=(
                    page_size if storage_page_size is None else storage_page_size
                ),
                split_kv_mode=(
                    "gmem_reduction_with_separate_kernel"
                    if requested_splits > 1
                    else "disabled"
                ),
                splits_kv=requested_splits,
                max_splits_kv=requested_splits,
                mask_type=mask_type,
                auto_tuner=False,
            )
        finally:
            config_module._validate_profile_support = original_validate
        return decode_module._decode_launch_spec_from_config(
            cfg,
            batch_size=batch_size,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            seq_len_q=seq_len_q,
            max_active_clusters=(
                config_module.get_max_active_clusters_for_cluster_size(1)
            ),
        )

    decode_module._resolve_decode_launch_spec = resolve_with_experimental_q4
    decode_module._get_compiled_decode.cache_clear()


@dataclass(frozen=True)
class Workload:
    phase: str
    batch_size: int
    seq_len_q: int
    context_length: int
    decode_end_tail: int | None = None

    @property
    def rows(self) -> int:
        return self.batch_size * self.seq_len_q


@dataclass(frozen=True)
class QSAInputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_indices: torch.Tensor
    logical_indices: torch.Tensor
    block_table: torch.Tensor
    token_to_req: torch.Tensor
    logical_positions: torch.Tensor
    sequence_lengths: torch.Tensor
    compact_lengths: torch.Tensor
    adjacent_topk_overlap: float


@dataclass(frozen=True)
class RealRouteTrace:
    logical_indices: torch.Tensor
    block_table: torch.Tensor
    token_to_req: torch.Tensor
    logical_positions: torch.Tensor
    storage_page_size: int
    num_storage_pages: int
    context_length: int


@dataclass(frozen=True)
class BenchmarkResult:
    workload: Workload
    tp_size: int
    route_order: str
    contiguous_cache_scope: str
    topk_pattern: str
    adjacent_topk_overlap: float
    compact_min: int
    compact_max: int
    selected_kv_tokens: int
    kv_bytes_per_token: int
    metadata_us: float
    legacy_metadata_us: float
    qsa_attention_us: float
    qsa_end_to_end_us: float
    triton_sparse_us: float
    triton_end_to_end_us: float
    contiguous_us: float
    triton_max_abs_diff: float

    @property
    def qsa_ratio(self) -> float:
        return self.qsa_attention_us / self.contiguous_us

    @property
    def triton_ratio(self) -> float:
        return self.triton_sparse_us / self.contiguous_us

    @property
    def end_to_end_ratio(self) -> float:
        return self.qsa_end_to_end_us / self.contiguous_us

    @property
    def triton_end_to_end_ratio(self) -> float:
        return self.triton_end_to_end_us / self.contiguous_us

    @property
    def prims_ts_speedup_over_triton(self) -> float:
        """Compare metadata+attention against expansion+attention."""

        return self.triton_end_to_end_us / self.qsa_end_to_end_us

    @property
    def selected_kv_gb(self) -> float:
        return _selected_kv_gb(
            self.selected_kv_tokens,
            self.kv_bytes_per_token,
        )

    @property
    def qsa_effective_kv_tbps(self) -> float:
        """Logical selected K+V bytes divided by attention latency."""

        return self.selected_kv_gb * 1000 / self.qsa_attention_us

    @property
    def contiguous_effective_kv_tbps(self) -> float:
        """Logical selected K+V bytes divided by contiguous latency."""

        return self.selected_kv_gb * 1000 / self.contiguous_us


@dataclass(frozen=True)
class UnionUpperBoundResult:
    tp_size: int
    group_size: int
    membership_mode: str
    row_scope: str
    groups: int
    splits_kv: int
    source_kv_tokens: int
    union_kv_tokens: int
    grouped_swa_kv_tokens: int
    kv_bytes_per_token: int
    mean_union_blocks: float
    max_union_blocks: int
    launch_max_seq_len: int
    ideal_load_reduction: float
    metadata_us: float
    legacy_metadata_us: float
    flattened_qsa_us: float
    union_upper_bound_us: float
    union_end_to_end_us: float
    triton_sparse_us: float
    triton_end_to_end_us: float
    grouped_swa_us: float
    max_abs_diff: float
    triton_max_abs_diff: float

    @property
    def speedup(self) -> float:
        return self.flattened_qsa_us / self.union_upper_bound_us

    @property
    def non_shared_kv_ratio(self) -> float:
        """Extra union KV relative to an exactly shared grouped SWA window."""

        return self.union_kv_tokens / self.grouped_swa_kv_tokens - 1.0

    @property
    def projected_grouped_swa_us(self) -> float:
        """Project grouped SWA latency to the sparse union's KV volume."""

        return self.grouped_swa_us * (self.union_kv_tokens / self.grouped_swa_kv_tokens)

    @property
    def grouped_swa_ratio(self) -> float:
        return self.union_upper_bound_us / self.grouped_swa_us

    @property
    def grouped_swa_end_to_end_ratio(self) -> float:
        return self.union_end_to_end_us / self.grouped_swa_us

    @property
    def triton_grouped_swa_ratio(self) -> float:
        return self.triton_sparse_us / self.grouped_swa_us

    @property
    def triton_grouped_swa_end_to_end_ratio(self) -> float:
        return self.triton_end_to_end_us / self.grouped_swa_us

    @property
    def union_triton_ratio(self) -> float:
        return self.union_upper_bound_us / self.triton_sparse_us

    @property
    def prims_ts_speedup_over_triton(self) -> float:
        """Compare compact metadata+attention against expansion+attention."""

        return self.triton_end_to_end_us / self.union_end_to_end_us

    @property
    def projected_contiguous_ratio(self) -> float:
        return self.union_upper_bound_us / self.projected_grouped_swa_us

    @property
    def projected_end_to_end_ratio(self) -> float:
        return self.union_end_to_end_us / self.projected_grouped_swa_us

    @property
    def source_kv_gb(self) -> float:
        return _selected_kv_gb(
            self.source_kv_tokens,
            self.kv_bytes_per_token,
        )

    @property
    def union_kv_gb(self) -> float:
        return _selected_kv_gb(
            self.union_kv_tokens,
            self.kv_bytes_per_token,
        )

    @property
    def flattened_logical_kv_tbps(self) -> float:
        """Logical selected K+V bytes divided by flattened latency."""

        return self.source_kv_gb * 1000 / self.flattened_qsa_us

    @property
    def union_logical_kv_tbps(self) -> float:
        """Logical union K+V bytes divided by union attention latency."""

        return self.union_kv_gb * 1000 / self.union_upper_bound_us


def _selected_kv_gb(
    selected_kv_tokens: int,
    kv_bytes_per_token: int,
) -> float:
    """Return nominal selected K+V traffic in decimal GB."""

    return selected_kv_tokens * kv_bytes_per_token / 1e9


def _l2_cache_size_bytes(device_index: int) -> int:
    """Return the runtime L2 capacity, with a conservative Blackwell fallback."""

    properties = torch.cuda.get_device_properties(device_index)
    for attribute in ("L2_cache_size", "l2_cache_size"):
        value = int(getattr(properties, attribute, 0))
        if value > 0:
            return value
    return _MIN_L2_FLUSH_BYTES // _L2_FLUSH_MULTIPLIER


def _l2_flush_buffer() -> torch.Tensor:
    """Return a persistent buffer large enough to evict the device L2."""

    device_index = torch.cuda.current_device()
    buffer = _L2_FLUSH_BUFFERS.get(device_index)
    if buffer is None:
        flush_bytes = max(
            _MIN_L2_FLUSH_BYTES,
            _L2_FLUSH_MULTIPLIER * _l2_cache_size_bytes(device_index),
        )
        buffer = torch.zeros(flush_bytes, dtype=torch.uint8, device=device_index)
        _L2_FLUSH_BUFFERS[device_index] = buffer
    return buffer


def _evict_l2() -> None:
    """Read and write more than one full L2 working set on the active stream."""

    _l2_flush_buffer().add_(1)


def _balanced_timing_order(
    names: tuple[str, ...],
    round_index: int,
) -> tuple[str, ...]:
    """Rotate and periodically reverse backends to balance timing position."""

    if not names:
        return ()
    cycle, shift = divmod(round_index, len(names))
    base = names if cycle % 2 == 0 else names[::-1]
    return base[shift:] + base[:shift]


def _time_cuda_interleaved(
    functions: dict[str, Callable[[], object]],
    *,
    warmup_iterations: int,
    iterations: int,
    use_cuda_graph: bool,
    cold_l2: bool,
) -> dict[str, float]:
    """Time comparable backends in one balanced CUDA replay schedule.

    Each backend runs once per round. Its position rotates every round and the
    rotation reverses after a complete cycle, distributing every backend over
    the same clock and thermal trajectory. In cold-L2 mode, a separately
    captured eviction graph runs immediately before every target replay, on
    the same stream and outside the target's CUDA-event interval.
    """

    if not functions:
        raise ValueError("at least one CUDA timing function is required")
    names = tuple(functions)

    for round_index in range(warmup_iterations):
        for name in _balanced_timing_order(names, round_index):
            if cold_l2:
                _evict_l2()
            functions[name]()
    torch.cuda.synchronize()

    measured = dict(functions)
    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    flush_measured: Callable[[], object] | None = _evict_l2 if cold_l2 else None
    flush_graph = None
    if use_cuda_graph:
        for name, fn in functions.items():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            graphs[name] = graph
            measured[name] = graph.replay
        if cold_l2:
            # Warm the pointwise implementation before capture so graph setup
            # cannot allocate or compile on the measured path.
            _evict_l2()
            torch.cuda.synchronize()
            flush_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(flush_graph):
                _evict_l2()
            flush_measured = flush_graph.replay
        for round_index in range(warmup_iterations):
            for name in _balanced_timing_order(names, round_index):
                if flush_measured is not None:
                    flush_measured()
                measured[name]()
        torch.cuda.synchronize()

    event_pairs = {
        name: [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(iterations)
        ]
        for name in names
    }
    for round_index in range(iterations):
        for name in _balanced_timing_order(names, round_index):
            if flush_measured is not None:
                flush_measured()
            start, end = event_pairs[name][round_index]
            start.record()
            measured[name]()
            end.record()
    torch.cuda.synchronize()
    return {
        name: sum(start.elapsed_time(end) for start, end in event_pairs[name])
        * (1000 / iterations)
        for name in names
    }


def _tp_heads(tp_size: int) -> tuple[int, int]:
    if _TOTAL_QUERY_HEADS % tp_size:
        raise ValueError(f"TP={tp_size} does not divide {_TOTAL_QUERY_HEADS} Q heads")
    if tp_size <= _TOTAL_KV_HEADS:
        if _TOTAL_KV_HEADS % tp_size:
            raise ValueError(f"TP={tp_size} does not divide {_TOTAL_KV_HEADS} KV heads")
    elif tp_size % _TOTAL_KV_HEADS:
        raise ValueError(f"TP={tp_size} cannot replicate {_TOTAL_KV_HEADS} KV heads")
    return _TOTAL_QUERY_HEADS // tp_size, max(1, _TOTAL_KV_HEADS // tp_size)


def _logical_positions(workload: Workload) -> torch.Tensor:
    if workload.phase == "prefill":
        per_request = torch.arange(
            workload.seq_len_q,
            dtype=torch.int64,
            device="cuda",
        )
    else:
        end_visible = workload.context_length
        if workload.phase == "decode" and workload.decode_end_tail is not None:
            # Choose the final query position so its compact KV has exactly
            # this many causal tail tokens after the 512 complete pages.
            end_visible -= (
                workload.context_length - workload.decode_end_tail
            ) % _COMPRESS_RATIO
        first_position = end_visible - workload.seq_len_q
        per_request = torch.arange(
            first_position,
            end_visible,
            dtype=torch.int64,
            device="cuda",
        )
    return per_request.repeat(workload.batch_size)


def _random_causal_block_topk(
    logical_positions: torch.Tensor,
    token_to_req: torch.Tensor,
    batch_size: int,
    context_length: int,
    topk_pattern: str,
) -> torch.Tensor:
    """Sample distinct compressed blocks from every row's causal prefix."""

    num_compressed_blocks = context_length // _COMPRESS_RATIO
    block_indices = torch.empty(
        (logical_positions.numel(), _BLOCK_TOPK),
        dtype=torch.int32,
        device="cuda",
    )
    block_columns = torch.arange(
        num_compressed_blocks,
        dtype=torch.int64,
        device="cuda",
    )
    shared_scores = None
    if topk_pattern == "shared-request":
        shared_scores = torch.rand(
            (batch_size, num_compressed_blocks),
            dtype=torch.float32,
            device="cuda",
        )
    elif topk_pattern != "independent":
        raise ValueError(f"unsupported top-k pattern: {topk_pattern}")
    for row_start in range(0, logical_positions.numel(), _TOPK_RANDOM_CHUNK_ROWS):
        row_end = min(
            row_start + _TOPK_RANDOM_CHUNK_ROWS,
            logical_positions.numel(),
        )
        visible_blocks = (
            (logical_positions[row_start:row_end] + 1) // _COMPRESS_RATIO
        ).clamp(max=num_compressed_blocks)
        if shared_scores is None:
            scores = torch.rand(
                (row_end - row_start, num_compressed_blocks),
                dtype=torch.float32,
                device="cuda",
            )
        else:
            scores = shared_scores[
                token_to_req[row_start:row_end].to(torch.int64)
            ].clone()
        scores.masked_fill_(
            block_columns.unsqueeze(0) >= visible_blocks.unsqueeze(1),
            -float("inf"),
        )
        block_indices[row_start:row_end] = torch.topk(
            scores,
            k=_BLOCK_TOPK,
            dim=1,
            sorted=False,
        ).indices.to(torch.int32)
    return block_indices


def _adjacent_topk_overlap(
    block_indices: torch.Tensor,
    logical_positions: torch.Tensor,
    token_to_req: torch.Tensor,
    context_length: int,
) -> float:
    """Return mean |A intersect B| / top-k for adjacent saturated routes."""

    if block_indices.shape[0] < 2:
        return float("nan")
    num_compressed_blocks = context_length // _COMPRESS_RATIO
    visible_blocks = ((logical_positions + 1) // _COMPRESS_RATIO).clamp(
        max=num_compressed_blocks
    )
    saturated = visible_blocks >= _BLOCK_TOPK
    adjacent = (token_to_req[1:] == token_to_req[:-1]) & saturated[1:] & saturated[:-1]
    if not bool(adjacent.any().item()):
        return float("nan")

    membership = torch.zeros(
        (block_indices.shape[0], num_compressed_blocks),
        dtype=torch.bool,
        device="cuda",
    )
    block_indices_i64 = block_indices.to(torch.int64)
    valid = (block_indices_i64 >= 0) & (block_indices_i64 < visible_blocks.unsqueeze(1))
    safe_indices = block_indices_i64.clamp(0, num_compressed_blocks - 1)
    membership.scatter_(1, safe_indices, valid)
    intersections = (membership[1:] & membership[:-1]).sum(dim=1)
    return float(intersections[adjacent].float().mean().item() / _BLOCK_TOPK)


def _load_real_route_trace(
    paths: list[Path],
    storage_page_size_override: int | None,
) -> RealRouteTrace:
    """Load contiguous chunks from one real request and compact page IDs."""

    if not paths:
        raise ValueError("at least one real top-k dump is required")
    required = {
        "token_topk",
        "compress_ratio",
        "main_storage_page_size",
        "selected_token_indices",
        "token_to_req",
        "logical_positions",
        "main_block_table",
    }
    payloads = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"{path} is missing fields: {', '.join(missing)}")
        if int(payload["token_topk"]) != _TOKEN_TOPK:
            raise ValueError(f"{path} has an incompatible token top-k")
        if int(payload["compress_ratio"]) != _COMPRESS_RATIO:
            raise ValueError(f"{path} has an incompatible compression ratio")
        payloads.append(payload)

    captured_page_sizes = {
        int(payload["main_storage_page_size"]) for payload in payloads
    }
    if storage_page_size_override is None:
        if len(captured_page_sizes) != 1:
            raise ValueError("real top-k dumps disagree on main-cache page size")
        storage_page_size = captured_page_sizes.pop()
    else:
        storage_page_size = storage_page_size_override
    if storage_page_size <= 0 or storage_page_size % _COMPRESS_RATIO:
        raise ValueError("real top-k storage page size must be a multiple of four")

    payloads.sort(key=lambda payload: int(payload["logical_positions"].min().item()))
    logical_indices = torch.cat(
        [payload["selected_token_indices"] for payload in payloads]
    ).to(torch.int32)
    token_to_req = torch.cat([payload["token_to_req"] for payload in payloads]).to(
        torch.int32
    )
    logical_positions = torch.cat(
        [payload["logical_positions"] for payload in payloads]
    ).to(torch.int64)
    if not logical_positions.numel():
        raise ValueError("real top-k dumps must contain at least one row")
    if logical_indices.shape != (logical_positions.numel(), _QSA_MAX_SEQ_LEN):
        raise ValueError("real top-k rows must have the QSA expanded width")
    if token_to_req.shape != logical_positions.shape or bool(
        (token_to_req != 0).any().item()
    ):
        raise ValueError("real-route replay currently requires one request")
    expected_positions = torch.arange(
        int(logical_positions[0].item()),
        int(logical_positions[-1].item()) + 1,
        dtype=torch.int64,
    )
    if not torch.equal(logical_positions, expected_positions):
        raise ValueError("real top-k dump chunks must cover contiguous positions")

    context_length = int(logical_positions[-1].item()) + 1
    required_pages = math.ceil(context_length / storage_page_size)
    last_payload = payloads[-1]
    raw_block_table = last_payload["main_block_table"].to(torch.int32)
    if raw_block_table.ndim != 2 or raw_block_table.shape[0] != 1:
        raise ValueError("real-route replay currently requires one page-table row")
    if raw_block_table.shape[1] < required_pages:
        raise ValueError("real top-k main block table is too short")
    raw_block_table = raw_block_table[:, :required_pages].clone()
    if bool((raw_block_table < 0).any().item()):
        raise ValueError("real top-k main block table has an unallocated live page")

    for payload in payloads:
        positions = payload["logical_positions"]
        payload_required_pages = math.ceil(
            (int(positions.max().item()) + 1) / storage_page_size
        )
        payload_table = payload["main_block_table"]
        if not torch.equal(
            payload_table[:1, :payload_required_pages].to(torch.int32),
            raw_block_table[:, :payload_required_pages],
        ):
            raise ValueError("real top-k chunks disagree on allocated main pages")

    physical_pages = raw_block_table.unique(sorted=True)
    dense_pages = torch.zeros_like(physical_pages)
    for index in range(1, physical_pages.numel()):
        gap = 1 if physical_pages[index] == physical_pages[index - 1] + 1 else 2
        dense_pages[index] = dense_pages[index - 1] + gap
    block_table = torch.full_like(raw_block_table, -1)
    for physical_page, dense_page in zip(physical_pages, dense_pages):
        block_table[raw_block_table == physical_page] = dense_page
    num_storage_pages = int(dense_pages[-1].item()) + 1
    return RealRouteTrace(
        logical_indices=logical_indices,
        block_table=block_table,
        token_to_req=token_to_req,
        logical_positions=logical_positions,
        storage_page_size=storage_page_size,
        num_storage_pages=num_storage_pages,
        context_length=context_length,
    )


def _make_real_qsa_inputs(
    trace: RealRouteTrace,
    num_query_heads: int,
    num_kv_heads: int,
    topk_pattern: str,
) -> QSAInputs:
    q = _randn_qkv(
        trace.logical_positions.numel(),
        num_query_heads,
        _HEAD_DIM,
    )
    packed_kv = _randn_qkv(
        trace.num_storage_pages,
        num_kv_heads,
        trace.storage_page_size,
        2 * _HEAD_DIM,
    )
    k_cache, v_cache = packed_kv.split(_HEAD_DIM, dim=-1)
    block_table = trace.block_table.to(device="cuda")
    token_to_req = trace.token_to_req.to(device="cuda")
    logical_positions = trace.logical_positions.to(device="cuda")
    sequence_lengths = torch.tensor(
        [trace.context_length], dtype=torch.int32, device="cuda"
    )
    if topk_pattern == "captured":
        logical_indices = trace.logical_indices.to(device="cuda")
    else:
        block_indices = _random_causal_block_topk(
            logical_positions,
            token_to_req,
            1,
            trace.context_length,
            topk_pattern,
        )
        block_indices = _order_block_indices(
            block_indices,
            "topk",
            block_table,
            token_to_req,
            logical_positions,
            trace.storage_page_size,
        )
        logical_indices = expand_qsa_block_indices_cuda(
            block_indices,
            logical_positions,
            sequence_lengths,
            token_to_req,
            _COMPRESS_RATIO,
            _TOKEN_TOPK,
        )
    block_indices = torch.div(
        logical_indices[:, :_TOKEN_TOPK:_COMPRESS_RATIO],
        _COMPRESS_RATIO,
        rounding_mode="floor",
    )
    complete_pages = torch.minimum(
        (logical_positions + 1) // _COMPRESS_RATIO,
        torch.tensor(_BLOCK_TOPK, dtype=torch.int64, device="cuda"),
    )
    block_indices = block_indices.masked_fill(
        torch.arange(_BLOCK_TOPK, device="cuda").unsqueeze(0)
        >= complete_pages.unsqueeze(1),
        -1,
    )
    adjacent_topk_overlap = _adjacent_topk_overlap(
        block_indices,
        logical_positions,
        token_to_req,
        trace.context_length,
    )
    visible_tokens = logical_positions + 1
    compact_lengths = (
        complete_pages * _COMPRESS_RATIO + visible_tokens % _COMPRESS_RATIO
    ).to(torch.int32)
    return QSAInputs(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_indices=block_indices,
        logical_indices=logical_indices,
        block_table=block_table,
        token_to_req=token_to_req,
        logical_positions=logical_positions,
        sequence_lengths=sequence_lengths,
        compact_lengths=compact_lengths,
        adjacent_topk_overlap=adjacent_topk_overlap,
    )


def _build_union_csr(
    trace: RealRouteTrace,
    group_size: int,
    membership_mode: str,
    row_scope: str,
    start_group: int = 0,
    max_groups: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Build packed shared unions for consecutive real-route rows."""

    if row_scope == "full":
        candidate_rows = torch.arange(trace.logical_positions.numel())
    elif row_scope == "saturated":
        candidate_rows = torch.nonzero(
            trace.logical_positions + 1 >= _TOKEN_TOPK,
            as_tuple=False,
        ).flatten()
    else:
        raise ValueError(f"unsupported union row scope: {row_scope}")
    candidate_rows = candidate_rows[start_group * group_size :]
    num_groups = candidate_rows.numel() // group_size
    if max_groups is not None:
        num_groups = min(num_groups, max_groups)
    if not num_groups:
        raise ValueError(f"real trace has no complete {row_scope} query group")
    row_indices = candidate_rows[: num_groups * group_size]
    grouped_rows = row_indices.reshape(num_groups, group_size)
    grouped_positions = trace.logical_positions[grouped_rows]
    if not bool(torch.all(grouped_positions[:, 1:] - grouped_positions[:, :-1] == 1)):
        raise ValueError("union groups require consecutive logical positions")
    grouped_requests = trace.token_to_req[grouped_rows]
    if not bool(torch.all(grouped_requests == grouped_requests[:, :1])):
        raise ValueError("union groups cannot cross request boundaries")
    subpages_per_storage_page = trace.storage_page_size // _COMPRESS_RATIO
    union_locators = []
    union_counts = torch.empty(num_groups, dtype=torch.int32)
    union_seq_lens = torch.empty(num_groups, dtype=torch.int32)
    source_page_count = 0
    for group_index in range(num_groups):
        row_blocks = []
        for row_index in grouped_rows[group_index].tolist():
            visible_tokens = int(trace.logical_positions[row_index].item()) + 1
            complete_pages = min(
                visible_tokens // _COMPRESS_RATIO,
                _BLOCK_TOPK,
            )
            tail_tokens = visible_tokens % _COMPRESS_RATIO
            live_pages = complete_pages + int(tail_tokens > 0)
            blocks = torch.div(
                trace.logical_indices[
                    row_index,
                    : live_pages * _COMPRESS_RATIO : _COMPRESS_RATIO,
                ],
                _COMPRESS_RATIO,
                rounding_mode="floor",
            )
            row_blocks.append(blocks)
            source_page_count += blocks.numel()
        union_blocks = torch.unique(torch.cat(row_blocks), sorted=True)
        membership = torch.zeros(union_blocks.numel(), dtype=torch.int32)
        if membership_mode == "full":
            membership.fill_((1 << group_size) - 1)
        else:
            for q_index, blocks in enumerate(row_blocks):
                union_positions = torch.searchsorted(union_blocks, blocks)
                membership[union_positions] |= 1 << q_index
        logical_tokens = union_blocks * _COMPRESS_RATIO
        logical_pages = torch.div(
            logical_tokens,
            trace.storage_page_size,
            rounding_mode="floor",
        )
        subpages = torch.div(
            logical_tokens.remainder(trace.storage_page_size),
            _COMPRESS_RATIO,
            rounding_mode="floor",
        )
        request_index = int(grouped_requests[group_index, 0].item())
        physical_pages = trace.block_table[request_index, logical_pages]
        locators = physical_pages * subpages_per_storage_page + subpages
        union_locators.append((locators << _QSA_PAGE_MEMBERSHIP_BITS) | membership)
        union_counts[group_index] = union_blocks.numel()
        tail_tokens = int((grouped_positions[group_index, -1].item() + 1) % 4)
        tail_padding = (4 - tail_tokens) % 4 if membership_mode == "masked" else 0
        union_seq_lens[group_index] = union_blocks.numel() * 4 - tail_padding
    paged_kv_indptr = torch.empty(num_groups + 1, dtype=torch.int32)
    paged_kv_indptr[0] = 0
    torch.cumsum(union_counts, dim=0, out=paged_kv_indptr[1:])
    paged_kv_indices = torch.cat(union_locators).to(torch.int32)
    ideal_load_reduction = 1.0 - float(union_counts.sum().item() / source_page_count)
    return (
        row_indices,
        paged_kv_indptr,
        paged_kv_indices,
        union_seq_lens,
        ideal_load_reduction,
    )


def _run_union_upper_bound(
    trace: RealRouteTrace,
    inputs: QSAInputs,
    tp_size: int,
    group_size: int,
    membership_mode: str,
    row_scope: str,
    warmup_iterations: int,
    iterations: int,
    use_cuda_graph: bool,
    cold_l2: bool,
    profile_component: str | None,
    start_group: int,
    max_groups: int | None,
    static_max_seq_len: int | None,
) -> UnionUpperBoundResult:
    """Time a full-mask control or a semantically masked shared union."""

    from flashinfer.decode import (
        get_prims_ts_batch_decode_workspace_size,
        prims_ts_batch_decode_with_kv_cache,
    )

    num_query_heads, num_kv_heads = _tp_heads(tp_size)
    (
        row_indices_cpu,
        union_indptr_cpu,
        union_indices_cpu,
        union_seq_lens_cpu,
        ideal_load_reduction,
    ) = _build_union_csr(
        trace,
        group_size,
        membership_mode,
        row_scope,
        start_group,
        max_groups,
    )
    row_indices = row_indices_cpu.to(device="cuda")
    flat_q = inputs.q.index_select(0, row_indices).contiguous()
    flat_logical_indices = inputs.logical_indices.index_select(
        0, row_indices
    ).contiguous()
    flat_block_indices = inputs.block_indices.index_select(0, row_indices).contiguous()
    flat_token_to_req = inputs.token_to_req.index_select(0, row_indices).contiguous()
    flat_positions = inputs.logical_positions.index_select(0, row_indices).contiguous()
    flat_compact_lengths = inputs.compact_lengths.index_select(
        0, row_indices
    ).contiguous()
    flat_rows = flat_q.shape[0]
    page_capacity = _BLOCK_TOPK + 1
    flat_indptr = torch.empty(flat_rows + 1, dtype=torch.int32, device="cuda")
    flat_indices = torch.empty(
        flat_rows * page_capacity,
        dtype=torch.int32,
        device="cuda",
    )
    flat_seq_lens = torch.empty(flat_rows, dtype=torch.int32, device="cuda")
    flat_expanded_indices = torch.empty_like(flat_logical_indices)
    qsa_prims_ts_build_page4_metadata(
        flat_block_indices,
        inputs.block_table,
        flat_token_to_req,
        flat_positions,
        trace.storage_page_size,
        1,
        None,
        flat_indptr,
        flat_indices,
        flat_seq_lens,
    )
    flat_workspace = torch.zeros(
        qsa_prims_ts_workspace_size(
            flat_q,
            inputs.k_cache,
            _QSA_MAX_SEQ_LEN,
            out_dtype=_output_dtype(),
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    flat_output = _empty_output_like(flat_q)
    triton_output = _empty_output_like(flat_q)
    triton_k_cache = inputs.k_cache.permute(0, 2, 1, 3)
    triton_v_cache = inputs.v_cache.permute(0, 2, 1, 3)

    def flattened_attention() -> None:
        with torch.cuda.nvtx.range("flattened_qsa_attention"):
            qsa_prims_ts_paged_attention(
                flat_q,
                inputs.k_cache,
                inputs.v_cache,
                flat_workspace,
                flat_indptr,
                flat_indices,
                flat_seq_lens,
                _QSA_MAX_SEQ_LEN,
                flat_output,
            )

    def triton_sparse_attention() -> None:
        with torch.cuda.nvtx.range("union_triton_sparse_attention"):
            qsa_sparse_paged_attention(
                flat_q,
                triton_k_cache,
                triton_v_cache,
                flat_logical_indices,
                inputs.block_table,
                flat_token_to_req,
                triton_output,
            )

    def triton_end_to_end() -> None:
        with torch.cuda.nvtx.range("union_triton_expand_indices"):
            expand_qsa_block_indices_cuda(
                flat_block_indices,
                flat_positions,
                inputs.sequence_lengths,
                flat_token_to_req,
                _COMPRESS_RATIO,
                _TOKEN_TOPK,
                flat_expanded_indices,
            )
        with torch.cuda.nvtx.range("union_triton_sparse_attention"):
            qsa_sparse_paged_attention(
                flat_q,
                triton_k_cache,
                triton_v_cache,
                flat_expanded_indices,
                inputs.block_table,
                flat_token_to_req,
                triton_output,
            )

    num_groups = flat_rows // group_size
    union_q = flat_q.view(
        num_groups,
        group_size,
        num_query_heads,
        _HEAD_DIM,
    )
    observed_union_max_seq_len = int(union_seq_lens_cpu.max().item())
    union_max_seq_len = (
        observed_union_max_seq_len if static_max_seq_len is None else static_max_seq_len
    )
    if union_max_seq_len < observed_union_max_seq_len:
        raise ValueError(
            "static union max sequence length must cover the observed union: "
            f"{union_max_seq_len} < {observed_union_max_seq_len}"
        )
    union_mask_type = "causal" if membership_mode == "masked" else "dense"
    from flashinfer.attention.prims_ts.decode import _resolve_decode_launch_spec

    qkv_dtype_key = _qkv_dtype_key()
    union_spec = _resolve_decode_launch_spec(
        torch.cuda.current_device(),
        num_groups,
        num_query_heads,
        num_kv_heads,
        _HEAD_DIM,
        _COMPRESS_RATIO,
        union_max_seq_len,
        group_size,
        qkv_dtype_key,
        qkv_dtype_key,
        _output_dtype_key(),
        "HND",
        union_mask_type,
        False,
        -1,
        trace.storage_page_size,
    )
    actual_splits_kv = (
        int(union_spec.config.splits_kv) if union_spec.config.use_split_kv else 1
    )
    union_metadata: Callable[[], object] | None = None
    union_legacy_metadata: Callable[[], object] | None = None
    if membership_mode == "masked":
        union_page_capacity = group_size * (_BLOCK_TOPK + 1)
        union_indptr = torch.empty(num_groups + 1, dtype=torch.int32, device="cuda")
        union_indices = torch.empty(
            num_groups * union_page_capacity,
            dtype=torch.int32,
            device="cuda",
        )
        union_seq_lens = torch.empty(num_groups, dtype=torch.int32, device="cuda")
        union_metadata_workspace = torch.empty(
            qsa_prims_ts_metadata_workspace_size(
                flat_rows,
                inputs.block_table,
                trace.storage_page_size,
                group_size,
            ),
            dtype=torch.uint8,
            device="cuda",
        )
        qsa_prims_ts_build_page4_metadata(
            flat_block_indices,
            inputs.block_table,
            flat_token_to_req,
            flat_positions,
            trace.storage_page_size,
            group_size,
            union_metadata_workspace,
            union_indptr,
            union_indices,
            union_seq_lens,
        )

        def union_metadata() -> object:
            with torch.cuda.nvtx.range("union_metadata"):
                return qsa_prims_ts_build_page4_metadata(
                    flat_block_indices,
                    inputs.block_table,
                    flat_token_to_req,
                    flat_positions,
                    trace.storage_page_size,
                    group_size,
                    union_metadata_workspace,
                    union_indptr,
                    union_indices,
                    union_seq_lens,
                )

        def union_legacy_metadata() -> object:
            with torch.cuda.nvtx.range("union_legacy_metadata"):
                expand_qsa_block_indices_cuda(
                    flat_block_indices,
                    flat_positions,
                    inputs.sequence_lengths,
                    flat_token_to_req,
                    _COMPRESS_RATIO,
                    _TOKEN_TOPK,
                    flat_expanded_indices,
                )
                return qsa_build_page4_grouped_paged_metadata(
                    flat_expanded_indices,
                    inputs.block_table,
                    flat_token_to_req,
                    flat_positions,
                    trace.storage_page_size,
                    group_size,
                    bitset_workspace=union_metadata_workspace.view(torch.int32),
                    paged_kv_indptr=union_indptr,
                    paged_kv_indices=union_indices,
                    seq_lens=union_seq_lens,
                )

        gpu_indptr = union_indptr.cpu()
        gpu_indices = union_indices.cpu()
        gpu_seq_lens = union_seq_lens.cpu()
        if not torch.equal(gpu_seq_lens, union_seq_lens_cpu):
            raise AssertionError("grouped GPU metadata sequence lengths mismatch")
        for group_index in range(num_groups):
            live_pages = int(union_indptr_cpu[group_index + 1].item()) - int(
                union_indptr_cpu[group_index].item()
            )
            cpu_begin = int(union_indptr_cpu[group_index].item())
            gpu_begin = int(gpu_indptr[group_index].item())
            if gpu_begin != group_index * union_page_capacity:
                raise AssertionError("grouped GPU metadata indptr mismatch")
            if not torch.equal(
                gpu_indices[gpu_begin : gpu_begin + live_pages],
                union_indices_cpu[cpu_begin : cpu_begin + live_pages],
            ):
                raise AssertionError("grouped GPU metadata locator mismatch")
    else:
        union_indptr = union_indptr_cpu.to(device="cuda")
        union_indices = union_indices_cpu.to(device="cuda")
        union_seq_lens = union_seq_lens_cpu.to(device="cuda")
    union_workspace = torch.zeros(
        get_prims_ts_batch_decode_workspace_size(
            num_groups,
            num_query_heads,
            num_kv_heads,
            _HEAD_DIM,
            _COMPRESS_RATIO,
            union_max_seq_len,
            seq_len_q=group_size,
            q_dtype=union_q.dtype,
            kv_dtype=inputs.k_cache.dtype,
            out_dtype=_output_dtype(),
            mask_type=union_mask_type,
            storage_page_size=trace.storage_page_size,
            device="cuda",
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    union_output = _empty_output_like(union_q)

    def union_attention() -> None:
        with torch.cuda.nvtx.range("union_attention"):
            prims_ts_batch_decode_with_kv_cache(
                union_q,
                (inputs.k_cache, inputs.v_cache),
                union_workspace,
                union_indptr,
                union_indices,
                union_seq_lens,
                union_max_seq_len,
                seq_len_q=group_size,
                bmm1_scale=_HEAD_DIM**-0.5,
                out=union_output,
                out_dtype=union_output.dtype,
                mask_type=union_mask_type,
                page_size=_COMPRESS_RATIO,
            )

    def union_end_to_end() -> None:
        if union_metadata is not None:
            union_metadata()
        union_attention()

    subset_inputs = QSAInputs(
        q=flat_q,
        k_cache=inputs.k_cache,
        v_cache=inputs.v_cache,
        block_indices=flat_block_indices,
        logical_indices=flat_logical_indices,
        block_table=inputs.block_table,
        token_to_req=flat_token_to_req,
        logical_positions=flat_positions,
        sequence_lengths=inputs.sequence_lengths,
        compact_lengths=flat_compact_lengths,
        adjacent_topk_overlap=inputs.adjacent_topk_overlap,
    )
    (
        baseline_k,
        baseline_v,
        baseline_indptr,
        baseline_indices,
        baseline_seq_lens,
        baseline_workspace,
        grouped_swa_kv_tokens,
    ) = _make_grouped_contiguous_baseline(
        subset_inputs,
        group_size,
        num_query_heads,
        num_kv_heads,
        trace.context_length,
    )
    baseline_output = torch.empty(
        union_q.shape,
        dtype=_grouped_swa_output_dtype(),
        device=union_q.device,
    )

    def grouped_swa_attention() -> None:
        with torch.cuda.nvtx.range("union_grouped_swa_attention"):
            prims_ts_batch_decode_with_kv_cache(
                union_q,
                (baseline_k, baseline_v),
                baseline_workspace,
                baseline_indptr,
                baseline_indices,
                baseline_seq_lens,
                trace.context_length,
                bmm1_scale=_HEAD_DIM**-0.5,
                out=baseline_output,
                out_dtype=baseline_output.dtype,
                seq_len_q=group_size,
                mask_type="causal",
                window_left=_BASELINE_WINDOW_LEFT,
                page_size=_BASELINE_PAGE_SIZE,
            )

    # Poison the output before the untimed correctness launch. This makes a
    # missing TileQ store distinguishable from arithmetic that happens to
    # leave finite allocator contents behind.
    union_output.fill_(float("nan"))
    benchmark_flattened = torch.bfloat16 == _QKV_DTYPE
    if benchmark_flattened:
        flattened_attention()
    triton_sparse_attention()
    if union_metadata is not None:
        union_metadata()
    union_attention()
    grouped_swa_attention()
    torch.cuda.synchronize()
    # PyTorch does not implement isfinite for float8 tensors.  Convert only for
    # validation; the measured kernel still writes the requested output dtype.
    finite_output = torch.isfinite(union_output.float())
    if not bool(finite_output.all().item()):
        bad_rows = torch.nonzero(
            ~finite_output.flatten(start_dim=3).all(dim=3),
            as_tuple=False,
        )
        nonfinite_by_q_head = (~finite_output).sum(dim=(0, 3)).cpu().tolist()
        raise AssertionError(
            "shared-union upper bound produced non-finite output: "
            f"nonfinite_values={int((~finite_output).sum().item())}, "
            f"bad_rows={bad_rows[:16].cpu().tolist()}, "
            f"nonfinite_by_q_head={nonfinite_by_q_head}"
        )
    comparison_atol = _comparison_atol()
    triton_max_abs_diff = math.nan
    if benchmark_flattened:
        triton_abs_diff = (triton_output.float() - flat_output.float()).abs()
        triton_max_abs_diff = float(triton_abs_diff.max().item())
        if (
            not math.isfinite(triton_max_abs_diff)
            or triton_max_abs_diff > comparison_atol
        ):
            raise AssertionError(
                "Triton sparse output disagrees with flattened PrimTS: "
                f"max diff {triton_max_abs_diff:.5f}"
            )
    max_abs_diff = math.nan
    if membership_mode.startswith("masked"):
        flat_reference = (
            flat_output if benchmark_flattened else triton_output
        ).view_as(union_output)
        abs_diff = (union_output.float() - flat_reference.float()).abs()
        max_abs_diff = float(abs_diff.max().item())
        if membership_mode == "masked" and max_abs_diff > comparison_atol:
            worst = torch.unravel_index(abs_diff.argmax(), abs_diff.shape)
            worst_group = int(worst[0].item())
            worst_query = int(worst[1].item())
            worst_flat_row = worst_group * group_size + worst_query
            logical = flat_logical_indices[worst_flat_row]
            logical = logical[logical >= 0].long()
            request = flat_token_to_req[worst_flat_row].long()
            pages = inputs.block_table[
                request,
                logical // trace.storage_page_size,
            ].long()
            offsets = logical % trace.storage_page_size
            repeats = num_query_heads // num_kv_heads
            keys = inputs.k_cache[pages, :, offsets].repeat_interleave(repeats, dim=1)
            values = inputs.v_cache[pages, :, offsets].repeat_interleave(repeats, dim=1)
            scores = torch.einsum(
                "hd,khd->hk",
                flat_q[worst_flat_row].float(),
                keys.float(),
            )
            probabilities = torch.softmax(scores * (_HEAD_DIM**-0.5), dim=-1)
            direct_reference = torch.einsum(
                "hk,khd->hd",
                probabilities,
                values.float(),
            )
            union_reference_diff = float(
                (union_output[worst_group, worst_query].float() - direct_reference)
                .abs()
                .max()
                .item()
            )
            flattened_reference_diff = float(
                (flat_reference[worst_group, worst_query].float() - direct_reference)
                .abs()
                .max()
                .item()
            )
            max_diff_by_q_head = abs_diff.amax(dim=(0, 3)).cpu().tolist()
            bad_values_by_q_head = (
                (abs_diff > comparison_atol).sum(dim=(0, 3)).cpu().tolist()
            )
            raise AssertionError(
                "shared-union membership mismatch: "
                f"max diff {max_abs_diff:.5f}, "
                f"worst_group={worst_group}, worst_query={worst_query}, "
                f"union-vs-reference={union_reference_diff:.5f}, "
                f"flattened-vs-reference={flattened_reference_diff:.5f}, "
                f"max_diff_by_q_head={max_diff_by_q_head}, "
                f"bad_values_by_q_head={bad_values_by_q_head}"
            )
    profile_functions = {
        "union": union_attention,
        "triton": triton_sparse_attention,
        "contiguous": grouped_swa_attention,
    }
    if benchmark_flattened:
        profile_functions["flattened"] = flattened_attention
    if union_metadata is not None:
        profile_functions["metadata"] = union_metadata
    if profile_component is not None:
        profile_function = profile_functions.get(profile_component)
        if profile_function is None:
            raise ValueError(
                f"cannot profile {profile_component} for membership mode "
                f"{membership_mode}"
            )
        if cold_l2:
            _evict_l2()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        profile_function()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    timing_args = {
        "warmup_iterations": warmup_iterations,
        "iterations": iterations,
        "use_cuda_graph": use_cuda_graph,
        "cold_l2": cold_l2,
    }
    timing_functions = {
        "union": union_attention,
        "union_end_to_end": union_end_to_end,
        "triton": triton_sparse_attention,
        "triton_end_to_end": triton_end_to_end,
        "contiguous": grouped_swa_attention,
    }
    if benchmark_flattened:
        timing_functions["flattened"] = flattened_attention
    if union_metadata is not None:
        assert union_legacy_metadata is not None
        timing_functions = {
            "metadata": union_metadata,
            "legacy_metadata": union_legacy_metadata,
            **timing_functions,
        }
    timings = _time_cuda_interleaved(timing_functions, **timing_args)
    return UnionUpperBoundResult(
        tp_size=tp_size,
        group_size=group_size,
        membership_mode=membership_mode,
        row_scope=row_scope,
        groups=num_groups,
        splits_kv=actual_splits_kv,
        source_kv_tokens=int(flat_compact_lengths.sum().item()),
        union_kv_tokens=int(union_seq_lens_cpu.sum().item()),
        grouped_swa_kv_tokens=grouped_swa_kv_tokens,
        kv_bytes_per_token=(
            2 * num_kv_heads * _HEAD_DIM * inputs.k_cache.element_size()
        ),
        mean_union_blocks=float(union_seq_lens_cpu.float().mean().item() / 4),
        max_union_blocks=observed_union_max_seq_len // 4,
        launch_max_seq_len=union_max_seq_len,
        ideal_load_reduction=ideal_load_reduction,
        metadata_us=timings.get("metadata", math.nan),
        legacy_metadata_us=timings.get("legacy_metadata", math.nan),
        flattened_qsa_us=timings.get("flattened", math.nan),
        union_upper_bound_us=timings["union"],
        union_end_to_end_us=timings["union_end_to_end"],
        triton_sparse_us=timings["triton"],
        triton_end_to_end_us=timings["triton_end_to_end"],
        grouped_swa_us=timings["contiguous"],
        max_abs_diff=max_abs_diff,
        triton_max_abs_diff=triton_max_abs_diff,
    )


def _order_block_indices(
    block_indices: torch.Tensor,
    route_order: str,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    storage_page_size: int,
) -> torch.Tensor:
    """Reorder selected blocks while keeping causal entries in the prefix."""

    block_indices_i64 = block_indices.to(torch.int64)
    visible_blocks = (logical_positions + 1) // _COMPRESS_RATIO
    causally_valid = block_indices_i64 < visible_blocks.unsqueeze(1)
    invalid_key = torch.iinfo(torch.int64).max
    if route_order == "topk":
        # ``topk(sorted=False)`` does not promise where the -inf padding lands
        # before 512 complete blocks are visible. Preserve the sampled order,
        # but move padding behind the causal prefix consumed by the adapter.
        sort_keys = (~causally_valid).to(torch.int64)
    elif route_order == "logical":
        sort_keys = block_indices_i64.masked_fill(~causally_valid, invalid_key)
    elif route_order == "physical":
        tokens = block_indices_i64 * _COMPRESS_RATIO
        logical_pages = tokens // storage_page_size
        subpages = (tokens % storage_page_size) // _COMPRESS_RATIO
        physical_pages = block_table[
            token_to_req.to(torch.int64).unsqueeze(1),
            logical_pages,
        ].to(torch.int64)
        subpages_per_storage_page = storage_page_size // _COMPRESS_RATIO
        sort_keys = (physical_pages * subpages_per_storage_page + subpages).masked_fill(
            ~causally_valid, invalid_key
        )
    else:
        raise ValueError(f"unsupported route order: {route_order}")

    order = sort_keys.argsort(dim=1, stable=True)
    return block_indices.gather(1, order)


def _make_qsa_inputs(
    workload: Workload,
    num_query_heads: int,
    num_kv_heads: int,
    storage_page_size: int,
    route_order: str,
    topk_pattern: str,
) -> QSAInputs:
    if workload.context_length % storage_page_size:
        raise ValueError("storage page size must divide the benchmark context")
    if workload.context_length % _COMPRESS_RATIO:
        raise ValueError("context length must be divisible by the compression ratio")

    pages_per_request = workload.context_length // storage_page_size
    num_storage_pages = workload.batch_size * pages_per_request
    q = _randn_qkv(
        workload.rows,
        num_query_heads,
        _HEAD_DIM,
    )
    packed_kv = _randn_qkv(
        num_storage_pages,
        num_kv_heads,
        storage_page_size,
        2 * _HEAD_DIM,
    )
    k_cache, v_cache = packed_kv.split(_HEAD_DIM, dim=-1)

    request_page_bases = (
        torch.arange(workload.batch_size, dtype=torch.int32, device="cuda")
        * pages_per_request
    )
    request_permutations = torch.rand(
        (workload.batch_size, pages_per_request),
        dtype=torch.float32,
        device="cuda",
    ).argsort(dim=1)
    block_table = request_permutations.to(torch.int32) + request_page_bases[:, None]
    token_to_req = torch.arange(
        workload.batch_size,
        dtype=torch.int32,
        device="cuda",
    ).repeat_interleave(workload.seq_len_q)
    logical_positions = _logical_positions(workload)
    sequence_lengths = torch.full(
        (workload.batch_size,),
        workload.context_length,
        dtype=torch.int32,
        device="cuda",
    )
    block_indices = _random_causal_block_topk(
        logical_positions,
        token_to_req,
        workload.batch_size,
        workload.context_length,
        topk_pattern,
    )
    adjacent_topk_overlap = _adjacent_topk_overlap(
        block_indices,
        logical_positions,
        token_to_req,
        workload.context_length,
    )
    block_indices = _order_block_indices(
        block_indices,
        route_order,
        block_table,
        token_to_req,
        logical_positions,
        storage_page_size,
    )
    logical_indices = expand_qsa_block_indices_cuda(
        block_indices,
        logical_positions,
        sequence_lengths,
        token_to_req,
        _COMPRESS_RATIO,
        _TOKEN_TOPK,
    )
    visible_tokens = logical_positions + 1
    complete_pages = torch.minimum(
        visible_tokens // _COMPRESS_RATIO,
        torch.tensor(_BLOCK_TOPK, dtype=torch.int64, device="cuda"),
    )
    compact_lengths = (
        complete_pages * _COMPRESS_RATIO + visible_tokens % _COMPRESS_RATIO
    ).to(torch.int32)
    return QSAInputs(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_indices=block_indices,
        logical_indices=logical_indices,
        block_table=block_table,
        token_to_req=token_to_req,
        logical_positions=logical_positions,
        sequence_lengths=sequence_lengths,
        compact_lengths=compact_lengths,
        adjacent_topk_overlap=adjacent_topk_overlap,
    )


def _make_synthetic_route_trace(
    inputs: QSAInputs,
    storage_page_size: int,
    context_length: int,
) -> RealRouteTrace:
    """Expose a synthetic multi-request decode case to the union harness."""

    return RealRouteTrace(
        logical_indices=inputs.logical_indices.cpu(),
        block_table=inputs.block_table.cpu(),
        token_to_req=inputs.token_to_req.cpu(),
        logical_positions=inputs.logical_positions.cpu(),
        storage_page_size=storage_page_size,
        num_storage_pages=inputs.k_cache.shape[0],
        context_length=context_length,
    )


def _make_contiguous_baseline(
    inputs: QSAInputs,
    batch_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    cache_scope: str,
    physical_seq_len: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    from flashinfer.decode import get_prims_ts_batch_decode_workspace_size

    if physical_seq_len <= 0:
        raise ValueError("physical contiguous sequence length must be positive")
    max_pages_per_request = (
        physical_seq_len + _BASELINE_PAGE_SIZE - 1
    ) // _BASELINE_PAGE_SIZE
    if cache_scope == "request":
        num_cache_rows = batch_size
        row_cache_owners = inputs.token_to_req
    elif cache_scope == "route":
        num_cache_rows = inputs.q.shape[0]
        row_cache_owners = torch.arange(
            num_cache_rows,
            dtype=torch.int32,
            device="cuda",
        )
    else:
        raise ValueError(f"unsupported contiguous cache scope: {cache_scope}")

    packed_kv = _randn_qkv(
        num_cache_rows * max_pages_per_request,
        num_kv_heads,
        _BASELINE_PAGE_SIZE,
        2 * _HEAD_DIM,
    )
    k_cache, v_cache = packed_kv.split(_HEAD_DIM, dim=-1)

    local_pages = torch.arange(
        max_pages_per_request,
        dtype=torch.int32,
        device="cuda",
    )
    row_page_bases = row_cache_owners[:, None] * max_pages_per_request
    row_pages = row_page_bases + local_pages.unsqueeze(0)
    paged_kv_indices = row_pages.reshape(-1).contiguous()
    paged_kv_indptr = (
        torch.arange(
            inputs.q.shape[0] + 1,
            dtype=torch.int32,
            device="cuda",
        )
        * max_pages_per_request
    )
    seq_lens = torch.full(
        (inputs.q.shape[0],),
        physical_seq_len,
        dtype=torch.int32,
        device="cuda",
    )
    workspace_bytes = get_prims_ts_batch_decode_workspace_size(
        inputs.q.shape[0],
        num_query_heads,
        num_kv_heads,
        _HEAD_DIM,
        _BASELINE_PAGE_SIZE,
        physical_seq_len,
        q_dtype=inputs.q.dtype,
        kv_dtype=k_cache.dtype,
        out_dtype=_output_dtype(),
        mask_type="causal",
        window_left=_BASELINE_WINDOW_LEFT,
        device="cuda",
    )
    workspace = torch.zeros(workspace_bytes, dtype=torch.uint8, device="cuda")
    return (
        k_cache,
        v_cache,
        paged_kv_indptr,
        paged_kv_indices,
        seq_lens,
        workspace,
    )


def _make_grouped_contiguous_baseline(
    inputs: QSAInputs,
    group_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    physical_seq_len: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    """Build an exact SQ-grouped causal SWA baseline and its live KV volume."""

    from flashinfer.decode import get_prims_ts_batch_decode_workspace_size

    rows = inputs.q.shape[0]
    if rows % group_size:
        raise ValueError("grouped contiguous baseline requires complete Q groups")
    groups = rows // group_size
    grouped_requests = inputs.token_to_req.view(groups, group_size)
    grouped_positions = inputs.logical_positions.view(groups, group_size)
    if not bool((grouped_requests == grouped_requests[:, :1]).all().item()):
        raise ValueError("grouped contiguous baseline cannot cross requests")
    if group_size > 1 and not bool(
        (grouped_positions[:, 1:] == grouped_positions[:, :-1] + 1).all().item()
    ):
        raise ValueError("grouped contiguous baseline requires adjacent Q positions")

    max_pages_per_request = (
        physical_seq_len + _BASELINE_PAGE_SIZE - 1
    ) // _BASELINE_PAGE_SIZE
    num_requests = int(inputs.token_to_req.max().item()) + 1
    packed_kv = _randn_qkv(
        num_requests * max_pages_per_request,
        num_kv_heads,
        _BASELINE_PAGE_SIZE,
        2 * _HEAD_DIM,
    )
    k_cache, v_cache = packed_kv.split(_HEAD_DIM, dim=-1)

    local_pages = torch.arange(
        max_pages_per_request,
        dtype=torch.int32,
        device="cuda",
    )
    request_page_bases = grouped_requests[:, :1] * max_pages_per_request
    paged_kv_indices = (request_page_bases + local_pages).reshape(-1).contiguous()
    paged_kv_indptr = (
        torch.arange(groups + 1, dtype=torch.int32, device="cuda")
        * max_pages_per_request
    )
    seq_lens = (grouped_positions[:, -1] + 1).to(torch.int32)
    if int(seq_lens.max().item()) > physical_seq_len:
        raise ValueError("grouped contiguous sequence exceeds the physical cache")

    # The union of four adjacent 2K causal windows starts at the first query's
    # left boundary and ends at the final query. This is the exact KV volume a
    # perfectly shared grouped SWA producer must stage.
    window_starts = torch.clamp(
        grouped_positions[:, 0] - _BASELINE_WINDOW_LEFT,
        min=0,
    )
    grouped_swa_kv_tokens = int(
        (grouped_positions[:, -1] - window_starts + 1).sum().item()
    )
    workspace_bytes = get_prims_ts_batch_decode_workspace_size(
        groups,
        num_query_heads,
        num_kv_heads,
        _HEAD_DIM,
        _BASELINE_PAGE_SIZE,
        physical_seq_len,
        seq_len_q=group_size,
        q_dtype=inputs.q.dtype,
        kv_dtype=k_cache.dtype,
        out_dtype=_grouped_swa_output_dtype(),
        mask_type="causal",
        window_left=_BASELINE_WINDOW_LEFT,
        device="cuda",
    )
    workspace = torch.zeros(workspace_bytes, dtype=torch.uint8, device="cuda")
    return (
        k_cache,
        v_cache,
        paged_kv_indptr,
        paged_kv_indices,
        seq_lens,
        workspace,
        grouped_swa_kv_tokens,
    )


def _run_case(
    workload: Workload,
    tp_size: int,
    storage_page_size: int,
    route_order: str,
    topk_pattern: str,
    contiguous_cache_scope: str,
    warmup_iterations: int,
    iterations: int,
    use_cuda_graph: bool,
    cold_l2: bool,
    prepared_inputs: QSAInputs | None = None,
) -> BenchmarkResult:
    from flashinfer.decode import prims_ts_batch_decode_with_kv_cache

    num_query_heads, num_kv_heads = _tp_heads(tp_size)
    inputs = prepared_inputs
    if inputs is None:
        inputs = _make_qsa_inputs(
            workload,
            num_query_heads,
            num_kv_heads,
            storage_page_size,
            route_order,
            topk_pattern,
        )
    page_capacity = _BLOCK_TOPK + 1
    paged_kv_indptr = torch.empty(
        workload.rows + 1,
        dtype=torch.int32,
        device="cuda",
    )
    paged_kv_indices = torch.empty(
        workload.rows * page_capacity,
        dtype=torch.int32,
        device="cuda",
    )
    seq_lens = torch.empty(workload.rows, dtype=torch.int32, device="cuda")
    expanded_indices = torch.empty_like(inputs.logical_indices)

    def build_metadata() -> None:
        with torch.cuda.nvtx.range("qsa_metadata"):
            qsa_prims_ts_build_page4_metadata(
                inputs.block_indices,
                inputs.block_table,
                inputs.token_to_req,
                inputs.logical_positions,
                storage_page_size,
                1,
                None,
                paged_kv_indptr,
                paged_kv_indices,
                seq_lens,
            )

    def build_legacy_metadata() -> None:
        with torch.cuda.nvtx.range("qsa_legacy_metadata"):
            expand_qsa_block_indices_cuda(
                inputs.block_indices,
                inputs.logical_positions,
                inputs.sequence_lengths,
                inputs.token_to_req,
                _COMPRESS_RATIO,
                _TOKEN_TOPK,
                expanded_indices,
            )
            qsa_build_page4_paged_metadata(
                expanded_indices,
                inputs.block_table,
                inputs.token_to_req,
                inputs.logical_positions,
                storage_page_size,
                paged_kv_indptr=paged_kv_indptr,
                paged_kv_indices=paged_kv_indices,
                seq_lens=seq_lens,
            )

    build_metadata()
    qsa_workspace = torch.zeros(
        qsa_prims_ts_workspace_size(
            inputs.q,
            inputs.k_cache,
            _QSA_MAX_SEQ_LEN,
            out_dtype=_output_dtype(),
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    qsa_output = _empty_output_like(inputs.q)
    triton_output = _empty_output_like(inputs.q)
    triton_k_cache = inputs.k_cache.permute(0, 2, 1, 3)
    triton_v_cache = inputs.v_cache.permute(0, 2, 1, 3)

    def qsa_attention() -> None:
        with torch.cuda.nvtx.range("qsa_attention"):
            qsa_prims_ts_paged_attention(
                inputs.q,
                inputs.k_cache,
                inputs.v_cache,
                qsa_workspace,
                paged_kv_indptr,
                paged_kv_indices,
                seq_lens,
                _QSA_MAX_SEQ_LEN,
                qsa_output,
            )

    def qsa_end_to_end() -> None:
        build_metadata()
        qsa_attention()

    def triton_sparse_attention() -> None:
        with torch.cuda.nvtx.range("triton_sparse_attention"):
            qsa_sparse_paged_attention(
                inputs.q,
                triton_k_cache,
                triton_v_cache,
                inputs.logical_indices,
                inputs.block_table,
                inputs.token_to_req,
                triton_output,
            )

    def triton_end_to_end() -> None:
        expand_qsa_block_indices_cuda(
            inputs.block_indices,
            inputs.logical_positions,
            inputs.sequence_lengths,
            inputs.token_to_req,
            _COMPRESS_RATIO,
            _TOKEN_TOPK,
            expanded_indices,
        )
        with torch.cuda.nvtx.range("triton_sparse_attention"):
            qsa_sparse_paged_attention(
                inputs.q,
                triton_k_cache,
                triton_v_cache,
                expanded_indices,
                inputs.block_table,
                inputs.token_to_req,
                triton_output,
            )

    (
        baseline_k,
        baseline_v,
        baseline_indptr,
        baseline_indices,
        baseline_seq_lens,
        baseline_workspace,
    ) = _make_contiguous_baseline(
        inputs,
        workload.batch_size,
        num_query_heads,
        num_kv_heads,
        contiguous_cache_scope,
        workload.context_length,
    )
    baseline_output = _empty_output_like(inputs.q)

    def contiguous_attention() -> None:
        with torch.cuda.nvtx.range("contiguous_attention"):
            prims_ts_batch_decode_with_kv_cache(
                inputs.q,
                (baseline_k, baseline_v),
                baseline_workspace,
                baseline_indptr,
                baseline_indices,
                baseline_seq_lens,
                workload.context_length,
                bmm1_scale=_HEAD_DIM**-0.5,
                out=baseline_output,
                out_dtype=baseline_output.dtype,
                mask_type="causal",
                window_left=_BASELINE_WINDOW_LEFT,
                page_size=_BASELINE_PAGE_SIZE,
            )

    qsa_attention()
    triton_sparse_attention()
    torch.cuda.synchronize()
    output_diff = (qsa_output.float() - triton_output.float()).abs()
    triton_max_abs_diff = float(output_diff.max().item())
    comparison_atol = _comparison_atol()
    if not math.isfinite(triton_max_abs_diff) or triton_max_abs_diff > comparison_atol:
        row_diffs = torch.nan_to_num(
            output_diff.flatten(1).amax(dim=1),
            nan=float("inf"),
        )
        worst_row = int(row_diffs.argmax().item())
        logical = inputs.logical_indices[worst_row]
        logical = logical[logical >= 0].long()
        request = inputs.token_to_req[worst_row].long()
        pages = inputs.block_table[
            request,
            logical // storage_page_size,
        ].long()
        offsets = logical % storage_page_size
        repeats = num_query_heads // num_kv_heads
        keys = triton_k_cache[pages, offsets].repeat_interleave(repeats, dim=1)
        values = triton_v_cache[pages, offsets].repeat_interleave(repeats, dim=1)
        scores = torch.einsum(
            "hd,khd->hk",
            inputs.q[worst_row].float(),
            keys.float(),
        )
        probabilities = torch.softmax(scores * (_HEAD_DIM**-0.5), dim=-1)
        reference = torch.einsum(
            "hk,khd->hd",
            probabilities,
            values.float(),
        )
        qsa_reference_diff = float(
            (qsa_output[worst_row].float() - reference).abs().max().item()
        )
        triton_reference_diff = float(
            (triton_output[worst_row].float() - reference).abs().max().item()
        )
        qsa_abs_max = float(qsa_output[worst_row].float().abs().max().item())
        triton_abs_max = float(triton_output[worst_row].float().abs().max().item())
        reference_abs_max = float(reference.abs().max().item())
        raise AssertionError(
            "QSA backend mismatch: "
            f"worst_row={worst_row}, request={int(request.item())}, "
            f"position={int(inputs.logical_positions[worst_row].item())}, "
            f"PrimTS-vs-Triton={triton_max_abs_diff:.6f}, "
            f"PrimTS-vs-reference={qsa_reference_diff:.6f}, "
            f"Triton-vs-reference={triton_reference_diff:.6f}, "
            f"absmax=(PrimTS={qsa_abs_max:.6f}, "
            f"Triton={triton_abs_max:.6f}, reference={reference_abs_max:.6f})"
        )
    torch.testing.assert_close(
        qsa_output.float(),
        triton_output.float(),
        rtol=comparison_atol,
        atol=comparison_atol,
    )

    timing_args = {
        "warmup_iterations": warmup_iterations,
        "iterations": iterations,
        "use_cuda_graph": use_cuda_graph,
        "cold_l2": cold_l2,
    }
    timings = _time_cuda_interleaved(
        {
            "metadata": build_metadata,
            "legacy_metadata": build_legacy_metadata,
            "qsa": qsa_attention,
            "qsa_end_to_end": qsa_end_to_end,
            "triton": triton_sparse_attention,
            "triton_end_to_end": triton_end_to_end,
            "contiguous": contiguous_attention,
        },
        **timing_args,
    )
    compact_min, compact_max = torch.aminmax(inputs.compact_lengths)
    return BenchmarkResult(
        workload=workload,
        tp_size=tp_size,
        route_order=route_order,
        contiguous_cache_scope=contiguous_cache_scope,
        topk_pattern=topk_pattern,
        adjacent_topk_overlap=inputs.adjacent_topk_overlap,
        compact_min=int(compact_min.item()),
        compact_max=int(compact_max.item()),
        selected_kv_tokens=int(inputs.compact_lengths.sum().item()),
        kv_bytes_per_token=(
            2 * num_kv_heads * _HEAD_DIM * inputs.k_cache.element_size()
        ),
        metadata_us=timings["metadata"],
        legacy_metadata_us=timings["legacy_metadata"],
        qsa_attention_us=timings["qsa"],
        qsa_end_to_end_us=timings["qsa_end_to_end"],
        triton_sparse_us=timings["triton"],
        triton_end_to_end_us=timings["triton_end_to_end"],
        contiguous_us=timings["contiguous"],
        triton_max_abs_diff=triton_max_abs_diff,
    )


def _print_header() -> None:
    header = (
        " phase    TP    route     topk  contig overlap tail   BS    SQ   rows "
        "compact_KV metadata_us legacy_meta_us qsa_us qsa_e2e_us "
        "triton_us triton_e2e_us contig_us "
        "selected_KV nominal_KV_GB qsa_KV_TBps contig_KV_TBps "
        "qsa/contig qsa_e2e/contig triton/contig triton_e2e/contig "
        "prims_speedup max_diff"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)


def _print_result(result: BenchmarkResult) -> None:
    workload = result.workload
    end_tail = (
        "-" if workload.decode_end_tail is None else str(workload.decode_end_tail)
    )
    compact_range = (
        str(result.compact_min)
        if result.compact_min == result.compact_max
        else f"{result.compact_min}-{result.compact_max}"
    )
    print(
        f"{workload.phase:>7} {result.tp_size:>5} {result.route_order:>8} "
        f"{result.topk_pattern:>8} {result.contiguous_cache_scope:>7} "
        f"{result.adjacent_topk_overlap:>7.3f} "
        f"{end_tail:>4} "
        f"{workload.batch_size:>4} {workload.seq_len_q:>5} "
        f"{workload.rows:>6} {compact_range:>10} "
        f"{result.metadata_us:>11.2f} {result.legacy_metadata_us:>14.2f} "
        f"{result.qsa_attention_us:>6.2f} "
        f"{result.qsa_end_to_end_us:>10.2f} "
        f"{result.triton_sparse_us:>9.2f} "
        f"{result.triton_end_to_end_us:>13.2f} "
        f"{result.contiguous_us:>9.2f} "
        f"{result.selected_kv_tokens:>11} "
        f"{result.selected_kv_gb:>13.2f} "
        f"{result.qsa_effective_kv_tbps:>11.2f} "
        f"{result.contiguous_effective_kv_tbps:>14.2f} "
        f"{result.qsa_ratio:>10.3f} {result.end_to_end_ratio:>10.3f} "
        f"{result.triton_ratio:>13.3f} "
        f"{result.triton_end_to_end_ratio:>18.3f} "
        f"{result.prims_ts_speedup_over_triton:>13.3f} "
        f"{result.triton_max_abs_diff:>8.5f}",
        flush=True,
    )


def _print_union_header() -> None:
    header = (
        " union TP group membership scope groups split mean_union max_union launch_KV "
        "load_reduction "
        "metadata_us legacy_meta_us flat_qsa_us union_us union_e2e_us "
        "triton_us triton_e2e_us swa4_us "
        "proj_swa_us speedup "
        "source_KV union_KV swa4_KV nonshared source_GB union_GB "
        "flat_logical_KV_TBps union_logical_KV_TBps "
        "union/swa4 e2e/swa4 triton/swa4 triton_e2e/swa4 "
        "union/triton prims_speedup union/proj e2e/proj "
        "union_diff triton_diff"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)


def _print_union_result(result: UnionUpperBoundResult) -> None:
    print(
        f" union {result.tp_size:>2} {result.group_size:>5} "
        f"{result.membership_mode:>10} {result.row_scope:>9} "
        f"{result.groups:>6} {result.splits_kv:>5} "
        f"{result.mean_union_blocks:>10.1f} "
        f"{result.max_union_blocks:>9} {result.launch_max_seq_len:>9} "
        f"{result.ideal_load_reduction:>14.3f} "
        f"{result.metadata_us:>11.2f} "
        f"{result.legacy_metadata_us:>14.2f} "
        f"{result.flattened_qsa_us:>11.2f} "
        f"{result.union_upper_bound_us:>8.2f} "
        f"{result.union_end_to_end_us:>12.2f} "
        f"{result.triton_sparse_us:>9.2f} "
        f"{result.triton_end_to_end_us:>13.2f} "
        f"{result.grouped_swa_us:>7.2f} "
        f"{result.projected_grouped_swa_us:>11.2f} {result.speedup:>7.3f} "
        f"{result.source_kv_tokens:>9} {result.union_kv_tokens:>8} "
        f"{result.grouped_swa_kv_tokens:>7} "
        f"{result.non_shared_kv_ratio:>9.3f} "
        f"{result.source_kv_gb:>9.2f} {result.union_kv_gb:>8.2f} "
        f"{result.flattened_logical_kv_tbps:>20.2f} "
        f"{result.union_logical_kv_tbps:>21.2f} "
        f"{result.grouped_swa_ratio:>10.3f} "
        f"{result.grouped_swa_end_to_end_ratio:>8.3f} "
        f"{result.triton_grouped_swa_ratio:>11.3f} "
        f"{result.triton_grouped_swa_end_to_end_ratio:>17.3f} "
        f"{result.union_triton_ratio:>12.3f} "
        f"{result.prims_ts_speedup_over_triton:>13.3f} "
        f"{result.projected_contiguous_ratio:>10.3f} "
        f"{result.projected_end_to_end_ratio:>8.3f} "
        f"{result.max_abs_diff:>10.5f} "
        f"{result.triton_max_abs_diff:>11.5f}",
        flush=True,
    )


def _make_workloads(args: argparse.Namespace) -> list[Workload]:
    workloads = []
    if "prefill" in args.phases:
        workloads.append(
            Workload(
                phase="prefill",
                batch_size=1,
                seq_len_q=args.prefill_seq_len,
                context_length=args.context_length,
            )
        )
    if "decode" in args.phases:
        workloads.extend(
            Workload(
                phase="decode",
                batch_size=batch_size,
                seq_len_q=seq_len_q,
                context_length=args.context_length,
                decode_end_tail=end_tail,
            )
            for batch_size in args.decode_batch_sizes
            for seq_len_q in args.decode_seq_lens_q
            for end_tail in args.decode_end_tails
        )
    return workloads


def main() -> None:
    global _HEAD_DIM, _QKV_DTYPE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=("prefill", "decode"),
        default=["prefill", "decode"],
    )
    parser.add_argument("--tp-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--prefill-seq-len", type=int, default=8192)
    parser.add_argument(
        "--qkv-dtype",
        choices=("bf16", "fp8"),
        default="bf16",
        help="Q/K/V dtype; both modes write model-facing BF16 output",
    )
    parser.add_argument(
        "--decode-batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 64, 256],
    )
    parser.add_argument(
        "--decode-seq-lens-q",
        type=int,
        nargs="+",
        default=[1, 4],
    )
    parser.add_argument(
        "--decode-union-group-sizes",
        type=int,
        nargs="+",
        choices=(2, 4),
        default=[],
        help=(
            "Also benchmark exact grouped QSA unions when decode SQ is "
            "divisible by the requested group size. Each SWA control keeps "
            "the same grouped SQ."
        ),
    )
    parser.add_argument(
        "--decode-union-only",
        action="store_true",
        help=(
            "For compatible decode SQ values, skip the ordinary flattened "
            "case; the union row still reports flattened PrimTS and Triton."
        ),
    )
    parser.add_argument(
        "--decode-end-tails",
        type=int,
        nargs="+",
        default=[0],
        help=(
            "Compact causal tail lengths for the final decode query "
            "(0..3; use 0 3 for exact tail qualification)"
        ),
    )
    parser.add_argument("--storage-page-size", type=int, default=256)
    parser.add_argument(
        "--topk-dumps",
        nargs="+",
        type=Path,
        help="Replay contiguous chunks from one captured real request.",
    )
    parser.add_argument(
        "--topk-dump-storage-page-size",
        type=int,
        help="Override the main-cache page size recorded by older dumps.",
    )
    parser.add_argument(
        "--topk-dump-patterns",
        nargs="+",
        choices=("captured", "independent", "shared-request"),
        default=["captured"],
        help="Replay captured routes or matched synthetic rankings.",
    )
    parser.add_argument(
        "--topk-dump-union-group-sizes",
        nargs="+",
        type=int,
        default=[],
        help="Time shared-union page loading for real routes.",
    )
    parser.add_argument(
        "--topk-dump-union-membership-modes",
        nargs="+",
        choices=("full", "masked-dense", "masked"),
        default=["masked"],
        help=(
            "Use the semantic full/masked modes or diagnostic controls that "
            "isolate causal masking from real per-query membership."
        ),
    )
    parser.add_argument(
        "--topk-dump-union-row-scope",
        choices=("full", "saturated"),
        default="saturated",
        help="Group the full captured prefill or only its saturated suffix.",
    )
    parser.add_argument(
        "--profile-union-component",
        choices=("metadata", "flattened", "union", "contiguous"),
        help="Bracket one warmed union component with CUDA profiler start/stop.",
    )
    parser.add_argument(
        "--topk-dump-union-only",
        action="store_true",
        help="Skip the ordinary captured-route replay when timing unions.",
    )
    parser.add_argument(
        "--topk-dump-union-max-groups",
        type=int,
        help="Limit real-route union groups for focused correctness diagnostics.",
    )
    parser.add_argument(
        "--topk-dump-union-group-counts",
        type=int,
        nargs="+",
        help=(
            "Benchmark several prefix group counts in one process; this is "
            "mutually exclusive with --topk-dump-union-max-groups."
        ),
    )
    parser.add_argument(
        "--topk-dump-union-start-group",
        type=int,
        default=0,
        help="Skip real-route union groups for focused correctness diagnostics.",
    )
    parser.add_argument(
        "--topk-dump-union-static-max-seq-len",
        type=int,
        help=(
            "Resolve and launch grouped attention with a graph-stable KV upper "
            "bound instead of the observed union maximum."
        ),
    )
    parser.add_argument(
        "--experimental-q4-keeps-issuers",
        type=int,
        choices=(1, 2, 4, 8),
        help=(
            "Force the unqualified Q64 Keeps profile for the dense TP2/Q4 "
            "real-route union control. Production dispatch is unchanged."
        ),
    )
    parser.add_argument(
        "--experimental-q4-keeps-kv256-d128",
        action="store_true",
        help=(
            "Force the existing two-instance Q64/KV256 D128 mma_ws profile "
            "for the dense TP2/Q4 real-route union control. Production "
            "dispatch is unchanged."
        ),
    )
    parser.add_argument(
        "--experimental-q4-keeps-kv256-d256",
        action="store_true",
        help=(
            "Force the one-instance Q64/KV256 D256 mma_ws profile for the "
            "TP2/Q4 real-route union experiment. Production dispatch is unchanged."
        ),
    )
    parser.add_argument(
        "--experimental-q4-keeps-kv256-load-warps",
        type=int,
        choices=(1, 2, 4, 8),
        default=1,
        help=(
            "Number of encoded page-4 TMA issuer warps in the experimental "
            "KV256 profile; one selects the profile default (four for D256)."
        ),
    )
    parser.add_argument(
        "--experimental-q4-keeps-kv128-splits",
        type=int,
        choices=(1, 2, 3, 4, 5, 8),
        default=1,
        help=(
            "Requested split-KV fanout cap for the experimental KV128/D256 "
            "profile; values above one use the standalone reduction kernel."
        ),
    )
    parser.add_argument(
        "--experimental-q4-keeps-kv256-splits",
        type=int,
        choices=(1, 2, 3, 4, 5, 8),
        default=1,
        help=(
            "Requested split-KV fanout cap for the experimental KV256/D256 "
            "profile; the common work cap may lower it, and values above one "
            "use the standalone reduction kernel."
        ),
    )
    parser.add_argument(
        "--route-order",
        choices=("topk", "logical", "physical"),
        default="topk",
        help="Order selected blocks before expansion; sorting cost is not timed.",
    )
    parser.add_argument(
        "--contiguous-cache-scope",
        choices=("request", "route"),
        default="request",
        help=(
            "Share full-context baseline pages across query tokens in a "
            "request, or allocate independent pages for every flattened "
            "route."
        ),
    )
    parser.add_argument(
        "--topk-pattern",
        choices=("independent", "shared-request"),
        default="independent",
        help="Use independent route rankings or one causal ranking per request.",
    )
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eager", action="store_true", help="Disable CUDA graphs")
    parser.add_argument(
        "--warm-l2",
        action="store_true",
        help="Do not evict L2 before each timed replay (default: cold L2)",
    )
    args = parser.parse_args()

    _QKV_DTYPE = torch.bfloat16 if args.qkv_dtype == "bf16" else torch.float8_e4m3fn

    if not torch.cuda.is_available():
        raise RuntimeError("QSA PrimTS benchmark requires CUDA")
    if not has_qsa_prims_ts_attention():
        raise RuntimeError("Installed FlashInfer lacks encoded page-4 PrimTS")
    if args.warmup_iterations < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations positive")
    if args.decode_union_only and not args.decode_union_group_sizes:
        raise ValueError("decode union-only mode requires union group sizes")
    if (
        args.topk_dump_union_max_groups is not None
        and args.topk_dump_union_max_groups <= 0
    ):
        raise ValueError("union max groups must be positive")
    if args.topk_dump_union_group_counts is not None:
        if args.topk_dump_union_max_groups is not None:
            raise ValueError("choose union max groups or group counts, not both")
        if any(count <= 0 for count in args.topk_dump_union_group_counts):
            raise ValueError("union group counts must be positive")
    if args.topk_dump_union_start_group < 0:
        raise ValueError("union start group must be non-negative")
    if (
        args.topk_dump_union_static_max_seq_len is not None
        and args.topk_dump_union_static_max_seq_len <= 0
    ):
        raise ValueError("static union max sequence length must be positive")
    experimental_q4_keeps = (
        args.experimental_q4_keeps_issuers is not None
        or args.experimental_q4_keeps_kv256_d128
        or args.experimental_q4_keeps_kv256_d256
    )
    experimental_profiles = sum(
        (
            args.experimental_q4_keeps_issuers is not None,
            args.experimental_q4_keeps_kv256_d128,
            args.experimental_q4_keeps_kv256_d256,
        )
    )
    if experimental_profiles > 1:
        raise ValueError(
            "choose one KV128 issuer, KV256/D128, or KV256/D256 experiment"
        )
    if args.experimental_q4_keeps_kv256_load_warps != 1:
        if not (
            args.experimental_q4_keeps_kv256_d128
            or args.experimental_q4_keeps_kv256_d256
        ):
            raise ValueError("KV256 load-warp tuning requires a KV256 profile")
        if (
            args.experimental_q4_keeps_kv256_d128
            and args.experimental_q4_keeps_kv256_load_warps == 8
        ):
            raise ValueError("KV256/D128 supports at most four load warps")
        if (
            args.experimental_q4_keeps_kv256_d256
            and args.experimental_q4_keeps_kv256_load_warps not in (1, 8)
        ):
            raise ValueError("KV256/D256 supports its four-warp default or eight")
    if args.experimental_q4_keeps_kv256_splits != 1 and not (
        args.experimental_q4_keeps_kv256_d256
    ):
        raise ValueError("KV256 split tuning requires the D256 profile")
    if args.experimental_q4_keeps_kv128_splits != 1 and (
        args.experimental_q4_keeps_issuers is None
    ):
        raise ValueError("KV128 split tuning requires the KV128/D256 profile")
    if experimental_q4_keeps:
        if not args.topk_dumps:
            raise ValueError("experimental Q4 Keeps requires real top-k dumps")
        if not args.tp_sizes or any(tp_size not in (1, 2) for tp_size in args.tp_sizes):
            raise ValueError("experimental Q4 Keeps supports TP1 and TP2 qualification")
        if args.topk_dump_union_group_sizes != [4]:
            raise ValueError("experimental Q4 Keeps requires union group size four")
        if args.experimental_q4_keeps_kv256_d128:
            _HEAD_DIM = 128
        _install_experimental_q4_keeps_selector(
            args.experimental_q4_keeps_issuers,
            kv128_splits=args.experimental_q4_keeps_kv128_splits,
            kv256_d128=args.experimental_q4_keeps_kv256_d128,
            kv256_d256=args.experimental_q4_keeps_kv256_d256,
            kv256_load_warps=args.experimental_q4_keeps_kv256_load_warps,
            kv256_splits=args.experimental_q4_keeps_kv256_splits,
        )

    torch.manual_seed(args.seed)
    cache_mode = "warm"
    if not args.warm_l2:
        cache_mode = f"cold ({_l2_flush_buffer().numel() / 2**20:.0f} MiB eviction)"
    print(
        f"# timing: L2={cache_mode}, CUDA graph={'off' if args.eager else 'on'}",
        flush=True,
    )
    output_dtype_name = "bf16" if _output_dtype() == torch.bfloat16 else "fp16"
    print(f"# qkv/output dtype: {args.qkv_dtype}->{output_dtype_name}", flush=True)
    if experimental_q4_keeps:
        d256_load_warps = (
            4
            if args.experimental_q4_keeps_kv256_load_warps == 1
            else args.experimental_q4_keeps_kv256_load_warps
        )
        profile = (
            "KV256/D128 two-instance mma_ws "
            f"load_warps={args.experimental_q4_keeps_kv256_load_warps}"
            if args.experimental_q4_keeps_kv256_d128
            else (
                "KV256/D256 one-instance full-width mma_ws "
                f"load_warps={d256_load_warps}, "
                f"split_request={args.experimental_q4_keeps_kv256_splits}"
                if args.experimental_q4_keeps_kv256_d256
                else (
                    f"KV128/D256 issuers={args.experimental_q4_keeps_issuers}, "
                    f"split_request={args.experimental_q4_keeps_kv128_splits}"
                )
            )
        )
        print(
            "# experimental Q64 Keeps: "
            f"profile={profile}, TP={','.join(map(str, args.tp_sizes))}, "
            f"membership={','.join(args.topk_dump_union_membership_modes)}",
            flush=True,
        )
    _print_header()
    if args.topk_dumps:
        trace = _load_real_route_trace(
            args.topk_dumps,
            args.topk_dump_storage_page_size,
        )
        workload = Workload(
            phase="trace",
            batch_size=1,
            seq_len_q=trace.logical_positions.numel(),
            context_length=trace.context_length,
        )
        if args.topk_dump_union_only and not args.topk_dump_union_group_sizes:
            raise ValueError("union-only mode requires union group sizes")
        if not args.topk_dump_union_only:
            for topk_pattern in args.topk_dump_patterns:
                for tp_size in args.tp_sizes:
                    num_query_heads, num_kv_heads = _tp_heads(tp_size)
                    inputs = _make_real_qsa_inputs(
                        trace,
                        num_query_heads,
                        num_kv_heads,
                        topk_pattern,
                    )
                    captured = topk_pattern == "captured"
                    result = _run_case(
                        workload,
                        tp_size,
                        trace.storage_page_size,
                        "captured" if captured else "topk",
                        "real" if captured else topk_pattern,
                        args.contiguous_cache_scope,
                        args.warmup_iterations,
                        args.iterations,
                        not args.eager,
                        not args.warm_l2,
                        prepared_inputs=inputs,
                    )
                    _print_result(result)
        if args.topk_dump_union_group_sizes:
            if any(size not in (2, 4) for size in args.topk_dump_union_group_sizes):
                raise ValueError("union group sizes must be two or four")
            _print_union_header()
            for tp_size in args.tp_sizes:
                num_query_heads, num_kv_heads = _tp_heads(tp_size)
                inputs = _make_real_qsa_inputs(
                    trace,
                    num_query_heads,
                    num_kv_heads,
                    "captured",
                )
                for membership_mode in args.topk_dump_union_membership_modes:
                    for group_size in args.topk_dump_union_group_sizes:
                        group_counts = args.topk_dump_union_group_counts or [
                            args.topk_dump_union_max_groups
                        ]
                        for max_groups in group_counts:
                            result = _run_union_upper_bound(
                                trace,
                                inputs,
                                tp_size,
                                group_size,
                                membership_mode,
                                args.topk_dump_union_row_scope,
                                args.warmup_iterations,
                                args.iterations,
                                not args.eager,
                                not args.warm_l2,
                                args.profile_union_component,
                                args.topk_dump_union_start_group,
                                max_groups,
                                args.topk_dump_union_static_max_seq_len,
                            )
                            _print_union_result(result)
        return

    if args.storage_page_size % _COMPRESS_RATIO:
        raise ValueError("storage page size must be divisible by four")
    if args.context_length < max(args.decode_seq_lens_q):
        raise ValueError("decode SQ cannot exceed the context length")
    if args.context_length < _TOKEN_TOPK:
        raise ValueError("context length must cover the 2048-token QSA top-k")
    if "prefill" in args.phases and args.prefill_seq_len != args.context_length:
        raise ValueError("prefill benchmark requires SQ=SKV")
    if args.context_length % args.storage_page_size:
        raise ValueError("storage page size must divide the context length")
    if args.context_length % _COMPRESS_RATIO:
        raise ValueError("context length must be divisible by four")
    if any(size <= 0 for size in args.decode_batch_sizes):
        raise ValueError("decode batch sizes must be positive")
    if any(length <= 0 for length in args.decode_seq_lens_q):
        raise ValueError("decode query lengths must be positive")
    if any(tail not in range(_COMPRESS_RATIO) for tail in args.decode_end_tails):
        raise ValueError("decode end tails must be in [0, 3]")
    for seq_len_q in args.decode_seq_lens_q:
        for end_tail in args.decode_end_tails:
            end_visible = (
                args.context_length - (args.context_length - end_tail) % _COMPRESS_RATIO
            )
            if seq_len_q > end_visible:
                raise ValueError(
                    "decode SQ cannot exceed the visible context selected by its tail"
                )

    if args.decode_union_group_sizes:
        _print_union_header()
    for tp_size in args.tp_sizes:
        for workload in _make_workloads(args):
            num_query_heads, num_kv_heads = _tp_heads(tp_size)
            inputs = _make_qsa_inputs(
                workload,
                num_query_heads,
                num_kv_heads,
                args.storage_page_size,
                args.route_order,
                args.topk_pattern,
            )
            grouped_sizes = tuple(
                group_size
                for group_size in args.decode_union_group_sizes
                if workload.phase == "decode" and workload.seq_len_q % group_size == 0
            )
            if not (args.decode_union_only and grouped_sizes):
                result = _run_case(
                    workload,
                    tp_size,
                    args.storage_page_size,
                    args.route_order,
                    args.topk_pattern,
                    args.contiguous_cache_scope,
                    args.warmup_iterations,
                    args.iterations,
                    not args.eager,
                    not args.warm_l2,
                    prepared_inputs=inputs,
                )
                _print_result(result)
            if grouped_sizes:
                trace = _make_synthetic_route_trace(
                    inputs,
                    args.storage_page_size,
                    workload.context_length,
                )
                for group_size in grouped_sizes:
                    union_result = _run_union_upper_bound(
                        trace,
                        inputs,
                        tp_size,
                        group_size,
                        "masked",
                        "full",
                        args.warmup_iterations,
                        args.iterations,
                        not args.eager,
                        not args.warm_l2,
                        None,
                        0,
                        None,
                        group_size * (_QSA_MAX_SEQ_LEN + 1),
                    )
                    _print_union_result(union_result)


if __name__ == "__main__":
    main()
