# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native D512 sparse MLA over vLLM's already mapped storage-row indices."""

import functools

import torch

from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kMxfp8Dynamic,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _compact_rows(
    source,
    source_lengths,
    indices,
    lengths,
    SOURCE_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    length = tl.load(source_lengths + row)
    value = tl.load(source + row * SOURCE_STRIDE + col, col < WIDTH, -1)
    valid = (col < WIDTH) & (col < length) & (value >= 0)
    dest = tl.cumsum(valid.to(tl.int32)) - 1
    count = tl.sum(valid.to(tl.int32))
    tl.store(indices + row * WIDTH + dest, value, valid)
    tl.store(indices + row * WIDTH + col, -1, (col >= count) & (col < WIDTH))
    tl.store(lengths + row, count)


@triton.jit
def _prepare_routes(
    primary,
    extra,
    primary_lengths,
    extra_lengths,
    routes,
    execution_lengths,
    counts,
    scale_params,
    sinks,
    positions,
    quant_positions,
    KS: tl.constexpr,
    KC: tl.constexpr,
    HEADS: tl.constexpr,
    CAPACITY: tl.constexpr,
    BMM1: tl.constexpr,
    BMM2: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    QUANTIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    ls = tl.load(primary_lengths + row)
    lc = tl.load(extra_lengths + row) if KC else 0
    span_s = tl.cdiv(ls, 128) * 128
    span_c = tl.cdiv(lc, 128) * 128
    offset = tl.arange(0, BLOCK)
    for start in range(0, CAPACITY, BLOCK):
        col = start + offset
        a = tl.load(primary + row * KS + col, col < ls, 0x7FFFFFFF)
        local = col - span_s
        if KC:
            b = tl.load(extra + row * KC + local, (local >= 0) & (local < lc), -1)
            b = b | -2147483648
        else:
            b = tl.full((BLOCK,), -1, tl.int32)
        tl.store(
            routes + row * CAPACITY + col, tl.where(col < span_s, a, b), col < CAPACITY
        )
    tl.store(execution_lengths + row, tl.maximum(span_s + span_c, 1))
    tl.store(counts + row, ls + lc)
    tl.store(scale_params + 2 + HEADS + row, (ls + lc).to(tl.float32))
    if QUANTIZE:
        tl.store(quant_positions + row, tl.load(positions + row).to(tl.int32))
    if row == 0:
        tl.store(scale_params, BMM1)
        tl.store(scale_params + 1, BMM2)
        h = tl.arange(0, triton.next_power_of_2(HEADS))
        sink = (
            tl.load(sinks + h, h < HEADS, -float("inf")) if HAS_SINKS else -float("inf")
        )
        tl.store(scale_params + 2 + h, sink, h < HEADS)


def _compact(source, source_lengths):
    rows, width = source.shape
    indices = torch.empty((rows, width), device=source.device, dtype=torch.int32)
    lengths = torch.empty((rows,), device=source.device, dtype=torch.int32)
    _compact_rows[(rows,)](
        source,
        source_lengths,
        indices,
        lengths,
        source.stride(0),
        width,
        triton.next_power_of_2(width),
    )
    return indices, lengths


# Layers on one stream execute serially and may share scratch. Old plans retain
# their allocations through graph replay; doubling keeps retained arenas below
# twice the largest workspace instead of allocating one arena per query shape.
_workspaces: dict[tuple[torch.device, int], torch.Tensor] = {}


@functools.cache
def _plan(
    device,
    stream,
    max_rows,
    heads,
    ks,
    kc,
    dtype,
    has_sinks,
    return_lse,
    quant_block,
    output_groups,
):
    from flashinfer.attention.prims_ts import (
        BatchSparseMLADecodePagedTSWrapper,
        get_prims_ts_sparse_mla_decode_workspace_size,
    )

    options = dict(
        max_topk=ks,
        max_extra_topk=kc,
        max_seq_len_q=max_rows,
        packed_query=True,
        q_data_type=dtype,
        has_sinks=has_sinks,
        return_lse=return_lse,
        output_quant_block_size=quant_block,
        num_output_groups=output_groups,
    )
    required = get_prims_ts_sparse_mla_decode_workspace_size(
        device, 1, heads, **options
    )
    key = (device, stream)
    workspace = _workspaces.get(key)
    if workspace is None or workspace.numel() < required:
        workspace = torch.empty(
            triton.next_power_of_2(required), device=device, dtype=torch.uint8
        )
        _workspaces[key] = workspace
    wrapper = BatchSparseMLADecodePagedTSWrapper(workspace)
    wrapper.plan(device, 1, heads, **options)
    return wrapper


def quantized_mla_output(rows, heads, groups, block_size, device):
    """Natural [token, group, channel] values and DeepGEMM's MN-major UE8M0."""
    channels = heads // groups * 512
    data = torch.empty(
        (rows, groups, channels), dtype=torch.float8_e4m3fn, device=device
    )
    scales = torch.empty(
        (groups, channels // block_size // 4, triton.cdiv(rows, 4) * 4),
        dtype=torch.int32,
        device=device,
    ).permute(2, 0, 1)[:rows]
    return QuantizedActivation(
        data=data,
        scale=scales,
        orig_dtype=torch.bfloat16,
        orig_shape=data.shape,
        quant_key=kMxfp8Dynamic if block_size == 32 else kFp8Dynamic128Sym,
    )


def sparse_mla(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    *,
    extra_kv_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_lengths: torch.Tensor | None = None,
    bmm1_scale: float,
    bmm2_scale: float = 1.0,
    sinks: torch.Tensor | None = None,
    out: torch.Tensor | QuantizedActivation | None = None,
    return_lse: bool = False,
    output_quant_block_size: int | None = None,
    token_positions: torch.Tensor | None = None,
    cos_sin_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run prefill/decode with the same per-query sparse attention contract.

    Inputs are native BF16/E4M3 Q and KV, D512, with independent INT32 lists
    for the two pools. ``lengths`` bounds each list; -1 holes are compacted.
    All query rows are independent: the model's sparse lists already encode
    causality, so one packed sequence is sufficient for the kernel scheduler.
    """
    from flashinfer.attention.prims_ts import SparseMLAPreparedMetadata

    rows, heads, _ = q.shape
    quantized = isinstance(out, QuantizedActivation)
    output_groups = None
    if isinstance(out, QuantizedActivation):
        assert output_quant_block_size is not None and token_positions is not None
        output_groups = out.data.shape[1]
    else:
        assert output_quant_block_size is None
    ks = indices.shape[1]
    kc = extra_indices.shape[1] if extra_indices is not None else 0
    wrapper = _plan(
        q.device,
        torch.cuda.current_stream(q.device).cuda_stream,
        triton.next_power_of_2(rows),
        heads,
        ks,
        kc,
        q.dtype,
        sinks is not None,
        return_lse,
        output_quant_block_size,
        output_groups,
    )
    si, sl = _compact(indices, lengths)
    ci, cl = _compact(extra_indices, extra_lengths) if kc else (None, None)
    capacity = max(256, (triton.cdiv(ks, 128) + triton.cdiv(kc, 128)) * 128)

    def empty(shape, dtype=torch.int32):
        return torch.empty(shape, device=q.device, dtype=dtype)

    metadata = SparseMLAPreparedMetadata(
        si,
        sl,
        ci,
        cl,
        empty((1, rows, capacity)),
        empty((1, rows)),
        empty((1, rows)),
        empty((1, 2 + heads + rows), torch.float32),
    )
    quant_positions = empty((rows,)) if quantized else None
    _prepare_routes[(rows,)](
        si,
        ci,
        sl,
        cl,
        metadata.routes,
        metadata.execution_lengths,
        metadata.valid_counts,
        metadata.scale_params,
        sinks,
        token_positions,
        quant_positions,
        ks,
        kc,
        heads,
        capacity,
        bmm1_scale,
        bmm2_scale,
        sinks is not None,
        quantized,
        256,
    )
    quant_kwargs = {}
    values = out
    if isinstance(out, QuantizedActivation):
        values = out.data[:rows]
        scales = out.scale[:rows]
        # A graph-padded allocation can have a larger MN stride than the live
        # query prefix. Only that case needs a compact temporary scale buffer.
        direct_scales = scales.stride(-1) == triton.cdiv(rows, 4) * 4
        quant_kwargs = dict(
            token_positions=quant_positions,
            cos_sin_cache=cos_sin_cache,
            out_scale=scales if direct_scales else None,
        )
    # Device-only construction also works while the custom op is captured.
    qo_indptr = torch.arange(2, dtype=torch.int32, device=q.device) * rows
    result = wrapper.run(
        q.contiguous(),
        kv_cache,
        metadata,
        extra_kv_cache,
        qo_indptr=qo_indptr,
        softmax_scale=bmm1_scale,
        output_scale=bmm2_scale,
        sinks=sinks,
        out=values,
        validate=False,
        **quant_kwargs,
    )
    if quantized:
        if not direct_scales:
            scales.copy_(result[1])
        return result[0], result[2] if return_lse else None
    return result if return_lse else (result, None)
