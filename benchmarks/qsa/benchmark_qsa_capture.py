"""Replay a captured QSA call with production metadata and cold-L2 timing.

This diagnostic consumes the opt-in backend-comparison captures produced by
the earlier Qwen3.8 integration.  It rebases retained cache pages, verifies
the current metadata adapter against the captured CSR, and compares PrimTS
attention, metadata plus PrimTS, and the native Triton sparse kernel.  Timing
uses CUDA graphs and evicts L2 outside each measured event.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import torch

from vllm.models.qwen4_exp.nvidia.ops.qsa import (
    qsa_build_page4_paged_metadata,
    qsa_prims_ts_paged_attention,
    qsa_prims_ts_workspace_size,
    qsa_sparse_paged_attention,
)

_SEMANTIC_PAGE_SIZE = 4
_MAX_SEQ_LEN = 2051


def _compact_block_table(payload: dict) -> torch.Tensor:
    block_table = payload["active_block_table"].clone()
    for new_page, old_page in enumerate(payload["physical_page_ids"].tolist()):
        block_table[payload["active_block_table"] == old_page] = new_page
    return block_table


def _rebase_captured_indices(payload: dict) -> torch.Tensor:
    """Rebase encoded semantic-page locators onto retained cache pages."""

    group_size = int(payload["group_size"])
    if group_size not in (1, 2, 4):
        raise ValueError(f"unsupported captured QSA group size {group_size}")
    storage_page_size = payload["key_cache_pages"].shape[1]
    subpages_per_storage_page = storage_page_size // _SEMANTIC_PAGE_SIZE
    indices = payload["paged_kv_indices"].clone()
    used_entries = int(payload["paged_kv_indptr"][-1])
    live = torch.arange(indices.numel()) < used_entries

    memberships = torch.zeros_like(indices)
    if group_size > 1:
        memberships[live] = indices[live] & 0xF
        indices[live] >>= 4
    old_pages = torch.div(
        indices.clamp_min(0), subpages_per_storage_page, rounding_mode="floor"
    )
    subpages = indices.clamp_min(0) % subpages_per_storage_page
    new_pages = torch.full_like(old_pages, -1)
    for new_page, old_page in enumerate(payload["physical_page_ids"].tolist()):
        new_pages[old_pages == old_page] = new_page
    if torch.any(live & (new_pages < 0)):
        raise ValueError("capture omitted a referenced physical page")
    indices[live] = new_pages[live] * subpages_per_storage_page + subpages[live]
    if group_size > 1:
        indices[live] = (indices[live] << 4) | memberships[live]
    return indices


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = actual.float() - expected.float()
    return {
        "max": float(difference.abs().max()),
        "mean": float(difference.abs().mean()),
        "rmse": float(difference.square().mean().sqrt()),
    }


def _balanced_order(names: tuple[str, ...], round_index: int) -> tuple[str, ...]:
    cycle, shift = divmod(round_index, len(names))
    base = names if cycle % 2 == 0 else names[::-1]
    return base[shift:] + base[:shift]


def _time_cuda_graphs(
    functions: dict[str, Callable[[], object]],
    *,
    warmups: int,
    iterations: int,
    l2_flush_mib: int,
) -> dict[str, float]:
    """Interleave cold-L2 CUDA-graph replays and return mean microseconds."""

    for _ in range(warmups):
        for function in functions.values():
            function()
    torch.cuda.synchronize()

    graphs = {}
    for name, function in functions.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            function()
        graphs[name] = graph

    flush_buffer = torch.zeros(
        l2_flush_mib * 1024 * 1024, dtype=torch.uint8, device="cuda"
    )
    flush_buffer.add_(1)
    torch.cuda.synchronize()
    flush_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(flush_graph):
        flush_buffer.add_(1)

    names = tuple(functions)
    for round_index in range(warmups):
        for name in _balanced_order(names, round_index):
            flush_graph.replay()
            graphs[name].replay()
    torch.cuda.synchronize()

    events = {
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
        for name in _balanced_order(names, round_index):
            flush_graph.replay()
            start, end = events[name][round_index]
            start.record()
            graphs[name].replay()
            end.record()
    torch.cuda.synchronize()
    return {
        name: sum(start.elapsed_time(end) for start, end in pairs)
        * 1000
        / iterations
        for name, pairs in events.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--row-limit", type=int)
    parser.add_argument(
        "--pad-to-rows",
        type=int,
        help="Append inert vLLM-style rows with logical_position=-1.",
    )
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--l2-flush-mib", type=int, default=512)
    parser.add_argument(
        "--calls-per-graph",
        type=int,
        default=1,
        help="Replay each backend this many times inside one captured graph.",
    )
    args = parser.parse_args()

    payload = torch.load(args.capture, map_location="cpu", weights_only=False)
    group_size = int(payload["group_size"])
    captured_rows = int(payload["query"].shape[0])
    row_end = (
        captured_rows
        if args.row_limit is None
        else args.row_start + args.row_limit
    )
    if not 0 <= args.row_start < row_end <= captured_rows:
        raise ValueError("requested row slice falls outside the capture")
    if args.row_start % group_size or row_end % group_size:
        raise ValueError("requested row slice must align to the Q group size")
    if args.row_start or row_end != captured_rows:
        payload = dict(payload)
        row_slice = slice(args.row_start, row_end)
        for field in (
            "query",
            "logical_indices",
            "token_to_req",
            "logical_positions",
            "seq_lens",
            "triton_output",
            "triton_p448_output",
            "prims_output",
        ):
            if field in payload:
                payload[field] = payload[field][row_slice]
        route_start = args.row_start // group_size
        route_end = row_end // group_size
        old_indptr = payload["paged_kv_indptr"]
        entry_start = int(old_indptr[route_start])
        entry_end = int(old_indptr[route_end])
        payload["paged_kv_indptr"] = (
            old_indptr[route_start : route_end + 1] - entry_start
        )
        payload["paged_kv_indices"] = payload["paged_kv_indices"][
            entry_start:entry_end
        ]
    source_rows = int(payload["query"].shape[0])
    if args.pad_to_rows is not None:
        if group_size != 1:
            raise ValueError("inert-row diagnostics currently require Q1")
        if args.pad_to_rows < source_rows:
            raise ValueError("pad-to rows cannot be smaller than the capture slice")
        padding_rows = args.pad_to_rows - source_rows
        if padding_rows:
            payload = dict(payload)
            payload["query"] = torch.cat(
                (
                    payload["query"],
                    torch.zeros(
                        padding_rows,
                        *payload["query"].shape[1:],
                        dtype=payload["query"].dtype,
                    ),
                )
            )
            payload["logical_indices"] = torch.cat(
                (
                    payload["logical_indices"],
                    torch.full(
                        (padding_rows, payload["logical_indices"].shape[1]),
                        -1,
                        dtype=torch.int32,
                    ),
                )
            )
            payload["token_to_req"] = torch.cat(
                (
                    payload["token_to_req"],
                    torch.zeros(padding_rows, dtype=torch.int32),
                )
            )
            payload["logical_positions"] = torch.cat(
                (
                    payload["logical_positions"],
                    torch.full((padding_rows,), -1, dtype=torch.int64),
                )
            )
    flat_query = payload["query"].cuda()
    rows = flat_query.shape[0]
    if rows % group_size:
        raise ValueError("captured rows do not align to the Q group size")
    query = (
        flat_query
        if group_size == 1
        else flat_query.view(
            rows // group_size,
            group_size,
            flat_query.shape[1],
            flat_query.shape[2],
        )
    )

    triton_k = payload["key_cache_pages"].contiguous().cuda()
    triton_v = payload["value_cache_pages"].contiguous().cuda()
    prims_k = triton_k.transpose(1, 2).contiguous()
    prims_v = triton_v.transpose(1, 2).contiguous()
    block_table = _compact_block_table(payload).cuda()
    logical_indices = payload["logical_indices"].cuda()
    token_to_req = payload["token_to_req"].cuda()
    logical_positions = payload["logical_positions"].cuda()

    if group_size != 1:
        raise NotImplementedError(
            "metadata rebuild timing currently targets captured Q1 decode/prefill"
        )

    page_capacity = (logical_indices.shape[1] + 3) // _SEMANTIC_PAGE_SIZE
    rebuilt_indptr = torch.empty(rows + 1, dtype=torch.int32, device="cuda")
    rebuilt_indices = torch.empty(
        rows * page_capacity, dtype=torch.int32, device="cuda"
    )
    rebuilt_seq_lens = torch.empty(rows, dtype=torch.int32, device="cuda")
    qsa_build_page4_paged_metadata(
        logical_indices,
        block_table,
        token_to_req,
        logical_positions,
        prims_k.shape[2],
        paged_kv_indptr=rebuilt_indptr,
        paged_kv_indices=rebuilt_indices,
        seq_lens=rebuilt_seq_lens,
    )
    torch.cuda.synchronize()
    if rows == source_rows:
        captured_indptr = payload["paged_kv_indptr"].cuda()
        captured_indices = _rebase_captured_indices(payload).cuda()
        captured_seq_lens = payload["seq_lens"].cuda()
    else:
        captured_indptr = rebuilt_indptr
        captured_indices = rebuilt_indices
        captured_seq_lens = rebuilt_seq_lens
    page_offsets = torch.arange(page_capacity, device="cuda").repeat(
        rebuilt_seq_lens.numel()
    )
    live_pages = (
        page_offsets * _SEMANTIC_PAGE_SIZE
        < rebuilt_seq_lens.repeat_interleave(page_capacity)
    )
    all_indices_match = rebuilt_indices == captured_indices
    live_indices_match = all_indices_match[live_pages]
    print(
        "metadata-match",
        {
            "indptr": bool(torch.equal(rebuilt_indptr, captured_indptr)),
            "indices": bool(live_indices_match.all()),
            "index_mismatches": int((~live_indices_match).sum()),
            "seq_lens": bool(torch.equal(rebuilt_seq_lens, captured_seq_lens)),
        },
    )
    if not bool(live_indices_match.all()):
        mismatch = torch.nonzero(
            live_pages & ~all_indices_match, as_tuple=False
        )[:8, 0]
        print(
            "metadata-index-mismatch-sample",
            [
                (
                    int(index),
                    int(rebuilt_indices[index]),
                    int(captured_indices[index]),
                )
                for index in mismatch
            ],
        )

    max_seq_len = _MAX_SEQ_LEN * group_size
    workspace = torch.zeros(
        qsa_prims_ts_workspace_size(
            query, prims_k, max_seq_len, out_dtype=torch.bfloat16
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    prims_output = torch.empty(query.shape, dtype=torch.bfloat16, device="cuda")
    e2e_output = torch.empty_like(prims_output)
    triton_output = torch.empty(
        flat_query.shape, dtype=torch.bfloat16, device="cuda"
    )

    def prims_attention() -> None:
        qsa_prims_ts_paged_attention(
            query,
            prims_k,
            prims_v,
            workspace,
            captured_indptr,
            captured_indices,
            captured_seq_lens,
            max_seq_len,
            prims_output,
            bmm1_scale=payload["bmm1_scale"],
            bmm2_scale=payload["bmm2_scale"],
        )

    def metadata_prims() -> None:
        qsa_build_page4_paged_metadata(
            logical_indices,
            block_table,
            token_to_req,
            logical_positions,
            prims_k.shape[2],
            paged_kv_indptr=rebuilt_indptr,
            paged_kv_indices=rebuilt_indices,
            seq_lens=rebuilt_seq_lens,
        )
        qsa_prims_ts_paged_attention(
            query,
            prims_k,
            prims_v,
            workspace,
            rebuilt_indptr,
            rebuilt_indices,
            rebuilt_seq_lens,
            max_seq_len,
            e2e_output,
            bmm1_scale=payload["bmm1_scale"],
            bmm2_scale=payload["bmm2_scale"],
        )

    def triton_attention() -> None:
        qsa_sparse_paged_attention(
            flat_query,
            triton_k,
            triton_v,
            logical_indices,
            block_table,
            token_to_req,
            triton_output,
            bmm1_scale=payload["bmm1_scale"],
            bmm2_scale=payload["bmm2_scale"],
        )

    for function in (prims_attention, metadata_prims, triton_attention):
        function()
    torch.cuda.synchronize()
    flat_prims = prims_output.view_as(triton_output)
    print(
        "prims-vs-saved-prims",
        _metrics(flat_prims[:source_rows].cpu(), payload["prims_output"]),
    )
    print("prims-vs-triton", _metrics(flat_prims, triton_output))
    print("e2e-vs-prims", _metrics(e2e_output.view_as(flat_prims), flat_prims))

    if args.calls_per_graph <= 0:
        raise ValueError("calls per graph must be positive")

    def repeat(function: Callable[[], object]) -> Callable[[], None]:
        def repeated() -> None:
            for _ in range(args.calls_per_graph):
                function()

        return repeated

    timings = _time_cuda_graphs(
        {
            "prims_attention": repeat(prims_attention),
            "metadata_plus_prims": repeat(metadata_prims),
            "triton_attention": repeat(triton_attention),
        },
        warmups=args.warmups,
        iterations=args.iterations,
        l2_flush_mib=args.l2_flush_mib,
    )
    selected_tokens = int(captured_seq_lens.sum())
    nominal_bytes = selected_tokens * 2 * prims_k.shape[-1] * prims_k.element_size()
    print(
        "shape",
        {
            "rows": rows,
            "group_size": group_size,
            "q_heads": flat_query.shape[1],
            "kv_heads": prims_k.shape[1],
            "head_dim": flat_query.shape[2],
            "selected_tokens": selected_tokens,
            "nominal_kv_bytes": nominal_bytes,
        },
    )
    print("calls_per_graph", args.calls_per_graph)
    print(
        "timings_us_per_call",
        {name: value / args.calls_per_graph for name, value in timings.items()},
    )
    print(
        "nominal_kv_tb_s",
        {
            name: nominal_bytes * args.calls_per_graph / latency_us / 1.0e6
            for name, latency_us in timings.items()
            if "prims" in name or "triton" in name
        },
    )


if __name__ == "__main__":
    main()
