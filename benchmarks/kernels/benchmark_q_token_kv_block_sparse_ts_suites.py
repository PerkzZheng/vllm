# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the recorded QToken-KvBlock-Sparse-Attention PrimTS standalone suites.

This runner consumes the manifests under ``qsa_bench/suites``.  It keeps plan
construction, validation, allocation, compilation, and graph capture outside
the measured region.  Every timed backend is a CUDA-graph replay preceded by
the same-stream cold-L2 eviction used by the legacy QSA benchmark.

Run from the vLLM repository root with the local FlashInfer checkout first on
``PYTHONPATH``::

    python benchmarks/kernels/benchmark_q_token_kv_block_sparse_ts_suites.py \
      ../qsa_bench/suites/q5_bf16_real_topk.json \
      --output-json ../qsa_bench/results/q5.json
"""

from __future__ import annotations

import argparse
import fnmatch
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from flashinfer.attention.prims_ts import (
    QTokenKvBlockSparsePagedTSWrapper,
    get_q_token_kv_block_sparse_workspace_size,
    make_q_token_kv_block_sparse_qo_indptr,
    suggest_q_token_kv_block_sparse_group_size,
    validate_q_token_kv_block_sparse_group_size,
)
from flashinfer.attention.prims_ts.decode import _resolve_decode_launch_spec
from flashinfer.attention.prims_ts.q_token_kv_block_sparse_metadata import (
    _build_q_token_kv_block_sparse_metadata,
    _get_q_token_kv_block_sparse_metadata_output_shapes,
)

from vllm.models.qwen4_exp.nvidia.ops.qsa import (
    expand_qsa_block_indices_cuda,
    qsa_sparse_paged_attention,
)

TOKEN_TOPK = 2048
SPARSE_BLOCK_SIZE = 4
BLOCK_TOPK = TOKEN_TOPK // SPARSE_BLOCK_SIZE
EXPANDED_WIDTH = TOKEN_TOPK + SPARSE_BLOCK_SIZE - 1
HEAD_DIM = 256
TOTAL_Q_HEADS = 24
TOTAL_KV_HEADS = 2
MIN_L2_FLUSH_BYTES = 256 * 1024 * 1024
L2_FLUSH_MULTIPLIER = 2
_L2_FLUSH_BUFFERS: dict[int, torch.Tensor] = {}


@dataclass(frozen=True)
class SuiteCase:
    case_id: str
    phase: str
    tp: int
    dtype_name: str
    batch_size: int
    seq_len_q: int
    kv_length: int
    group_size: int
    query_layout: str
    trace_paths: tuple[Path, ...]
    q5_start_position: int | None = None


@dataclass(frozen=True)
class RouteTrace:
    block_indices: torch.Tensor
    tail_token_indices: torch.Tensor
    logical_positions: torch.Tensor
    block_table: torch.Tensor
    storage_page_size: int
    num_physical_pages: int
    context_length: int
    source_format: str
    source_paths: tuple[Path, ...]
    source_sha256: tuple[str, ...]


@dataclass
class CaseTensors:
    query: torch.Tensor
    query_flat: torch.Tensor
    attention_query: torch.Tensor
    prims_k: torch.Tensor
    prims_v: torch.Tensor
    triton_k: torch.Tensor
    triton_v: torch.Tensor
    block_indices: torch.Tensor
    tail_token_indices: torch.Tensor
    block_table: torch.Tensor
    token_to_request: torch.Tensor
    query_positions: torch.Tensor
    sequence_lengths: torch.Tensor
    qo_indptr: torch.Tensor | None
    output: torch.Tensor
    output_flat: torch.Tensor
    attention_output: torch.Tensor
    triton_output: torch.Tensor
    expanded_indices: torch.Tensor
    metadata_page_indices: torch.Tensor
    metadata_page_memberships: torch.Tensor
    metadata_seq_lens: torch.Tensor
    attention_workspace: torch.Tensor
    attention_plan: Any
    max_seq_len_kv: int


@dataclass(frozen=True)
class TimingDistribution:
    samples_us: tuple[float, ...]
    mean_us: float
    median_us: float
    p95_us: float
    cv: float

    def as_json(self) -> dict[str, Any]:
        return {
            "mean_us": self.mean_us,
            "median_us": self.median_us,
            "p95_us": self.p95_us,
            "cv": self.cv,
            "samples_us": list(self.samples_us),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stable_case_seed(case_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(case_id.encode()).digest()
    return (base_seed + int.from_bytes(digest[:4], "little")) % (2**31)


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name in ("fp8", "fp8_e4m3"):
        return torch.float8_e4m3fn
    raise ValueError(f"unsupported QSA dtype: {name}")


def _dtype_key(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float8_e4m3fn:
        return "float8_e4m3fn"
    raise ValueError(f"unsupported dtype: {dtype}")


def _tp_heads(tp: int) -> tuple[int, int]:
    if TOTAL_Q_HEADS % tp:
        raise ValueError(f"TP={tp} does not divide {TOTAL_Q_HEADS} query heads")
    if tp <= TOTAL_KV_HEADS:
        if TOTAL_KV_HEADS % tp:
            raise ValueError(f"TP={tp} does not divide {TOTAL_KV_HEADS} KV heads")
    elif tp % TOTAL_KV_HEADS:
        raise ValueError(f"TP={tp} cannot replicate {TOTAL_KV_HEADS} KV heads")
    return TOTAL_Q_HEADS // tp, max(1, TOTAL_KV_HEADS // tp)


def _resolve_workspace_root(manifest_path: Path) -> Path:
    manifest_path = manifest_path.resolve()
    if (
        manifest_path.parent.name == "suites"
        and manifest_path.parent.parent.name == "qsa_bench"
    ):
        return manifest_path.parents[2]
    raise ValueError("suite manifest must live under <workspace>/qsa_bench/suites")


def _validate_declared_trace_hashes(topk: dict[str, Any], workspace_root: Path) -> None:
    """Fail closed when a recorded route artifact differs from its manifest."""

    declared: dict[str, str]
    if "files" in topk:
        files = list(topk["files"])
        hashes = list(topk.get("sha256", []))
        if len(files) != len(hashes):
            raise ValueError("top-k files and sha256 lists must have equal length")
        declared = dict(zip(files, hashes, strict=True))
    else:
        files = [
            value
            for trace_set in topk.get("trace_sets", {}).values()
            for value in trace_set
        ]
        declared = dict(topk.get("trace_sha256", {}))
        if set(files) != set(declared):
            raise ValueError(
                "top-k trace sets and trace_sha256 keys must match exactly"
            )

    for relative_path, expected in declared.items():
        path = workspace_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing top-k trace: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"top-k trace checksum mismatch for {path}: "
                f"expected {expected}, got {actual}"
            )


def _expand_manifest(manifest_path: Path) -> tuple[dict[str, Any], list[SuiteCase]]:
    manifest_path = manifest_path.resolve()
    workspace_root = _resolve_workspace_root(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported QSA suite schema")
    if not isinstance(manifest.get("model_len"), int) or manifest["model_len"] <= 0:
        raise ValueError("QSA suite model_len must be a positive integer")
    topk = manifest["topk_source"]
    _validate_declared_trace_hashes(topk, workspace_root)
    cases: list[SuiteCase] = []
    if "cases" in manifest:
        geometry = manifest["geometry"]
        paths = tuple(workspace_root / value for value in topk["files"])
        for value in manifest["cases"]:
            cases.append(
                SuiteCase(
                    case_id=value["case_id"],
                    phase=value["phase"],
                    tp=int(value["tp"]),
                    dtype_name=value["dtype"],
                    batch_size=int(value["batch_size"]),
                    seq_len_q=int(geometry["query_group_size"]),
                    kv_length=int(geometry["context_length"]),
                    group_size=int(geometry["query_group_size"]),
                    query_layout=geometry["query_layout"],
                    trace_paths=paths,
                    q5_start_position=int(geometry["first_logical_position"]),
                )
            )
    else:
        geometry = manifest["geometry"]
        workloads = manifest["workloads"]
        trace_sets = topk["trace_sets"]
        tp = int(geometry["tp"])
        for dtype_name in geometry["qkv_dtypes"]:
            for length in workloads["prefill"]["query_lengths"]:
                cases.append(
                    SuiteCase(
                        case_id=f"prefill-tp{tp}-{dtype_name}-q{length}-kv{length}-bs1-g4",
                        phase="prefill",
                        tp=tp,
                        dtype_name=dtype_name,
                        batch_size=1,
                        seq_len_q=int(length),
                        kv_length=int(length),
                        group_size=int(workloads["prefill"]["query_group_size"]),
                        query_layout=workloads["prefill"]["query_layout"],
                        trace_paths=tuple(
                            workspace_root / value for value in trace_sets[str(length)]
                        ),
                    )
                )
        for dtype_name in geometry["qkv_dtypes"]:
            for length in workloads["decode"]["kv_lengths"]:
                for mtp, group_size in workloads["decode"][
                    "mtp_to_query_group_size"
                ].items():
                    for batch_size in workloads["decode"]["batch_sizes"]:
                        cases.append(
                            SuiteCase(
                                case_id=(
                                    f"decode-tp{tp}-{dtype_name}-kv{length}-"
                                    f"bs{batch_size}-mtp{mtp}-g{group_size}"
                                ),
                                phase="decode",
                                tp=tp,
                                dtype_name=dtype_name,
                                batch_size=int(batch_size),
                                seq_len_q=int(group_size),
                                kv_length=int(length),
                                group_size=int(group_size),
                                query_layout=workloads["decode"]["query_layout"],
                                trace_paths=tuple(
                                    workspace_root / value
                                    for value in trace_sets[str(length)]
                                ),
                            )
                        )
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("suite contains duplicate case IDs")
    expected = manifest.get("matrix", {}).get("total_cases")
    if expected is not None and len(cases) != int(expected):
        raise ValueError(
            f"manifest declares {expected} cases but expands to {len(cases)}"
        )
    for case in cases:
        if case.phase == "prefill":
            expected_layout = "packed"
        elif case.phase == "decode":
            expected_layout = "fixed_unpacked_[B,num_groups,G,Hq,D]"
        elif case.phase == "grouped_decode_proxy":
            expected_layout = "fixed_grouped"
        else:
            raise ValueError(f"unsupported suite phase: {case.phase}")
        if case.query_layout != expected_layout:
            raise ValueError(
                f"{case.case_id}: query_layout={case.query_layout!r}, "
                f"expected {expected_layout!r} for {case.phase}"
            )
        missing = [str(path) for path in case.trace_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{case.case_id}: missing trace files: {missing}")
    return manifest, cases


def _dense_gap_remap(raw_table: torch.Tensor) -> tuple[torch.Tensor, int]:
    physical_pages = raw_table.unique(sorted=True)
    if not physical_pages.numel() or bool((physical_pages < 0).any().item()):
        raise ValueError("live source block table contains invalid physical pages")
    dense_pages = torch.zeros_like(physical_pages)
    for index in range(1, physical_pages.numel()):
        step = 1 if physical_pages[index] == physical_pages[index - 1] + 1 else 2
        dense_pages[index] = dense_pages[index - 1] + step
    remapped = torch.full_like(raw_table, -1)
    for source, target in zip(physical_pages, dense_pages, strict=True):
        remapped[raw_table == source] = target
    return remapped.contiguous(), int(dense_pages[-1].item()) + 1


def _validate_compact_rows(
    blocks: torch.Tensor,
    tails: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    if blocks.shape != (positions.numel(), BLOCK_TOPK) or blocks.dtype != torch.int32:
        raise ValueError("compact trace must contain int32 [rows,512] block indices")
    if tails.shape != (positions.numel(), SPARSE_BLOCK_SIZE - 1):
        raise ValueError("compact trace must contain [rows,3] causal tail indices")
    columns = torch.arange(BLOCK_TOPK, dtype=torch.int64)
    valid_counts = torch.minimum(
        (positions + 1) // SPARSE_BLOCK_SIZE,
        torch.tensor(BLOCK_TOPK, dtype=torch.int64),
    )
    expected_live = columns.unsqueeze(0) < valid_counts.unsqueeze(1)
    if not torch.equal(blocks >= 0, expected_live):
        raise ValueError(
            "compact trace block validity does not match causal visibility"
        )
    visible_complete = (positions + 1) // SPARSE_BLOCK_SIZE
    if bool(
        ((blocks.to(torch.int64) >= visible_complete.unsqueeze(1)) & expected_live)
        .any()
        .item()
    ):
        raise ValueError("compact trace selects a causal tail/future block")
    sentinel = torch.iinfo(torch.int32).max
    sorted_blocks = torch.sort(
        blocks.masked_fill(~expected_live, sentinel), dim=1
    ).values
    duplicate = (sorted_blocks[:, 1:] == sorted_blocks[:, :-1]) & (
        sorted_blocks[:, 1:] != sentinel
    )
    if bool(duplicate.any().item()):
        raise ValueError("compact trace contains duplicate selected blocks")

    tail_counts = (positions + 1) % SPARSE_BLOCK_SIZE
    tail_columns = torch.arange(SPARSE_BLOCK_SIZE - 1, dtype=torch.int64)
    tail_live = tail_columns.unsqueeze(0) < tail_counts.unsqueeze(1)
    tail_begin = positions + 1 - tail_counts
    expected_tails = tail_begin.unsqueeze(1) + tail_columns.unsqueeze(0)
    expected_tails = expected_tails.masked_fill(~tail_live, -1).to(torch.int32)
    if not torch.equal(tails, expected_tails):
        raise ValueError("compact trace causal tail indices are inconsistent")


def _expand_compact_indices(
    blocks: torch.Tensor,
    tails: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Materialize the canonical compact live-prefix representation."""

    block_offsets = torch.arange(
        SPARSE_BLOCK_SIZE, dtype=torch.int32, device=blocks.device
    )
    expanded_blocks = blocks.unsqueeze(2) * SPARSE_BLOCK_SIZE + block_offsets
    expanded_blocks = expanded_blocks.masked_fill(blocks.unsqueeze(2) < 0, -1)
    expanded = torch.full(
        (blocks.shape[0], EXPANDED_WIDTH),
        -1,
        dtype=torch.int32,
        device=blocks.device,
    )
    expanded[:, :TOKEN_TOPK] = expanded_blocks.flatten(1)
    complete_blocks = (blocks >= 0).sum(dim=1)
    tail_counts = (positions + 1) % SPARSE_BLOCK_SIZE
    for tail_offset in range(SPARSE_BLOCK_SIZE - 1):
        rows = torch.nonzero(tail_offset < tail_counts, as_tuple=False).flatten()
        columns = complete_blocks[rows] * SPARSE_BLOCK_SIZE + tail_offset
        expanded[rows, columns] = tails[rows, tail_offset]
    return expanded


def _load_trace(paths: Sequence[Path], source_format: str) -> RouteTrace:
    required_common = {
        "token_topk",
        "compress_ratio",
        "main_storage_page_size",
        "token_to_req",
        "logical_positions",
        "main_block_table",
    }
    payloads: list[dict[str, Any]] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        missing = required_common - payload.keys()
        if missing:
            raise ValueError(f"{path}: missing trace fields {sorted(missing)}")
        if int(payload["token_topk"]) != TOKEN_TOPK:
            raise ValueError(f"{path}: token_topk mismatch")
        if int(payload["compress_ratio"]) != SPARSE_BLOCK_SIZE:
            raise ValueError(f"{path}: sparse block size mismatch")
        payloads.append(payload)
    payloads.sort(key=lambda item: int(item["logical_positions"].min().item()))
    page_sizes = {int(item["main_storage_page_size"]) for item in payloads}
    if len(page_sizes) != 1:
        raise ValueError("trace chunks disagree on storage page size")
    storage_page_size = page_sizes.pop()
    if storage_page_size % SPARSE_BLOCK_SIZE:
        raise ValueError("storage page size must be divisible by sparse block size")

    positions = torch.cat([item["logical_positions"] for item in payloads]).to(
        torch.int64
    )
    requests = torch.cat([item["token_to_req"] for item in payloads]).to(torch.int32)
    if bool((requests != 0).any().item()):
        raise ValueError("recorded route replay expects one source request")
    expected_positions = torch.arange(positions.numel(), dtype=torch.int64)
    if not torch.equal(positions, expected_positions):
        raise ValueError("trace chunks must cover each position from zero exactly once")
    context_length = positions.numel()

    if source_format == "compact_blocks_v1":
        if any(item.get("dump_format") != source_format for item in payloads):
            raise ValueError("compact trace dump_format mismatch")
        blocks = torch.cat([item["selected_block_indices"] for item in payloads]).to(
            torch.int32
        )
        tails = torch.cat(
            [item["selected_tail_token_indices"] for item in payloads]
        ).to(torch.int32)
        _validate_compact_rows(blocks, tails, positions)
    elif source_format == "expanded_tokens_v1":
        logical = torch.cat([item["selected_token_indices"] for item in payloads]).to(
            torch.int32
        )
        if logical.shape != (positions.numel(), EXPANDED_WIDTH):
            raise ValueError("expanded trace must have [rows,2051] token indices")
        blocks = torch.div(
            logical[:, :TOKEN_TOPK:SPARSE_BLOCK_SIZE],
            SPARSE_BLOCK_SIZE,
            rounding_mode="floor",
        )
        valid_counts = torch.minimum(
            (positions + 1) // SPARSE_BLOCK_SIZE,
            torch.tensor(BLOCK_TOPK, dtype=torch.int64),
        )
        blocks = blocks.masked_fill(
            torch.arange(BLOCK_TOPK).unsqueeze(0) >= valid_counts.unsqueeze(1), -1
        ).to(torch.int32)
        # Older expanded captures compact their live prefix instead of
        # reserving the final three columns for the causal tail. Reconstruct
        # the tail from the semantic position, exactly as the production
        # expansion and PrimTS metadata kernels do.
        tail_counts = (positions + 1) % SPARSE_BLOCK_SIZE
        tail_columns = torch.arange(SPARSE_BLOCK_SIZE - 1, dtype=torch.int64)
        tail_live = tail_columns.unsqueeze(0) < tail_counts.unsqueeze(1)
        tail_begin = positions + 1 - tail_counts
        tails = (
            (tail_begin.unsqueeze(1) + tail_columns.unsqueeze(0))
            .masked_fill(~tail_live, -1)
            .to(torch.int32)
        )
        _validate_compact_rows(blocks, tails, positions)
        reconstructed = _expand_compact_indices(blocks, tails, positions)
        if not torch.equal(logical, reconstructed):
            raise ValueError(
                "expanded trace is not canonical contiguous block-4 data with "
                "an exact causal-tail live prefix"
            )
    else:
        raise ValueError(f"unsupported route format: {source_format}")

    live_pages = math.ceil(context_length / storage_page_size)
    final_table = payloads[-1]["main_block_table"].to(torch.int32)
    if final_table.ndim != 2 or final_table.shape[0] != 1:
        raise ValueError("source trace requires one main block-table row")
    raw_table = final_table[:, :live_pages].clone()
    for payload in payloads:
        chunk_live_pages = math.ceil(
            (int(payload["logical_positions"].max().item()) + 1) / storage_page_size
        )
        if not torch.equal(
            payload["main_block_table"][:1, :chunk_live_pages].to(torch.int32),
            raw_table[:, :chunk_live_pages],
        ):
            raise ValueError("trace chunks disagree on live main block-table entries")
    block_table, num_physical_pages = _dense_gap_remap(raw_table)
    return RouteTrace(
        block_indices=blocks.contiguous(),
        tail_token_indices=tails.contiguous(),
        logical_positions=positions.contiguous(),
        block_table=block_table,
        storage_page_size=storage_page_size,
        num_physical_pages=num_physical_pages,
        context_length=context_length,
        source_format=source_format,
        source_paths=tuple(path.resolve() for path in paths),
        source_sha256=tuple(_sha256(path) for path in paths),
    )


def _random_values(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    values = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    if dtype == torch.float8_e4m3fn:
        values.mul_(0.25)
        values = values.to(dtype)
    return values


def _flatten_fixed(tensor: torch.Tensor, group_size: int) -> torch.Tensor:
    flattened = tensor.flatten(0, 1)
    return flattened.squeeze(1) if group_size == 1 else flattened


def _select_case_routes(
    case: SuiteCase,
    trace: RouteTrace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return CPU blocks/tails/positions/requests and physical replication."""

    if trace.context_length != case.kv_length:
        raise ValueError(
            f"{case.case_id}: trace length {trace.context_length} != {case.kv_length}"
        )
    if case.q5_start_position is not None:
        start = case.q5_start_position
        rows = case.batch_size * case.group_size
        end = start + rows
        if end > trace.context_length:
            raise ValueError(f"{case.case_id}: Q5 route slice exceeds trace")
        positions = trace.logical_positions[start:end]
        if not torch.equal(positions, torch.arange(start, end, dtype=torch.int64)):
            raise ValueError(f"{case.case_id}: Q5 positions are not consecutive")
        return (
            trace.block_indices[start:end],
            trace.tail_token_indices[start:end],
            positions,
            torch.zeros(rows, dtype=torch.int32),
            1,
        )
    if case.phase == "prefill":
        return (
            trace.block_indices,
            trace.tail_token_indices,
            trace.logical_positions,
            torch.zeros(trace.context_length, dtype=torch.int32),
            1,
        )
    if case.phase != "decode":
        raise ValueError(f"unsupported suite phase: {case.phase}")
    begin = trace.context_length - case.seq_len_q
    source_blocks = trace.block_indices[begin:]
    source_tails = trace.tail_token_indices[begin:]
    source_positions = trace.logical_positions[begin:]
    blocks = source_blocks.repeat(case.batch_size, 1)
    tails = source_tails.repeat(case.batch_size, 1)
    positions = source_positions.repeat(case.batch_size)
    requests = torch.arange(case.batch_size, dtype=torch.int32).repeat_interleave(
        case.seq_len_q
    )
    return blocks, tails, positions, requests, case.batch_size


def _replicate_block_table(
    trace: RouteTrace,
    request_copies: int,
) -> torch.Tensor:
    offsets = (
        torch.arange(request_copies, dtype=torch.int32).unsqueeze(1)
        * trace.num_physical_pages
    )
    table = trace.block_table.repeat(request_copies, 1) + offsets
    live_sets = [set(row.tolist()) for row in table]
    for left in range(len(live_sets)):
        for right in range(left):
            if live_sets[left] & live_sets[right]:
                raise AssertionError("replicated request page tables overlap")
    return table.contiguous()


def _pad_block_table_for_metadata_bound(
    block_table: torch.Tensor,
    max_seq_len_kv: int,
    storage_page_size: int,
) -> torch.Tensor:
    """Extend each dense row for a larger metadata-only logical bound.

    Padding repeats a physical page already owned by the same request. Real
    selected indices and query positions remain bounded by the workload KV
    length, so neither PrimTS nor Triton attention consumes these entries.
    """

    required_pages = (max_seq_len_kv + storage_page_size - 1) // storage_page_size
    padding_pages = required_pages - int(block_table.shape[1])
    if padding_pages <= 0:
        return block_table
    if block_table.shape[1] == 0:
        raise ValueError("cannot pad an empty dense block table")
    request_local_page = block_table[:, :1]
    if bool(torch.any(request_local_page < 0).item()):
        raise ValueError("dense block-table padding requires a valid physical page")
    padding = request_local_page.expand(-1, padding_pages)
    return torch.cat((block_table, padding), dim=1).contiguous()


def _prepare_case_tensors(
    case: SuiteCase,
    trace: RouteTrace,
    base_seed: int,
    model_len: int,
) -> CaseTensors:
    num_q_heads, num_kv_heads = _tp_heads(case.tp)
    dtype = _dtype(case.dtype_name)
    torch.manual_seed(_stable_case_seed(case.case_id, base_seed))
    blocks_cpu, tails_cpu, positions_cpu, requests_cpu, request_copies = (
        _select_case_routes(case, trace)
    )
    rows = blocks_cpu.shape[0]
    query_start_loc = None
    if case.group_size > 1:
        if case.phase == "decode":
            query_start_loc = torch.arange(
                0, rows + 1, case.group_size, dtype=torch.int32
            )
        else:
            query_start_loc = torch.tensor([0, rows], dtype=torch.int32)
    validate_q_token_kv_block_sparse_group_size(
        query_start_loc,
        rows,
        num_q_heads,
        num_kv_heads,
        group_size=case.group_size,
    )

    block_table_cpu = _replicate_block_table(trace, request_copies)
    max_seq_len_kv = model_len
    if max_seq_len_kv < case.kv_length:
        raise AssertionError("model_len is smaller than the workload context")
    block_table_cpu = _pad_block_table_for_metadata_bound(
        block_table_cpu,
        max_seq_len_kv,
        trace.storage_page_size,
    )
    block_table = block_table_cpu.to(device="cuda")
    block_indices = blocks_cpu.to(device="cuda")
    tail_token_indices = tails_cpu.to(device="cuda")
    query_positions = positions_cpu.to(device="cuda")
    token_to_request = requests_cpu.to(device="cuda")
    sequence_lengths = torch.full(
        (request_copies,),
        case.kv_length,
        dtype=torch.int32,
        device="cuda",
    )

    base_k = _random_values(
        (trace.num_physical_pages, num_kv_heads, trace.storage_page_size, HEAD_DIM),
        dtype,
    )
    base_v = _random_values(
        (trace.num_physical_pages, num_kv_heads, trace.storage_page_size, HEAD_DIM),
        dtype,
    )
    if request_copies > 1:
        prims_k = base_k.repeat(request_copies, 1, 1, 1).contiguous()
        prims_v = base_v.repeat(request_copies, 1, 1, 1).contiguous()
        del base_k, base_v
    else:
        prims_k, prims_v = base_k, base_v
    triton_k_view = prims_k.permute(0, 2, 1, 3)
    triton_v_view = prims_v.permute(0, 2, 1, 3)
    triton_k = (
        triton_k_view if triton_k_view.is_contiguous() else triton_k_view.contiguous()
    )
    triton_v = (
        triton_v_view if triton_v_view.is_contiguous() else triton_v_view.contiguous()
    )
    if not prims_k.is_contiguous() or not prims_v.is_contiguous():
        raise AssertionError("PrimTS HND cache is not native contiguous storage")
    if not triton_k.is_contiguous() or not triton_v.is_contiguous():
        raise AssertionError("Triton NHD cache is not native contiguous storage")

    if case.phase == "decode":
        if case.seq_len_q % case.group_size:
            raise ValueError(
                f"{case.case_id}: fixed benchmark layout requires explicit "
                "semantic padding when seq_len_q is not divisible by group_size"
            )
        num_query_groups = case.seq_len_q // case.group_size
        base_query = _random_values(
            (1, num_query_groups, case.group_size, num_q_heads, HEAD_DIM), dtype
        )
        query = base_query.repeat(case.batch_size, 1, 1, 1, 1).contiguous()
    elif case.q5_start_position is not None:
        query = _random_values(
            (case.batch_size, 1, case.group_size, num_q_heads, HEAD_DIM), dtype
        )
    else:
        query = _random_values((rows, num_q_heads, HEAD_DIM), dtype)

    qo_indptr = None
    if case.phase == "prefill":
        qo_indptr = make_q_token_kv_block_sparse_qo_indptr(
            torch.tensor([0, rows], dtype=torch.int32),
            rows,
            group_size=case.group_size,
            device=query.device,
        )
        attention_query = query
    else:
        attention_query = _flatten_fixed(query, case.group_size)
    query_flat = query.view(rows, num_q_heads, HEAD_DIM)
    if query_flat.shape != (rows, num_q_heads, HEAD_DIM):
        raise AssertionError("flattened query shape does not match route rows")

    output = torch.empty(query.shape, dtype=torch.bfloat16, device="cuda")
    attention_output = (
        output if case.phase == "prefill" else _flatten_fixed(output, case.group_size)
    )
    output_flat = output.view(rows, num_q_heads, HEAD_DIM)
    triton_output = torch.empty_like(output_flat)
    expanded_indices = torch.empty(
        (rows, EXPANDED_WIDTH), dtype=torch.int32, device="cuda"
    )
    groups = (
        int(qo_indptr.numel()) - 1 if qo_indptr is not None else rows // case.group_size
    )
    (
        metadata_page_indices_shape,
        metadata_page_memberships_shape,
        metadata_seq_lens_shape,
    ) = _get_q_token_kv_block_sparse_metadata_output_shapes(
        rows,
        BLOCK_TOPK,
        case.group_size,
        num_query_groups=groups,
        sparse_block_size=SPARSE_BLOCK_SIZE,
    )
    metadata_page_indices = torch.empty(
        metadata_page_indices_shape, dtype=torch.int32, device="cuda"
    )
    metadata_page_memberships = torch.empty(
        metadata_page_memberships_shape, dtype=torch.int32, device="cuda"
    )
    metadata_seq_lens = torch.empty(
        metadata_seq_lens_shape, dtype=torch.int32, device="cuda"
    )
    attention_bytes = get_q_token_kv_block_sparse_workspace_size(
        query,
        prims_k,
        block_table,
        block_topk=BLOCK_TOPK,
        max_seq_len_kv=max_seq_len_kv,
        o_data_type=torch.bfloat16,
        qo_indptr=qo_indptr,
        seq_len_q=case.group_size if qo_indptr is not None else None,
        kv_block_size=SPARSE_BLOCK_SIZE,
    )
    attention_workspace = torch.zeros(attention_bytes, dtype=torch.uint8, device="cuda")
    attention_plan = QTokenKvBlockSparsePagedTSWrapper()
    attention_plan.plan(
        groups,
        case.group_size,
        num_q_heads,
        num_kv_heads,
        HEAD_DIM,
        SPARSE_BLOCK_SIZE,
        trace.storage_page_size,
        BLOCK_TOPK,
        max_seq_len_kv,
        device=query.device,
        workspace_buffer=attention_workspace,
        use_packed_q=qo_indptr is not None,
        q_data_type=query.dtype,
        kv_data_type=prims_k.dtype,
        o_data_type=output.dtype,
    )
    attention_plan.run(
        query,
        (prims_k, prims_v),
        block_table,
        block_indices,
        token_to_request,
        query_positions,
        out=output,
        qo_indptr=qo_indptr,
    )
    return CaseTensors(
        query=query,
        query_flat=query_flat,
        attention_query=attention_query,
        prims_k=prims_k,
        prims_v=prims_v,
        triton_k=triton_k,
        triton_v=triton_v,
        block_indices=block_indices,
        tail_token_indices=tail_token_indices,
        block_table=block_table,
        token_to_request=token_to_request,
        query_positions=query_positions,
        sequence_lengths=sequence_lengths,
        qo_indptr=qo_indptr,
        output=output,
        output_flat=output_flat,
        attention_output=attention_output,
        triton_output=triton_output,
        expanded_indices=expanded_indices,
        metadata_page_indices=metadata_page_indices,
        metadata_page_memberships=metadata_page_memberships,
        metadata_seq_lens=metadata_seq_lens,
        attention_workspace=attention_workspace,
        attention_plan=attention_plan,
        max_seq_len_kv=max_seq_len_kv,
    )


def _l2_cache_size_bytes(device_index: int) -> int:
    properties = torch.cuda.get_device_properties(device_index)
    for attribute in ("L2_cache_size", "l2_cache_size"):
        value = int(getattr(properties, attribute, 0))
        if value > 0:
            return value
    return MIN_L2_FLUSH_BYTES // L2_FLUSH_MULTIPLIER


def _l2_flush_buffer() -> torch.Tensor:
    device_index = torch.accelerator.current_device_index()
    buffer = _L2_FLUSH_BUFFERS.get(device_index)
    if buffer is None:
        flush_bytes = max(
            MIN_L2_FLUSH_BYTES,
            L2_FLUSH_MULTIPLIER * _l2_cache_size_bytes(device_index),
        )
        buffer = torch.zeros(flush_bytes, dtype=torch.uint8, device=device_index)
        _L2_FLUSH_BUFFERS[device_index] = buffer
    return buffer


def _evict_l2() -> None:
    _l2_flush_buffer().add_(1)


def _balanced_order(names: tuple[str, ...], round_index: int) -> tuple[str, ...]:
    cycle, shift = divmod(round_index, len(names))
    base = names if cycle % 2 == 0 else names[::-1]
    return base[shift:] + base[:shift]


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(samples_us: Sequence[float]) -> TimingDistribution:
    samples = tuple(float(value) for value in samples_us)
    mean = statistics.fmean(samples)
    return TimingDistribution(
        samples_us=samples,
        mean_us=mean,
        median_us=statistics.median(samples),
        p95_us=_percentile(samples, 0.95),
        cv=statistics.pstdev(samples) / mean if mean else 0.0,
    )


def _time_cuda_graphs(
    functions: dict[str, Callable[[], object]],
    *,
    warmup_iterations: int,
    iterations: int,
) -> dict[str, TimingDistribution]:
    """Capture and time functions with cold-L2 balanced backend ordering."""

    names = tuple(functions)
    if not names:
        raise ValueError("at least one timing function is required")
    for round_index in range(warmup_iterations):
        for name in _balanced_order(names, round_index):
            _evict_l2()
            functions[name]()
    torch.accelerator.synchronize()

    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    for name, function in functions.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            function()
        graphs[name] = graph
    _evict_l2()
    torch.accelerator.synchronize()
    flush_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(flush_graph):
        _evict_l2()
    for round_index in range(warmup_iterations):
        for name in _balanced_order(names, round_index):
            flush_graph.replay()
            graphs[name].replay()
    torch.accelerator.synchronize()

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
            begin, end = events[name][round_index]
            begin.record()
            graphs[name].replay()
            end.record()
    torch.accelerator.synchronize()
    return {
        name: _distribution(
            [begin.elapsed_time(end) * 1000 for begin, end in events[name]]
        )
        for name in names
    }


def _expected_expanded_indices(tensors: CaseTensors) -> torch.Tensor:
    return _expand_compact_indices(
        tensors.block_indices,
        tensors.tail_token_indices,
        tensors.query_positions,
    )


def _assert_metadata_equal(
    case_id: str,
    left_page_indices: torch.Tensor,
    left_page_memberships: torch.Tensor,
    left_lengths: torch.Tensor,
    right_page_indices: torch.Tensor,
    right_page_memberships: torch.Tensor,
    right_lengths: torch.Tensor,
) -> None:
    """Compare only metadata covered by the per-route live-length contract.

    Memberships are byte-packed four per Int32.  The final word can contain
    unspecified padding bytes, so compare its live bytes rather than whole
    words.
    """

    if not torch.equal(left_lengths, right_lengths):
        raise AssertionError(f"{case_id}: QSA metadata sequence lengths differ")
    if left_page_indices.shape != right_page_indices.shape:
        raise AssertionError(f"{case_id}: QSA page-index shapes differ")
    if left_page_memberships.shape != right_page_memberships.shape:
        raise AssertionError(f"{case_id}: QSA page-membership shapes differ")
    live_locators = torch.div(
        left_lengths + SPARSE_BLOCK_SIZE - 1,
        SPARSE_BLOCK_SIZE,
        rounding_mode="floor",
    ).tolist()
    left_membership_bytes = left_page_memberships.view(torch.uint8)
    right_membership_bytes = right_page_memberships.view(torch.uint8)
    for row, count in enumerate(live_locators):
        if not torch.equal(
            left_page_indices[row, :count], right_page_indices[row, :count]
        ):
            raise AssertionError(
                f"{case_id}: QSA page-index live prefix differs in route {row}"
            )
        if left_membership_bytes.shape[1] and not torch.equal(
            left_membership_bytes[row, :count],
            right_membership_bytes[row, :count],
        ):
            raise AssertionError(
                f"{case_id}: QSA page-membership live prefix differs in route {row}"
            )


def _resolve_qsa_config(case: SuiteCase, tensors: CaseTensors) -> Any:
    num_q_heads, num_kv_heads = _tp_heads(case.tp)
    routes = (
        int(tensors.qo_indptr.numel()) - 1
        if tensors.qo_indptr is not None
        else tensors.block_indices.shape[0] // case.group_size
    )
    dtype_key = _dtype_key(tensors.query.dtype)
    max_seq_len = (
        BLOCK_TOPK * SPARSE_BLOCK_SIZE + SPARSE_BLOCK_SIZE - 1
        if case.group_size == 1
        else case.group_size * (BLOCK_TOPK + 1) * SPARSE_BLOCK_SIZE
    )
    return _resolve_decode_launch_spec(
        torch.accelerator.current_device_index(),
        routes,
        num_q_heads,
        num_kv_heads,
        HEAD_DIM,
        SPARSE_BLOCK_SIZE,
        max_seq_len,
        case.group_size,
        dtype_key,
        dtype_key,
        "bfloat16",
        "HND",
        "causal",
        tensors.qo_indptr is not None,
        -1,
        tensors.prims_k.shape[2],
        True,
        True,
    ).config


def _resolve_triton_config(tensors: CaseTensors) -> dict[str, int]:
    """Mirror the checked-in vLLM Triton baseline's launch policy."""

    q = tensors.query_flat
    num_kv_heads = int(tensors.triton_k.shape[2])
    heads_per_kv = int(q.shape[1]) // num_kv_heads
    block_m = 1 << (heads_per_kv - 1).bit_length()
    base_programs = int(q.shape[0]) * num_kv_heads
    small_profile_limit = 8 if block_m <= 8 else 4
    if base_programs <= small_profile_limit:
        block_n, target_splits, partial_warps = 16, 64, 4
    elif base_programs < 32:
        block_n, target_splits, partial_warps = 16, 32, 4
    elif base_programs <= 256:
        block_n, target_splits, partial_warps = 64, 8, 2
    elif base_programs <= 512:
        block_n, target_splits, partial_warps = 64, 4, 2
    else:
        block_n, target_splits, partial_warps = 64, 1, 2
    if q.dtype == torch.float8_e4m3fn:
        block_n = max(block_n, 32)
    num_tiles = math.ceil(EXPANDED_WIDTH / block_n)
    max_useful_splits = 1 << (num_tiles.bit_length() - 1)
    return {
        "block_m": block_m,
        "block_n": block_n,
        "partial_warps": partial_warps,
        "splits_kv": min(max_useful_splits, target_splits),
    }


def _error_statistics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = actual.float() - expected.float()
    absolute = difference.abs()
    result = {
        "max_abs": float(absolute.max().item()),
        "mean_abs": float(absolute.mean().item()),
        "rmse": float(torch.sqrt(torch.mean(difference.square())).item()),
    }
    del difference, absolute
    return result


def _assert_backend_outputs(
    case: SuiteCase,
    tensors: CaseTensors,
    *,
    tolerance: float,
    stage: str,
) -> dict[str, float]:
    """Validate backend agreement and deterministic replicated requests."""

    errors = _error_statistics(tensors.output_flat, tensors.triton_output)
    torch.testing.assert_close(
        tensors.output_flat.float(),
        tensors.triton_output.float(),
        rtol=tolerance,
        atol=tolerance,
        msg=lambda message: f"{case.case_id}: {stage}: {message}",
    )
    if case.phase == "decode" and case.batch_size > 1:
        num_q_heads, _ = _tp_heads(case.tp)
        prims_rows = tensors.output_flat.view(
            case.batch_size, case.group_size, num_q_heads, HEAD_DIM
        )
        triton_rows = tensors.triton_output.view_as(prims_rows)
        torch.testing.assert_close(
            prims_rows,
            prims_rows[:1].expand_as(prims_rows),
            rtol=0,
            atol=0,
            msg=lambda message: f"{case.case_id}: {stage} PrimTS: {message}",
        )
        torch.testing.assert_close(
            triton_rows,
            triton_rows[:1].expand_as(triton_rows),
            rtol=0,
            atol=0,
            msg=lambda message: f"{case.case_id}: {stage} Triton: {message}",
        )
    return errors


def _sample_fp32_oracle(
    case: SuiteCase,
    tensors: CaseTensors,
    *,
    required: bool,
    tolerance: float,
) -> dict[str, Any] | None:
    if not required:
        return None
    rows = tensors.query_flat.shape[0]
    sampled_rows = sorted({0, rows // 2, rows - 1})
    num_q_heads, num_kv_heads = _tp_heads(case.tp)
    head_repeats = num_q_heads // num_kv_heads
    prims_errors: list[float] = []
    triton_errors: list[float] = []
    for row in sampled_rows:
        logical = tensors.expanded_indices[row]
        logical = logical[logical >= 0].to(torch.int64)
        request = int(tensors.token_to_request[row].item())
        logical_pages = torch.div(
            logical, tensors.prims_k.shape[2], rounding_mode="floor"
        )
        offsets = logical.remainder(tensors.prims_k.shape[2])
        physical_pages = tensors.block_table[request, logical_pages].to(torch.int64)
        keys = tensors.prims_k[physical_pages, :, offsets, :]
        values = tensors.prims_v[physical_pages, :, offsets, :]
        keys = keys.repeat_interleave(head_repeats, dim=1).float()
        values = values.repeat_interleave(head_repeats, dim=1).float()
        query = tensors.query_flat[row].float()
        scores = torch.einsum("hd,thd->ht", query, keys) * (HEAD_DIM**-0.5)
        probabilities = torch.softmax(scores, dim=-1)
        reference = torch.einsum("ht,thd->hd", probabilities, values)
        torch.testing.assert_close(
            tensors.output_flat[row].float(),
            reference,
            rtol=tolerance,
            atol=tolerance,
            msg=lambda message: f"{case.case_id}: FP32 PrimTS oracle: {message}",
        )
        torch.testing.assert_close(
            tensors.triton_output[row].float(),
            reference,
            rtol=tolerance,
            atol=tolerance,
            msg=lambda message: f"{case.case_id}: FP32 Triton oracle: {message}",
        )
        prims_errors.append(
            float((tensors.output_flat[row].float() - reference).abs().max().item())
        )
        triton_errors.append(
            float((tensors.triton_output[row].float() - reference).abs().max().item())
        )
    return {
        "rows": sampled_rows,
        "prims_ts_max_abs": max(prims_errors),
        "triton_max_abs": max(triton_errors),
        "prims_ts_per_row_max_abs": prims_errors,
        "triton_per_row_max_abs": triton_errors,
    }


def _run_case(
    case: SuiteCase,
    trace: RouteTrace,
    *,
    warmup_iterations: int,
    iterations: int,
    base_seed: int,
    model_len: int,
) -> dict[str, Any]:
    tensors = _prepare_case_tensors(
        case,
        trace,
        base_seed,
        model_len,
    )
    _, num_kv_heads = _tp_heads(case.tp)
    groups = (
        int(tensors.qo_indptr.numel()) - 1
        if tensors.qo_indptr is not None
        else tensors.block_indices.shape[0] // case.group_size
    )

    def metadata() -> object:
        return _build_q_token_kv_block_sparse_metadata(
            tensors.block_indices,
            tensors.block_table,
            tensors.token_to_request,
            tensors.query_positions,
            group_size=case.group_size,
            storage_page_size=trace.storage_page_size,
            max_seq_len_kv=tensors.max_seq_len_kv,
            sparse_block_size=SPARSE_BLOCK_SIZE,
            qo_indptr=tensors.qo_indptr,
            out=(
                tensors.metadata_page_indices,
                tensors.metadata_page_memberships,
                tensors.metadata_seq_lens,
            ),
        )

    def prims_attention() -> object:
        assert tensors.attention_plan._prepared_plan is not None
        return tensors.attention_plan._prepared_plan._attention_plan.run(
            tensors.attention_query,
            out=tensors.attention_output,
            bmm1_scale=HEAD_DIM**-0.5,
        )

    def prims_combined() -> object:
        return tensors.attention_plan.run(
            tensors.query,
            (tensors.prims_k, tensors.prims_v),
            tensors.block_table,
            tensors.block_indices,
            tensors.token_to_request,
            tensors.query_positions,
            qo_indptr=tensors.qo_indptr,
            out=tensors.output,
        )

    def triton_expand() -> object:
        return expand_qsa_block_indices_cuda(
            tensors.block_indices,
            tensors.query_positions,
            tensors.sequence_lengths,
            tensors.token_to_request,
            SPARSE_BLOCK_SIZE,
            TOKEN_TOPK,
            tensors.expanded_indices,
        )

    def triton_attention() -> object:
        return qsa_sparse_paged_attention(
            tensors.query_flat,
            tensors.triton_k,
            tensors.triton_v,
            tensors.expanded_indices,
            tensors.block_table,
            tensors.token_to_request,
            tensors.triton_output,
            bmm1_scale=HEAD_DIM**-0.5,
        )

    def triton_combined() -> object:
        triton_expand()
        return triton_attention()

    metadata()
    prims_combined()
    triton_expand()
    expected_expanded = _expected_expanded_indices(tensors)
    if not torch.equal(tensors.expanded_indices, expected_expanded):
        raise AssertionError(f"{case.case_id}: Triton expansion disagrees with trace")
    del expected_expanded
    _assert_metadata_equal(
        case.case_id,
        tensors.metadata_page_indices,
        tensors.metadata_page_memberships,
        tensors.metadata_seq_lens,
        tensors.attention_plan._prepared_plan._metadata_plan.q_token_kv_block_sparse_page_indices,
        tensors.attention_plan._prepared_plan._metadata_plan.q_token_kv_block_sparse_page_memberships,
        tensors.attention_plan._prepared_plan._metadata_plan.seq_lens,
    )
    triton_attention()
    torch.accelerator.synchronize()

    tolerance = 0.02 if tensors.query.dtype == torch.bfloat16 else 0.05
    _assert_backend_outputs(
        case,
        tensors,
        tolerance=tolerance,
        stage="eager",
    )

    config = _resolve_qsa_config(case, tensors)
    splits_kv = int(config.splits_kv) if config.use_split_kv else 1
    triton_config = _resolve_triton_config(tensors)
    timings = _time_cuda_graphs(
        {
            "metadata": metadata,
            "prims_ts_attention": prims_attention,
            "prims_ts_combined": prims_combined,
            "triton_expand_indices": triton_expand,
            "triton_attention": triton_attention,
            "triton_combined": triton_combined,
        },
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )
    errors = _assert_backend_outputs(
        case,
        tensors,
        tolerance=tolerance,
        stage="post-CUDA-graph replay",
    )
    expected_expanded = _expected_expanded_indices(tensors)
    if not torch.equal(tensors.expanded_indices, expected_expanded):
        raise AssertionError(
            f"{case.case_id}: post-graph Triton expansion disagrees with trace"
        )
    del expected_expanded
    _assert_metadata_equal(
        case.case_id,
        tensors.metadata_page_indices,
        tensors.metadata_page_memberships,
        tensors.metadata_seq_lens,
        tensors.attention_plan._prepared_plan._metadata_plan.q_token_kv_block_sparse_page_indices,
        tensors.attention_plan._prepared_plan._metadata_plan.q_token_kv_block_sparse_page_memberships,
        tensors.attention_plan._prepared_plan._metadata_plan.seq_lens,
    )
    fp32_oracle = _sample_fp32_oracle(
        case,
        tensors,
        required=splits_kv > 1 or triton_config["splits_kv"] > 1,
        tolerance=tolerance,
    )
    visible = tensors.query_positions + 1
    selected_blocks = torch.minimum(
        visible // SPARSE_BLOCK_SIZE,
        torch.tensor(BLOCK_TOPK, device="cuda", dtype=visible.dtype),
    )
    source_selected_tokens = int(
        (selected_blocks * SPARSE_BLOCK_SIZE + visible % SPARSE_BLOCK_SIZE).sum().item()
    )
    union_selected_tokens = int(tensors.metadata_seq_lens.sum().item())
    logical_kv_bytes = (
        union_selected_tokens
        * 2
        * num_kv_heads
        * HEAD_DIM
        * tensors.prims_k.element_size()
    )
    result = {
        "case_id": case.case_id,
        "phase": case.phase,
        "tp": case.tp,
        "dtype": case.dtype_name,
        "batch_size": case.batch_size,
        "seq_len_q": case.seq_len_q,
        "kv_length": case.kv_length,
        "model_len": model_len,
        "metadata_max_seq_len_kv": tensors.max_seq_len_kv,
        "query_group_size": case.group_size,
        "query_layout": case.query_layout,
        "num_routes": groups,
        "storage_page_size": trace.storage_page_size,
        "dense_block_table_pages": tensors.block_table.shape[1],
        "num_physical_pages": tensors.prims_k.shape[0],
        "resolved_splits_kv": splits_kv,
        "resolved_triton_splits_kv": triton_config["splits_kv"],
        "tile_size_q": int(config.tile_size_q),
        "tile_size_kv": int(config.tile_size_kv),
        "triton_block_m": triton_config["block_m"],
        "triton_block_n": triton_config["block_n"],
        "triton_partial_warps": triton_config["partial_warps"],
        "source_selected_kv_tokens": source_selected_tokens,
        "union_selected_kv_tokens": union_selected_tokens,
        "logical_kv_bytes": logical_kv_bytes,
        "triton_over_prims_ts": (
            timings["triton_combined"].mean_us / timings["prims_ts_combined"].mean_us
        ),
        "incremental_metadata_us": (
            timings["prims_ts_combined"].mean_us - timings["prims_ts_attention"].mean_us
        ),
        "correctness": {
            **errors,
            "rtol": tolerance,
            "atol": tolerance,
            "post_cuda_graph_replay": True,
            "sampled_fp32_oracle": fp32_oracle,
        },
        "timings": {name: value.as_json() for name, value in timings.items()},
        "trace": {
            "format": trace.source_format,
            "paths": [str(path) for path in trace.source_paths],
            "sha256": list(trace.source_sha256),
        },
    }
    return result


def _git_state(path: Path) -> dict[str, Any]:
    """Record committed identity plus the tracked diff used by the run."""

    command = ["git", "-c", f"safe.directory={path}"]
    try:
        commit = subprocess.check_output(
            [*command, "rev-parse", "HEAD"],
            cwd=path,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            [*command, "status", "--short"],
            cwd=path,
            stderr=subprocess.DEVNULL,
        )
        tracked_diff = subprocess.check_output(
            [*command, "diff", "--binary", "HEAD", "--"],
            cwd=path,
            stderr=subprocess.DEVNULL,
        )
        return {
            "commit": commit,
            "dirty": bool(status),
            "status": status.decode(errors="replace").splitlines(),
            "status_sha256": _sha256_bytes(status),
            "tracked_diff_sha256": _sha256_bytes(tracked_diff),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot record Git provenance for {path}") from error


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _new_result_document(
    manifest_path: Path,
    manifest: dict[str, Any],
    selected_cases: Sequence[SuiteCase],
    warmup_iterations: int,
    iterations: int,
    base_seed: int,
    model_len: int,
) -> dict[str, Any]:
    import flashinfer
    import triton
    from flashinfer.attention.prims_ts import q_token_kv_block_sparse_metadata

    runner_path = Path(__file__).resolve()
    vllm_root = runner_path.parents[2]
    # Framework overlays may keep the image package while loading attention
    # from another checkout. Record the implementation actually benchmarked.
    flashinfer_root = (
        Path(q_token_kv_block_sparse_metadata.__file__).resolve().parents[3]
    )
    properties = torch.cuda.get_device_properties(
        torch.accelerator.current_device_index()
    )
    manifest_sha256 = _sha256(manifest_path)
    runner_sha256 = _sha256(runner_path)
    flashinfer_state = _git_state(flashinfer_root)
    vllm_state = _git_state(vllm_root)
    source = {
        "flashinfer_commit": flashinfer_state["commit"],
        "vllm_commit": vllm_state["commit"],
        "flashinfer_root": str(flashinfer_root),
        "vllm_root": str(vllm_root),
        "flashinfer_git_state": flashinfer_state,
        "vllm_git_state": vllm_state,
    }
    environment = {
        "hostname": os.uname().nodename,
        "gpu": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability()),
        "gpu_memory_bytes": int(properties.total_memory),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "flashinfer_package_root": str(
            Path(flashinfer.__file__).resolve().parent.parent
        ),
        "device_index": torch.accelerator.current_device_index(),
    }
    timing_contract = {
        "cuda_graph": True,
        "cold_l2": True,
        "l2_eviction_bytes": int(_l2_flush_buffer().numel()),
        "warmup_iterations": warmup_iterations,
        "measured_iterations": iterations,
        "backend_order": "rotating_and_periodically_reversed",
        "primary_statistic": "mean_cuda_event_us",
        "raw_samples_retained": True,
        "model_len": model_len,
    }
    selected_case_ids = [case.case_id for case in selected_cases]
    resume_identity = {
        "manifest_sha256": manifest_sha256,
        "runner_sha256": runner_sha256,
        "source": source,
        "gpu": environment["gpu"],
        "compute_capability": environment["compute_capability"],
        "gpu_memory_bytes": environment["gpu_memory_bytes"],
        "torch": environment["torch"],
        "cuda": environment["cuda"],
        "triton": environment["triton"],
        "device_index": environment["device_index"],
        "timing_contract": timing_contract,
        "seed": base_seed,
        "model_len": model_len,
        "selected_case_ids": selected_case_ids,
    }
    return {
        "schema_version": 1,
        "suite_id": manifest["suite_id"],
        "status": "running",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "runner": str(runner_path),
        "runner_sha256": runner_sha256,
        "source": source,
        "environment": environment,
        "timing_contract": timing_contract,
        "resume_identity": resume_identity,
        "seed": base_seed,
        "model_len": model_len,
        "selected_case_ids": selected_case_ids,
        "started_unix_time": time.time(),
        "cases": [],
        "failures": [],
    }


def _select_cases(
    cases: Sequence[SuiteCase],
    patterns: Sequence[str],
    max_cases: int | None,
) -> list[SuiteCase]:
    selected = [
        case
        for case in cases
        if not patterns
        or any(fnmatch.fnmatchcase(case.case_id, pattern) for pattern in patterns)
    ]
    if max_cases is not None:
        selected = selected[:max_cases]
    if not selected:
        raise ValueError("case filters selected no benchmark cases")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Repeatable fnmatch pattern over manifest case IDs.",
    )
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--warmup-iterations", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--auto-group-size",
        action="store_true",
        help=(
            "Use suggest_q_token_kv_block_sparse_group_size for each selected "
            "case. "
            "The runner queries the selected device's SM count once and passes it."
        ),
    )
    parser.add_argument(
        "--model-len",
        "--metadata-max-seq-len-kv",
        dest="model_len",
        type=int,
        help=(
            "Override the manifest's per-request model_len used as the QSA "
            "metadata bound. This is never the global physical KV-cache capacity."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and list selected cases only."
    )
    args = parser.parse_args()
    if args.max_cases is not None and args.max_cases <= 0:
        raise ValueError("--max-cases must be positive")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    manifest, cases = _expand_manifest(args.manifest)
    model_len = int(manifest["model_len"]) if args.model_len is None else args.model_len
    if model_len <= 0:
        raise ValueError("model_len must be positive")
    selected = _select_cases(cases, args.case_id, args.max_cases)
    too_long = [case.case_id for case in selected if case.kv_length > model_len]
    if too_long:
        raise ValueError(
            "model_len must be at least each selected case's KV length; too "
            f"small for {', '.join(too_long)}"
        )
    timing = manifest["timing"]
    warmup_iterations = (
        int(timing["warmup_iterations"])
        if args.warmup_iterations is None
        else args.warmup_iterations
    )
    iterations = (
        int(timing["measured_iterations"])
        if args.iterations is None
        else args.iterations
    )
    if warmup_iterations < 0 or iterations <= 0:
        raise ValueError("timing iterations must be nonnegative/positive")
    if args.auto_group_size:
        if not torch.cuda.is_available():
            raise RuntimeError("automatic QSA grouping requires CUDA")
        torch.accelerator.set_device_index(args.device)
        multi_processor_count = torch.cuda.get_device_properties(
            args.device
        ).multi_processor_count
        selected_with_suggestions = []
        for case in selected:
            num_q_heads, num_kv_heads = _tp_heads(case.tp)
            group_size = suggest_q_token_kv_block_sparse_group_size(
                case.batch_size,
                case.seq_len_q,
                EXPANDED_WIDTH,
                num_q_heads,
                num_kv_heads,
                multi_processor_count,
            )
            selected_with_suggestions.append(
                replace(
                    case,
                    case_id=f"{case.case_id}-auto-g{group_size}",
                    group_size=group_size,
                )
            )
        selected = selected_with_suggestions
    for case in selected:
        print(case.case_id, flush=True)
    if args.dry_run:
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("canonical QSA suites require CUDA")
    torch.accelerator.set_device_index(args.device)
    output_path = args.output_json.resolve()
    if output_path.exists() and not (args.resume or args.overwrite):
        raise FileExistsError(
            f"{output_path} exists; pass --resume or --overwrite explicitly"
        )
    if args.resume and output_path.exists():
        document = json.loads(output_path.read_text(encoding="utf-8"))
        current = _new_result_document(
            args.manifest,
            manifest,
            selected,
            warmup_iterations,
            iterations,
            args.seed,
            model_len,
        )
        if document.get("resume_identity") != current["resume_identity"]:
            raise ValueError(
                "resume identity differs in manifest, runner/source state, "
                "GPU/software, timing, seed, device, or selected cases"
            )
        document["status"] = "running"
    else:
        document = _new_result_document(
            args.manifest,
            manifest,
            selected,
            warmup_iterations,
            iterations,
            args.seed,
            model_len,
        )
        _write_result(output_path, document)

    complete_ids = {case["case_id"] for case in document.get("cases", [])}
    source_format = manifest["topk_source"]["format"]
    trace_cache: dict[tuple[Path, ...], RouteTrace] = {}
    for case in selected:
        if case.case_id in complete_ids:
            print(f"SKIP completed {case.case_id}", flush=True)
            continue
        print(f"RUN {case.case_id}", flush=True)
        document["failures"] = [
            failure
            for failure in document.get("failures", [])
            if failure.get("case_id") != case.case_id
        ]
        try:
            trace = trace_cache.get(case.trace_paths)
            if trace is None:
                trace = _load_trace(case.trace_paths, source_format)
                trace_cache[case.trace_paths] = trace
            result = _run_case(
                case,
                trace,
                warmup_iterations=warmup_iterations,
                iterations=iterations,
                base_seed=args.seed,
                model_len=model_len,
            )
            document["cases"].append(result)
            _write_result(output_path, document)
            print(
                f"PASS {case.case_id}: PrimTS "
                f"{result['timings']['prims_ts_combined']['mean_us']:.3f} us, "
                f"Triton {result['timings']['triton_combined']['mean_us']:.3f} us, "
                f"speedup {result['triton_over_prims_ts']:.3f}x, "
                f"max diff {result['correctness']['max_abs']:.6f}",
                flush=True,
            )
        except Exception as error:
            document["status"] = "failed"
            document["failures"].append(
                {
                    "case_id": case.case_id,
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
            document["finished_unix_time"] = time.time()
            _write_result(output_path, document)
            raise
        finally:
            gc.collect()
            torch.accelerator.empty_cache()

    selected_ids = set(document["selected_case_ids"])
    complete_ids = {case["case_id"] for case in document.get("cases", [])}
    document["status"] = "complete" if selected_ids <= complete_ids else "partial"
    document["finished_unix_time"] = time.time()
    _write_result(output_path, document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
