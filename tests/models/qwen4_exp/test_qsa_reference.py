# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import math
import weakref
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from vllm.models.qwen4_exp.common import qsa_cache
from vllm.models.qwen4_exp.common.qsa_cache import QSAMetadataBuilder
from vllm.models.qwen4_exp.nvidia import indexer_qsa
from vllm.models.qwen4_exp.nvidia import (
    model as _qwen4_exp_model,  # noqa: F401
)
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

requires_qsa_kernels = pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="QSA kernels require CUDA and Triton",
)


def _has_real_qsa_prims_ts_kernels() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return (
            current_platform.is_device_capability(100)
            or current_platform.is_device_capability(103)
        ) and qsa_ops.has_qsa_prims_ts_attention()
    except (AttributeError, ImportError, RuntimeError):
        return False


requires_real_qsa_prims_ts = pytest.mark.skipif(
    not _has_real_qsa_prims_ts_kernels(),
    reason=(
        "real PrimTS QSA integration requires SM100 or SM103 and its FlashInfer API"
    ),
)


def test_qsa_mtp_index_share_updates_cache_but_skips_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = torch.tensor([[3, 1, -1], [5, 2, 0]], dtype=torch.int32)
    raw_metadata = SimpleNamespace(
        num_actual_tokens=2,
        slot_mapping=torch.arange(2),
        block_table=torch.empty(0),
        query_start_loc=torch.arange(3),
        logical_positions=torch.arange(2),
    )
    compressed_metadata = SimpleNamespace(
        num_actual_tokens=2,
        slot_mapping=torch.arange(2),
        k_work_metadata=torch.empty(0),
    )
    updates = []
    selections = []
    indexer = SimpleNamespace(
        skip_topk=True,
        _metadata=lambda: (raw_metadata, compressed_metadata),
        index_qk_proj=lambda hidden: (torch.zeros(2, 2), None),
        index_n_heads=1,
        index_kv_heads=1,
        index_head_dim=1,
        raw_key_cache=SimpleNamespace(
            kv_cache=torch.empty(0),
            rope_position_cache=None,
            rope_position_offset=0,
        ),
        compressed_key_cache=SimpleNamespace(kv_cache=torch.empty(0)),
        use_fused_pre_indexer=True,
        rotary_emb=SimpleNamespace(cos_sin_cache=torch.empty(0)),
        q_layernorm=SimpleNamespace(weight=torch.ones(1), variance_epsilon=1e-6),
        k_layernorm=SimpleNamespace(weight=torch.ones(1)),
        compress_ratio=2,
    )

    monkeypatch.setattr(
        indexer_qsa,
        "qsa_pre_indexer",
        lambda *args, **kwargs: updates.append((args, kwargs)),
    )
    monkeypatch.setattr(
        qsa_ops,
        "qsa_select_paged_tokens",
        lambda *args, **kwargs: selections.append((args, kwargs)),
    )

    actual = indexer_qsa.QSAIndexer.forward(
        indexer,
        torch.zeros(2, 4),
        torch.tensor([7, 8]),
        rows,
    )

    assert actual is rows
    assert len(updates) == 1
    assert not selections


def _qsa_mqa_paged_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    visible_lengths: torch.Tensor,
) -> torch.Tensor:
    pages = page_table.index_select(0, token_to_req.long()).long()
    keys = k_cache[pages, :, 0, :].flatten(1, 2)
    scores = torch.einsum("rhd,rnd->rnh", q.float(), keys.float())
    logits = torch.relu(scores).sum(dim=-1) / math.sqrt(q.shape[-1])
    positions = torch.arange(keys.shape[1], device=q.device).unsqueeze(0)
    return logits.masked_fill(positions >= visible_lengths.unsqueeze(1), -torch.inf)


def _qsa_relative_topk_reference(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    output = torch.full(
        (logits.shape[0], topk), -1, dtype=torch.int32, device=logits.device
    )
    for row in range(logits.shape[0]):
        start = int(row_starts[row].item())
        length = int((row_ends[row] - row_starts[row]).item())
        width = min(length, topk)
        if width:
            output[row, :width] = torch.topk(
                logits[row, start : start + length], width
            ).indices.to(torch.int32)
    return output


def _expand_qsa_indices_reference(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
) -> torch.Tensor:
    rows = block_indices.shape[0]
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    offsets = torch.arange(compress_ratio, device=block_indices.device)
    blocks = block_indices.long()
    expanded = blocks.unsqueeze(-1) * compress_ratio + offsets
    expanded = torch.where(
        blocks.unsqueeze(-1) >= 0, expanded, torch.full_like(expanded, -1)
    ).reshape(rows, block_topk * compress_ratio)
    expanded = expanded[:, :token_topk]
    expanded = torch.where(
        (expanded >= 0) & (expanded < sequence_lengths.unsqueeze(1)),
        expanded,
        torch.full_like(expanded, -1),
    )

    tail_offsets = torch.arange(compress_ratio - 1, device=block_indices.device)
    visible_tokens = query_positions + 1
    tail_start = visible_tokens // compress_ratio * compress_ratio
    tail = tail_start.unsqueeze(1) + tail_offsets.unsqueeze(0)
    tail_count = (visible_tokens - tail_start).unsqueeze(1)
    tail_valid = (tail_offsets.unsqueeze(0) < tail_count) & (
        tail < sequence_lengths.unsqueeze(1)
    )
    tail = torch.where(tail_valid, tail, torch.full_like(tail, -1))

    result = torch.cat((expanded, tail), dim=1)
    order = torch.arange(output_width, device=result.device).expand(rows, -1)
    sort_key = torch.where(result >= 0, order, order + output_width)
    return result.gather(1, torch.argsort(sort_key, dim=1, stable=True)).to(torch.int32)


def _qsa_select_paged_tokens_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
) -> torch.Tensor:
    row_sequence_lengths = sequence_lengths.index_select(0, token_to_req.long())
    visible_blocks = torch.minimum(
        (query_positions + 1) // compress_ratio,
        row_sequence_lengths // compress_ratio,
    ).to(torch.int32)
    logits = _qsa_mqa_paged_reference(
        q,
        k_cache,
        page_table,
        token_to_req,
        visible_blocks,
    )
    starts = torch.zeros_like(visible_blocks)
    blocks = _qsa_relative_topk_reference(
        logits,
        starts,
        visible_blocks,
        token_topk // compress_ratio,
    )
    return _expand_qsa_indices_reference(
        blocks,
        query_positions,
        row_sequence_lengths,
        compress_ratio,
        token_topk,
    )


def _qsa_sparse_paged_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    output = torch.zeros_like(q)
    repeats = q.shape[1] // k_cache.shape[2]
    page_size = k_cache.shape[1]
    for row in range(q.shape[0]):
        logical = logical_indices[row]
        logical = logical[logical >= 0].long()
        if not logical.numel():
            continue
        request = token_to_req[row].long()
        pages = block_table[request, logical // page_size].long()
        offsets = logical % page_size
        keys = k_cache[pages, offsets].repeat_interleave(repeats, dim=1)
        values = v_cache[pages, offsets].repeat_interleave(repeats, dim=1)
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys.float())
        probabilities = torch.softmax(scores * softmax_scale, dim=-1)
        output[row] = torch.einsum("hk,khd->hd", probabilities, values.float()).to(
            q.dtype
        )
    return output


@pytest.mark.parametrize(
    (
        "query_start_offsets",
        "num_query_tokens",
        "max_query_len",
        "has_prefill",
        "expected",
    ),
    [
        pytest.param(
            (0, 4, 8, 8, 8),
            16,
            4,
            False,
            4,
            id="uniform-g4-with-graph-padding",
        ),
        pytest.param((0, 1, 4), 4, 4, False, None, id="unsafe-one-plus-three"),
        pytest.param((0, 4, 8), 8, 4, True, None, id="prefill"),
        pytest.param(None, 8, 4, False, None, id="device-only-boundaries"),
    ],
)
def test_uniform_qsa_decode_query_len_requires_request_safe_groups(
    query_start_offsets: tuple[int, ...] | None,
    num_query_tokens: int,
    max_query_len: int,
    has_prefill: bool,
    expected: int | None,
) -> None:
    assert (
        qsa_cache._uniform_qsa_decode_query_len(
            query_start_offsets,
            num_query_tokens=num_query_tokens,
            max_query_len=max_query_len,
            has_prefill=has_prefill,
        )
        == expected
    )


@requires_qsa_kernels
def test_qsa_side_metadata_marks_cudagraph_padding_inert() -> None:
    device = torch.device("cuda")
    builder = QSAMetadataBuilder.__new__(QSAMetadataBuilder)
    builder.compress_ratio = 1
    builder.is_circular_buffer = False
    builder.decode_group_size = 4
    builder.full_decode_graph_token_counts = frozenset({16})
    builder.storage_block_size = 64
    builder.token_to_req_buffer = torch.empty(16, dtype=torch.int32, device=device)
    builder.slot_mapping_buffer = torch.empty(16, dtype=torch.int64, device=device)
    builder.logical_positions_buffer = torch.empty(16, dtype=torch.int64, device=device)
    builder.k_work_metadata_buffer = torch.empty(0, 2, dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 4, 8, 12, 12], dtype=torch.int32, device=device)
    token_to_req = torch.tensor([0] * 4 + [1] * 4 + [2] * 4 + [0] * 4, device=device)
    common = SimpleNamespace(
        num_actual_tokens=16,
        num_reqs=3,
        max_query_len=4,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor([68, 68, 68, 0], dtype=torch.int32, device=device),
        slot_mapping=torch.tensor(list(range(12)) + [-1] * 4, device=device),
        block_table_tensor=torch.empty((4, 0), dtype=torch.int32, device=device),
        token_to_req_indices=lambda buffer: buffer.copy_(token_to_req),
        is_prefilling=torch.zeros(4, dtype=torch.bool),
    )

    metadata = builder.build(0, common)

    assert metadata.logical_positions.tolist() == [
        64,
        65,
        66,
        67,
        64,
        65,
        66,
        67,
        64,
        65,
        66,
        67,
        -1,
        -1,
        -1,
        -1,
    ]
    assert metadata.slot_mapping.tolist() == list(range(12)) + [-1] * 4
    assert metadata.query_start_offsets == (0, 4, 8, 12)
    assert metadata.uniform_decode_query_len == 4
    assert metadata.prepare_cudagraph_plan

    builder.full_decode_graph_token_counts = frozenset()
    capture_metadata = builder.build_for_cudagraph_capture(common)
    assert capture_metadata.prepare_cudagraph_plan

    # Fast drafting has no authoritative CPU boundaries and therefore uses
    # the request-independent Q1 route, including during graph warmup.
    builder.full_decode_graph_token_counts = frozenset({16})
    fast_metadata = builder.build(0, common, fast_build=True)
    assert fast_metadata.query_start_offsets is None
    assert fast_metadata.uniform_decode_query_len is None
    assert fast_metadata.prepare_cudagraph_plan

    common.is_prefilling = torch.tensor([True, True, True, False])
    prefill_metadata = builder.build(0, common)
    assert prefill_metadata.query_start_offsets == (0, 4, 8, 12)
    assert prefill_metadata.uniform_decode_query_len is None
    assert not prefill_metadata.prepare_cudagraph_plan


@requires_qsa_kernels
def test_qsa_circular_buffer_metadata_keeps_only_each_requests_suffix() -> None:
    device = torch.device("cuda")
    builder = QSAMetadataBuilder.__new__(QSAMetadataBuilder)
    builder.compress_ratio = 4
    builder.is_circular_buffer = True
    builder.kv_cache_spec = SimpleNamespace(block_size=4)
    builder.storage_block_size = 4
    builder.token_to_req_buffer = torch.empty(16, dtype=torch.int32, device=device)
    builder.slot_mapping_buffer = torch.empty(16, dtype=torch.int64, device=device)
    builder.logical_positions_buffer = torch.empty(16, dtype=torch.int64, device=device)
    builder.k_work_metadata_buffer = torch.empty(0, 2, dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 7, 13, 13], dtype=torch.int32, device=device)
    token_to_req = torch.tensor([0] * 7 + [1] * 6 + [0] * 3, device=device)
    block_table = torch.tensor([[1], [0], [2]], dtype=torch.int32, device=device)
    common = SimpleNamespace(
        num_actual_tokens=16,
        num_reqs=2,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor([9, 11, 0], dtype=torch.int32, device=device),
        slot_mapping=torch.full((16,), -1, dtype=torch.int64, device=device),
        block_table_tensor=block_table,
        token_to_req_indices=lambda buffer: buffer.copy_(token_to_req),
    )

    metadata = builder.build(0, common)
    expected = [
        -1,
        -1,
        -1,
        5,
        6,
        7,
        4,
        -1,
        -1,
        3,
        0,
        1,
        2,
        -1,
        -1,
        -1,
    ]

    assert metadata.slot_mapping.tolist() == expected


@pytest.mark.parametrize("chunk_start", list(range(8)))
def test_qsa_circular_buffer_survives_one_speculative_step(chunk_start: int) -> None:
    """A speculative step must not overwrite the open group's committed keys.

    The step stores every row it computes, drafts included, before acceptance
    is known, while the next step still reads the earlier members of the group
    being compressed from the ring. A ring sized at the compression ratio makes
    those rows alias, so a rejected draft silently replaces a committed key.
    """
    compress_ratio = 4
    num_spec = 3
    capacity = compress_ratio * -(-(compress_ratio + num_spec) // compress_ratio)
    query_len = num_spec + 1

    slots = qsa_cache.circular_qsa_slot_mapping(
        torch.tensor([[0]], dtype=torch.int32),
        torch.zeros(query_len, dtype=torch.int32),
        torch.arange(chunk_start, chunk_start + query_len),
        capacity,
        query_start_loc=torch.tensor([0, query_len], dtype=torch.int32),
    )

    committed = torch.arange(chunk_start - chunk_start % compress_ratio, chunk_start)
    assert set(slots.tolist()).isdisjoint((committed % capacity).tolist())


def _qsa_key_cache(
    block_size: int,
    compress_ratio: int,
    *,
    cache_rope_positions: bool = False,
) -> qsa_cache.QSAKeyStateCache:
    return qsa_cache.QSAKeyStateCache(
        head_size=64,
        dtype=torch.bfloat16,
        cache_config=SimpleNamespace(block_size=block_size),
        prefix=f"raw.{block_size}.{compress_ratio}",
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={})
        ),
        compress_ratio=compress_ratio,
        cache_rope_positions=cache_rope_positions,
    )


def test_qsa_state_caches_adapt_the_unified_logical_layout() -> None:
    raw_cache = _qsa_key_cache(block_size=32, compress_ratio=4)
    compressed_cache = qsa_cache.QSACompressedKeyCache(
        head_size=64,
        dtype=torch.bfloat16,
        cache_config=SimpleNamespace(block_size=32),
        prefix="compressed.bind",
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={})
        ),
        compress_ratio=4,
    )
    raw_view = torch.empty(2, 1, 8, 64, dtype=torch.bfloat16)
    compressed_view = torch.empty(2, 1, 8, 64, dtype=torch.bfloat16)

    raw_cache.bind_kv_cache(raw_view)
    compressed_cache.bind_kv_cache(compressed_view)

    assert raw_cache.kv_cache.shape == (2, 8, 1, 64)
    assert compressed_cache.kv_cache.shape == (2, 8, 1, 64)
    assert raw_cache.kv_cache.data_ptr() == raw_view.data_ptr()
    assert compressed_cache.kv_cache.data_ptr() == compressed_view.data_ptr()


def test_qsa_key_state_cache_teardown_releases_derived_views() -> None:
    from vllm.v1.worker.utils import clear_layer_kv_caches

    cache = _qsa_key_cache(
        block_size=32,
        compress_ratio=4,
        cache_rope_positions=True,
    )
    raw_view = torch.empty(2, 1, 8, cache.head_size, dtype=torch.bfloat16)
    cache.bind_kv_cache(raw_view)
    key_cache = cache.key_cache
    rope_position_cache = cache.rope_position_cache
    raw_ref = weakref.ref(raw_view)
    key_ref = weakref.ref(key_cache)
    rope_ref = weakref.ref(rope_position_cache)
    del raw_view, key_cache, rope_position_cache

    clear_layer_kv_caches([cache])
    gc.collect()

    assert cache.kv_cache.numel() == 0
    assert cache.key_cache is None
    assert cache.rope_position_cache is None
    assert raw_ref() is None
    assert key_ref() is None
    assert rope_ref() is None


@pytest.mark.parametrize(
    ("compress_ratio", "num_spec", "expected"),
    [(4, 0, 4), (4, 1, 8), (4, 3, 8), (4, 4, 8), (4, 5, 12), (2, 3, 6)],
)
def test_qsa_ring_capacity_covers_one_speculative_step(
    compress_ratio: int, num_spec: int, expected: int
) -> None:
    """Capacity spans the open group plus one speculative step, in whole groups."""
    spec = _qsa_key_cache(
        block_size=48, compress_ratio=compress_ratio
    ).get_kv_cache_spec(SimpleNamespace(num_speculative_tokens=num_spec))
    assert spec.block_size == expected


@requires_qsa_kernels
def test_qsa_compressed_metadata_keeps_dummy_slots_inert() -> None:
    device = torch.device("cuda")
    builder = QSAMetadataBuilder.__new__(QSAMetadataBuilder)
    builder.compress_ratio = 4
    builder.is_circular_buffer = False
    builder.storage_block_size = 16
    builder.token_to_req_buffer = torch.empty(8, dtype=torch.int32, device=device)
    builder.slot_mapping_buffer = torch.empty(8, dtype=torch.int64, device=device)
    builder.logical_positions_buffer = torch.empty(8, dtype=torch.int64, device=device)
    # Simulate max_num_seqs exceeding the three live requests below.
    builder.request_capacity = 8
    builder.k_work_metadata_buffer = torch.empty(4, 2, dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 3, 3, 8], dtype=torch.int32, device=device)
    token_to_req = torch.tensor(
        [0, 0, 0, 2, 2, 2, 2, 2], dtype=torch.int32, device=device
    )
    common = SimpleNamespace(
        num_actual_tokens=8,
        num_reqs=3,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor([7, 0, 12], dtype=torch.int32, device=device),
        slot_mapping=torch.full((8,), -1, dtype=torch.int64, device=device),
        block_table_tensor=torch.zeros((3, 1), dtype=torch.int32, device=device),
        token_to_req_indices=lambda buffer: buffer.copy_(token_to_req),
    )

    metadata = builder.build(0, common)

    assert metadata.slot_mapping.tolist() == [-1] * 8
    assert metadata.k_work_metadata.tolist() == [[0, 0], [2, 0], [2, 1], [-1, -1]]


@requires_qsa_kernels
@pytest.mark.parametrize("compress_ratio", [1, 4])
@pytest.mark.parametrize("num_reqs", [2, 3, 4, 7, 8, 9])
def test_qsa_triton_metadata_matches_pytorch(
    compress_ratio: int, num_reqs: int
) -> None:
    device = torch.device("cuda")
    num_tokens = 8
    query_start_loc = torch.tensor(
        [0, 3, *([3] * (num_reqs - 2)), 8], dtype=torch.int32, device=device
    )
    token_to_req = torch.tensor(
        [0, 0, 0, *([num_reqs - 1] * 5)],
        dtype=torch.int32,
        device=device,
    )
    block_table_rows = torch.tensor(
        [
            [4, -1, 8, -1, 12, -1],
            [1, -1, 2, -1, 3, -1],
            [7, -1, 9, -1, 11, -1],
        ],
        dtype=torch.int32,
        device=device,
    )
    block_table_storage = block_table_rows[
        torch.arange(num_reqs, device=device) % block_table_rows.shape[0]
    ]
    seq_lens = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    seq_lens[0] = 10
    seq_lens[-1] = 20
    common = SimpleNamespace(
        num_actual_tokens=num_tokens,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=seq_lens,
        slot_mapping=torch.tensor(
            [0, 1, -1, 3, 4, -1, -1, -1], dtype=torch.int64, device=device
        ),
        block_table_tensor=block_table_storage[:, ::2],
        token_to_req_indices=lambda buffer: buffer.copy_(token_to_req),
    )

    def make_buffers() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.empty(num_tokens, dtype=torch.int32, device=device),
            torch.empty(num_tokens, dtype=torch.int64, device=device),
            torch.empty(num_tokens, dtype=torch.int64, device=device),
        )

    max_num_work = (
        (num_tokens + (compress_ratio - 1) * num_reqs) // compress_ratio
        if compress_ratio != 1
        else 0
    )
    actual_k_work = (
        torch.empty(max_num_work, 2, dtype=torch.int32, device=device)
        if max_num_work
        else None
    )
    actual_buffers = make_buffers()
    actual = qsa_cache.build_qsa_metadata_triton(
        common,
        *actual_buffers,
        storage_block_size=2,
        compress_ratio=compress_ratio,
        k_work_metadata_buffer=actual_k_work,
        request_capacity=num_reqs,
    )

    expected_k_work = (
        torch.empty_like(actual_k_work) if actual_k_work is not None else None
    )
    expected_buffers = make_buffers()
    expected = qsa_cache._build_qsa_metadata_torch(
        common,
        *expected_buffers,
        storage_block_size=2,
        compress_ratio=compress_ratio,
        k_work_metadata_buffer=expected_k_work,
        request_capacity=num_reqs,
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)
    if actual_k_work is not None:
        torch.testing.assert_close(actual_k_work, expected_k_work)


@requires_qsa_kernels
def test_qsa_fused_metadata_matches_pytorch_for_large_padded_prefill() -> None:
    device = torch.device("cuda")
    num_mapped_tokens = 4096
    num_tokens = 4224
    query_start_loc = torch.tensor(
        [0, num_mapped_tokens], dtype=torch.int32, device=device
    )
    common = SimpleNamespace(
        num_actual_tokens=num_tokens,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor(
            [num_mapped_tokens + 32], dtype=torch.int32, device=device
        ),
        block_table_tensor=torch.arange(256, dtype=torch.int32, device=device)[None],
        slot_mapping=torch.tensor(
            [0] * num_mapped_tokens + [-1] * (num_tokens - num_mapped_tokens),
            dtype=torch.int64,
            device=device,
        ),
        token_to_req_indices=lambda buffer: buffer.zero_(),
    )

    def make_buffers() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.empty(num_tokens, dtype=torch.int32, device=device),
            torch.empty(num_tokens, dtype=torch.int64, device=device),
            torch.empty(num_tokens, dtype=torch.int64, device=device),
        )

    max_num_work = (num_tokens + 3) // 4
    actual_k_work = torch.empty(max_num_work, 2, dtype=torch.int32, device=device)
    expected_k_work = torch.empty_like(actual_k_work)
    actual = qsa_cache.build_qsa_metadata_triton(
        common,
        *make_buffers(),
        storage_block_size=8,
        compress_ratio=4,
        k_work_metadata_buffer=actual_k_work,
    )
    expected = qsa_cache._build_qsa_metadata_torch(
        common,
        *make_buffers(),
        storage_block_size=8,
        compress_ratio=4,
        k_work_metadata_buffer=expected_k_work,
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)
    torch.testing.assert_close(actual_k_work, expected_k_work)


@requires_qsa_kernels
@pytest.mark.parametrize(
    "num_rows",
    [
        pytest.param(3, id="one_tile_per_program"),
        pytest.param(33, id="looped_tiles"),
    ],
)
def test_qsa_mqa_paged_matches_test_reference(num_rows: int) -> None:
    torch.manual_seed(1)
    head_dim = 128
    q = torch.randn(num_rows, 4, head_dim, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(40, 4, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    page_table = torch.randperm(40, device="cuda", dtype=torch.int32).reshape(2, 20)
    request_rows = [(num_rows + 1) // 2, num_rows // 2]
    token_to_req = torch.repeat_interleave(
        torch.arange(2, device="cuda", dtype=torch.int32),
        torch.tensor(request_rows, device="cuda"),
    )
    sequence_length_values = [320, 264]
    sequence_lengths = torch.tensor(
        sequence_length_values, device="cuda", dtype=torch.int32
    )
    query_positions = torch.cat(
        [
            torch.arange(length - rows, length, device="cuda", dtype=torch.int32)
            for rows, length in zip(request_rows, sequence_length_values, strict=True)
        ]
    )
    compress_ratio = 4
    visible_lengths = (query_positions + 1) // compress_ratio

    actual, actual_visible_blocks = qsa_ops.qsa_mqa_paged(
        q,
        cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        compress_ratio=compress_ratio,
    )
    expected = _qsa_mqa_paged_reference(
        q, cache, page_table, token_to_req, visible_lengths
    )

    torch.testing.assert_close(actual_visible_blocks, visible_lengths)
    # Top-k is bounded by visible_blocks; later columns are intentionally unwritten.
    columns = torch.arange(actual.shape[1], device=actual.device)
    visible = columns[None, :] < actual_visible_blocks[:, None]
    torch.testing.assert_close(actual[visible], expected[visible], rtol=1e-3, atol=1e-3)


@requires_qsa_kernels
def test_qsa_block_expansion_matches_test_reference() -> None:
    blocks = torch.tensor([[0, -1], [1, 0]], device="cuda", dtype=torch.int32)
    query_positions = torch.tensor([5, 10], device="cuda")
    sequence_lengths = torch.tensor([6, 11], device="cuda")
    token_to_req = torch.tensor([0, 1], device="cuda", dtype=torch.int32)

    actual = qsa_ops.expand_qsa_block_indices_cuda(
        blocks,
        query_positions,
        sequence_lengths,
        token_to_req,
        compress_ratio=4,
        token_topk=8,
    )
    expected = _expand_qsa_indices_reference(
        blocks,
        query_positions,
        sequence_lengths,
        compress_ratio=4,
        token_topk=8,
    )

    torch.testing.assert_close(actual, expected)


@requires_qsa_kernels
@pytest.mark.parametrize(
    ("num_rows", "num_query_heads", "num_kv_heads", "page_size"),
    [
        # Kernel-visible pages with --block-size 256 and hybrid-cache alignment.
        pytest.param(1, 24, 2, 1792, id="tp1_split64"),
        pytest.param(16, 12, 1, 1792, id="tp2_split32"),
        pytest.param(32, 6, 1, 1024, id="tp4_split8"),
        pytest.param(257, 6, 1, 1024, id="tp4_split4"),
        pytest.param(513, 6, 1, 1024, id="tp4_split1"),
    ],
)
def test_qsa_sparse_paged_attention_matches_test_reference(
    num_rows: int,
    num_query_heads: int,
    num_kv_heads: int,
    page_size: int,
) -> None:
    torch.manual_seed(2)
    head_dim = 256
    num_requests = 2
    num_selected_pages = 64
    # Keep the newest page outside the synthetic top-k as causal headroom.
    num_pages_per_request = num_selected_pages + 1
    num_cache_blocks = num_requests * num_pages_per_request
    indexer_budget = 2048
    indexer_compress_ratio = 4
    selection_width = indexer_budget + indexer_compress_ratio - 1
    q = torch.randn(
        num_rows, num_query_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    kv_cache = torch.randn(
        num_cache_blocks,
        page_size,
        num_kv_heads,
        2 * head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k_cache, v_cache = kv_cache.split(head_dim, dim=-1)
    block_table = (
        torch.randperm(num_cache_blocks, device="cuda")
        .reshape(num_requests, num_pages_per_request)
        .to(torch.int32)
    )
    rows_per_request = math.ceil(num_rows / num_requests)
    row_indices = torch.arange(num_rows, device="cuda", dtype=torch.int32)
    token_to_req = row_indices // rows_per_request
    request_row_counts = torch.tensor(
        [rows_per_request, num_rows - rows_per_request],
        device="cuda",
        dtype=torch.int32,
    )

    context_length = num_pages_per_request * page_size - 1
    block_topk = indexer_budget // indexer_compress_ratio
    compressed_blocks_per_page = page_size // indexer_compress_ratio
    selection = torch.arange(block_topk, device="cuda")
    selected_pages = selection % num_selected_pages
    selected_offsets = selection // num_selected_pages
    row_shifts = 2 * row_indices.unsqueeze(1)
    # Eight blocks per page; adjacent rows overlap by six of those eight.
    selected_offsets = (selected_offsets + row_shifts) % compressed_blocks_per_page
    block_indices = (selected_pages * compressed_blocks_per_page + selected_offsets).to(
        torch.int32
    )
    rows_within_request = row_indices % rows_per_request
    query_positions = (
        context_length - request_row_counts[token_to_req.long()] + rows_within_request
    ).to(torch.int64)
    sequence_lengths = torch.full(
        (num_requests,), context_length, device="cuda", dtype=torch.int32
    )
    logical_indices = qsa_ops.expand_qsa_block_indices_cuda(
        block_indices,
        query_positions,
        sequence_lengths,
        token_to_req,
        indexer_compress_ratio,
        indexer_budget,
    )
    assert logical_indices.shape == (num_rows, selection_width)
    scale = q.shape[-1] ** -0.5

    actual = qsa_ops.qsa_sparse_paged_attention(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
    )
    expected = _qsa_sparse_paged_attention_reference(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        scale,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@requires_qsa_kernels
def test_qsa_selection_chunks_workspace_and_matches_test_reference(
    monkeypatch: pytest.MonkeyPatch,
    workspace_init,
) -> None:
    rows, keys, heads, head_dim = 65, 640, 4, 16
    token_topk, compress_ratio = 2048, 4
    torch.manual_seed(3)
    q = torch.randn(rows, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(40, 16, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    page_table = torch.randperm(40, device="cuda", dtype=torch.int32).unsqueeze(0)
    token_to_req = torch.zeros(rows, device="cuda", dtype=torch.int32)
    query_positions = torch.full((rows,), 2559, device="cuda", dtype=torch.int32)
    sequence_lengths = torch.tensor([2560], device="cuda", dtype=torch.int32)
    monkeypatch.setattr(qsa_ops, "_LOGITS_WORKSPACE_BYTES", 32 * keys * 4)
    original_score = qsa_ops.qsa_mqa_paged
    scored_row_counts = []

    def record_score(query: torch.Tensor, *args, **kwargs):
        scored_row_counts.append(query.shape[0])
        return original_score(query, *args, **kwargs)

    monkeypatch.setattr(qsa_ops, "qsa_mqa_paged", record_score)

    actual = qsa_ops.qsa_select_paged_tokens(
        q,
        cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        token_topk,
        compress_ratio,
    )
    compact_out = torch.empty(
        rows,
        token_topk // compress_ratio,
        dtype=torch.int32,
        device="cuda",
    )
    compact = qsa_ops.qsa_select_paged_tokens(
        q,
        cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        token_topk,
        compress_ratio,
        compact_out,
        expand_blocks=False,
    )
    expected = _qsa_select_paged_tokens_reference(
        q,
        cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        token_topk,
        compress_ratio,
    )

    torch.testing.assert_close(actual.sort().values, expected.sort().values)
    expected_blocks = expected[:, :token_topk:compress_ratio] // compress_ratio
    torch.testing.assert_close(compact.sort().values, expected_blocks.sort().values)
    assert compact.data_ptr() == compact_out.data_ptr()
    assert max(scored_row_counts) <= 32
    assert sum(scored_row_counts) == 2 * rows


@requires_qsa_kernels
def test_qsa_selection_handles_no_complete_compressed_blocks(workspace_init) -> None:
    q = torch.zeros(2, 4, 8, device="cuda", dtype=torch.bfloat16)
    cache = torch.zeros(1, 16, 1, 8, device="cuda", dtype=torch.bfloat16)
    page_table = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
    token_to_req = torch.zeros(2, device="cuda", dtype=torch.int32)
    query_positions = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    sequence_lengths = torch.tensor([3], device="cuda", dtype=torch.int32)

    selected = qsa_ops.qsa_select_paged_tokens(
        q,
        cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        token_topk=2048,
        compress_ratio=4,
    )

    assert selected[0, :2].tolist() == [0, 1]
    assert selected[1, :3].tolist() == [0, 1, 2]
    assert torch.all(selected[0, 2:] == -1)
    assert torch.all(selected[1, 3:] == -1)


@requires_qsa_kernels
def test_qsa_streaming_compression_and_compressor_state_store_match_reference() -> None:
    head_dim = 8
    current_pairs = [
        *((0, position) for position in range(2, 9)),
        *((1, position) for position in range(5, 11)),
    ]

    def key_row(request: int, position: int) -> torch.Tensor:
        return (
            torch.arange(head_dim, dtype=torch.float32) + request * 1000 + position * 10
        )

    def position_row(request: int, position: int) -> torch.Tensor:
        return torch.tensor(
            [
                request * 1000 + position,
                request * 1000 + position + 100,
                request * 1000 + position + 200,
            ],
            dtype=torch.int64,
        )

    raw_keys = (
        torch.stack([key_row(request, position) for request, position in current_pairs])
        .unsqueeze(1)
        .to(device="cuda", dtype=torch.bfloat16)
    )
    raw_positions = (
        torch.stack(
            [position_row(request, position) for request, position in current_pairs]
        )
        .unsqueeze(1)
        .to(device="cuda")
    )
    token_to_req = torch.tensor(
        [request for request, _ in current_pairs],
        dtype=torch.int32,
        device="cuda",
    )
    logical_positions = torch.tensor(
        [position for _, position in current_pairs],
        dtype=torch.int64,
        device="cuda",
    )
    query_start_loc = torch.tensor([0, 7, 13], dtype=torch.int32, device="cuda")
    compressor_state_block_table = torch.tensor(
        [[1], [0]], dtype=torch.int32, device="cuda"
    )
    compressor_state_cache = torch.zeros(
        2, 4, 1, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    rope_cache = torch.zeros(2, 4, 1, 3, dtype=torch.int64, device="cuda")
    for request, position, block in ((0, 0, 1), (0, 1, 1), (1, 4, 0)):
        compressor_state_cache[block, position % 4, 0] = key_row(request, position).to(
            device="cuda", dtype=torch.bfloat16
        )
        rope_cache[block, position % 4, 0] = position_row(request, position).to("cuda")

    compressed_slots = torch.full(
        (len(current_pairs),), -1, dtype=torch.int64, device="cuda"
    )
    valid_rows = torch.tensor([1, 5, 9], dtype=torch.int64, device="cuda")
    compressed_slots[valid_rows] = torch.arange(3, device="cuda")
    pooled, first_positions = qsa_ops.qsa_compress_groups_with_ratio(
        raw_keys,
        raw_positions,
        compressor_state_cache,
        compressor_state_block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        compress_ratio=4,
        rope_cache=rope_cache,
    )
    pooled_without_rope, scalar_first_positions = (
        qsa_ops.qsa_compress_groups_with_ratio(
            raw_keys,
            raw_positions,
            compressor_state_cache,
            compressor_state_block_table,
            token_to_req,
            query_start_loc,
            logical_positions,
            compressed_slots,
            compress_ratio=4,
        )
    )

    groups = [
        [(0, position) for position in range(0, 4)],
        [(0, position) for position in range(4, 8)],
        [(1, position) for position in range(4, 8)],
    ]
    expected_pooled = (
        torch.stack(
            [
                torch.stack([key_row(*pair) for pair in group]).mean(dim=0)
                for group in groups
            ]
        )
        .unsqueeze(1)
        .to(device="cuda", dtype=torch.bfloat16)
    )
    expected_positions = torch.stack(
        [position_row(0, 0), position_row(0, 4), position_row(1, 4)]
    ).to("cuda")
    expected_scalar_positions = torch.tensor(
        [[0, 0, 0], [4, 4, 4], [4, 4, 4]],
        dtype=torch.int64,
        device="cuda",
    )

    torch.testing.assert_close(pooled[valid_rows], expected_pooled)
    torch.testing.assert_close(pooled_without_rope[valid_rows], expected_pooled)
    torch.testing.assert_close(first_positions[valid_rows], expected_positions)
    torch.testing.assert_close(
        scalar_first_positions[valid_rows], expected_scalar_positions
    )

    compressor_state_slots = torch.tensor(
        [-1, -1, -1, 5, 6, 7, 4, -1, -1, 3, 0, 1, 2],
        dtype=torch.int64,
        device="cuda",
    )
    qsa_ops.qsa_store_cache_rows(
        compressor_state_cache, compressor_state_slots, raw_keys
    )
    qsa_ops.qsa_store_cache_rows(rope_cache, compressor_state_slots, raw_positions)
    for request, positions, block in ((0, range(5, 9), 1), (1, range(7, 11), 0)):
        for position in positions:
            torch.testing.assert_close(
                compressor_state_cache[block, position % 4, 0],
                key_row(request, position).to(device="cuda", dtype=torch.bfloat16),
            )
            torch.testing.assert_close(
                rope_cache[block, position % 4, 0],
                position_row(request, position).to("cuda"),
            )


@pytest.mark.parametrize(
    ("backend", "capable", "available", "expected", "expected_probes"),
    [
        ("auto", True, True, True, 1),
        ("auto", True, False, False, 1),
        ("triton", True, True, False, 0),
        ("prims_ts", True, True, True, 1),
    ],
)
def test_qsa_attention_backend_selection(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    capable: bool,
    available: bool,
    expected: bool,
    expected_probes: int,
) -> None:
    from vllm.models.qwen4_exp.nvidia import qsa as qsa_model

    monkeypatch.setattr(
        qsa_model.envs,
        "VLLM_QSA_ATTENTION_BACKEND",
        backend,
    )
    probes = 0

    def availability_probe() -> bool:
        nonlocal probes
        probes += 1
        return available

    assert (
        qsa_model._resolve_qsa_prims_ts_backend(
            capable=capable,
            availability_probe=availability_probe,
        )
        is expected
    )
    assert probes == expected_probes


def test_qsa_attention_backend_selection_rejects_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.models.qwen4_exp.nvidia import qsa as qsa_model

    monkeypatch.setattr(
        qsa_model.envs,
        "VLLM_QSA_ATTENTION_BACKEND",
        "prims_ts",
    )
    with pytest.raises(RuntimeError, match="SM100 or SM103"):
        qsa_model._resolve_qsa_prims_ts_backend(
            capable=False,
            availability_probe=lambda: pytest.fail(
                "unsupported geometry must not probe FlashInfer"
            ),
        )


def test_qsa_prims_ts_geometry_capability() -> None:
    from vllm.models.qwen4_exp.nvidia import qsa as qsa_model

    def supported(
        *,
        num_heads: int = 24,
        head_size: int = 256,
        sparse_block_size: int = 4,
    ) -> bool:
        return qsa_model._supports_qsa_prims_ts_geometry(
            num_heads=num_heads,
            num_kv_heads=2,
            head_size=head_size,
            sparse_block_size=sparse_block_size,
            max_group_size=5,
        )

    assert supported()
    assert not supported(head_size=128)
    assert not supported(sparse_block_size=8)
    assert not supported(num_heads=26)


def test_qsa_prims_ts_device_capability_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.models.qwen4_exp.nvidia import qsa as qsa_model

    for actual, expected in ((100, True), (103, True), (101, False)):
        monkeypatch.setattr(
            qsa_model.current_platform,
            "is_device_capability",
            lambda requested, actual=actual: requested == actual,
        )
        assert qsa_model._supports_qsa_prims_ts_device() is expected


def _make_qsa_prims_ts_owner_case(
    group_size: int,
    query_starts: tuple[int, ...],
    kv_cache_dtype: str,
) -> SimpleNamespace:
    # Use the TP1 geometry so the owner test detects an accidental swap of
    # the non-singleton K/V head and storage-page axes.
    num_query_heads = 24
    num_kv_heads = 2
    head_dim = 256
    storage_page_size = 16
    block_topk = 512
    num_tokens = query_starts[-1]
    num_requests = len(query_starts) - 1
    num_storage_pages = 4 * num_requests

    query = torch.zeros(
        num_tokens,
        num_query_heads,
        head_dim,
        dtype=torch.bfloat16,
    )
    if kv_cache_dtype == "fp8":
        kv_cache = torch.zeros(
            num_storage_pages,
            num_kv_heads,
            storage_page_size,
            2 * head_dim,
            dtype=torch.float8_e4m3fn,
        ).view(torch.uint8)
        fp8_query_buffer = torch.empty(
            num_tokens,
            num_query_heads,
            head_dim,
            dtype=torch.float8_e4m3fn,
        )
    else:
        kv_cache = torch.zeros(
            num_storage_pages,
            num_kv_heads,
            storage_page_size,
            2 * head_dim,
            dtype=torch.bfloat16,
        )
        fp8_query_buffer = None

    request_lengths = torch.diff(torch.tensor(query_starts, dtype=torch.int32))
    token_to_req = torch.repeat_interleave(
        torch.arange(num_requests, dtype=torch.int32),
        request_lengths,
    )
    logical_positions = torch.cat(
        [
            torch.arange(
                2 * storage_page_size - int(length),
                2 * storage_page_size,
                dtype=torch.int64,
            )
            for length in request_lengths
        ]
    )
    block_table = torch.arange(
        num_storage_pages,
        dtype=torch.int32,
    ).reshape(num_requests, 4)
    output = torch.empty_like(query)
    layer = SimpleNamespace(
        qsa_indices_are_blocks=True,
        qsa_prims_ts_decode_group_size=group_size,
        indexer=SimpleNamespace(compress_ratio=4),
        topk_indices_buffer=torch.zeros(
            num_tokens,
            block_topk,
            dtype=torch.int32,
        ),
        _qsa_fp8_query_buffer=fp8_query_buffer,
        _q_scale=torch.tensor(2.0),
        _q_scale_float=2.0,
        _k_scale_float=3.0,
        _v_scale_float=4.0,
        _qsa_prims_ts_graph_states={},
        _qsa_prims_ts_eager_state=None,
    )
    metadata = SimpleNamespace(
        num_actual_tokens=num_tokens,
        block_table=block_table,
        max_seq_len=2 * storage_page_size,
    )
    return SimpleNamespace(
        query=query,
        kv_cache=kv_cache,
        output=output,
        layer=layer,
        metadata=metadata,
        token_to_req=token_to_req,
        logical_positions=logical_positions,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        storage_page_size=storage_page_size,
        block_topk=block_topk,
    )


@requires_real_qsa_prims_ts
@pytest.mark.parametrize(
    ("kv_cache_dtype", "group_size", "query_lengths", "has_prefill"),
    [
        pytest.param("bfloat16", 4, (5, 3), True, id="bf16-packed-g4"),
        pytest.param("bfloat16", 1, (1, 1), False, id="bf16-fixed-g1"),
        pytest.param("bfloat16", 4, (4, 4), False, id="bf16-fixed-g4"),
        pytest.param("bfloat16", 5, (5, 5), False, id="bf16-fixed-g5"),
        pytest.param("fp8", 4, (4, 4), False, id="fp8-fixed-g4"),
    ],
)
def test_qsa_prims_ts_owner_real_cuda_matches_reference_and_graph(
    kv_cache_dtype: str,
    group_size: int,
    query_lengths: tuple[int, int],
    has_prefill: bool,
) -> None:
    """Exercise the real vLLM owner boundary, including fixed CUDA graphs."""

    from vllm.models.qwen4_exp.nvidia import qsa as qsa_model

    device = torch.device("cuda")
    num_query_heads = 24
    num_kv_heads = 2
    head_dim = 256
    block_topk = 512
    sparse_block_size = 4
    storage_page_size = 16
    first_query_position = block_topk * sparse_block_size - 1
    pages_per_request = 129
    num_requests = len(query_lengths)
    num_tokens = sum(query_lengths)
    generator = torch.Generator(device=device).manual_seed(
        1701 + group_size + (100 if kv_cache_dtype == "fp8" else 0)
    )

    query = (
        torch.randn(
            num_tokens,
            num_query_heads,
            head_dim,
            generator=generator,
            device=device,
        )
        .mul_(0.125)
        .to(torch.bfloat16)
    )
    num_pages = num_requests * pages_per_request
    cache_dtype = (
        current_platform.fp8_dtype() if kv_cache_dtype == "fp8" else torch.bfloat16
    )
    typed_kv_cache = torch.empty(
        num_pages,
        num_kv_heads,
        storage_page_size,
        2 * head_dim,
        dtype=cache_dtype,
        device=device,
    )
    keys = torch.randn(
        num_pages,
        num_kv_heads,
        storage_page_size,
        head_dim,
        generator=generator,
        device=device,
    ).mul_(0.125)
    head_bias = torch.tensor((-0.375, 0.375), device=device).view(1, 2, 1, 1)
    values = (
        torch.randn(
            keys.shape,
            generator=generator,
            device=device,
        ).mul_(0.05)
        + head_bias
    )
    typed_kv_cache[..., :head_dim].copy_(keys)
    typed_kv_cache[..., head_dim:].copy_(values)
    kv_cache = (
        typed_kv_cache.view(torch.uint8) if kv_cache_dtype == "fp8" else typed_kv_cache
    )
    block_table = torch.arange(num_pages, dtype=torch.int32, device=device).reshape(
        num_requests, pages_per_request
    )
    token_to_req = torch.repeat_interleave(
        torch.arange(num_requests, dtype=torch.int32, device=device),
        torch.tensor(query_lengths, dtype=torch.int64, device=device),
    )
    logical_positions = torch.cat(
        [
            torch.arange(
                first_query_position,
                first_query_position + length,
                dtype=torch.int64,
                device=device,
            )
            for length in query_lengths
        ]
    )
    sequence_lengths = torch.tensor(
        [first_query_position + length for length in query_lengths],
        dtype=torch.int64,
        device=device,
    )
    base_blocks = torch.arange(block_topk, dtype=torch.int32, device=device)
    compact_blocks = torch.stack(
        [torch.roll(base_blocks, row % 7) for row in range(num_tokens)]
    ).contiguous()
    query_start_offsets = (0, query_lengths[0], num_tokens)
    output = torch.empty_like(query)
    layer = SimpleNamespace(
        qsa_indices_are_blocks=True,
        qsa_prims_ts_decode_group_size=group_size,
        indexer=SimpleNamespace(compress_ratio=sparse_block_size),
        topk_indices_buffer=compact_blocks,
        _qsa_fp8_query_buffer=(
            torch.empty_like(query, dtype=cache_dtype)
            if kv_cache_dtype == "fp8"
            else None
        ),
        _q_scale=torch.tensor(1.0, dtype=torch.float32, device=device),
        _q_scale_float=1.0,
        _k_scale_float=1.0,
        _v_scale_float=1.0,
        _qsa_prims_ts_graph_states={},
        _qsa_prims_ts_eager_state=None,
    )
    metadata = SimpleNamespace(
        num_actual_tokens=num_tokens,
        block_table=block_table,
        max_seq_len=int(sequence_lengths.max().item()),
    )
    impl = qsa_model.Qwen4ExpQSAFlashAttentionImpl.__new__(
        qsa_model.Qwen4ExpQSAFlashAttentionImpl
    )
    impl.head_size = head_dim
    impl.scale = head_dim**-0.5
    impl.kv_cache_dtype = kv_cache_dtype
    impl.alibi_slopes = None
    impl.sinks = None
    impl.sliding_window = (-1, -1)
    impl.use_qsa_prims_ts = True

    def run() -> torch.Tensor:
        return impl.forward_qsa(
            layer,
            query,
            query,
            query,
            kv_cache,
            metadata,
            output,
            token_to_req,
            logical_positions,
            query_start_offsets=query_start_offsets,
            has_prefill=has_prefill,
            uniform_decode_query_len=None if has_prefill else group_size,
            persistent_plan=not has_prefill,
        )

    actual = run()
    torch.accelerator.synchronize()
    assert actual is output
    assert compact_blocks.shape == (num_tokens, block_topk)

    row_sequence_lengths = sequence_lengths.index_select(0, token_to_req.long())
    # This is the 2,051-token representation consumed by the vLLM Triton
    # path and by the independent FP32 oracle; PrimTS receives only 512 blocks.
    expanded_tokens = _expand_qsa_indices_reference(
        compact_blocks,
        logical_positions,
        row_sequence_lengths,
        sparse_block_size,
        block_topk * sparse_block_size,
    )
    reference_query = (
        layer._qsa_fp8_query_buffer.float()
        if kv_cache_dtype == "fp8"
        else query.float()
    )
    reference = _qsa_sparse_paged_attention_reference(
        reference_query,
        typed_kv_cache[..., :head_dim].transpose(1, 2),
        typed_kv_cache[..., head_dim:].transpose(1, 2),
        expanded_tokens,
        block_table,
        token_to_req,
        impl.scale,
    )
    tolerance = 3e-2 if kv_cache_dtype == "fp8" else 1e-2
    torch.testing.assert_close(
        actual.float(), reference, rtol=tolerance, atol=tolerance
    )

    if has_prefill:
        assert not layer._qsa_prims_ts_graph_states
        assert layer._qsa_prims_ts_eager_state is not None
        return

    prepared_state = next(iter(layer._qsa_prims_ts_graph_states.values()))
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    assert prepared_state in layer._qsa_prims_ts_graph_states.values()
    output.fill_(torch.nan)
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        output.float(), reference, rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize(
    (
        "kv_cache_dtype",
        "has_prefill",
        "uniform_decode_query_len",
        "group_size",
        "query_starts",
        "query_start_offsets",
        "expected_qo_indptr",
    ),
    [
        pytest.param(
            "bfloat16",
            True,
            None,
            4,
            (0, 5, 8),
            (0, 5, 8),
            (0, 4, 5, 8),
            id="packed-prefill-g4",
        ),
        pytest.param(
            "bfloat16",
            False,
            1,
            1,
            (0, 1, 2),
            (0, 1, 2),
            None,
            id="fixed-decode-g1",
        ),
        pytest.param(
            "bfloat16",
            False,
            4,
            4,
            (0, 4, 8),
            (0, 4, 8),
            None,
            id="fixed-decode-g4",
        ),
        pytest.param(
            "bfloat16",
            False,
            5,
            5,
            (0, 5, 10),
            (0, 5, 10),
            None,
            id="fixed-decode-g5",
        ),
        pytest.param(
            "fp8",
            False,
            4,
            4,
            (0, 4, 8),
            (0, 4, 8),
            None,
            id="fp8-fixed-decode-g4",
        ),
        pytest.param(
            "bfloat16",
            False,
            None,
            4,
            (0, 1, 4),
            (0, 1, 4),
            (0, 1, 4),
            id="packed-irregular-decode",
        ),
        pytest.param(
            "bfloat16",
            True,
            None,
            4,
            (0, 5, 8),
            None,
            tuple(range(9)),
            id="packed-prefill-device-boundaries",
        ),
    ],
)
def test_qsa_prims_ts_owner_routes_fixed_and_packed_queries(
    monkeypatch: pytest.MonkeyPatch,
    kv_cache_dtype: str,
    has_prefill: bool,
    uniform_decode_query_len: int | None,
    group_size: int,
    query_starts: tuple[int, ...],
    query_start_offsets: tuple[int, ...] | None,
    expected_qo_indptr: tuple[int, ...] | None,
) -> None:
    from vllm.models.qwen4_exp.nvidia import qsa as qsa_model

    case = _make_qsa_prims_ts_owner_case(
        group_size,
        query_starts,
        kv_cache_dtype,
    )
    num_tokens = query_starts[-1]
    route_starts = (
        (0, num_tokens) if has_prefill and query_start_offsets is None else query_starts
    )
    route_group_size = 1 if has_prefill and query_start_offsets is None else group_size
    num_query_groups = (
        len(expected_qo_indptr) - 1
        if expected_qo_indptr is not None
        else num_tokens // group_size
    )
    expected_query_shape = (
        (num_tokens, case.num_query_heads, case.head_dim)
        if expected_qo_indptr is not None
        else (
            num_query_groups,
            1,
            group_size,
            case.num_query_heads,
            case.head_dim,
        )
    )
    captured: dict[str, object] = {}
    prepared_plan = object()

    if expected_qo_indptr is None:
        monkeypatch.setattr(
            qsa_ops,
            "qsa_prims_ts_qo_indptr",
            lambda *_args, **_kwargs: pytest.fail(
                "fixed decode must not build packed query offsets"
            ),
        )
    else:

        def fake_qo_indptr(
            actual_starts: tuple[int, ...],
            actual_num_tokens: int,
            actual_group_size: int,
        ) -> torch.Tensor:
            captured["packed_route"] = (
                actual_starts,
                actual_num_tokens,
                actual_group_size,
            )
            return torch.tensor(expected_qo_indptr, dtype=torch.int32)

        monkeypatch.setattr(qsa_ops, "qsa_prims_ts_qo_indptr", fake_qo_indptr)

    def fake_get_state(
        _layer: object,
        route_query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_indices: torch.Tensor,
        block_table: torch.Tensor,
        token_to_req: torch.Tensor,
        logical_positions: torch.Tensor,
        route_output: torch.Tensor,
        bmm1_scale: float,
        bmm2_scale: float,
        qo_indptr: torch.Tensor | None,
        qo_topology: tuple[int, ...] | None,
        route_group_size: int,
        sparse_block_size: int,
        *,
        persistent: bool,
    ) -> SimpleNamespace:
        state = SimpleNamespace(
            query=route_query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_indices=block_indices,
            block_table=block_table,
            token_to_req=token_to_req,
            logical_positions=logical_positions,
            output=route_output,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            qo_indptr=qo_indptr,
            qo_topology=qo_topology,
            group_size=route_group_size,
            sparse_block_size=sparse_block_size,
            persistent=persistent,
            plan=prepared_plan,
        )
        captured["state"] = state
        return state

    def fake_run(
        plan: object,
        actual_query: torch.Tensor,
        block_indices: torch.Tensor,
        actual_block_table: torch.Tensor,
        actual_token_to_req: torch.Tensor,
        actual_positions: torch.Tensor,
        actual_output: torch.Tensor,
    ) -> torch.Tensor:
        assert plan is prepared_plan
        captured["run"] = (
            actual_query,
            block_indices,
            actual_block_table,
            actual_token_to_req,
            actual_positions,
        )
        actual_output.fill_(7)
        return actual_output

    monkeypatch.setattr(qsa_ops, "qsa_prims_ts_run_prepared", fake_run)

    if kv_cache_dtype == "fp8":
        monkeypatch.setattr(
            qsa_model,
            "current_platform",
            SimpleNamespace(fp8_dtype=lambda: torch.float8_e4m3fn),
        )

        def fake_scaled_fp8_quant(
            actual_query: torch.Tensor,
            *,
            scale: torch.Tensor,
            output: torch.Tensor,
        ) -> None:
            assert scale is case.layer._q_scale
            assert actual_query.dtype == torch.bfloat16
            assert output.dtype == torch.float8_e4m3fn
            output.zero_()
            captured["quantized"] = output

        monkeypatch.setattr(
            qsa_model.custom_ops,
            "scaled_fp8_quant",
            fake_scaled_fp8_quant,
        )

    impl = qsa_model.Qwen4ExpQSAFlashAttentionImpl.__new__(
        qsa_model.Qwen4ExpQSAFlashAttentionImpl
    )
    impl.head_size = case.head_dim
    impl.scale = case.head_dim**-0.5
    impl.kv_cache_dtype = kv_cache_dtype
    impl.alibi_slopes = None
    impl.sinks = None
    impl.sliding_window = (-1, -1)
    impl.use_qsa_prims_ts = True
    monkeypatch.setattr(impl, "_get_qsa_prims_ts_prepared_state", fake_get_state)

    result = impl.forward_qsa(
        case.layer,
        case.query,
        case.query,
        case.query,
        case.kv_cache,
        case.metadata,
        case.output,
        case.token_to_req,
        case.logical_positions,
        query_start_offsets=query_start_offsets,
        has_prefill=has_prefill,
        uniform_decode_query_len=uniform_decode_query_len,
        persistent_plan=expected_qo_indptr is None,
    )

    assert result is case.output
    assert torch.all(result == 7)

    state = cast(SimpleNamespace, captured["state"])
    assert state.query.shape == expected_query_shape
    assert state.output.shape == expected_query_shape
    assert (
        state.key_cache.shape
        == state.value_cache.shape
        == (
            4 * (len(query_starts) - 1),
            case.num_kv_heads,
            case.storage_page_size,
            case.head_dim,
        )
    )
    expected_table_width = 2 if expected_qo_indptr is not None else 4
    assert state.block_table.shape[1] == expected_table_width
    assert state.block_indices.shape == (num_tokens, case.block_topk)
    assert state.group_size == route_group_size
    assert state.sparse_block_size == 4
    assert state.persistent is (expected_qo_indptr is None)
    if expected_qo_indptr is not None:
        assert captured["packed_route"] == (
            route_starts,
            num_tokens,
            route_group_size,
        )
        assert tuple(state.qo_indptr.tolist()) == expected_qo_indptr
        assert state.qo_topology == route_starts
    else:
        assert state.qo_indptr is None
        assert state.qo_topology is None

    if kv_cache_dtype == "fp8":
        assert state.query.dtype == state.key_cache.dtype == torch.float8_e4m3fn
        quantized_query = captured["quantized"]
        assert isinstance(quantized_query, torch.Tensor)
        assert quantized_query.data_ptr() == state.query.data_ptr()
        assert state.bmm1_scale == pytest.approx(impl.scale * 2.0 * 3.0)
        assert state.bmm2_scale == 4.0
    else:
        assert state.query.dtype == state.key_cache.dtype == torch.bfloat16
        assert state.bmm1_scale == impl.scale
        assert state.bmm2_scale == 1.0


def test_qsa_prims_ts_public_adapter_forwards_grouped_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: set[str] = set()

    class FakePlan:
        def run(
            self,
            q: torch.Tensor,
            block_indices: torch.Tensor,
            block_table: torch.Tensor,
            token_to_req: torch.Tensor,
            logical_positions: torch.Tensor,
            *,
            out: torch.Tensor,
        ) -> torch.Tensor:
            assert q.shape[0] == block_indices.shape[0] == token_to_req.shape[0]
            assert block_table.ndim == 2
            assert logical_positions.shape == token_to_req.shape
            calls.add("run")
            out.fill_(5)
            return out

    def workspace_size(
        _q: torch.Tensor,
        _k_cache: torch.Tensor,
        _block_table: torch.Tensor,
        **options: object,
    ) -> int:
        assert options["sparse_block_size"] == 4
        assert options["max_seq_len_q"] == 2
        calls.add("workspace")
        return 64

    def make_qo_indptr(
        starts: torch.Tensor,
        num_tokens: int,
        *,
        group_size: int,
        device: str,
    ) -> torch.Tensor:
        assert (tuple(starts.tolist()), num_tokens, group_size, device) == (
            (0, 1, 3),
            3,
            2,
            "cpu",
        )
        calls.add("qo_indptr")
        return torch.tensor([0, 1, 3], dtype=torch.int32)

    def prepare(
        q: torch.Tensor,
        paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
        block_indices: torch.Tensor,
        _block_table: torch.Tensor,
        _token_to_req: torch.Tensor,
        _logical_positions: torch.Tensor,
        workspace: torch.Tensor,
        **options: object,
    ) -> FakePlan:
        assert paged_kv_cache[0].shape == paged_kv_cache[1].shape
        assert q.shape[0] == block_indices.shape[0]
        assert workspace.numel() == 64
        assert options["max_seq_len_q"] == 2
        assert options["sparse_block_size"] == 4
        calls.add("prepare")
        return FakePlan()

    monkeypatch.setattr(
        qsa_ops,
        "_qsa_prims_ts_apis",
        lambda: (
            lambda *_args, **_kwargs: None,
            workspace_size,
            make_qo_indptr,
            prepare,
        ),
    )
    qsa_ops._qsa_prims_ts_qo_indptr_cached.cache_clear()
    q = torch.empty(3, 12, 256, dtype=torch.bfloat16)
    cache = torch.empty(2, 1, 16, 256, dtype=torch.bfloat16)
    blocks = torch.empty(3, 512, dtype=torch.int32)
    block_table = torch.empty(1, 2, dtype=torch.int32)
    token_to_req = torch.zeros(3, dtype=torch.int32)
    positions = torch.arange(3, dtype=torch.int64)
    workspace = torch.empty(64, dtype=torch.uint8)
    output = torch.empty_like(q)

    qo_indptr = qsa_ops.qsa_prims_ts_qo_indptr((0, 1, 3), 3, 2)
    required = qsa_ops.qsa_prims_ts_combined_workspace_size(
        q,
        cache,
        block_table,
        512,
        qo_indptr=qo_indptr,
        max_seq_len_q=2,
        sparse_block_size=4,
    )
    plan = qsa_ops.qsa_prims_ts_prepare_attention(
        q,
        cache,
        cache,
        blocks,
        block_table,
        token_to_req,
        positions,
        workspace,
        output,
        qo_indptr=qo_indptr,
        max_seq_len_q=2,
        sparse_block_size=4,
    )
    result = qsa_ops.qsa_prims_ts_run_prepared(
        plan,
        q,
        blocks,
        block_table,
        token_to_req,
        positions,
        output,
    )

    assert required == workspace.numel()
    assert calls == {"qo_indptr", "workspace", "prepare", "run"}
    assert result is output
    assert torch.all(output == 5)


def test_qsa_kv_cache_rebind_clears_prepared_storage() -> None:
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAAttention

    layer = Qwen4ExpQSAAttention.__new__(Qwen4ExpQSAAttention)
    torch.nn.Module.__init__(layer)
    layer._qsa_prims_ts_graph_states = {"profiling": object()}
    layer._qsa_prims_ts_eager_state = object()
    layer.kv_cache = torch.empty(0)
    real_kv_cache = torch.empty(2, 1, 16, 512)

    Qwen4ExpQSAAttention.bind_kv_cache(layer, real_kv_cache)

    assert layer.kv_cache is real_kv_cache
    assert not layer._qsa_prims_ts_graph_states
    assert layer._qsa_prims_ts_eager_state is None


def test_qsa_generic_kv_cache_teardown_releases_prepared_storage() -> None:
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAAttention
    from vllm.v1.worker.utils import clear_layer_kv_caches

    layer = Qwen4ExpQSAAttention.__new__(Qwen4ExpQSAAttention)
    torch.nn.Module.__init__(layer)
    layer.kv_cache = torch.empty(2, 1, 16, 512)
    key_cache, value_cache = layer.kv_cache.split(256, dim=-1)
    workspace = torch.empty(128, dtype=torch.uint8)
    plan = SimpleNamespace(key_cache=key_cache, value_cache=value_cache)
    state = SimpleNamespace(key=("graph",), workspace=workspace, plan=plan)
    key_ref = weakref.ref(key_cache)
    value_ref = weakref.ref(value_cache)
    workspace_ref = weakref.ref(workspace)
    layer._qsa_prims_ts_graph_states = {"graph": state}
    layer._qsa_prims_ts_eager_state = state
    del key_cache, value_cache, workspace, plan, state

    clear_layer_kv_caches([layer])
    gc.collect()

    assert layer.kv_cache.numel() == 0
    assert not layer._qsa_prims_ts_graph_states
    assert layer._qsa_prims_ts_eager_state is None
    assert key_ref() is None
    assert value_ref() is None
    assert workspace_ref() is None
