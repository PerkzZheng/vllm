# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

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
requires_qsa_triton = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_TRITON,
    reason="QSA metadata kernels require a CUDA-capable Triton runtime",
)
requires_qsa_prims_ts = pytest.mark.skipif(
    not torch.cuda.is_available() or not qsa_ops.has_qsa_prims_ts_attention(),
    reason="QSA attention requires page-4 FlashInfer PrimTS",
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


@requires_qsa_kernels
def test_qsa_side_metadata_marks_cudagraph_padding_inert() -> None:
    device = torch.device("cuda")
    builder = QSAMetadataBuilder.__new__(QSAMetadataBuilder)
    builder.compress_ratio = 1
    builder.is_circular_buffer = False
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
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor([68, 68, 68, 0], dtype=torch.int32, device=device),
        slot_mapping=torch.tensor(list(range(12)) + [-1] * 4, device=device),
        block_table_tensor=torch.empty((4, 0), dtype=torch.int32, device=device),
        token_to_req_indices=lambda buffer: buffer.copy_(token_to_req),
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


def _qsa_key_cache(block_size: int, compress_ratio: int) -> qsa_cache.QSAKeyStateCache:
    return qsa_cache.QSAKeyStateCache(
        head_size=64,
        dtype=torch.bfloat16,
        cache_config=SimpleNamespace(block_size=block_size),
        prefix=f"raw.{block_size}.{compress_ratio}",
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={})
        ),
        compress_ratio=compress_ratio,
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
    assert scored_row_counts == [32, 32, 1, 32, 32, 1]


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
    ("backend", "capable", "available", "expected"),
    [
        (None, True, True, True),
        ("auto", True, False, False),
        ("triton", True, True, False),
        ("prims_ts", True, True, True),
    ],
)
def test_qsa_attention_backend_selection(
    monkeypatch: pytest.MonkeyPatch,
    backend: str | None,
    capable: bool,
    available: bool,
    expected: bool,
) -> None:
    from vllm.models.qwen4_exp.nvidia.qsa import (
        _QSA_BACKEND_ENV,
        _resolve_qsa_prims_ts_backend,
    )

    if backend is None:
        monkeypatch.delenv(_QSA_BACKEND_ENV, raising=False)
    else:
        monkeypatch.setenv(_QSA_BACKEND_ENV, backend)
    assert (
        _resolve_qsa_prims_ts_backend(capable=capable, available=available) is expected
    )


def test_qsa_attention_backend_selection_rejects_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.models.qwen4_exp.nvidia.qsa import (
        _QSA_BACKEND_ENV,
        _resolve_qsa_prims_ts_backend,
    )

    monkeypatch.setenv(_QSA_BACKEND_ENV, "prims_ts")
    with pytest.raises(RuntimeError, match="SM100-family"):
        _resolve_qsa_prims_ts_backend(capable=False, available=True)

    monkeypatch.setenv(_QSA_BACKEND_ENV, "unknown")
    with pytest.raises(ValueError, match=_QSA_BACKEND_ENV):
        _resolve_qsa_prims_ts_backend(capable=True, available=True)


def test_qsa_prims_ts_batch_capacity_uses_bounded_power_of_two_buckets() -> None:
    from vllm.models.qwen4_exp.nvidia.qsa import _qsa_prims_ts_batch_capacity

    assert _qsa_prims_ts_batch_capacity(1, 16384) == 1
    assert _qsa_prims_ts_batch_capacity(64, 16384) == 64
    assert _qsa_prims_ts_batch_capacity(65, 16384) == 128
    assert _qsa_prims_ts_batch_capacity(12000, 12000) == 12000
    with pytest.raises(ValueError, match="fit the staging capacity"):
        _qsa_prims_ts_batch_capacity(0, 16384)
    with pytest.raises(ValueError, match="fit the staging capacity"):
        _qsa_prims_ts_batch_capacity(16385, 16384)


@pytest.mark.parametrize("indices_are_blocks", [False, True])
def test_qsa_attention_owner_preserves_pr53896_cache_layout(
    monkeypatch: pytest.MonkeyPatch,
    indices_are_blocks: bool,
) -> None:
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAFlashAttentionImpl

    rows = 3
    route_capacity = 4
    num_query_heads = 6
    num_kv_heads = 1
    head_dim = 256
    storage_page_size = 16
    selection_width = 512 if indices_are_blocks else 2051
    page_capacity = 513
    query = torch.zeros(rows, num_query_heads, head_dim, dtype=torch.bfloat16)
    kv_cache = torch.zeros(
        3,
        num_kv_heads,
        storage_page_size,
        2 * head_dim,
        dtype=torch.bfloat16,
    )
    output = torch.full_like(query, torch.nan)
    block_table = torch.zeros(2, 256, dtype=torch.int32)
    token_to_req = torch.tensor([0, 1, 0], dtype=torch.int32)
    logical_positions = torch.tensor([2, 3, 4], dtype=torch.int64)
    layer = SimpleNamespace(
        qsa_indices_are_blocks=indices_are_blocks,
        topk_indices_buffer=torch.full(
            (route_capacity, selection_width), -1, dtype=torch.int32
        ),
        qsa_paged_kv_indptr_buffer=torch.empty(route_capacity + 1, dtype=torch.int32),
        qsa_paged_kv_indices_buffer=torch.empty(
            route_capacity * page_capacity, dtype=torch.int32
        ),
        qsa_seq_lens_buffer=torch.empty(route_capacity, dtype=torch.int32),
        _qsa_prims_ts_token_to_req_buffer=torch.zeros(
            route_capacity, dtype=torch.int32
        ),
        _qsa_prims_ts_logical_positions_buffer=torch.full(
            (route_capacity,), -1, dtype=torch.int64
        ),
        _qsa_prims_ts_bf16_query_buffer=torch.zeros(
            route_capacity,
            num_query_heads,
            head_dim,
            dtype=torch.bfloat16,
        ),
        _qsa_prims_ts_output_buffer=torch.empty(
            route_capacity,
            num_query_heads,
            head_dim,
            dtype=torch.bfloat16,
        ),
        _qsa_prims_ts_workspace=None,
    )
    metadata = SimpleNamespace(num_actual_tokens=rows, block_table=block_table)
    calls: dict[str, object] = {}

    def fake_build_metadata(*_args, **buffers) -> None:
        assert buffers["indices_are_blocks"] is indices_are_blocks
        buffers["paged_kv_indptr"].copy_(
            torch.arange(route_capacity + 1, dtype=torch.int32) * page_capacity
        )
        buffers["paged_kv_indices"].fill_(-1)
        buffers["seq_lens"].copy_(torch.tensor([3, 1, 1, 1], dtype=torch.int32))
        assert _args[0].shape[0] == route_capacity
        assert _args[2].shape == _args[3].shape == (route_capacity,)
        assert int(_args[3][-1]) == -1

    def fake_workspace_size(
        actual_query: torch.Tensor,
        key_cache: torch.Tensor,
        max_seq_len: int,
        *,
        out_dtype: torch.dtype,
    ) -> int:
        assert actual_query.shape == (
            route_capacity,
            num_query_heads,
            head_dim,
        )
        assert key_cache.shape == (3, 1, storage_page_size, head_dim)
        assert key_cache.stride(-1) == 1
        assert max_seq_len == 2051
        assert out_dtype == torch.bfloat16
        return 128

    def fake_attention(
        actual_query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        workspace: torch.Tensor,
        _paged_kv_indptr: torch.Tensor,
        _paged_kv_indices: torch.Tensor,
        _seq_lens: torch.Tensor,
        _max_seq_len: int,
        actual_output: torch.Tensor,
        *,
        bmm1_scale: float,
        bmm2_scale: float,
    ) -> torch.Tensor:
        torch.testing.assert_close(actual_query[:rows], query)
        assert (
            key_cache.shape
            == value_cache.shape
            == (
                3,
                1,
                storage_page_size,
                head_dim,
            )
        )
        assert workspace.shape == (128,)
        assert bmm1_scale == head_dim**-0.5
        assert bmm2_scale == 1.0
        actual_output.fill_(5)
        calls["workspace"] = workspace
        return actual_output

    monkeypatch.setattr(qsa_ops, "qsa_build_page4_paged_metadata", fake_build_metadata)
    monkeypatch.setattr(qsa_ops, "qsa_prims_ts_group_size", lambda *_args: 1)
    monkeypatch.setattr(qsa_ops, "qsa_prims_ts_workspace_size", fake_workspace_size)
    monkeypatch.setattr(qsa_ops, "qsa_prims_ts_paged_attention", fake_attention)
    impl = Qwen4ExpQSAFlashAttentionImpl.__new__(Qwen4ExpQSAFlashAttentionImpl)
    impl.head_size = head_dim
    impl.scale = head_dim**-0.5
    impl.kv_cache_dtype = "auto"
    impl.alibi_slopes = None
    impl.sinks = None
    impl.sliding_window = (-1, -1)
    impl.use_qsa_prims_ts = True

    result = impl.forward_qsa(
        layer,
        query,
        query,
        query,
        kv_cache,
        metadata,
        output,
        token_to_req,
        logical_positions,
    )

    assert result is output
    assert torch.all(output == 5)
    assert layer._qsa_prims_ts_workspace is calls["workspace"]


def test_qsa_prims_ts_workspace_reuse_does_not_record_shape_memsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAFlashAttentionImpl

    def fake_workspace_size(
        query: torch.Tensor,
        _key_cache: torch.Tensor,
        _max_seq_len: int,
        *,
        out_dtype: torch.dtype,
    ) -> int:
        assert out_dtype == torch.bfloat16
        return 128 if query.shape[0] <= 3 else 256

    monkeypatch.setattr(qsa_ops, "qsa_prims_ts_workspace_size", fake_workspace_size)
    impl = Qwen4ExpQSAFlashAttentionImpl.__new__(Qwen4ExpQSAFlashAttentionImpl)
    layer = SimpleNamespace(_qsa_prims_ts_workspace=None)
    key_cache = torch.empty(1, 1, 16, 256, dtype=torch.bfloat16)

    first = impl._get_qsa_prims_ts_workspace(
        layer,
        torch.empty(2, 6, 256, dtype=torch.bfloat16),
        key_cache,
        2051,
        torch.bfloat16,
    )
    first.fill_(0xA5)
    second = impl._get_qsa_prims_ts_workspace(
        layer,
        torch.empty(3, 6, 256, dtype=torch.bfloat16),
        key_cache,
        2051,
        torch.bfloat16,
    )

    assert second.data_ptr() == first.data_ptr()
    assert torch.all(second == 0xA5)

    larger = impl._get_qsa_prims_ts_workspace(
        layer,
        torch.empty(4, 6, 256, dtype=torch.bfloat16),
        key_cache,
        2051,
        torch.bfloat16,
    )
    assert larger.numel() == 256


def test_qsa_prims_ts_wrappers_forward_grouped_sq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups = 3
    group_size = 4
    num_query_heads = 6
    head_dim = 256
    query = torch.zeros(
        groups,
        group_size,
        num_query_heads,
        head_dim,
        dtype=torch.bfloat16,
    )
    key_cache = torch.zeros(8, 1, 16, head_dim, dtype=torch.bfloat16)
    value_cache = torch.zeros_like(key_cache)
    calls: dict[str, object] = {}

    def fake_workspace_size(**kwargs) -> int:
        calls["workspace"] = kwargs
        return 320

    def fake_attention(*args, **kwargs) -> torch.Tensor:
        calls["attention_args"] = args
        calls["attention_kwargs"] = kwargs
        return kwargs["out"]

    monkeypatch.setattr(
        qsa_ops,
        "_qsa_prims_ts_apis",
        lambda: (lambda *_args, **_kwargs: 1, fake_workspace_size, fake_attention),
    )
    assert (
        qsa_ops.qsa_prims_ts_workspace_size(
            query, key_cache, 8208, out_dtype=torch.bfloat16
        )
        == 320
    )
    workspace_kwargs = calls["workspace"]
    assert isinstance(workspace_kwargs, dict)
    assert workspace_kwargs["batch_size"] == groups
    assert workspace_kwargs["seq_len_q"] == group_size
    assert workspace_kwargs["num_qo_heads"] == num_query_heads
    assert workspace_kwargs["out_dtype"] == torch.bfloat16

    workspace = torch.empty(320, dtype=torch.uint8)
    indptr = torch.arange(groups + 1, dtype=torch.int32)
    indices = torch.zeros(groups * group_size * 513, dtype=torch.int32)
    seq_lens = torch.full((groups,), 2051, dtype=torch.int32)
    output = torch.empty_like(query)
    assert (
        qsa_ops.qsa_prims_ts_paged_attention(
            query,
            key_cache,
            value_cache,
            workspace,
            indptr,
            indices,
            seq_lens,
            8208,
            output,
            bmm1_scale=0.125,
            bmm2_scale=1.25,
        )
        is output
    )
    attention_kwargs = calls["attention_kwargs"]
    assert isinstance(attention_kwargs, dict)
    assert attention_kwargs["seq_len_q"] == group_size
    assert attention_kwargs["mask_type"] == "causal"
    assert attention_kwargs["bmm1_scale"] == 0.125
    assert attention_kwargs["bmm2_scale"] == 1.25


def _qsa_page4_paged_metadata_reference(
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    storage_page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output_device = logical_indices.device
    logical_indices = logical_indices.cpu()
    block_table = block_table.cpu()
    token_to_req = token_to_req.cpu()
    logical_positions = logical_positions.cpu()
    rows, output_width = logical_indices.shape
    token_topk = output_width - 3
    page_capacity = token_topk // 4 + 1
    indptr = torch.arange(rows + 1, dtype=torch.int32) * page_capacity
    locators = torch.full((rows, page_capacity), -1, dtype=torch.int32)
    seq_lens = torch.ones(rows, dtype=torch.int32)
    subpages_per_storage_page = storage_page_size // 4
    for row in range(rows):
        position = int(logical_positions[row].item())
        request = int(token_to_req[row].item())
        if position < 0 or request < 0 or request >= block_table.shape[0]:
            continue
        visible_tokens = position + 1
        complete_pages = min(visible_tokens // 4, token_topk // 4)
        tail_tokens = visible_tokens % 4
        seq_lens[row] = complete_pages * 4 + tail_tokens
        live_pages = complete_pages + int(tail_tokens > 0)
        for page_rank in range(live_pages):
            logical_token = int(logical_indices[row, 4 * page_rank].item())
            logical_storage_page, token_offset = divmod(
                logical_token, storage_page_size
            )
            physical_page = int(block_table[request, logical_storage_page].item())
            locators[row, page_rank] = (
                physical_page * subpages_per_storage_page + token_offset // 4
            )
    return tuple(
        tensor.to(output_device) for tensor in (indptr, locators.flatten(), seq_lens)
    )


@requires_qsa_triton
@pytest.mark.parametrize("storage_page_size", [16, 256])
def test_qsa_page4_metadata_matches_test_reference(
    storage_page_size: int,
) -> None:
    token_topk = 2048
    compress_ratio = 4
    block_topk = token_topk // compress_ratio
    logical_positions = torch.tensor(
        [0, 1, 2, 3, 6, 2048, 2049, 2050, 2051, 2052, -1],
        dtype=torch.int64,
        device="cuda",
    )
    token_to_req = torch.tensor(
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2],
        dtype=torch.int32,
        device="cuda",
    )
    sequence_lengths = torch.tensor([4096, 4096, 0], dtype=torch.int32, device="cuda")
    blocks = torch.arange(block_topk, dtype=torch.int32, device="cuda")
    blocks = blocks.unsqueeze(0).expand(logical_positions.numel(), -1).contiguous()
    logical_indices = qsa_ops.expand_qsa_block_indices_cuda(
        blocks,
        logical_positions,
        sequence_lengths,
        token_to_req,
        compress_ratio,
        token_topk,
    )
    table_width = math.ceil(4096 / storage_page_size)
    block_table = (
        torch.arange(3 * table_width, dtype=torch.int32, device="cuda")
        .reshape(3, table_width)
        .flip(1)
        .contiguous()
    )

    actual = qsa_ops.qsa_build_page4_paged_metadata(
        logical_indices,
        block_table,
        token_to_req,
        logical_positions,
        storage_page_size,
    )
    compact_actual = qsa_ops.qsa_build_page4_paged_metadata(
        blocks,
        block_table,
        token_to_req,
        logical_positions,
        storage_page_size,
        indices_are_blocks=True,
    )
    expected = _qsa_page4_paged_metadata_reference(
        logical_indices,
        block_table,
        token_to_req,
        logical_positions,
        storage_page_size,
    )

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor)
    for actual_tensor, expected_tensor in zip(compact_actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor)


@requires_qsa_triton
def test_qsa_grouped_page4_metadata_reinitializes_owned_bitsets() -> None:
    group_size = 4
    token_topk = 2048
    output_width = token_topk + 3
    storage_page_size = 16
    positions = torch.tensor(
        [12, 13, 14, 15, 20, 21, 22, 23], dtype=torch.int64, device="cuda"
    )
    requests = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32, device="cuda")
    selected_blocks = (
        (0, 2, 4, 6),
        (1, 2, 4, 6),
        (0, 3, 4, 6),
        (0, 2, 5, 6),
        (8, 9, 10, 11, 12, 13),
        (8, 9, 10, 11, 12, 14),
        (8, 9, 10, 11, 15, 14),
        (8, 9, 10, 16, 15, 14),
    )
    logical_indices = torch.full(
        (len(selected_blocks), output_width),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    for row, blocks in enumerate(selected_blocks):
        logical_indices[row, : 4 * len(blocks) : 4] = (
            torch.tensor(blocks, dtype=torch.int32, device="cuda") * 4
        )
    block_table = torch.arange(16, dtype=torch.int32, device="cuda").reshape(2, 8)
    workspace, indptr, locators, seq_lens = (
        qsa_ops.qsa_build_page4_grouped_paged_metadata(
            logical_indices,
            block_table,
            requests,
            positions,
            storage_page_size,
            group_size,
        )
    )

    workspace.fill_(0x55555555)
    indptr.fill_(-1)
    locators.fill_(-1)
    seq_lens.fill_(-1)
    qsa_ops.qsa_build_page4_grouped_paged_metadata(
        logical_indices,
        block_table,
        requests,
        positions,
        storage_page_size,
        group_size,
        bitset_workspace=workspace,
        paged_kv_indptr=indptr,
        paged_kv_indices=locators,
        seq_lens=seq_lens,
    )
    page_capacity = group_size * (token_topk // 4 + 1)
    assert indptr.tolist() == [0, page_capacity, 2 * page_capacity]
    for group in range(2):
        membership_by_block: dict[int, int] = {}
        for query_index in range(group_size):
            row = group * group_size + query_index
            for logical_block in selected_blocks[row]:
                membership_by_block[logical_block] = membership_by_block.get(
                    logical_block, 0
                ) | (1 << query_index)
        expected = []
        for logical_block, membership in sorted(membership_by_block.items()):
            logical_token = logical_block * 4
            storage_page = logical_token // storage_page_size
            subpage = logical_block % (storage_page_size // 4)
            physical_page = int(block_table[group, storage_page].item())
            locator = physical_page * (storage_page_size // 4) + subpage
            expected.append((locator << 4) | membership)
        begin = group * page_capacity
        torch.testing.assert_close(
            locators[begin : begin + len(expected)].cpu(),
            torch.tensor(expected, dtype=torch.int32),
        )
        assert int(seq_lens[group].item()) == 4 * len(expected)


@requires_qsa_triton
def test_qsa_grouped_compact_metadata_matches_real_expansion() -> None:
    group_size = 4
    token_topk = 2048
    compress_ratio = 4
    storage_page_size = 16
    positions = torch.tensor(
        [2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054],
        dtype=torch.int64,
        device="cuda",
    )
    requests = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32, device="cuda")
    sequence_lengths = torch.tensor([4096, 4096], dtype=torch.int32, device="cuda")
    base_blocks = torch.arange(token_topk // 4, dtype=torch.int32, device="cuda")
    compact_blocks = torch.stack(
        [torch.roll(base_blocks, row) for row in range(positions.numel())]
    )
    expanded_tokens = qsa_ops.expand_qsa_block_indices_cuda(
        compact_blocks,
        positions,
        sequence_lengths,
        requests,
        compress_ratio,
        token_topk,
    )
    table_width = 4096 // storage_page_size
    block_table = torch.arange(
        2 * table_width, dtype=torch.int32, device="cuda"
    ).reshape(2, table_width)

    expanded = qsa_ops.qsa_build_page4_grouped_paged_metadata(
        expanded_tokens,
        block_table,
        requests,
        positions,
        storage_page_size,
        group_size,
    )
    compact = qsa_ops.qsa_build_page4_grouped_paged_metadata(
        compact_blocks,
        block_table,
        requests,
        positions,
        storage_page_size,
        group_size,
        indices_are_blocks=True,
    )

    _, expanded_indptr, expanded_locators, expanded_seq_lens = expanded
    _, compact_indptr, compact_locators, compact_seq_lens = compact
    torch.testing.assert_close(compact_indptr, expanded_indptr)
    torch.testing.assert_close(compact_seq_lens, expanded_seq_lens)
    for group in range(positions.numel() // group_size):
        begin = int(expanded_indptr[group].item())
        live_pages = (int(expanded_seq_lens[group].item()) + 3) // 4
        torch.testing.assert_close(
            compact_locators[begin : begin + live_pages],
            expanded_locators[begin : begin + live_pages],
        )
