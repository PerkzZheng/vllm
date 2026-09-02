# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for the Qwen4Exp weight-free QSA path."""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache
from inspect import signature

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton

_LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024
_TOPK_WORKSPACE_BYTES = 1024 * 1024
_QSA_SEMANTIC_PAGE_SIZE = 4
_QSA_PAGE_MEMBERSHIP_BITS = 4
_QSAPrimsTSAPIs = tuple[
    Callable[..., int],
    Callable[..., int],
    Callable[..., object],
]


@triton.jit
def _qsa_mqa_paged_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_table_req,
    stride_table_page,
    stride_logits_row,
    num_rows,
    num_columns,
    num_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
    STAGES: tl.constexpr,
    MAX_N: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    heads = tl.arange(0, MAX_N)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + row)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    visible = tl.minimum(
        (query_position + 1) // COMPRESS_RATIO,
        sequence_length // COMPRESS_RATIO,
    )
    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + row, visible)
    tile_start = tl.program_id(1) * TILES_PER_PROG
    # Top-k is bounded by visible_blocks, so columns beyond it need no value.
    if tile_start * BLOCK_N >= visible:
        return
    tile_end = tl.minimum(tile_start + TILES_PER_PROG, tl.cdiv(visible, BLOCK_N))
    tile_end = tl.minimum(tile_end, tl.cdiv(num_columns, BLOCK_N))

    # Pad the small head axis to a tensor-core-compatible N dimension.
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + heads[None, :] * stride_q_head
        + dims[:, None] * stride_q_dim,
        mask=(heads[None, :] < NUM_HEADS) & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(tile_start, tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < visible
        logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr
            + safe_request * stride_table_req
            + logical_page * stride_table_page,
            mask=live,
            other=-1,
        )
        page_valid = live & (physical_page >= 0) & (physical_page < num_pages)
        # physical_page * block stride can overflow int32 for large caches.
        safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[:, None] * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=page_valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.where(heads[None, :] < NUM_HEADS, tl.maximum(scores, 0.0), 0.0)
        score = tl.sum(scores, axis=1) / score_divisor
        tl.store(
            logits_ptr + row * stride_logits_row + columns,
            tl.where(page_valid, score, -float("inf")),
            mask=live & (columns < num_columns),
        )


@triton.jit
def _expand_qsa_indices_kernel(
    block_indices_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    token_to_req_ptr,
    output_ptr,
    stride_blocks_row,
    stride_blocks_column,
    stride_output_row,
    stride_output_column,
    rows,
    num_requests,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TOKEN_TOPK: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    COLUMN_BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    columns = tl.program_id(1) * COLUMN_BLOCK + tl.arange(0, COLUMN_BLOCK)
    query_position = tl.load(query_positions_ptr + row)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    complete_blocks = tl.minimum(
        tl.minimum(
            (query_position + 1) // COMPRESS_RATIO,
            sequence_length // COMPRESS_RATIO,
        ),
        BLOCK_TOPK,
    )
    expanded_count = complete_blocks * COMPRESS_RATIO
    tail_start = ((query_position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_count = (query_position + 1) - tail_start

    is_expanded = columns < expanded_count
    block_rank = columns // COMPRESS_RATIO
    offset = columns % COMPRESS_RATIO
    safe_rank = tl.minimum(block_rank, BLOCK_TOPK - 1)
    block = tl.load(
        block_indices_ptr + row * stride_blocks_row + safe_rank * stride_blocks_column,
        mask=(row < rows) & is_expanded,
        other=-1,
    )
    expanded = block * COMPRESS_RATIO + offset
    tail_offset = columns - expanded_count
    is_tail = (
        (columns >= expanded_count)
        & (tail_offset < tail_count)
        & (tail_offset < COMPRESS_RATIO - 1)
    )
    token = tl.where(is_expanded, expanded, tail_start + tail_offset)
    valid = (
        (row < rows)
        & (columns < OUTPUT_WIDTH)
        & (is_expanded | is_tail)
        & (token >= 0)
        & (token < sequence_length)
    )
    tl.store(
        output_ptr + row * stride_output_row + columns * stride_output_column,
        tl.where(valid, token, -1),
        mask=(row < rows) & (columns < OUTPUT_WIDTH),
    )


@triton.jit
def _build_qsa_page4_paged_metadata_kernel(
    logical_indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    logical_positions_ptr,
    paged_kv_indptr_ptr,
    paged_kv_indices_ptr,
    seq_lens_ptr,
    stride_indices_row,
    stride_indices_column,
    stride_table_req,
    stride_table_page,
    rows,
    num_requests,
    TOKEN_TOPK: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    SEMANTIC_PAGE_SIZE: tl.constexpr,
    STORAGE_PAGE_SIZE: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
    INDICES_ARE_BLOCKS: tl.constexpr,
) -> None:
    """Convert one compact QSA token row into one fixed-capacity CSR row."""

    row = tl.program_id(0)
    page_ranks = tl.arange(0, BLOCK_PAGES)
    request = tl.load(token_to_req_ptr + row)
    logical_position = tl.load(logical_positions_ptr + row)
    active = (
        (row < rows)
        & (logical_position >= 0)
        & (request >= 0)
        & (request < num_requests)
    )
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    visible_tokens = tl.maximum(logical_position + 1, 0)
    complete_pages = tl.minimum(
        visible_tokens // SEMANTIC_PAGE_SIZE,
        TOKEN_TOPK // SEMANTIC_PAGE_SIZE,
    )
    tail_tokens = visible_tokens % SEMANTIC_PAGE_SIZE
    compact_length = complete_pages * SEMANTIC_PAGE_SIZE + tail_tokens
    live_pages = complete_pages + (tail_tokens > 0)

    # PrimTS can consume the indexer's compact block IDs directly and avoid
    # the otherwise separate 512 -> 2,051 token-expansion launch. Retain the
    # expanded-token mode for the native Triton attention path and diagnostics.
    page_live = active & (page_ranks < live_pages)
    if INDICES_ARE_BLOCKS:
        selected_page = page_ranks < complete_pages
        selected_block = tl.load(
            logical_indices_ptr
            + row * stride_indices_row
            + page_ranks * stride_indices_column,
            mask=active & selected_page,
            other=-1,
        )
        is_tail_page = (tail_tokens > 0) & (page_ranks == complete_pages)
        logical_token = tl.where(
            is_tail_page,
            (visible_tokens // SEMANTIC_PAGE_SIZE) * SEMANTIC_PAGE_SIZE,
            selected_block * SEMANTIC_PAGE_SIZE,
        )
    else:
        # Every four-token group begins at column 4 * page_rank. This includes
        # the optional causal tail appended after all selected complete groups.
        logical_token = tl.load(
            logical_indices_ptr
            + row * stride_indices_row
            + page_ranks * SEMANTIC_PAGE_SIZE * stride_indices_column,
            mask=page_live & (page_ranks < PAGE_CAPACITY),
            other=-1,
        )
    logical_storage_page = tl.maximum(logical_token, 0) // STORAGE_PAGE_SIZE
    table_entry_live = (
        page_live & (logical_token >= 0) & (logical_storage_page < PAGE_TABLE_WIDTH)
    )
    physical_page = tl.load(
        block_table_ptr
        + safe_request * stride_table_req
        + tl.minimum(logical_storage_page, PAGE_TABLE_WIDTH - 1) * stride_table_page,
        mask=table_entry_live,
        other=-1,
    )
    subpage = (tl.maximum(logical_token, 0) % STORAGE_PAGE_SIZE) // SEMANTIC_PAGE_SIZE
    subpages_per_storage_page: tl.constexpr = STORAGE_PAGE_SIZE // SEMANTIC_PAGE_SIZE
    locator = physical_page * subpages_per_storage_page + subpage
    locator_live = table_entry_live & (physical_page >= 0)

    # CUDA-graph padding rows have logical_position=-1. Their positive dummy
    # length preserves the decode scheduler contract while locator -1 selects
    # PrimTS's TMA out-of-bounds zero page, producing an inert output row.
    output_locator = tl.where(locator_live, locator, -1)
    tl.store(
        paged_kv_indices_ptr + row * PAGE_CAPACITY + page_ranks,
        output_locator,
        mask=(row < rows) & (page_ranks < PAGE_CAPACITY),
    )
    if row < rows:
        tl.store(paged_kv_indptr_ptr + row, row * PAGE_CAPACITY)
        tl.store(seq_lens_ptr + row, tl.maximum(compact_length, 1))
        if row == rows - 1:
            tl.store(paged_kv_indptr_ptr + rows, rows * PAGE_CAPACITY)


@triton.jit
def _qsa_popcount_u32(value):
    """Return a vectorized uint32 population count."""

    return tl.inline_asm_elementwise(
        asm="popc.b32 $0, $1;",
        constraints="=r,r",
        args=[value.to(tl.uint32)],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _build_qsa_grouped_page_bitsets_kernel(
    logical_indices_ptr,
    token_to_req_ptr,
    logical_positions_ptr,
    bitsets_ptr,
    stride_indices_row,
    stride_indices_column,
    rows,
    num_requests,
    TOKEN_TOPK: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SEMANTIC_PAGE_SIZE: tl.constexpr,
    BITSET_WORDS: tl.constexpr,
    BITSET_BLOCK: tl.constexpr,
    CLEAR_IN_CTA: tl.constexpr,
    INDICES_ARE_BLOCKS: tl.constexpr,
) -> None:
    """Materialize one logical-page bitmap for each query in a fused group."""

    group = tl.program_id(0)
    q_index = tl.program_id(1)
    row = group * GROUP_SIZE + q_index
    bitset_base = (group * GROUP_SIZE + q_index) * BITSET_WORDS
    offsets = tl.arange(0, BITSET_BLOCK)

    if CLEAR_IN_CTA:
        # For a small grouped grid, saving the separate fill launch is worth
        # the per-CTA clear and barrier. Large grids are cleared once by the
        # caller so every builder CTA can begin issuing loads immediately.
        tl.store(
            bitsets_ptr + bitset_base + offsets,
            0,
            mask=offsets < BITSET_WORDS,
        )
        tl.debug_barrier()

    request = tl.load(token_to_req_ptr + row, mask=row < rows, other=-1)
    logical_position = tl.load(
        logical_positions_ptr + row,
        mask=row < rows,
        other=-1,
    )
    active = (
        (row < rows)
        & (logical_position >= 0)
        & (request >= 0)
        & (request < num_requests)
    )
    visible_tokens = tl.maximum(logical_position + 1, 0)
    complete_pages = tl.minimum(
        visible_tokens // SEMANTIC_PAGE_SIZE,
        TOKEN_TOPK // SEMANTIC_PAGE_SIZE,
    )
    tail_tokens = visible_tokens % SEMANTIC_PAGE_SIZE
    page_ranks = offsets
    selected_index = tl.load(
        logical_indices_ptr
        + row * stride_indices_row
        + page_ranks
        * (1 if INDICES_ARE_BLOCKS else SEMANTIC_PAGE_SIZE)
        * stride_indices_column,
        mask=active & (page_ranks < complete_pages),
        other=-1,
    )
    logical_block = (
        selected_index if INDICES_ARE_BLOCKS else selected_index // SEMANTIC_PAGE_SIZE
    )
    word = logical_block // 32
    bit = logical_block % 32
    page_live = (
        active
        & (page_ranks < complete_pages)
        & (selected_index >= 0)
        & (word >= 0)
        & (word < BITSET_WORDS)
    )
    bit_value = (1 << bit).to(tl.int32)
    # This CTA exclusively owns its query bitmap. CTA-scope atomicity resolves
    # lane collisions; the following stream-ordered pack kernel provides the
    # inter-kernel visibility boundary.
    tl.atomic_or(
        bitsets_ptr + bitset_base + tl.maximum(word, 0),
        bit_value,
        mask=page_live,
        sem="relaxed",
        scope="cta",
    )

    # Keep the optional 1--3-token causal tail out of the vector block. Including
    # its 513th page would round this kernel from 512 to 1024 lanes even though
    # every other lane in the upper half is inactive.
    if INDICES_ARE_BLOCKS:
        tail_logical_block = visible_tokens // SEMANTIC_PAGE_SIZE
    else:
        tail_logical_token = tl.load(
            logical_indices_ptr
            + row * stride_indices_row
            + complete_pages * SEMANTIC_PAGE_SIZE * stride_indices_column,
            mask=active & (tail_tokens > 0),
            other=-1,
        )
        tail_logical_block = tail_logical_token // SEMANTIC_PAGE_SIZE
    tail_word = tail_logical_block // 32
    tail_bit = tail_logical_block % 32
    tail_live = (
        active
        & (tail_tokens > 0)
        & (tail_logical_block >= 0)
        & (tail_word >= 0)
        & (tail_word < BITSET_WORDS)
    )
    tl.atomic_or(
        bitsets_ptr + bitset_base + tl.maximum(tail_word, 0),
        (1 << tail_bit).to(tl.int32),
        mask=tail_live,
        sem="relaxed",
        scope="cta",
    )


@triton.jit
def _pack_qsa_grouped_page_union_kernel(
    bitsets_ptr,
    block_table_ptr,
    token_to_req_ptr,
    logical_positions_ptr,
    paged_kv_indptr_ptr,
    paged_kv_indices_ptr,
    seq_lens_ptr,
    stride_table_req,
    stride_table_page,
    groups,
    num_requests,
    GROUP_SIZE: tl.constexpr,
    BITSET_WORDS: tl.constexpr,
    BITSET_BLOCK: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    SEMANTIC_PAGE_SIZE: tl.constexpr,
    PAGE_MEMBERSHIP_BITS: tl.constexpr,
    STORAGE_PAGE_SIZE: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
) -> None:
    """Scan grouped bitsets into sorted packed locator/membership entries."""

    group = tl.program_id(0)
    words = tl.arange(0, BITSET_BLOCK)
    word_live = words < BITSET_WORDS
    bitset_base = group * GROUP_SIZE * BITSET_WORDS
    q_word_0 = tl.load(
        bitsets_ptr + bitset_base + words,
        mask=(group < groups) & word_live,
        other=0,
    ).to(tl.uint32)
    q_word_1 = tl.load(
        bitsets_ptr + bitset_base + BITSET_WORDS + words,
        mask=(group < groups) & word_live,
        other=0,
    ).to(tl.uint32)
    q_word_2 = tl.zeros((BITSET_BLOCK,), dtype=tl.uint32)
    q_word_3 = tl.zeros((BITSET_BLOCK,), dtype=tl.uint32)
    if GROUP_SIZE == 4:
        q_word_2 = tl.load(
            bitsets_ptr + bitset_base + 2 * BITSET_WORDS + words,
            mask=(group < groups) & word_live,
            other=0,
        ).to(tl.uint32)
        q_word_3 = tl.load(
            bitsets_ptr + bitset_base + 3 * BITSET_WORDS + words,
            mask=(group < groups) & word_live,
            other=0,
        ).to(tl.uint32)
    union_word = q_word_0 | q_word_1 | q_word_2 | q_word_3
    word_counts = _qsa_popcount_u32(union_word)
    word_offsets = tl.cumsum(word_counts, axis=0) - word_counts
    union_pages = tl.sum(word_counts, axis=0)

    first_row = group * GROUP_SIZE
    request = tl.load(token_to_req_ptr + first_row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    group_valid = (group < groups) & (request >= 0) & (request < num_requests)
    first_position = tl.load(logical_positions_ptr + first_row)
    last_position = tl.load(logical_positions_ptr + first_row + GROUP_SIZE - 1)
    group_valid &= last_position == first_position + GROUP_SIZE - 1
    for q_index in tl.static_range(1, GROUP_SIZE):
        q_request = tl.load(token_to_req_ptr + first_row + q_index)
        q_position = tl.load(logical_positions_ptr + first_row + q_index)
        group_valid &= (q_request == request) & (q_position == first_position + q_index)

    subpages_per_storage_page: tl.constexpr = STORAGE_PAGE_SIZE // SEMANTIC_PAGE_SIZE
    for bit_index in tl.static_range(0, 32):
        selected = word_live & ((union_word & (1 << bit_index)) != 0)
        rank = word_offsets + _qsa_popcount_u32(union_word & ((1 << bit_index) - 1))
        logical_block = words * 32 + bit_index
        logical_token = logical_block * SEMANTIC_PAGE_SIZE
        logical_storage_page = logical_token // STORAGE_PAGE_SIZE
        table_live = (
            group_valid
            & selected
            & (rank < PAGE_CAPACITY)
            & (logical_storage_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + tl.minimum(logical_storage_page, PAGE_TABLE_WIDTH - 1)
            * stride_table_page,
            mask=table_live,
            other=-1,
        )
        subpage = (logical_token % STORAGE_PAGE_SIZE) // SEMANTIC_PAGE_SIZE
        locator = physical_page * subpages_per_storage_page + subpage
        membership = (
            ((q_word_0 >> bit_index) & 1).to(tl.int32)
            | (((q_word_1 >> bit_index) & 1).to(tl.int32) << 1)
            | (((q_word_2 >> bit_index) & 1).to(tl.int32) << 2)
            | (((q_word_3 >> bit_index) & 1).to(tl.int32) << 3)
        )
        packed = (locator << PAGE_MEMBERSHIP_BITS) | membership
        tl.store(
            paged_kv_indices_ptr + group * PAGE_CAPACITY + rank,
            packed,
            mask=table_live & (physical_page >= 0),
        )

    if group < groups:
        tail_tokens = (last_position + 1) % SEMANTIC_PAGE_SIZE
        tail_padding = tl.where(
            tail_tokens == 0,
            0,
            SEMANTIC_PAGE_SIZE - tail_tokens,
        )
        seq_len = union_pages * SEMANTIC_PAGE_SIZE - tail_padding
        tl.store(paged_kv_indptr_ptr + group, group * PAGE_CAPACITY)
        tl.store(seq_lens_ptr + group, tl.where(group_valid, seq_len, 1))
        if group == groups - 1:
            tl.store(paged_kv_indptr_ptr + groups, groups * PAGE_CAPACITY)
        tl.store(
            paged_kv_indices_ptr + group * PAGE_CAPACITY,
            -1,
            mask=~group_valid,
        )


@triton.jit
def _qsa_sparse_paged_gqa_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    bmm1_scale,
    bmm2_scale,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2 = bmm1_scale * 1.4426950408889634

    # Dynamic bounds avoid padded main-loop iterations for uneven splits.
    split_tile_start = split_id * NUM_TILES // NUM_SPLITS
    split_tile_end = (split_id + 1) * NUM_TILES // NUM_SPLITS
    for tile in range(split_tile_start, split_tile_end):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns,
            mask=columns < TOPK,
            other=-1,
        )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (
            (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        # physical_page * block stride can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None],
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.dot(query, keys)
        # Scaling scores avoids re-quantizing a scaled query to BF16.
        scores *= softmax_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    normalized_output *= bmm2_scale
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        tl.store(
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        tl.store(
            partial_output_ptr
            + (
                (split_id * num_rows + row) * NUM_QUERY_HEADS
                + first_head
                + head_offsets[:, None]
            )
            * HEAD_DIM
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr
            + (split_id * num_rows + row) * NUM_QUERY_HEADS
            + first_head
            + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


@triton.jit
def _qsa_merge_splitk_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_output_row,
    stride_output_head,
    num_rows,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < NUM_SPLITS
    lse = tl.load(
        partial_lse_ptr + (split_offsets * num_rows + row) * NUM_QUERY_HEADS + head,
        mask=split_mask,
        other=-float("inf"),
    )
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    shifted = tl.where(split_mask & has_values, lse - lse_max, -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    partial_output = tl.load(
        partial_output_ptr
        + ((split_offsets[:, None] * num_rows + row) * NUM_QUERY_HEADS + head)
        * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged,
    )


@triton.jit
def _store_qsa_rows_kernel(
    cache_ptr,
    slots_ptr,
    rows_ptr,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_rows_row,
    stride_rows_dim,
    num_rows,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    slot = tl.load(slots_ptr + row)
    valid = (row < num_rows) & (slot >= 0) & (slot < num_blocks * PAGE_SIZE)
    block = tl.maximum(slot, 0) // PAGE_SIZE
    token = tl.maximum(slot, 0) % PAGE_SIZE
    values = tl.load(
        rows_ptr + row * stride_rows_row + dims * stride_rows_dim,
        mask=valid & (dims < WIDTH),
        other=0,
    )
    tl.store(
        cache_ptr
        + block * stride_cache_block
        + token * stride_cache_token
        + dims * stride_cache_dim,
        values,
        mask=valid & (dims < WIDTH),
    )


@triton.jit
def _compress_qsa_groups_kernel(
    raw_keys_ptr,  # this step's raw key rows, straight from activations
    raw_positions_ptr,  # this step's per-token positions
    compressor_state_cache_ptr,  # per-request ring of previous raw keys
    rope_cache_ptr,  # packed RoPE position tail of the ring
    compressor_state_table_ptr,
    token_to_req_ptr,
    query_start_loc_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    pooled_ptr,
    first_positions_ptr,
    stride_raw_row,
    stride_raw_dim,
    stride_raw_positions_row,
    stride_raw_positions_dim,
    stride_compressor_state_block,
    stride_compressor_state_token,
    stride_compressor_state_dim,
    stride_rope_block,
    stride_rope_token,
    stride_rope_dim,
    stride_compressor_state_table_req,
    stride_pooled_row,
    stride_pooled_dim,
    stride_positions_row,
    stride_positions_dim,
    num_rows,
    num_compressor_state_blocks,
    num_requests,
    COMPRESSOR_STATE_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOAD_ROPE_POSITIONS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    end_position = tl.load(logical_positions_ptr + row)
    compressed_slot = tl.load(compressed_slots_ptr + row)
    valid_request = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_row_start = tl.load(
        query_start_loc_ptr + safe_request, mask=valid_request, other=0
    )
    query_row_end = tl.load(
        query_start_loc_ptr + safe_request + 1, mask=valid_request, other=0
    )
    chunk_start_position = end_position - (row - query_row_start)
    compressor_state_block = tl.load(
        compressor_state_table_ptr + safe_request * stride_compressor_state_table_req,
        mask=valid_request,
        other=-1,
    )
    valid_compressor_state_block = (compressor_state_block >= 0) & (
        compressor_state_block < num_compressor_state_blocks
    )
    valid_row = (
        (row < num_rows)
        & valid_request
        & (row >= query_row_start)
        & (row < query_row_end)
        & (end_position >= COMPRESS_RATIO - 1)
        & (compressed_slot >= 0)
    )
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # A group can span the compressor-state ring (older members) and this
    # step's raw rows (members at positions >= chunk_start_position).
    for group_offset in tl.range(0, COMPRESS_RATIO):
        position = end_position - (COMPRESS_RATIO - 1 - group_offset)
        use_raw = position >= chunk_start_position
        raw_row = query_row_start + position - chunk_start_position
        raw_values = tl.load(
            raw_keys_ptr + raw_row * stride_raw_row + dims * stride_raw_dim,
            mask=valid_row
            & use_raw
            & (raw_row >= query_row_start)
            & (raw_row < query_row_end)
            & (raw_row < num_rows)
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        compressor_state_values = tl.load(
            compressor_state_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64)
            * stride_compressor_state_block
            + (position % COMPRESSOR_STATE_SIZE) * stride_compressor_state_token
            + dims * stride_compressor_state_dim,
            mask=valid_row
            & ~use_raw
            & valid_compressor_state_block
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.where(use_raw, raw_values, compressor_state_values)

    tl.store(
        pooled_ptr + row * stride_pooled_row + dims * stride_pooled_dim,
        accumulator / COMPRESS_RATIO,
        mask=(row < num_rows) & (dims < HEAD_DIM),
    )

    position_dims = tl.arange(0, 4)
    first_position = end_position - COMPRESS_RATIO + 1
    if LOAD_ROPE_POSITIONS:
        first_from_raw = first_position >= chunk_start_position
        raw_first_row = query_row_start + first_position - chunk_start_position
        raw_position_values = tl.load(
            raw_positions_ptr
            + raw_first_row * stride_raw_positions_row
            + position_dims * stride_raw_positions_dim,
            mask=valid_row
            & first_from_raw
            & (raw_first_row >= query_row_start)
            & (raw_first_row < query_row_end)
            & (raw_first_row < num_rows)
            & (position_dims < 3),
            other=0,
        )
        compressor_state_position_values = tl.load(
            rope_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64) * stride_rope_block
            + (first_position % COMPRESSOR_STATE_SIZE) * stride_rope_token
            + position_dims * stride_rope_dim,
            mask=valid_row
            & ~first_from_raw
            & valid_compressor_state_block
            & (position_dims < 3),
            other=0,
        )
        position_values = tl.where(
            first_from_raw,
            raw_position_values,
            compressor_state_position_values,
        )
    else:
        position_values = tl.where(valid_row, first_position, 0)
    tl.store(
        first_positions_ptr
        + row * stride_positions_row
        + position_dims * stride_positions_dim,
        position_values,
        mask=(row < num_rows) & (position_dims < 3),
    )


def _validate_mqa(q: torch.Tensor) -> None:
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("QSA query must be [rows, heads, head_dim]")


def qsa_mqa_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    num_columns: int | None = None,
    score_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute QSA scores directly from a paged compressed-key cache."""

    _validate_mqa(q)
    if not q.is_cuda or not HAS_TRITON:
        raise RuntimeError("paged QSA scoring requires CUDA and Triton")
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, head_dim]")
    if k_cache.shape[3] != q.shape[2]:
        raise ValueError("QSA query and cache dimensions must match")
    if page_table.ndim != 2:
        raise ValueError("QSA page table must be two-dimensional")
    if q.shape[0] and (not all(k_cache.shape[:2]) or not all(page_table.shape)):
        raise ValueError("QSA paged scoring cache and page table must be nonempty")
    if token_to_req.shape != (q.shape[0],):
        raise ValueError("QSA request mapping must match query rows")
    if query_positions.shape != (q.shape[0],):
        raise ValueError("QSA query positions must match query rows")
    if sequence_lengths.shape != (page_table.shape[0],):
        raise ValueError("QSA sequence lengths must match page-table requests")
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    score_divisor = math.sqrt(q.shape[2]) if score_scale is None else score_scale
    if score_divisor <= 0:
        raise ValueError("QSA score scale must be positive")

    capacity = page_table.shape[1] * k_cache.shape[1]
    columns = capacity if num_columns is None else num_columns
    if columns < 0:
        raise ValueError("QSA score width must be non-negative")
    logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    visible_blocks = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    if not q.shape[0] or not columns:
        return logits, visible_blocks
    BLOCK_N = 64
    BLOCK_D = max(16, triton.next_power_of_2(q.shape[2]))
    MAX_N = max(16, triton.next_power_of_2(q.shape[1]))
    # Tuned on GB300: larger row batches provide enough parallelism to reuse Q.
    tiles_per_program = 1 if q.shape[0] <= 32 else 8
    _qsa_mqa_paged_kernel[
        (q.shape[0], triton.cdiv(columns, BLOCK_N * tiles_per_program))
    ](
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(3),
        page_table.stride(0),
        page_table.stride(1),
        logits.stride(0),
        q.shape[0],
        columns,
        k_cache.shape[0],
        page_table.shape[0],
        float(score_divisor),
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        TILES_PER_PROG=tiles_per_program,
        STAGES=2,
        MAX_N=MAX_N,
        COMPRESS_RATIO=compress_ratio,
        num_warps=2,
    )
    return logits, visible_blocks


def expand_qsa_block_indices_cuda(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_req: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand compressed blocks and compact the causal tail of the open group."""

    if not block_indices.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA CUDA expansion requires Triton")
    if token_topk % compress_ratio:
        raise ValueError("QSA token top-k must be divisible by compression ratio")
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    if block_indices.shape != (query_positions.numel(), block_topk):
        raise ValueError("QSA compressed top-k has an invalid shape")
    if token_to_req.shape != query_positions.shape:
        raise ValueError("QSA request mapping must match query positions")
    if sequence_lengths.ndim != 1 or not sequence_lengths.shape[0]:
        raise ValueError("QSA request sequence lengths must be nonempty")
    if out is None:
        out = torch.empty(
            (block_indices.shape[0], output_width),
            dtype=torch.int32,
            device=block_indices.device,
        )
    elif out.shape != (block_indices.shape[0], output_width):
        raise ValueError("QSA expansion output has an invalid shape")
    if not block_indices.shape[0]:
        return out
    column_block = 256
    _expand_qsa_indices_kernel[
        (block_indices.shape[0], triton.cdiv(output_width, column_block))
    ](
        block_indices,
        query_positions,
        sequence_lengths,
        token_to_req,
        out,
        block_indices.stride(0),
        block_indices.stride(1),
        out.stride(0),
        out.stride(1),
        block_indices.shape[0],
        sequence_lengths.shape[0],
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress_ratio,
        TOKEN_TOPK=token_topk,
        OUTPUT_WIDTH=output_width,
        COLUMN_BLOCK=column_block,
        num_warps=4,
    )
    return out


def qsa_build_page4_paged_metadata(
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    storage_page_size: int,
    *,
    indices_are_blocks: bool = False,
    paged_kv_indptr: torch.Tensor | None = None,
    paged_kv_indices: torch.Tensor | None = None,
    seq_lens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build native page-4 CSR metadata for flattened SQ=1 QSA rows.

    By default, ``logical_indices`` keeps the existing expanded-token indexer
    interface: each selected four-token group is consecutive and the optional
    causal tail follows the complete groups. With ``indices_are_blocks=True``,
    each input entry is one four-token logical block and the kernel derives the
    tail from ``logical_positions``. This compact mode avoids a separate token
    expansion while producing the same native PrimTS CSR interface.

    Every query token becomes an independent CSR row with capacity
    ``token_topk / 4 + 1``. The returned indices tensor is flattened, and the
    row offsets therefore advance by that fixed capacity. Runtime ``seq_lens``
    select only the live prefix, including an optional tail page.
    """

    if not logical_indices.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA page-4 metadata requires CUDA and Triton")
    if logical_indices.ndim != 2 or logical_indices.dtype != torch.int32:
        raise ValueError("QSA logical indices must be a rank-two int32 tensor")
    rows, input_width = logical_indices.shape
    token_topk = (
        input_width * _QSA_SEMANTIC_PAGE_SIZE
        if indices_are_blocks
        else input_width - (_QSA_SEMANTIC_PAGE_SIZE - 1)
    )
    if token_topk <= 0 or token_topk % _QSA_SEMANTIC_PAGE_SIZE:
        raise ValueError(
            "QSA index width must encode a positive token_topk divisible by four"
        )
    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        raise ValueError("QSA block table must be a rank-two int32 tensor")
    if not all(block_table.shape):
        raise ValueError("QSA block table must be nonempty")
    if token_to_req.shape != (rows,) or token_to_req.dtype != torch.int32:
        raise ValueError("QSA request mapping must be int32 with one entry per row")
    if logical_positions.shape != (rows,) or logical_positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("QSA logical positions must be int32/int64 per-row values")
    if (
        not isinstance(storage_page_size, int)
        or isinstance(storage_page_size, bool)
        or storage_page_size < _QSA_SEMANTIC_PAGE_SIZE
        or storage_page_size % _QSA_SEMANTIC_PAGE_SIZE
    ):
        raise ValueError("QSA storage page size must be a positive multiple of four")
    tensors = (block_table, token_to_req, logical_positions)
    if any(tensor.device != logical_indices.device for tensor in tensors):
        raise ValueError("QSA page-4 metadata inputs must share one CUDA device")
    if logical_indices.stride(1) != 1 or block_table.stride(1) != 1:
        raise ValueError("QSA indices and block-table rows must be contiguous")
    if token_to_req.stride(0) != 1 or logical_positions.stride(0) != 1:
        raise ValueError("QSA per-row metadata must be contiguous")

    page_capacity = token_topk // _QSA_SEMANTIC_PAGE_SIZE + 1
    expected_shapes = (
        (rows + 1,),
        (rows * page_capacity,),
        (rows,),
    )
    outputs = [paged_kv_indptr, paged_kv_indices, seq_lens]
    for output_index, (output, shape) in enumerate(zip(outputs, expected_shapes)):
        if output is None:
            outputs[output_index] = torch.empty(
                shape, dtype=torch.int32, device=logical_indices.device
            )
        elif (
            output.shape != shape
            or output.dtype != torch.int32
            or output.device != logical_indices.device
            or not output.is_contiguous()
        ):
            raise ValueError(
                "QSA page-4 output buffers must be contiguous int32 tensors "
                f"with shapes {expected_shapes}"
            )
    paged_kv_indptr, paged_kv_indices, seq_lens = outputs
    assert paged_kv_indptr is not None
    assert paged_kv_indices is not None
    assert seq_lens is not None
    if not rows:
        paged_kv_indptr.zero_()
        return paged_kv_indptr, paged_kv_indices, seq_lens

    _build_qsa_page4_paged_metadata_kernel[(rows,)](
        logical_indices,
        block_table,
        token_to_req,
        logical_positions,
        paged_kv_indptr,
        paged_kv_indices,
        seq_lens,
        logical_indices.stride(0),
        logical_indices.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        rows,
        block_table.shape[0],
        TOKEN_TOPK=token_topk,
        PAGE_TABLE_WIDTH=block_table.shape[1],
        SEMANTIC_PAGE_SIZE=_QSA_SEMANTIC_PAGE_SIZE,
        STORAGE_PAGE_SIZE=storage_page_size,
        PAGE_CAPACITY=page_capacity,
        BLOCK_PAGES=triton.next_power_of_2(page_capacity),
        INDICES_ARE_BLOCKS=indices_are_blocks,
        num_warps=4,
    )
    return paged_kv_indptr, paged_kv_indices, seq_lens


def qsa_build_page4_grouped_paged_metadata(
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    storage_page_size: int,
    group_size: int,
    *,
    indices_are_blocks: bool = False,
    bitset_workspace: torch.Tensor | None = None,
    paged_kv_indptr: torch.Tensor | None = None,
    paged_kv_indices: torch.Tensor | None = None,
    seq_lens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build Q2/Q4 union CSR with packed per-query page membership.

    The model-facing ``logical_indices`` tensor is unchanged. An internal
    per-query logical-page bitmap forms each consecutive same-request union.
    Every output CSR word uses ``(locator << 4) | membership``; PrimTS strips
    the low nibble for TMA and applies it to score rows before softmax. The
    final partial page remains governed by the grouped causal mask.
    """

    if group_size not in (2, 4):
        raise ValueError("QSA grouped page metadata supports group size two or four")
    if not logical_indices.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA grouped page metadata requires CUDA and Triton")
    if logical_indices.ndim != 2 or logical_indices.dtype != torch.int32:
        raise ValueError("QSA logical indices must be a rank-two int32 tensor")
    rows, input_width = logical_indices.shape
    if rows % group_size:
        raise ValueError("QSA grouped page metadata requires complete query groups")
    token_topk = (
        input_width * _QSA_SEMANTIC_PAGE_SIZE
        if indices_are_blocks
        else input_width - (_QSA_SEMANTIC_PAGE_SIZE - 1)
    )
    if token_topk <= 0 or token_topk % _QSA_SEMANTIC_PAGE_SIZE:
        raise ValueError(
            "QSA index width must encode a positive token_topk divisible by four"
        )
    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        raise ValueError("QSA block table must be a rank-two int32 tensor")
    if not all(block_table.shape):
        raise ValueError("QSA block table must be nonempty")
    if token_to_req.shape != (rows,) or token_to_req.dtype != torch.int32:
        raise ValueError("QSA request mapping must be int32 with one entry per row")
    if logical_positions.shape != (rows,) or logical_positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("QSA logical positions must be int32/int64 per-row values")
    if (
        not isinstance(storage_page_size, int)
        or isinstance(storage_page_size, bool)
        or storage_page_size < _QSA_SEMANTIC_PAGE_SIZE
        or storage_page_size % _QSA_SEMANTIC_PAGE_SIZE
    ):
        raise ValueError("QSA storage page size must be a positive multiple of four")
    tensors = (block_table, token_to_req, logical_positions)
    if any(tensor.device != logical_indices.device for tensor in tensors):
        raise ValueError("QSA grouped page metadata inputs must share one CUDA device")
    if logical_indices.stride(1) != 1 or block_table.stride(1) != 1:
        raise ValueError("QSA indices and block-table rows must be contiguous")

    groups = rows // group_size
    page_capacity = token_topk // _QSA_SEMANTIC_PAGE_SIZE + 1
    union_page_capacity = group_size * page_capacity
    logical_block_capacity = (
        block_table.shape[1] * storage_page_size // _QSA_SEMANTIC_PAGE_SIZE
    )
    bitset_words = (logical_block_capacity + 31) // 32
    expected_bitset_shape = (groups * group_size * bitset_words,)
    expected_output_shapes = (
        (groups + 1,),
        (groups * union_page_capacity,),
        (groups,),
    )
    if bitset_workspace is None:
        bitset_workspace = torch.empty(
            expected_bitset_shape,
            dtype=torch.int32,
            device=logical_indices.device,
        )
    elif (
        bitset_workspace.shape != expected_bitset_shape
        or bitset_workspace.dtype != torch.int32
        or bitset_workspace.device != logical_indices.device
        or not bitset_workspace.is_contiguous()
    ):
        raise ValueError(
            "QSA grouped bitset workspace must be contiguous int32 with shape "
            f"{expected_bitset_shape}"
        )

    outputs = [paged_kv_indptr, paged_kv_indices, seq_lens]
    for output_index, (output, shape) in enumerate(
        zip(outputs, expected_output_shapes)
    ):
        if output is None:
            outputs[output_index] = torch.empty(
                shape,
                dtype=torch.int32,
                device=logical_indices.device,
            )
        elif (
            output.shape != shape
            or output.dtype != torch.int32
            or output.device != logical_indices.device
            or not output.is_contiguous()
        ):
            raise ValueError(
                "QSA grouped page outputs must be contiguous int32 tensors "
                f"with shapes {expected_output_shapes}"
            )
    paged_kv_indptr, paged_kv_indices, seq_lens = outputs
    assert paged_kv_indptr is not None
    assert paged_kv_indices is not None
    assert seq_lens is not None
    if not groups:
        paged_kv_indptr.zero_()
        return bitset_workspace, paged_kv_indptr, paged_kv_indices, seq_lens

    bitset_block = triton.next_power_of_2(token_topk // _QSA_SEMANTIC_PAGE_SIZE)
    # A CTA can clear only the bitmap words covered by its selected-page
    # lanes. Longer-context bitmaps and large grids use one stream-ordered
    # device fill, which also avoids repeating the clear/barrier per query.
    clear_in_cta = rows < 4096 and bitset_words <= bitset_block
    if not clear_in_cta:
        bitset_workspace.zero_()
    _build_qsa_grouped_page_bitsets_kernel[(groups, group_size)](
        logical_indices,
        token_to_req,
        logical_positions,
        bitset_workspace,
        logical_indices.stride(0),
        logical_indices.stride(1),
        rows,
        block_table.shape[0],
        TOKEN_TOPK=token_topk,
        GROUP_SIZE=group_size,
        SEMANTIC_PAGE_SIZE=_QSA_SEMANTIC_PAGE_SIZE,
        BITSET_WORDS=bitset_words,
        BITSET_BLOCK=bitset_block,
        CLEAR_IN_CTA=clear_in_cta,
        INDICES_ARE_BLOCKS=indices_are_blocks,
        num_warps=4,
    )
    _pack_qsa_grouped_page_union_kernel[(groups,)](
        bitset_workspace,
        block_table,
        token_to_req,
        logical_positions,
        paged_kv_indptr,
        paged_kv_indices,
        seq_lens,
        block_table.stride(0),
        block_table.stride(1),
        groups,
        block_table.shape[0],
        GROUP_SIZE=group_size,
        BITSET_WORDS=bitset_words,
        BITSET_BLOCK=triton.next_power_of_2(bitset_words),
        PAGE_TABLE_WIDTH=block_table.shape[1],
        SEMANTIC_PAGE_SIZE=_QSA_SEMANTIC_PAGE_SIZE,
        PAGE_MEMBERSHIP_BITS=_QSA_PAGE_MEMBERSHIP_BITS,
        STORAGE_PAGE_SIZE=storage_page_size,
        PAGE_CAPACITY=union_page_capacity,
        num_warps=4,
    )
    return bitset_workspace, paged_kv_indptr, paged_kv_indices, seq_lens


@lru_cache
def _qsa_prims_ts_apis() -> _QSAPrimsTSAPIs | None:
    """Resolve the page-4 capable FlashInfer API without a hard dependency."""

    try:
        from flashinfer.decode import (
            get_prims_ts_qsa_group_size,
            get_prims_ts_qsa_workspace_size,
            prepare_prims_ts_qsa_attention,
        )
    except (AttributeError, ImportError):
        return None
    try:
        workspace_parameters = signature(
            get_prims_ts_qsa_workspace_size
        ).parameters
    except (TypeError, ValueError):
        return None
    if "block_topk" not in workspace_parameters:
        return None
    return (
        get_prims_ts_qsa_group_size,
        get_prims_ts_qsa_workspace_size,
        prepare_prims_ts_qsa_attention,
    )


def has_qsa_prims_ts_attention() -> bool:
    """Return whether FlashInfer exposes encoded page-4 PrimTS decode."""

    return _qsa_prims_ts_apis() is not None


def qsa_prims_ts_combined_workspace_size(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    block_topk: int,
    *,
    out_dtype: torch.dtype | None = None,
) -> int:
    """Return byte workspace required for a causal Q1/Q2/Q4 launch."""

    apis = _qsa_prims_ts_apis()
    if apis is None:
        raise RuntimeError("FlashInfer does not provide page-4 PrimTS attention")
    _, get_workspace_size, _ = apis
    return int(
        get_workspace_size(
            q,
            k_cache,
            block_table,
            block_topk=block_topk,
            out_dtype=out_dtype,
        )
    )


def qsa_prims_ts_workspace_size(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    max_seq_len: int,
    *,
    out_dtype: torch.dtype | None = None,
) -> int:
    """Return raw attention workspace bytes for diagnostic callers."""

    from flashinfer.decode import get_prims_ts_batch_decode_workspace_size

    if q.ndim == 3:
        batch_size, num_qo_heads, head_dim = q.shape
        seq_len_q = 1
    elif q.ndim == 4 and q.shape[1] in (2, 4):
        batch_size, seq_len_q, num_qo_heads, head_dim = q.shape
    else:
        raise ValueError("QSA PrimTS expects Q [R,Hq,D] or grouped Q [B,2|4,Hq,D]")
    if out_dtype is None:
        out_dtype = q.dtype
    return int(
        get_prims_ts_batch_decode_workspace_size(
            batch_size=batch_size,
            num_qo_heads=num_qo_heads,
            num_kv_heads=k_cache.shape[1],
            head_dim=head_dim,
            page_size=_QSA_SEMANTIC_PAGE_SIZE,
            storage_page_size=k_cache.shape[2],
            max_seq_len=max_seq_len,
            seq_len_q=seq_len_q,
            q_dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            out_dtype=out_dtype,
            mask_type="causal",
            device=q.device,
        )
    )


def qsa_prims_ts_group_size(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    query_start_loc_cpu: torch.Tensor | None,
) -> int:
    """Ask FlashInfer how many adjacent Q rows each metadata route should own."""

    if q.ndim != 3 or k_cache.ndim != 4:
        raise ValueError("QSA grouping policy expects Q [R,Hq,D] and K [P,Hkv,N,D]")
    if not q.is_cuda or q.device != k_cache.device:
        raise ValueError("QSA grouping policy expects Q and K on one CUDA device")
    if q.dtype != k_cache.dtype or q.dtype not in (
        torch.bfloat16,
        torch.float8_e4m3fn,
    ):
        raise ValueError("QSA grouping policy requires matching BF16 or FP8 Q/K")
    if q.shape[2] != k_cache.shape[3]:
        raise ValueError("QSA grouping policy requires matching head dimensions")

    apis = _qsa_prims_ts_apis()
    if apis is None:
        raise RuntimeError("FlashInfer does not provide page-4 PrimTS attention")
    get_group_size, _, _ = apis
    return int(
        get_group_size(
            query_start_loc_cpu,
            q.shape[0],
            q.shape[1],
            k_cache.shape[1],
            device=q.device,
        )
    )


def qsa_prims_ts_metadata_workspace_size(
    num_query_tokens: int,
    block_table: torch.Tensor,
    storage_page_size: int,
    group_size: int,
) -> int:
    """Return metadata-only workspace bytes for diagnostic callers."""

    from flashinfer.decode import get_prims_ts_qsa_metadata_workspace_size

    return int(
        get_prims_ts_qsa_metadata_workspace_size(
            num_query_tokens,
            block_table.shape[1],
            storage_page_size,
            group_size,
        )
    )


def qsa_prims_ts_build_page4_metadata(
    block_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    storage_page_size: int,
    group_size: int,
    workspace_buffer: torch.Tensor | None,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    enable_pdl: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build metadata separately for diagnostics and component benchmarks."""

    from flashinfer.decode import build_prims_ts_qsa_page4_metadata

    return build_prims_ts_qsa_page4_metadata(
        block_indices,
        block_table,
        token_to_req,
        logical_positions,
        workspace_buffer,
        group_size=group_size,
        storage_page_size=storage_page_size,
        out=(paged_kv_indptr, paged_kv_indices, seq_lens),
        enable_pdl=enable_pdl,
    )


def qsa_prims_ts_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    out: torch.Tensor,
    *,
    bmm1_scale: float | None = None,
    bmm2_scale: float = 1.0,
) -> torch.Tensor:
    """Run raw PrimTS page-4 attention for component benchmarks."""

    from flashinfer.decode import prims_ts_batch_decode_with_kv_cache

    seq_len_q = 1 if q.ndim == 3 else q.shape[1]
    return prims_ts_batch_decode_with_kv_cache(
        q,
        (k_cache, v_cache),
        workspace_buffer,
        paged_kv_indptr,
        paged_kv_indices,
        seq_lens,
        max_seq_len,
        seq_len_q=seq_len_q,
        bmm1_scale=q.shape[-1] ** -0.5 if bmm1_scale is None else bmm1_scale,
        bmm2_scale=bmm2_scale,
        out=out,
        out_dtype=out.dtype,
        mask_type="causal",
        page_size=_QSA_SEMANTIC_PAGE_SIZE,
    )


def qsa_prims_ts_prepare_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    out: torch.Tensor,
) -> object:
    """Prepare raw attention for component benchmarks."""

    from flashinfer.decode import prepare_prims_ts_batch_decode_with_kv_cache

    return prepare_prims_ts_batch_decode_with_kv_cache(
        q,
        (k_cache, v_cache),
        workspace_buffer,
        paged_kv_indptr,
        paged_kv_indices,
        seq_lens,
        max_seq_len,
        out=out,
        seq_len_q=1 if q.ndim == 3 else q.shape[1],
        out_dtype=out.dtype,
        mask_type="causal",
        page_size=_QSA_SEMANTIC_PAGE_SIZE,
    )


def qsa_prims_ts_run_prepared_attention(
    plan: object,
    q: torch.Tensor,
    out: torch.Tensor,
    *,
    bmm1_scale: float,
    bmm2_scale: float,
) -> torch.Tensor:
    """Run a raw attention-only plan for component benchmarks."""

    return getattr(plan, "run")(
        q,
        out=out,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
    )


def qsa_prims_ts_prepare_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace_buffer: torch.Tensor,
    out: torch.Tensor,
    *,
    bmm1_scale: float | None = None,
    bmm2_scale: float = 1.0,
) -> object:
    """Prepare metadata and attention as one framework-owned QSA plan."""

    apis = _qsa_prims_ts_apis()
    if apis is None:
        raise RuntimeError("FlashInfer does not provide page-4 PrimTS attention")
    _, _, prepare_attention = apis
    return prepare_attention(
        q,
        (k_cache, v_cache),
        block_indices,
        block_table,
        token_to_req,
        logical_positions,
        paged_kv_indptr,
        seq_lens,
        workspace_buffer,
        out=out,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
    )


def qsa_prims_ts_run_prepared(
    plan: object,
    q: torch.Tensor,
    block_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Launch an unchecked framework-owned metadata-plus-attention plan."""

    return getattr(plan, "run")(
        q,
        block_indices,
        block_table,
        token_to_req,
        logical_positions,
        out=out,
    )


def qsa_select_paged_tokens(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
    *,
    expand_blocks: bool = True,
) -> torch.Tensor:
    """Score and select QSA blocks, optionally expanding them to token IDs."""

    rows = q.shape[0]
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1 if expand_blocks else block_topk
    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=q.device)
    if out.shape != (rows, output_width):
        raise ValueError("QSA selection output has an invalid shape")
    if not rows:
        return out

    columns = page_table.shape[1] * k_cache.shape[1]
    rows_per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1))
    chunk_rows = min(rows, rows_per_chunk)
    blocks_buffer = (
        torch.empty((chunk_rows, block_topk), dtype=torch.int32, device=q.device)
        if expand_blocks
        else None
    )
    topk_workspace = torch.empty(
        (_TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device=q.device
    )
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        row_slice = slice(row_start, row_end)
        logits, visible_blocks = qsa_mqa_paged(
            q[row_slice],
            k_cache,
            page_table,
            token_to_req[row_slice],
            query_positions[row_slice],
            sequence_lengths,
            compress_ratio,
        )
        blocks = (
            blocks_buffer[: row_end - row_start]
            if blocks_buffer is not None
            else out[row_slice]
        )
        use_cooperative_topk = (
            blocks.shape[0] <= 32
            and logits.stride(0) % 4 == 0
            and current_platform.has_device_capability(90)
            and not current_platform.is_device_capability_family(120)
        )
        topk_op = (
            torch.ops._C.cooperative_topk
            if use_cooperative_topk
            else torch.ops._C.persistent_topk
        )
        topk_op(logits, visible_blocks, blocks, topk_workspace, block_topk, columns)
        if expand_blocks:
            expand_qsa_block_indices_cuda(
                blocks,
                query_positions[row_slice],
                sequence_lengths,
                token_to_req[row_slice],
                compress_ratio,
                token_topk,
                out[row_slice],
            )
    return out


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    bmm1_scale: float | None = None,
    bmm2_scale: float = 1.0,
) -> torch.Tensor:
    """Run sparse GQA directly over paged BF16 or FP8-E4M3 K/V caches."""

    if not q.is_cuda or not HAS_TRITON:
        raise RuntimeError("paged QSA sparse attention requires CUDA and Triton")
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA sparse attention received invalid Q/K/V shapes")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0]:
        raise ValueError("QSA indices must have one row per query")
    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA sparse attention metadata has invalid shapes")
    if not all(k_cache.shape[:3]) or not all(block_table.shape):
        raise ValueError("QSA sparse attention cache and block table must be nonempty")
    if logical_indices.shape[1] <= 0:
        raise ValueError("QSA sparse attention requires a positive selection width")
    if q.shape[2] != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
        raise ValueError("QSA sparse attention requires valid grouped-query heads")
    head_dim = q.shape[2]
    assert head_dim >= 16 and (head_dim & (head_dim - 1)) == 0
    if q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
        raise ValueError("QSA sparse attention requires matching Q/K/V dtypes")
    if q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("QSA sparse attention supports BF16 and FP8-E4M3 Q/K/V")
    assert logical_indices.dtype == block_table.dtype == torch.int32
    assert token_to_req.dtype == torch.int32
    assert q.device == k_cache.device == v_cache.device
    assert q.device == logical_indices.device == block_table.device
    assert q.device == token_to_req.device
    assert q.stride(2) == k_cache.stride(3) == v_cache.stride(3) == 1
    assert logical_indices.stride(1) == block_table.stride(1) == 1
    assert token_to_req.stride(0) == 1
    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape:
        raise ValueError("QSA sparse output must match its query")
    output_dtype_supported = out.dtype == q.dtype or (
        q.dtype == torch.float8_e4m3fn and out.dtype in (torch.float16, torch.bfloat16)
    )
    if not output_dtype_supported or out.device != q.device:
        raise ValueError(
            "QSA sparse output must use the query dtype, or FP16/BF16 for FP8 "
            "inputs, and reside on the query device"
        )
    assert out.stride(2) == 1
    if not q.shape[0]:
        return out
    if bmm1_scale is None:
        bmm1_scale = head_dim**-0.5

    group_size = q.shape[1] // k_cache.shape[2]
    block_m = triton.next_power_of_2(group_size)
    base_programs = q.shape[0] * k_cache.shape[2]
    small_profile_limit = 8 if block_m <= 8 else 4

    # Tuned on GB300 for the Qwen-Air TP1, TP2, and TP4 attention shapes.
    # Narrow tiles favor decode; wide tiles improve throughput for prefill.
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
        # Triton's FP8 dot requires K >= 32 for the P@V MMA. BF16 retains the
        # tuned BLOCK_N=16 low-grid profiles above.
        block_n = max(block_n, 32)

    num_tiles = triton.cdiv(logical_indices.shape[1], block_n)
    # Avoid empty splits when the selection width is smaller than the profile.
    max_useful_splits = 1 << (num_tiles.bit_length() - 1)
    num_splits = min(max_useful_splits, target_splits)

    # Split=1 writes output directly and compiles out all workspace accesses.
    if num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        # FP32 partials preserve accuracy when merging independently normalized
        # splits.
        partial_output = torch.empty(
            (num_splits, *q.shape), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (num_splits, q.shape[0], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )

    partial_grid = (q.shape[0], k_cache.shape[2], num_splits)
    _qsa_sparse_paged_gqa_splitk_kernel[partial_grid](
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        partial_output,
        partial_lse,
        out,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        logical_indices.stride(0),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        bmm1_scale,
        bmm2_scale,
        TOPK=logical_indices.shape[1],
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=partial_warps,
        num_stages=2,
    )
    if num_splits == 1:
        return out

    _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
        partial_output,
        partial_lse,
        out,
        out.stride(0),
        out.stride(1),
        q.shape[0],
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        num_warps=2,
        num_stages=1,
    )
    return out


def qsa_store_cache_rows(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Store fixed-width rows in a QSA cache without boolean indexing."""

    if not cache.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA CUDA cache stores require Triton")
    if cache.ndim != 4 or cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, width]")
    if not all(cache.shape):
        raise ValueError("QSA cache dimensions must be nonzero")
    if rows.ndim == 3:
        if rows.shape[1] != 1:
            raise ValueError("QSA cache rows must have one head")
        rows = rows[:, 0]
    if rows.shape != (slot_mapping.numel(), cache.shape[3]):
        raise ValueError("QSA cache rows and slots have incompatible shapes")
    if not rows.shape[0]:
        return
    _store_qsa_rows_kernel[(rows.shape[0],)](
        cache,
        slot_mapping,
        rows,
        cache.stride(0),
        cache.stride(1),
        cache.stride(3),
        rows.stride(0),
        rows.stride(1),
        rows.shape[0],
        cache.shape[0],
        PAGE_SIZE=cache.shape[1],
        WIDTH=cache.shape[3],
        BLOCK_D=triton.next_power_of_2(cache.shape[3]),
        num_warps=4,
    )


def qsa_compress_groups_with_ratio(
    raw_keys: torch.Tensor,  # this step's raw key rows [rows, 1, head_size]
    raw_positions: torch.Tensor,  # this step's positions [rows, 1, 3] int64
    compressor_state_cache: torch.Tensor,
    compressor_state_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compress_ratio: int,
    rope_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool completed groups from the compressor-state ring and raw token rows."""

    if not raw_keys.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA CUDA compression requires Triton")
    rows = token_to_req.numel()
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    if raw_keys.ndim != 3 or raw_keys.shape[:2] != (rows, 1):
        raise ValueError("QSA raw keys must be [rows, 1, head_size]")
    if raw_positions.shape != (rows, 1, 3) or raw_positions.dtype != torch.int64:
        raise ValueError("QSA raw positions must be [rows, 1, 3] int64")
    if logical_positions.shape != (rows,) or compressed_slots.shape != (rows,):
        raise ValueError("QSA compression metadata must match token rows")
    if compressor_state_cache.ndim != 4 or compressor_state_cache.shape[2] != 1:
        raise ValueError("QSA compressor-state cache has an invalid shape")
    if (
        # The ring is wider than one group so speculative rows cannot alias
        # onto the committed keys of the group still being collected.
        compressor_state_cache.shape[1] < compress_ratio
        or compressor_state_cache.shape[3] != raw_keys.shape[2]
        or compressor_state_cache.dtype != raw_keys.dtype
    ):
        raise ValueError(
            "QSA compressor-state cache does not match the compression layout"
        )
    if (
        compressor_state_block_table.ndim != 2
        or compressor_state_block_table.shape[1] < 1
    ):
        raise ValueError(
            "QSA compressor-state block table must contain one block per request"
        )
    if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 2:
        raise ValueError("QSA query starts must contain a terminal offset")
    num_requests = query_start_loc.shape[0] - 1
    if compressor_state_block_table.shape[0] < num_requests:
        raise ValueError("QSA compressor-state block table has too few request rows")
    if rope_cache is not None and (
        rope_cache.ndim != 4
        or rope_cache.shape[:3] != compressor_state_cache.shape[:3]
        or rope_cache.shape[3] != 3
        or rope_cache.dtype != torch.int64
    ):
        raise ValueError("QSA packed position view has an invalid shape or dtype")
    if rows and (
        not all(compressor_state_cache.shape)
        or not all(compressor_state_block_table.shape)
    ):
        raise ValueError("QSA compressor-state cache and block table must be nonempty")
    pooled = torch.empty(
        (rows, 1, raw_keys.shape[2]),
        dtype=raw_keys.dtype,
        device=raw_keys.device,
    )
    first_positions = torch.empty((rows, 3), dtype=torch.int64, device=raw_keys.device)
    if not rows:
        return pooled, first_positions
    if rope_cache is None:
        rope_cache = compressor_state_cache
        load_rope_positions = False
    else:
        load_rope_positions = True
    _compress_qsa_groups_kernel[(rows,)](
        raw_keys,
        raw_positions,
        compressor_state_cache,
        rope_cache,
        compressor_state_block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        pooled,
        first_positions,
        raw_keys.stride(0),
        raw_keys.stride(2),
        raw_positions.stride(0),
        raw_positions.stride(2),
        compressor_state_cache.stride(0),
        compressor_state_cache.stride(1),
        compressor_state_cache.stride(3),
        rope_cache.stride(0),
        rope_cache.stride(1),
        rope_cache.stride(3),
        compressor_state_block_table.stride(0),
        pooled.stride(0),
        pooled.stride(2),
        first_positions.stride(0),
        first_positions.stride(1),
        rows,
        compressor_state_cache.shape[0],
        num_requests,
        COMPRESSOR_STATE_SIZE=compressor_state_cache.shape[1],
        COMPRESS_RATIO=compress_ratio,
        HEAD_DIM=raw_keys.shape[2],
        LOAD_ROPE_POSITIONS=load_rope_positions,
        BLOCK_D=triton.next_power_of_2(raw_keys.shape[2]),
        num_warps=4,
    )
    return pooled, first_positions


__all__ = [
    "expand_qsa_block_indices_cuda",
    "qsa_compress_groups_with_ratio",
    "qsa_mqa_paged",
    "qsa_prims_ts_build_page4_metadata",
    "qsa_prims_ts_combined_workspace_size",
    "qsa_prims_ts_group_size",
    "qsa_prims_ts_metadata_workspace_size",
    "qsa_prims_ts_paged_attention",
    "qsa_prims_ts_prepare_attention",
    "qsa_prims_ts_prepare_paged_attention",
    "qsa_prims_ts_run_prepared",
    "qsa_prims_ts_run_prepared_attention",
    "qsa_prims_ts_workspace_size",
    "qsa_select_paged_tokens",
    "qsa_sparse_paged_attention",
    "qsa_store_cache_rows",
]
