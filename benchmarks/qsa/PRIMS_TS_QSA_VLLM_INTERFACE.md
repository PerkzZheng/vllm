# QToken-KvBlock-Sparse-Attention in vLLM

This note describes the current local FlashInfer and vLLM integration from a
framework user's perspective. The interface is the same for prefill, ordinary
decode, and MTP decode: choose a query layout, size one byte workspace, prepare
the metadata-plus-attention plan, and run it repeatedly.

## Selecting the vLLM backend

Applications do not allocate QSA metadata directly. Select the implementation
before starting vLLM:

```bash
export VLLM_QSA_ATTENTION_BACKEND=prims_ts
```

The accepted values are:

- `auto`: use PrimTS when the GPU, imported FlashInfer, and QSA geometry support
  it, otherwise use Triton;
- `triton`: force vLLM's expanded-index Triton QSA path; and
- `prims_ts`: require PrimTS and fail during model construction if the runtime
  or configured geometry is unsupported.

The current PrimTS implementation targets SM100 and SM103. This integration
has runtime coverage on SM103; SM100 runtime qualification remains pending.
The vLLM model path exposes BF16 Q/output with BF16 or FP8-E4M3 K/V cache.

## Semantic inputs

Let `R` be the number of flattened query tokens, `G` the caller-selected query
group size, `K` the compact top-k width in sparse blocks, `P` the number of
physical storage pages, and `N` the storage page size.

| Value | Shape and dtype | Meaning |
| --- | --- | --- |
| packed `query` | `[R, Hq, D]` | Prefill request-safe routes |
| fixed `query` | `[B, Nq, G, Hq, D]` | Complete uniform decode groups |
| `qo_indptr` | `[num_routes + 1]`, Int32 | Packed route boundaries; omitted for fixed Q |
| `k_cache`, `v_cache` | `[P, Hkv, N, D]` | Logical HND K/V cache views |
| `indexer_block_ids` | `[R, K]`, Int32 | Selected logical sparse-block IDs |
| `block_table` | `[num_requests, max_storage_pages]`, Int32 | Dense logical-to-physical storage-page table |
| `token_to_request` | `[R]`, Int32 | Owning request for every flattened query |
| `query_positions` | `[R]`, Int32 or Int64 | Absolute query positions |
| `out` | Same shape as `query` | Caller-owned output |

The input block table is dense, not CSR. FlashInfer converts the compact
selected blocks into its own dense route table inside the workspace. The
caller does not allocate or pass a separate QSA page table, sequence-length
tensor, or metadata scratch buffer.

vLLM stores K and V together as `[P, Hkv, N, 2D]`. It supplies FlashInfer with
zero-copy `[P, Hkv, N, D]` K/V views; integration does not repack the cache.
The public name `kv_block_size` describes the logical sparse block size.
The sizing helper defaults it to four; `plan()` takes it explicitly.
The value must be a positive power of two, but the current kernels deliberately
reject every value except `4`. The physical storage page size must be a
multiple of the sparse block size.

## Query layouts and grouping

The framework chooses `G`; FlashInfer does not infer it from batch size,
occupancy, or SM count. The currently qualified groups are Q1, Q2, Q4, and Q5,
subject to `G * (Hq / Hkv) <= 64`.

The vLLM policy is:

- Prefill and mixed batches use packed `[R, Hq, D]` Q with `G=4` when exact
  CPU request boundaries are available. `make_q_token_kv_block_sparse_qo_indptr` chunks
  each request independently, so a short request tail never joins the next
  request.
- Decode requests use `G=MTP+1` when that group is supported. A fixed
  `[B, Nq, G, Hq, D]` zero-copy view is used only after vLLM proves from exact
  CPU boundaries that every live request contributes exactly `G` adjacent
  rows. Zero-length graph-padding requests are allowed.
- Decode always uses fixed Q. If the runtime shape cannot prove one complete
  configured group per live request, vLLM uses request-independent fixed Q1.
- Fast drafting metadata without authoritative CPU request boundaries uses
  request-independent Q1. Unsupported groups such as Q3 also fall back to
  fixed Q1 for decode.

Total-row divisibility is not sufficient proof for fixed grouping. For example,
request lengths `[1, 3]` must not be viewed as one Q4 group even though their
sum is divisible by four.

## Public FlashInfer calls

The wrapper and helpers are exported from `flashinfer.decode`:

```python
from flashinfer.decode import (
    QTokenKvBlockSparsePagedTSWrapper,
    get_q_token_kv_block_sparse_workspace_size,
    make_q_token_kv_block_sparse_qo_indptr,
)
```

`suggest_q_token_kv_block_sparse_group_size` is an optional host-only helper
that accepts the SM count explicitly. It does not override a caller's group
size. vLLM uses the fixed phase policy above.

### Packed prefill example

```python
import torch

# q/out:                 [R, Hq, D]
# k_cache/v_cache:       [P, Hkv, N, D]
# indexer_block_ids:     [R, K] int32 compact sparse-block IDs
# block_table:           [num_requests, max_storage_pages] int32, dense
# token_to_request:      [R] int32
# query_positions:       [R] int32/int64
# query_start_loc_cpu:   CPU cumulative request boundaries

max_seq_len_kv = model_config.max_model_len
group_size = 4
qo_indptr_cpu = make_q_token_kv_block_sparse_qo_indptr(
    query_start_loc_cpu,
    q.shape[0],
    group_size=group_size,
    device="cpu",
)
qo_indptr = qo_indptr_cpu.to(q.device)
out = torch.empty_like(q, dtype=torch.bfloat16)

workspace_bytes = get_q_token_kv_block_sparse_workspace_size(
    q,
    k_cache,
    block_table,
    block_topk=indexer_block_ids.shape[1],
    max_seq_len_kv=max_seq_len_kv,
    o_data_type=out.dtype,
    qo_indptr=qo_indptr_cpu,
    seq_len_q=group_size,
    kv_block_size=4,
)
workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=q.device)

wrapper = QTokenKvBlockSparsePagedTSWrapper()
wrapper.plan(
    batch_size=qo_indptr_cpu.numel() - 1,  # Routes, not original requests.
    seq_len_q=group_size,
    num_qo_heads=q.shape[-2],
    num_kv_heads=k_cache.shape[1],
    head_dim=q.shape[-1],
    kv_block_size=4,
    page_size=k_cache.shape[2],
    block_topk=indexer_block_ids.shape[1],
    max_seq_len_kv=max_seq_len_kv,
    device=q.device,
    workspace_buffer=workspace,
    use_packed_q=True,
    q_data_type=q.dtype,
    kv_data_type=k_cache.dtype,
    o_data_type=out.dtype,
)

wrapper.run(
    q,
    (k_cache, v_cache),
    block_table,
    indexer_block_ids,
    token_to_request,
    query_positions,
    qo_indptr=qo_indptr,
    out=out,
)
```

The CPU `qo_indptr` is sufficient for sizing because only its shape is needed.
The first `run()` binds a stable CUDA Int32 copy and compiles the live tensor
geometry. Run once outside capture before recording subsequent calls. The
prepared hot path does not copy route values back to the host.

### Fixed uniform decode example

After the framework has proved that every live request owns one complete
group, fixed decode omits `qo_indptr`:

```python
# flat_q is contiguous [B * Nq * G, Hq, D].
group_size = mtp_num_speculative_tokens + 1
decode_q = flat_q.view(batch_size, num_groups, group_size, hq, head_dim)
decode_out = flat_out.view_as(decode_q)

workspace_bytes = get_q_token_kv_block_sparse_workspace_size(
    decode_q,
    k_cache,
    block_table,
    block_topk=indexer_block_ids.shape[1],
    max_seq_len_kv=model_config.max_model_len,
    o_data_type=decode_out.dtype,
    kv_block_size=4,
)
workspace = torch.empty(
    workspace_bytes,
    dtype=torch.uint8,
    device=decode_q.device,
)
wrapper = QTokenKvBlockSparsePagedTSWrapper()
wrapper.plan(
    batch_size=batch_size * num_groups,
    seq_len_q=group_size,
    num_qo_heads=hq,
    num_kv_heads=k_cache.shape[1],
    head_dim=head_dim,
    kv_block_size=4,
    page_size=k_cache.shape[2],
    block_topk=indexer_block_ids.shape[1],
    max_seq_len_kv=model_config.max_model_len,
    device=decode_q.device,
    workspace_buffer=workspace,
    q_data_type=decode_q.dtype,
    kv_data_type=k_cache.dtype,
    o_data_type=decode_out.dtype,
)
wrapper.run(
    decode_q,
    (k_cache, v_cache),
    block_table,
    indexer_block_ids,
    token_to_request,
    query_positions,
    out=decode_out,
)
```

The view does not move or duplicate Q. vLLM currently uses `Nq=1`; the public
five-dimensional layout keeps `Nq` explicit for frameworks that can provide
multiple complete groups per request.

These snippets assume already-prepared semantic tensors. With FP8 cache,
vLLM first quantizes Q into its FP8 buffer and passes live `sm_scale` and
`v_scale` to `run()`: `(D**-0.5) * q_scale * k_scale` and `v_scale`,
respectively. Output stays BF16. Scale changes do not require replanning.

## vLLM adapter surface

`vllm/models/qwen4_exp/nvidia/ops/qsa.py` intentionally keeps a small internal
adapter:

- `q_token_kv_block_sparse_ts_combined_workspace_size`
- `q_token_kv_block_sparse_ts_qo_indptr`
- `q_token_kv_block_sparse_ts_prepare_attention`
- `q_token_kv_block_sparse_ts_run_prepared`

There is no vLLM group-selection wrapper. The adapter imports
`validate_q_token_kv_block_sparse_group_size` only as an ABI-availability sentinel;
FlashInfer workspace sizing and preparation perform the authoritative kernel
validation. vLLM owns the phase policy described above.

## Workspace ownership

`get_q_token_kv_block_sparse_workspace_size(...)` returns one byte count covering:

```text
combined QSA workspace
├── dense QSA route table [num_routes, G * (K + 1)]
├── packed membership bytes [num_routes, ceil(G * (K + 1) / 4)] int32
├── compact route sequence lengths [num_routes]
└── attention scratch (disjoint from metadata)
    ├── direct policy: small ABI placeholder storage
    └── split-KV policy: partial output, statistics, and counters
```

The internal route table is `q_token_kv_block_sparse_page_indices`; its companion
`q_token_kv_block_sparse_page_memberships` packs four membership bytes in each
Int32 word. There is no separately allocated page-index tensor, membership
tensor, or CSR indptr. The metadata and split-KV regions never alias. In
particular, vLLM does not reuse metadata storage for split-KV counters.

The allocation must be a contiguous CUDA `torch.int8` or `torch.uint8` tensor
with the required alignment. Concurrent execution requires disjoint storage;
sequential layers may reuse it. Call the sizing API rather than reproducing its formula: the
attention suffix depends on the resolved direct/split-KV policy.

For Q1, the indexer must provide a valid selected-block prefix of length
`min(K, (query_position + 1) // 4)` with no missing entries inside that prefix.
The direct path does not compact arbitrary `-1` holes. vLLM's indexer meets
this contract; entries beyond the required prefix are ignored.

Q1 maps selected logical blocks and the causal tail directly with one CUDA C++
CTA per route. Grouped Q2/Q4/Q5 use one CTA per route to CUB radix-sort at most
`G * (K + 1)` tagged selected/tail candidates, unique equal logical IDs while
OR-reducing query-membership bits, and map the compact union through the dense
block table. Temporary sort/scan state is CTA-local shared memory, not caller
workspace. The separate persistent membership bytes preserve per-query
visibility, and attention applies membership plus the causal tail mask before
softmax.

## Prepared-plan and CUDA-graph lifetime

`plan()` publishes capacity, dtype and workspace geometry. The first eager
`run()` validates the live tensors, binds workspace views and K/V TensorMaps,
compiles the selected kernels, and initializes split-KV state when needed.
Subsequent `run()` calls reuse this state and launch metadata, attention and
any reducer. Changing retained K/V or packed-offset storage requires another
eager run before capture.

For CUDA graphs:

1. allocate all semantic tensors, output, K/V cache, and workspace at stable
   addresses;
2. prepare and warm the exact semantic geometry outside capture;
3. capture only the prepared run; and
4. retain every referenced allocation for the lifetime of the graph.

vLLM maintains a finite dictionary of per-layer prepared states for FULL
decode graph buckets and a four-entry least-recently-used cache for eager
packed prefill, piecewise execution, and fixed Q1 decode fallbacks. Prepared identity
includes `model_config.max_model_len`; all paths retain the full dense block
table rather than slicing it to a live request width.
Every eager lookup checks CUDA capture state before consulting the cache, so an
eager workspace cannot be captured and later evicted. Graph warmup metadata
explicitly marks capture bucket shapes so persistent plans are prepared before
capture. This marker is also required by the standalone MTP graph manager,
whose recorded body deliberately reports runtime graph mode `NONE`. A
persistent cache hit remains valid during capture; a missing persistent state
discovered while the CUDA stream is capturing is an error, because capture is
never allowed to allocate or prepare a new plan.

Up to four eager geometry plans reference one growable arena per layer. If a larger
geometry needs more storage, vLLM clears the plans before replacing the arena;
otherwise plans reuse it sequentially on the current stream. Thus retained
eager storage is the largest arena, not the sum of four arenas.

All sparse layers in one model's static forward context, including MTP,
share a weak allocation pool. Matching graph geometries and equal-size eager
arenas reuse the same storage across ordered layers. Different graph
geometries remain disjoint; plans and K/V TensorMaps remain layer-specific.
The weak pool does not keep old allocations alive after their last owner
drops them. DBO/microbatching is rejected because unordered overlap would
violate this ownership contract.

Metadata output
capacity depends on route count, group size, and top-k width; it is independent
of physical KV-cache capacity and unused model-context suffixes. Concurrent QSA
streams or microbatches would require separate workspace ownership; this model
currently rejects both.
KV-cache unbind/rebind clears the graph-state dictionary, eager LRU, and shared
arena, preventing plans from retaining stale K/V views or profiling
allocations.

Piecewise prefill currently executes QSA metadata and attention eagerly at the
opaque attention boundary. Full decode graphs capture the prepared metadata
and attention launches. The metadata-to-attention handoff and optional
attention-to-reducer handoff may use PDL on supported devices. Older devices
retain ordinary same-stream ordering; neither path requires a
framework-managed event.

Large predictable prefill chunks can opt into an exact-only piecewise graph
without adding a high-padding endpoint to the ordinary capture ladder:

```bash
vllm serve ... \
  --cudagraph-capture-sizes 2 4 8 16 24 32 40 48 56 64 \
  --compilation-config \
  '{"piecewise_cudagraph_exact_capture_sizes":[8192]}'
```

An exact size matches only the same scheduled token count. It does not cause
65--8,191-token batches to pad to 8K and does not add a FULL decode graph. Exact
sizes augment a nonempty ordinary capture-size list; this preserves existing
full-graph memory sizing and capture-backend assumptions. Sizes must be
positive and compatible with a piecewise graph mode; entries larger than
`max_num_batched_tokens` are warned about and dropped. Sizes must also be
TP-divisible when sequence parallelism is enabled. The 8K graph keeps QSA and
its metadata eager while graphing the safe regions between splitting ops.
On the measured TP2 BF16 model it reduced available KV memory by about
2.31 GiB per rank, so frameworks should opt in only for stable high-value
chunk sizes rather than populate a dense long-context ladder.

## Current constraints

- SM100/SM103 target, with current runtime qualification on SM103;
- `kv_block_size=4` only, with generic power-of-two API naming reserved for
  future specializations;
- Q1/Q2/Q4/Q5, with TileQ64 head-capacity validation;
- fixed contiguous `[B, Nq, G, Hq, D]` or packed `[R, Hq, D]` with CUDA Int32
  `qo_indptr`;
- dense CUDA Int32 block tables and compact block IDs with contiguous rows;
- head dimension 256;
- BF16 vLLM output with BF16 or FP8-E4M3 K/V cache; and
- stable prepared geometry and exclusive workspace ownership.
- no dual-batch overlap or microbatching; Qwen4Exp PLE/QSA rejects both during
  model configuration.

Fused output quantization, ALiBi, attention sinks, sliding-window QSA, decode
context parallelism, and KV connectors are not supported by this vLLM QSA
owner.
