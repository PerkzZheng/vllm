# PrimTS QSA integration

This document describes the Qwen3.8-Flash-Next query-sparse-attention (QSA)
integration with FlashInfer PrimTS. It records the maintained runtime contract;
benchmark campaign logs and machine-specific procedures are intentionally kept
outside the product source tree.

## FlashInfer dependency

The integration currently requires the experimental FlashInfer API published
at commit `efd44507e797ee44c1c94a652066bef13deaf2cf` on
`PerkzZheng/flashinfer:qsa-packed-query`. The released FlashInfer version in
vLLM's normal dependency set does not yet provide this dense prepared-plan ABI.

Backend discovery requires `build_prims_ts_qsa_metadata` in addition to the
workspace, grouping, and preparation functions. This sentinel prevents an
older experimental page-four API from being selected and then failing on its
first request.

## Backend selection

Set `VLLM_QSA_ATTENTION_BACKEND` before constructing the model:

- `auto` selects PrimTS only when the imported FlashInfer API, QSA geometry,
  and a runtime-qualified architecture all match. SM103 is currently the only
  auto-qualified architecture.
- `triton` always uses vLLM's existing expanded-index Triton QSA path.
- `prims_ts` requires PrimTS and raises during model construction if the
  architecture, API, or geometry is unsupported. The kernels structurally
  support SM100 and SM103, but SM100 runtime qualification remains pending.

The vLLM path supports head dimension 256, sparse block size 4, BF16 query and
output, and BF16 or FP8-E4M3 K/V cache. The configured group must be one of
Q1, Q2, Q4, or Q5 and must satisfy `G * (Hq / Hkv) <= 64`.

## Inputs and metadata

The indexer emits compact logical sparse-block IDs, not expanded token IDs:

```text
block_indices:    [R, block_topk] int32
block_table:      [B, max_storage_pages] int32
token_to_request: [R] int32
query_positions:  [R] int32 or int64
K/V cache:        [physical_page, Hkv, storage_page_size, D]
```

`block_table` is the model's normal dense logical-to-physical storage-page
table. FlashInfer maps each selected sparse block through that table and owns
the generated dense QSA route table, route lengths, metadata scratch, and any
split-KV scratch inside one caller-provided byte workspace. vLLM does not
allocate a second page-index tensor or run Triton's token-expansion kernel on
the PrimTS path.

For grouped Q2/Q4/Q5 routes, metadata packs the sorted union of the selected
pages and encodes query membership in each locator. The attention kernel tests
membership and applies the causal tail mask before softmax. Q1 uses the same
prepared interface without grouped-union bitmap scratch.

## Query layouts

The framework chooses the query group; FlashInfer does not infer it from batch
size, SM count, or a performance threshold.

- Prefill uses packed `[R, Hq, D]` query/output and caller-fixed Q4 routes.
  Route boundaries are derived from exact CPU request boundaries, so a partial
  request tail never joins the next request.
- Uniform decode uses `[B, num_groups, G, Hq, D]`, with `G=MTP+1`, only after
  vLLM proves that every live request contributes exactly `G` adjacent rows
  and any graph-padding suffix is group-aligned.
- Irregular or mixed decode stays packed. Fast drafting metadata that lacks
  authoritative CPU request boundaries uses request-independent Q1.
- Unsupported group sizes, such as Q3, fall back to Q1 rather than silently
  changing request grouping.

Total token-count divisibility is not sufficient evidence for fixed grouping:
for example, request lengths `[1, 3]` cannot be reinterpreted as one Q4 route.

## Prepared execution and CUDA graphs

The internal adapter has four operations:

```text
qsa_prims_ts_combined_workspace_size
qsa_prims_ts_qo_indptr
qsa_prims_ts_prepare_attention
qsa_prims_ts_run_prepared
```

Preparation validates tensor geometry, binds workspace views, resolves direct
or split-KV execution, and initializes persistent state. The hot `run` path
updates metadata and launches attention without allocation or host readback.

FULL decode graph buckets retain separate prepared plans because replay needs
stable addresses. Eager and piecewise launches use a bounded LRU of exact
geometries over one growable per-layer workspace arena. Arena growth first
invalidates every plan bound to the old allocation. KV-cache unbind or rebind
clears both persistent and eager states, preventing plans from retaining stale
cache or profiling allocations.

Graph capture must prepare and warm the exact geometry before capture. A
missing persistent plan discovered while the stream is capturing is an error;
capture never creates a plan or grows a workspace. Metadata and attention use
ordinary same-stream ordering. The grouped bitmap-to-pack metadata dependency
may use FlashInfer's architecture-gated programmatic dependent launch.

Large stable prefill chunks may be added as exact-only piecewise capture sizes.
An exact size matches only that scheduled token count and does not extend the
ordinary padding ladder or create another FULL decode graph.

## Current limitations

- Auto selection is runtime-qualified on SM103; explicit SM100 support still
  requires runtime qualification before it becomes an automatic default.
- Sparse block size 4 is the only compiled specialization.
- Head dimension is 256 and Q/output are BF16.
- Fixed decode requires complete contiguous request groups; other layouts use
  packed routing.
- Decode context parallelism, dual-batch overlap, and QSA microbatching are not
  supported.
- Fused output quantization, ALiBi, attention sinks, sliding-window QSA, and KV
  connectors are not supported by this owner.
