# QToken-KvBlock-Sparse-Attention in the Qwen owner

The PrimTS path consumes the indexer's compact logical four-token block IDs.
It skips Triton's token-index expansion. Select it with
`VLLM_QSA_ATTENTION_BACKEND=prims_ts`; `triton` selects the reference path.
`auto` uses PrimTS when the installed FlashInfer API, GPU, and geometry support it.

See the [framework interface guide](../../../../benchmarks/qsa/PRIMS_TS_QSA_VLLM_INTERFACE.md)
for complete `plan/run` examples, input shapes, and graph-lifetime rules.

## Query and cache layouts

- Prefill and mixed batches use packed `[total_q, Hq, D]` queries and outputs.
  CPU request boundaries form Q4 routes without joining adjacent requests.
- Uniform decode uses a zero-copy `[B, Nq, G, Hq, D]` view with `G=MTP+1`.
  vLLM currently uses `Nq=1`. Fixed queries omit `qo_indptr`.
- Fixed grouping requires proof that each live request owns a complete group.
  Unsupported groups, nonuniform decode, and drafting without authoritative
  CPU boundaries use request-independent fixed Q1.
- The qualified groups are Q1/Q2/Q4/Q5, with `G * (Hq/Hkv) <= 64`.
  These are caller-selected groups, not a batch-size heuristic.
- PR 53896 stores combined K/V as `[P,Hkv,page_size,2D]`. PrimTS receives
  zero-copy `[P,Hkv,page_size,D]` K and V views.
- The framework block table remains dense `[num_requests,max_storage_pages]`.
  No CSR input or cache repacking is needed.

The model-facing path uses BF16 Q/output and BF16 or FP8-E4M3 cache.
FP8 cache uses a preallocated quantized-Q buffer and live Q/K/V descales.
Source supports SM100/SM103; runtime qualification is on SM103.

## Metadata and attention

One prepared FlashInfer wrapper runs metadata, attention, and any split-KV
reduction. Its public inputs are Q, K/V, the dense block table, compact
`indexer_block_ids`, `token_to_request`, and `query_positions`.
Packed routes additionally provide `qo_indptr`.

Q1 maps selected blocks and its causal partial block directly. Grouped routes
CUB radix-sort at most `G * (block_topk + 1)` candidates, then unique-reduce
logical block IDs while OR-reducing membership bits. Physical locators and
four-byte-packed membership words are separate workspace views.
Per-query membership and causal masking preserve the indexer's selections.

`max_seq_len_kv` is `model_config.max_model_len`, not aggregate physical cache
capacity. Temporary sort state is CTA shared memory. Persistent metadata
capacity depends on selected candidates, not a full-context bitmap.

## Prepared storage and graph lifetime

Each layer keeps its own plans and K/V TensorMaps. Ordered target/MTP layers
share a model-scoped weak allocation pool. Matching graph geometries share
storage; different geometries remain disjoint. Eager plans use a separate
growable arena with a bounded four-entry plan cache.

Do not repurpose one graph's storage for another graph: fewer queries can
require more split-KV scratch. Metadata and attention scratch never alias.
The current separate-reducer policy does not consume split counters; a future
counter-consuming policy must preserve counter initialization across reuse.

KV-cache unbind/rebind clears every derived view and prepared state.
The weak pool cannot retain an allocation after its last owner releases it.
DBO/microbatching is rejected because it can violate ordered workspace use.

Warm the exact plan outside CUDA-graph capture. Capture only prepared runs
and retain their inputs/workspace for graph lifetime. Piecewise prefill keeps
sparse metadata and attention eager; full decode graphs capture both.
PDL orders the metadata/attention/reducer dependency chain.

## Validation

Run the maintained reference suite from the configured vLLM environment:

```bash
.venv/bin/python -m pytest tests/models/qwen4_exp/test_qsa_reference.py -q
```

It covers compact index selection, BF16/FP8, query boundaries, graph padding,
fixed/packed layouts, shared storage, and cache-lifecycle replay.
The [interface guide](../../../../benchmarks/qsa/PRIMS_TS_QSA_VLLM_INTERFACE.md)
describes the matched model evaluation and profiling workflow.
