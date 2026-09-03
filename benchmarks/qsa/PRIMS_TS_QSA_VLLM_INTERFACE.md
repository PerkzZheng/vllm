# PrimTS QSA in vLLM: the integration contract

This note describes the interface implemented by the current local FlashInfer
and vLLM branches. It is written from the perspective of two users:

- a vLLM operator selecting the QSA backend; and
- a framework author integrating the FlashInfer metadata-plus-attention path.

Items labelled **Current** are present in the checked-out code. Items labelled
**TODO** are proposals or qualification work and must not be assumed by another
framework.

## The vLLM operator interface

**Current.** An application using vLLM does not allocate QSA metadata or call
FlashInfer. Select the implementation before starting the server:

```bash
export VLLM_QSA_ATTENTION_BACKEND=prims_ts
```

The accepted values are:

- `auto`: use PrimTS when the GPU and imported FlashInfer support it, otherwise
  use Triton;
- `triton`: force vLLM's expanded-index Triton sparse-attention path; and
- `prims_ts`: require PrimTS and fail during model construction if unavailable.

The local PrimTS path requires an SM100-family GPU and a FlashInfer build that
exports its encoded page-4 QSA APIs. The current FlashInfer implementation is
restricted to SM100 and SM103. vLLM exposes BF16 output with either BF16 or
FP8-E4M3 KV cache. For FP8 cache it quantizes Q to E4M3 before attention and
applies the Q/K/V scales inside the PrimTS plan.

## What vLLM passes to FlashInfer

Let:

- `R` be the number of flattened query tokens in the current forward;
- `G` be the selected query grouping, one of 1, 2, 4, or 5;
- `M = R / G` be the number of attention routes;
- `K` be the compact top-k width in page-4 blocks (512 for this model);
- `P` be the number of physical storage pages;
- `N` be the physical storage-page size; and
- `Hq`, `Hkv`, and `D` be the local Q heads, KV heads, and head dimension.

The real inputs and caller-owned outputs are:

| Value | Shape and dtype | Meaning |
|---|---|---|
| `query` before grouping | `[R, Hq, D]` | Flattened Q rows local to this TP rank |
| `route_query`, Q1 | `[R, Hq, D]` | One query per attention route |
| `route_query`, Q2/Q4/Q5 | `[M, G, Hq, D]` | `G` adjacent queries sharing one exact page union |
| `k_cache`, `v_cache` | `[P, Hkv, N, D]` | Logical HND K/V views; vLLM creates these without copying |
| `block_indices` | `[R, K]`, Int32 | Logical IDs of selected four-token blocks |
| `block_table` | `[B, max_storage_pages]`, Int32 | Per-request logical-to-physical storage-page table |
| `token_to_request` | `[R]`, Int32 | Request ID for every flattened query |
| `query_positions` | `[R]`, Int32 or Int64 | Absolute position of every query |
| `out` | same shape as `route_query` | Caller-owned attention output |

The caller does **not** allocate or pass `qsa_page_indptr`,
`qsa_page_indices`, or `seq_lens` to the combined QSA API. They are internal
producer-consumer tensors inside the byte workspace: metadata writes them and
attention immediately reads them. Semantic inputs (`block_indices`, block
table, request map, and positions) and the output remain explicit tensors.
Prepared plans expose the metadata-output views for diagnostics; callers must
not modify them while the plan is running.

vLLM's native cache is combined storage. The QSA owner constructs the HND K/V
views with transpose/split operations and canonicalizes singleton strides; it
does not rearrange or copy the cache.

## End-to-end call flow

**Current.** The model path is:

```text
QSA indexer
  |  PrimTS: compact logical page-4 IDs [R, 512]
  |  Triton: expanded logical token IDs [R, 2051]
  v
caller-provided fixed group -> validate G in {1, 2, 4, 5}
  v
reshape Q/O to [M,G,Hq,D] when G > 1
  v
workspace-size query -> persistent byte allocation
  v
prepare one metadata + attention plan for this semantic geometry
  v
each run: page-4 metadata launch(es) -> PrimTS attention -> optional reducer
```

PrimTS therefore removes Triton's separate 512-to-2051 index-expansion launch.
The causal tail of zero to three tokens is derived from each query position by
the metadata builder.

## Minimal framework example

The public functions are currently exported lazily from `flashinfer.decode`.
The following is the actual prepared-plan shape of the API, with
application-specific tensor production omitted:

```python
import torch
from flashinfer.decode import (
    get_prims_ts_qsa_group_size,
    get_prims_ts_qsa_workspace_size,
    prepare_prims_ts_qsa_attention,
)

# q:                    [R, Hq, D]
# k_cache, v_cache:     [P, Hkv, N, D]
# block_indices:        [R, K] int32 compact logical page-4 IDs
# block_table:          [num_requests, max_storage_pages] int32
# token_to_request:     [R] int32
# query_positions:      [R] int32/int64
# query_start_loc_cpu:  CPU cumulative request boundaries, [num_requests + 1]

# Fixed caller choice. It is not inferred from batch size or SM occupancy.
requested_group_size = 4
group_size = get_prims_ts_qsa_group_size(
    query_start_loc_cpu,
    block_indices.shape[0],
    q.shape[1],
    k_cache.shape[1],
    group_size=requested_group_size,
)
num_groups = block_indices.shape[0] // group_size

route_q = (
    q
    if group_size == 1
    else q.view(num_groups, group_size, q.shape[1], q.shape[2])
)
route_out = torch.empty_like(route_q, dtype=torch.bfloat16)

workspace_bytes = get_prims_ts_qsa_workspace_size(
    route_q,
    k_cache,
    block_table,
    block_topk=block_indices.shape[1],
    out_dtype=route_out.dtype,
)
workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=q.device)
qk_scale = q.shape[-1] ** -0.5
v_scale = 1.0  # Include the cache value descale here for quantized K/V.

plan = prepare_prims_ts_qsa_attention(
    route_q,
    (k_cache, v_cache),
    block_indices,
    block_table,
    token_to_request,
    query_positions,
    workspace,
    out=route_out,
    bmm1_scale=qk_scale,
    bmm2_scale=v_scale,
)

# Repeated hot-path call. All tensors must preserve the prepared geometry.
plan.run(
    route_q,
    block_indices,
    block_table,
    token_to_request,
    query_positions,
    out=route_out,
)
```

The vLLM wrappers in
`vllm/models/qwen4_exp/nvidia/ops/qsa.py` present the same operations as
`qsa_prims_ts_group_size`, `qsa_prims_ts_combined_workspace_size`,
`qsa_prims_ts_prepare_attention`, and `qsa_prims_ts_run_prepared`. Those names
are vLLM-internal adapter functions, not an additional application API.

## Workspace ownership and sizing

**Current.** `get_prims_ts_qsa_workspace_size(...)` returns one byte count for
three non-overlapping regions:

```text
QSA byte workspace
├── metadata outputs
│     qsa_page_indptr, qsa_page_indices, and seq_lens
│     remain live until attention consumes them
├── metadata scratch
│     Q1: empty
│     Q2/Q4/Q5: one logical-page bitmap per original query
└── attention scratch
      direct policy: uniform-ABI placeholder views only
      split-KV policy: partial O, partial statistics, and completion counters
```

All section starts are 256-byte aligned. The workspace must be a contiguous
CUDA `torch.int8` or `torch.uint8` tensor with a 32-byte-aligned address. It is
exclusive to one in-flight plan or captured graph.

The internal page-index capacity is graph stable:

```text
qsa_page_indices_numel = R * (K + 1)
qsa_page_indices_bytes = R * (K + 1) * 4
```

Grouping does not add another multiplier: `R` already includes all member
queries. Each of the `M` CSR rows reserves `G * (K + 1)` entries, and
`qsa_page_indptr` advances by that fixed capacity. `seq_lens` exposes only the
live packed prefix. The workspace also owns `M + 1` Int32 indptr entries and
`M` Int32 sequence lengths; each region begins at a 256-byte-aligned offset.

For Q2/Q4/Q5, the bitmap bound is:

```text
logical_page4_capacity = max_storage_pages * N / 4
bitset_words = ceil(logical_page4_capacity / 32)
metadata_scratch_bytes = R * bitset_words * 4
```

Here `max_storage_pages` is `block_table.shape[1]`, not the selected-page count.
A selected logical block may occur anywhere in the pre-sparse context, so the
bitmap must cover the full logical range represented by the supplied table.
vLLM slices the table to
`ceil(attn_metadata.max_seq_len / N)` for eager prefill, avoiding a bitmap sized
for the model's full maximum context when the live request is shorter. A static
CUDA-graph bucket must instead retain a bound large enough for every replay.

For `R=8192`, `K=512`, and an exactly 128K-token logical KV bound,
`qsa_page_indices` uses 16,809,984 bytes and grouped bitmaps use 33,554,432
bytes. The CSR header adds only about 16 KiB in this case. The
attention-scratch suffix is policy dependent and must be included by using the
size-query API rather than a hand-written formula.

For the concrete BF16 TP2 prefill geometry used by the local 8K/128K sizing
test (`Hq=24`, `Hkv=2`, `D=256`, storage page 64, Q4), the complete allocation
is 50,382,336 bytes (48.05 MiB): 16,809,984 bytes of page indices, 16,388 bytes
of CSR headers, 33,554,432 bytes of grouped bitmaps, alignment padding, and a
1,280-byte direct-attention ABI suffix. `split_kv_workspace_bytes` is zero.

## Prepared-plan lifetime

**Current.** Preparation performs the work excluded from the hot launch:

1. validate tensor shapes, dtypes, devices, strides, workspace size/alignment,
   output capacity, and non-aliasing requirements;
2. bind the CSR outputs, bitmap, and attention-scratch workspace views;
3. freeze the Q1/Q2/Q4/Q5 metadata geometry and tensor strides;
4. resolve the PrimTS tile and split-KV policy;
5. compile or retrieve the attention callables;
6. validate and retain the QK/V scale values; and
7. initialize the split-KV completion counter once when the selected policy
   uses split-KV.

`PrimsTSQSAPlan.run(...)` then launches prepared metadata and calls the
attention plan's unchecked path. It does not repeat workspace binding, policy
selection, compilation, scale conversion, alias proofs, or counter memset.

The plan retains K/V cache storage, CSR output storage, workspace views,
compiled callables, and scalar scales. Eager callers may pass new Q, output,
block-index, block-table, request-map, and position storage, but those tensors
must preserve the exact prepared shapes, strides, dtypes, and device. The
combined plan deliberately omits hot-path checks, so violating this contract is
a framework error.

vLLM maintains per-layer workspace and plan dictionaries. The workspace key
contains route count, group size, storage-page extent, block-table width,
top-k, input/output dtype, and device. The plan key additionally freezes the
relevant shapes and strides plus scale values. Model head geometry and K/V
storage identity are invariant for a layer. A new semantic geometry obtains a
new allocation and plan; it is not reinterpreted through an incompatible old
layout.

## Metadata semantics

### Q1

**Current.** Q1 uses one metadata kernel and one CSR row per query. It:

1. limits selected complete blocks by `query_position + 1`;
2. maps each logical page-4 block through the request's block table;
3. encodes the physical storage page plus its four-token subpage in one Int32
   locator;
4. derives and appends an optional one-to-three-token causal tail page; and
5. writes the fixed CSR offset and compact token length.

An inactive CUDA-graph padding row uses `seq_lens=1` and locator `-1`. PrimTS's
TMA out-of-bounds path supplies zero K/V, producing an inert output row without
another output-mask kernel.

### Q2, Q4, and Q5: bitmap plus pack

**Current.** Grouping shares K/V loads without changing per-query visibility.
The first kernel creates one dense logical-page bitmap for every original
query, including its causal tail block. The second kernel ORs the `G` bitmaps,
uses popcount prefix ranks to pack only nonzero union pages, maps them through
the block table, and writes:

```text
packed_page_entry = (encoded_page4_locator << 8) | membership
```

Membership bit `i` says whether query `i` selected that page. A page with zero
membership is absent from the union; packing is parallel and does not
use a serial append counter. The attention kernel strips the low byte before
decoding the page locator and applies the membership plus causal token mask
before softmax. The final partial page is therefore masked exactly for each
query, including early positions with fewer than 2051 visible compact tokens.

The grouped value contract is important: every group must contain consecutive
query positions from one request. The GPU packer checks this without a host
readback. An invalid group is made inert; the framework's fixed-group
validation is responsible for preventing such groups in normal execution.

## Caller-owned fixed grouping for prefill and decode

**Current.** The framework supplies one fixed QSA group size for each prepared
semantic geometry. FlashInfer does not choose Q1/Q2/Q4/Q5 from the batch size,
SM count, or a performance threshold. The query tensor rank communicates that
caller choice to workspace sizing and preparation:

- `[M, Hq, D]` means Q1;
- `[M, 2, Hq, D]` means Q2;
- `[M, 4, Hq, D]` means Q4; and
- `[M, 5, Hq, D]` means Q5.

`get_prims_ts_qsa_group_size(...)` is now a boundary/capacity validator, not a
selector. Its `group_size=` argument is mandatory. It returns that exact group
when three correctness gates pass:

1. CPU request boundaries must be present, valid, nondecreasing, and cover no
   more than the padded query extent. Missing or unsafe boundaries return Q1.
2. Every nonempty request length, the real-token boundary, and the padded
   extent must be divisible by `G`; no group may cross requests or the
   real/padding boundary.
3. `G * (Hq / Hkv) <= 64`, so the complete token/head group fits TileQ64.
The low byte can represent up to eight query members, but only Q1/Q2/Q4/Q5 are
currently qualified. Unsafe request boundaries or TileQ64 overflow return Q1;
they never trigger a different grouped size. The attention policy reuses the
dense PrimTS grouped-Q candidate geometry and selects the smallest canonical
tile that holds the caller's complete group.

Consequences by phase:

- ordinary decode supplies Q1;
- MTP3 supplies Q4 and MTP4 supplies Q5, independent of batch occupancy;
- prefill supplies its configured fixed chunk size; and
- geometry without authoritative aligned boundaries uses Q1.

In the vLLM integration, `qsa_query_group_size` is the explicit model-config
override. When it is absent, the user-selected uniform MTP query width is used
when that width is one of Q1/Q2/Q4/Q5; unsupported widths use Q1.

## Split-KV behavior

**Current.** After the caller's fixed group is known, split-KV is selected by
the prepared attention policy from the grouped logical grid. It does not feed
back into group selection. For the qualified ratio-12 BF16 Q4 route, the
bounded wave rule chooses S8 for a 16-union grid and S4 for a 32-union grid.
Split routes receive a dedicated attention-scratch region containing partial
output, softmax statistics, and a completion counter. The bitmap and split-KV
regions never alias.

The fused split reducer uses a wrapping completion counter. Preparation zeros
it once; the final arriving CTA restores it to zero for the next eager call or
graph replay. A policy with a separate reducer does not consume the counter.
`run()` consequently performs no per-request counter memset. Direct plans keep
small placeholder views solely for the uniform compiled-call ABI; these are
not usable split-KV storage.

## CUDA graphs and the current vLLM boundary

**Current.** The FlashInfer plan is CUDA-graph compatible when:

- workspace, K/V, and all captured input/output addresses remain alive and
  stable;
- shapes, strides, dtypes, device, group size, head geometry, page sizes,
  top-k, block-table width, and scale values retain the prepared semantics;
- tensor contents are updated only between completed launches or replays; and
- one workspace is not used concurrently by multiple plans or graphs.

Prepare and warm the semantic key outside capture. Capture only `plan.run(...)`;
replay then executes the recorded metadata, attention, and optional reducer
nodes without Python plan lookup or allocation.

The current vLLM placement is phase dependent:

- full decode CUDA graphs capture the prepared metadata and attention launches;
- piecewise prefill treats QSA as an opaque eager attention boundary, so its
  metadata and attention launches are currently eager even though surrounding
  model pieces are graphed.

This eager prefill behavior applies equally to the backend comparison: the
Triton index expansion and sparse-attention kernels are also outside a CUDA
graph at that boundary. Graph compatibility of the FlashInfer API does not by
itself imply that vLLM captures the piecewise prefill region.

The prepared vLLM integration uses ordinary same-stream ordering between
metadata stages and attention. It explicitly disables metadata-to-attention
PDL. Grouped bitmap-to-pack PDL is also disabled because sustained CUDA-graph
MTP replay exposed stale metadata even though standalone tests passed. The
separate diagnostic metadata API still accepts an `enable_pdl` argument, but
that is not the ordering used by the integrated plan.

## Current constraints and failure modes

Framework authors should treat the following as part of the current contract:

- SM100/SM103 only for this PrimTS implementation;
- semantic QSA page size fixed at four; physical storage page size must be a
  positive multiple of four;
- Q grouping limited to 1, 2, 4, or 5 and `R` divisible by `G`;
- head dimension supported by the underlying path: 64, 128, or 256;
- valid grouped-query head geometry (`Hq` divisible by `Hkv`, with
  `1 <= Hq/Hkv <= 32`);
- CUDA Int32 compact block IDs and block tables with contiguous rows;
- CUDA contiguous per-row request mapping and query positions;
- matching K/V shapes, devices, and dtypes in logical HND views;
- stable prepared geometry and exclusive workspace ownership; and
- positive live sequence lengths no larger than the static maximum encoded by
  the plan.

In the vLLM model integration, public Q and output are BF16. The supported KV
cache modes are BF16 and FP8-E4M3. Fused output quantization, ALiBi, attention
sinks, sliding-window QSA, decode context parallelism, and KV connectors are
not supported by this QSA owner.

## Public-integration status

**Current in this local FlashInfer branch:**

- fixed-group validation is exposed as `get_prims_ts_qsa_group_size`;
- metadata output shapes and metadata-only sizing/build helpers are exported;
- one combined workspace-size query owns all metadata outputs plus disjoint
  metadata/attention scratch;
- `prepare_prims_ts_qsa_attention` returns a reusable combined plan; and
- the combined plan is allocation-free and host-readback-free on its run path.

The simplified workspace-owned-metadata interface was qualified on SM103 with
CUTLASS DSL 4.7.1 by all 38 tests in
`tests/attention/test_attention_ts_qsa_metadata.py`, the standalone
CUDA-graph example in `examples/prims_ts/qsa_page4_attention.py`, and an FP8
TP2/MTP3 vLLM smoke run with four 4K-context concurrent requests (8/8 measured
requests completed). The smoke artifact is
`/workspace/qsa_e2e_perf/job643892-api/api-workspace-smoke/`.

**TODO before treating this as a broadly supported upstream interface:**

- finish public API naming, documentation, examples, and compatibility policy;
- qualify the prepared path across the intended framework graph lifecycles and
  supported SM100/SM103 environments;
- replace the wave-only Q1/Q2/Q4/Q5 heuristic with a measured QSA-aware cost model;
- requalify PDL before enabling it in the prepared path; and
- expose any useful policy/workspace diagnostics without making frameworks
  depend on private layout objects.

From a framework user's perspective, the intended surface is small: choose a
request-safe grouping, query and allocate one byte workspace, prepare once per
semantic geometry, and run repeatedly. Prefill, normal decode, and MTP use that
same interface.
