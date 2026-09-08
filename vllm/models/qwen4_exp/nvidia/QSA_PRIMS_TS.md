# QSA PrimTS integration

This integration keeps the QSA indexer's existing fixed-width token output and
adapts it to FlashInfer PrimTS native paged-KV metadata. It does not add a
kernel-specific top-k argument.

## Current query-layout policy

The current implementation supersedes the older occupancy-based grouping
history retained later in this document:

- pure prefill and mixed batches with exact CPU request boundaries use packed
  `[R,Hq,D]` Q/O with Q4 routes;
- uniform decode uses fixed `[B,Nq,G,Hq,D]` Q/O, where `G=MTP+1`;
  current vLLM decode calls have `Nq=1`;
- fixed decode omits `qo_indptr`; FlashInfer flattens `[B,Nq]` into its internal
  route axis as a zero-copy view;
- fixed decode is selected only after proving every live request has exactly
  `G` rows and any graph-padding suffix is group aligned; and
- decode shapes that cannot prove the configured complete group use fixed Q1;
  decode never switches to packed storage; and
- fast drafting metadata without authoritative CPU request boundaries uses
  request-independent fixed Q1.

The compiled groups are Q1/Q2/Q4/Q5. MTP2 therefore falls back to Q1 until a
Q3 configuration is implemented. Group size is not selected from batch size,
SM count, or measured thresholds; split-KV is resolved independently after the
route grid is known.

## Metadata route

The indexer supplies compact page-four block IDs as Int32
`[num_query_tokens, block_topk]`; vLLM does not run the Triton token-expansion
kernel on the PrimTS path. FlashInfer maps each logical page through the
request's existing block table:

```text
subpages_per_storage_page = storage_page_size / 4
locator = physical_page * subpages_per_storage_page + token_offset / 4
```

Q1 maps one dense route row per query directly. Q2/Q4/Q5 form at most
`G * (block_topk + 1)` tagged candidates per route, radix-sort them in one CUDA
C++ CTA, and unique equal logical IDs while OR-reducing query-membership bits.
Locators and query membership are separate:
`qsa_page_indices` contains plain encoded physical subpage locators, while
`qsa_page_memberships` packs four consecutive 8-bit membership masks into each
Int32 word. The optional causal tail is inserted into the same union. Pages
with zero membership are omitted.

The public prepared plan owns the following views inside one caller-provided
byte workspace:

```text
qsa_page_indices: [num_routes, G * (block_topk + 1)]
qsa_page_memberships: [num_routes, ceil(G * (block_topk + 1) / 4)]
seq_lens: [num_routes]
attention and optional split-KV scratch
```

Prefill passes packed Q/O plus `qo_indptr`; each request is chunked into routes
of at most four rows, so a partial request tail never crosses into another
request. Uniform decode instead passes fixed `[B,Nq,G,Hq,D]` Q/O and omits
query offsets; current vLLM decode calls use `Nq=1`. Both layouts feed the same
metadata and attention kernels.

CUDA-graph padding rows have logical position `-1`. The adapter gives them the
reserved PrimTS inert-row encoding, `seq_len=1` and locator `-1`. Its TMA
out-of-bounds K/V is zero, so the row produces exact zero output without a
second masking launch.

## Attention owner

On SM100-family GPUs, the QSA owner selects PrimTS when the installed
FlashInfer exposes encoded page-4 support. PR 53896 exposes the combined vLLM
cache as `[physical_page, Hkv, storage_page_size, 2D]`. The owner creates a
zero-copy transposed view, splits K/V on the final dimension, and
canonicalizes the resulting `[physical_page, Hkv, storage_page_size, D]`
strides. It prepares the metadata and attention plan for each semantic launch
key outside the hot path, and then calls:

```text
plan.run(query, block_indices, block_table,
         token_to_request, query_positions, out=output)
```

Workspace is allocated lazily at the exact public size. Every PrimTS plan uses
`model_config.max_model_len` as its per-request logical K/V bound. That value
is independent of the number of physical pages allocated across all requests;
the dense block-table row width is used only to prove that each request can
address the model-length bound. Using one static bound makes graph warmup,
capture, replay, piecewise, and eager plan keys agree as live sequence lengths
change. Only the finite set of full-CUDA-graph decode geometries is retained
until cache rebind. Eager and piecewise execution retain the four most recently
used query geometries per QSA layer. KV-cache rebind and teardown clear both
prepared-state collections.

The four eager plans share one grow-only workspace arena per QSA layer.
Preparing a geometry that exceeds the arena first invalidates all plans bound
to its old storage, releases it, and allocates one larger arena. Subsequent
smaller geometries bind different typed views into that same allocation. This
is legal because Qwen4Exp rejects dual-batch overlap and microbatching, so QSA
launches are sequential on the current stream; callers must not reuse an arena
for concurrent launches. Persistent CUDA-graph plans retain their own stable
workspaces instead of sharing this eager arena.

For 8K packed Q4 with block-top-k 512, the persistent metadata outputs use
roughly 16 MiB for locators and 4 MiB for packed memberships. Radix-sort and
union temporary state is CTA-local shared memory, so the allocation is
independent of a 128K model bound and has no context-sized bitmap suffix. The
shared arena therefore retains about 20 MiB plus attention scratch per QSA
layer rather than the sum of four eager workspaces. Metadata outputs are fully
overwritten on every run. Prefill does not allocate split-KV scratch.
Decode split counters occupy a disjoint workspace section, are initialized
during plan preparation, and are restored by the qualified reducer after each
launch. The previous Triton sparse-attention path remains the fallback on
unsupported architectures or when the complete FlashInfer sparse-block API is
absent.

CPU request boundaries prove whether a decode batch is uniform. Every live
request must contribute exactly `G` adjacent rows and any inert graph-padding
suffix must be group aligned. `fast_build` drafting metadata omits CPU
boundaries because only its total is authoritative, so it deliberately uses
request-independent fixed Q1. This is a correctness fallback, not an occupancy
policy. The compact immutable CPU boundary tuple is built once in shared QSA
metadata and reused as the packed-route cache key by every layer; no layer
converts the expanded `qo_indptr` back to Python values.

## Historical validation and performance log

The metadata kernel is covered on SM103 with both 16-token and 256-token
physical cache pages. The tests include early causal positions, all tail sizes,
the 2048-token saturation boundary, encoded subpages, variable rows, and an
inactive graph-padding row. The complete adapter-to-PrimTS path matches an
independent sparse-attention reference for every valid Qwen TP geometry:
`1, 2, 4, 6, 8, 12, 24`.

`benchmarks/kernels/benchmark_qsa_prims_ts.py` measures metadata conversion,
attention alone, the captured end-to-end route, the vLLM Triton
`_qsa_sparse_paged_gqa_splitk_kernel` path, and an independent effective
contiguous-KV PrimTS baseline. Its default matrix covers TP `1, 2, 4, 8`, an
8K prefill, and decode/MTP batch sizes `1, 8, 64, 256` with SQ `1, 4`:

```shell
uv run python benchmarks/kernels/benchmark_qsa_prims_ts.py \
    --phases prefill decode --tp-sizes 1 2 4 8 \
    --context-length 8192 --prefill-seq-len 8192 \
    --decode-batch-sizes 1 8 64 256 --decode-seq-lens-q 1 4
```

Timing now defaults to CUDA-graph replay and a cold L2 for every sample. A
persistent eviction buffer is at least twice the runtime-reported L2 capacity
(258 MiB on GB300); its graph replay is ordered before the target on the same
stream but outside the target's CUDA-event interval. `--warm-l2` is an
explicit diagnostic opt-out. The benchmark also reports exact logical
selected-token counts and effective selected-KV TB/s. That logical rate is
not a profiler measurement of HBM traffic when routes share physical cache
lines; physical achieved bandwidth still requires matched DRAM counters.

Historical timing tables below that predate the cold-L2 checkpoint used warmed
L2 state even when they used CUDA graphs. They remain experiment history, not
the current performance signoff.

### Current grouped standalone signoff

The standalone harness builds exact Q2/Q4 page unions with packed per-query
membership. The original Q4 signoff compared one shared sparse producer with
a flattened `SQ=1` sliding-window baseline. That baseline launched two to four
query CTAs per four-token group and reloaded the same K/V window, so its result
was not a fair contiguous target.

The corrected control presents Q as `[groups, 4, Hq, D]`, uses page-128 K/V,
and applies an exact causal 2K sliding window. Both paths now use one Q CTA per
`(group, KV head)`. Native page-128 staging uses one TMA issuer because one
page covers the KV128 tile; page-4 QSA retains eight issuers to distribute its
independent fragments. On the full layer-3 8K trace, with 10 warmups, 100
CUDA-graph replays, and a 258-MiB L2 eviction before every replay:

| TP | Metadata (us) | Q4 PrimTS (us) | Q4 E2E (us) | Triton sparse (us) | Grouped SWA2K (us) | PrimTS / SWA | E2E / SWA | Triton / SWA | PrimTS / Triton | Volume-adjusted PrimTS / SWA | Volume-adjusted E2E / SWA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41.43 | 908.24 | 942.89 | 3444.27 | 566.29 | 1.604x | 1.665x | 6.082x | 0.264x | 1.395x | 1.448x |
| 2 | 42.64 | 470.76 | 503.41 | 1744.63 | 293.59 | 1.603x | 1.715x | 5.942x | 0.270x | 1.395x | 1.491x |
| 4 | 41.44 | 501.64 | 535.52 | 1966.31 | 274.64 | 1.827x | 1.950x | 7.160x | 0.255x | 1.589x | 1.696x |
| 8 | 41.57 | 463.38 | 497.11 | 2703.03 | 243.15 | 1.906x | 2.044x | 11.117x | 0.171x | 1.658x | 1.778x |

The Q4 unions contain 4,226,204 logical KV tokens. Exact grouped causal SWA
contains 3,675,648, so real top-k unions add 15.0 percent non-shared semantic
work. The volume-adjusted columns multiply measured SWA latency by 1.150; they
are a linear projection, not a hardware counter or a replacement for the
measured grouped baseline. Page-boundary rounding and cross-CTA cache reuse
remain outside that projection.

This result supersedes the older flattened-contiguous Q4 ratios in the
historical sections below. The 20-percent target is not yet met: the remaining
attention gap is 39.5--65.8 percent after the requested semantic-volume
adjustment. The dominant structural difference is transaction granularity.
For each D128 K or V stage, native page-128 SWA issues two contiguous D64 TMA
copies; page-4 QSA must issue 32 independently addressed page fragments per
D64 chunk. Grouping removes redundant K/V consumption across four queries but
does not remove those page-table reads or TMA instructions.

The Triton column is vLLM's existing
`_qsa_sparse_paged_gqa_splitk_kernel`. It consumes the original per-token
top-k rows because that interface has no grouped-union route. Grouped PrimTS is
3.79x/3.71x/3.92x/5.83x faster than Triton at TP1/2/4/8 while preserving the
same per-query semantics. Both implementations match flattened PrimTS within
`3.91e-3` on this checkpoint.

The reported Q4 logical rates are 9.53/9.19/8.63/9.32 TB/s at TP1/2/4/8.
They are not achieved HBM bandwidth. Cold-L2 eviction removes cache state from
the previous replay, but adjacent groups can still reuse the same physical K/V
lines through L2 during one kernel execution; consequently this logical rate
can exceed GB300's roughly 8-TB/s physical HBM limit.

#### BF16/FP8 TP1/TP2 audit

The latest dtype comparison uses the same SM103 GB300, 258-MiB cold-L2
eviction, CUDA graphs, 10 warmups, and 100 timed replays. Prefill replays the
full layer-3 8K route dump. Decode uses causal randomized routes at the two
page-boundary tails, and every row below reports the tail with the worse
`Triton / PrimTS E2E` ratio. FP8 means E4M3 Q/K/V, values scaled by 0.25, and
FP16 output. The selected framework policy is Q1 for SQ1 and the underfilled
BS1/BS8 SQ4 shapes, Q4 for BS64/BS256 SQ4, and Q4 for prefill. In the speedup
column, a value greater than one means PrimTS is faster.

| Dtype | TP | Phase | BS | SQ | Route | Tail | PrimTS attention (us) | PrimTS E2E (us) | Triton (us) | SWA2K (us) | Triton / PrimTS E2E | PrimTS E2E / SWA2K |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 1 | prefill | 1 | 8192 | Q4 | - | 850.42 | 884.35 | 3297.16 | 540.39 | 3.728x | 1.637x |
| BF16 | 2 | prefill | 1 | 8192 | Q4 | - | 439.08 | 471.83 | 1665.43 | 279.71 | 3.530x | 1.687x |
| BF16 | 1 | decode | 1 | 1 | Q1 | 0 | 17.41 | 20.09 | 17.84 | 26.60 | 0.888x | 0.755x |
| BF16 | 1 | decode | 1 | 4 | Q1 | 3 | 18.62 | 21.27 | 24.85 | 25.84 | 1.168x | 0.823x |
| BF16 | 1 | decode | 8 | 1 | Q1 | 0 | 21.51 | 23.99 | 29.53 | 28.77 | 1.231x | 0.834x |
| BF16 | 1 | decode | 8 | 4 | Q1 | 0 | 38.66 | 41.62 | 50.38 | 28.77 | 1.210x | 1.447x |
| BF16 | 1 | decode | 64 | 1 | Q1 | 3 | 65.21 | 67.99 | 79.56 | 61.26 | 1.170x | 1.110x |
| BF16 | 1 | decode | 64 | 4 | Q4 | 3 | 145.84 | 152.07 | 183.44 | 65.48 | 1.206x | 2.322x |
| BF16 | 1 | decode | 256 | 1 | Q1 | 3 | 197.16 | 200.70 | 212.58 | 182.03 | 1.059x | 1.103x |
| BF16 | 1 | decode | 256 | 4 | Q4 | 3 | 483.65 | 491.81 | 564.94 | 201.39 | 1.149x | 2.442x |
| BF16 | 2 | decode | 1 | 1 | Q1 | 0 | 17.30 | 19.77 | 16.47 | 26.43 | 0.833x | 0.748x |
| BF16 | 2 | decode | 1 | 4 | Q1 | 3 | 16.74 | 19.36 | 18.72 | 24.81 | 0.967x | 0.780x |
| BF16 | 2 | decode | 8 | 1 | Q1 | 0 | 19.08 | 21.46 | 24.78 | 28.35 | 1.155x | 0.757x |
| BF16 | 2 | decode | 8 | 4 | Q1 | 3 | 26.86 | 29.29 | 32.82 | 26.82 | 1.121x | 1.092x |
| BF16 | 2 | decode | 64 | 1 | Q1 | 3 | 42.35 | 45.92 | 56.72 | 40.74 | 1.235x | 1.127x |
| BF16 | 2 | decode | 64 | 4 | Q4 | 3 | 87.25 | 93.15 | 111.48 | 44.53 | 1.197x | 2.092x |
| BF16 | 2 | decode | 256 | 1 | Q1 | 3 | 110.67 | 114.09 | 126.20 | 104.06 | 1.106x | 1.096x |
| BF16 | 2 | decode | 256 | 4 | Q4 | 3 | 249.48 | 257.70 | 332.53 | 111.64 | 1.290x | 2.308x |
| FP8 | 1 | prefill | 1 | 8192 | Q4 | - | 817.49 | 850.68 | 5967.37 | 384.15 | 7.015x | 2.214x |
| FP8 | 2 | prefill | 1 | 8192 | Q4 | - | 423.91 | 455.22 | 3007.36 | 200.25 | 6.606x | 2.273x |
| FP8 | 1 | decode | 1 | 1 | Q1 | 0 | 45.79 | 48.05 | 16.28 | 18.19 | 0.339x | 2.642x |
| FP8 | 1 | decode | 1 | 4 | Q1 | 3 | 45.62 | 48.08 | 22.56 | 18.23 | 0.469x | 2.637x |
| FP8 | 1 | decode | 8 | 1 | Q1 | 0 | 46.49 | 48.50 | 31.51 | 18.45 | 0.650x | 2.629x |
| FP8 | 1 | decode | 8 | 4 | Q1 | 0 | 60.24 | 62.25 | 49.67 | 18.79 | 0.798x | 3.313x |
| FP8 | 1 | decode | 64 | 1 | Q1 | 3 | 134.57 | 136.58 | 89.90 | 36.66 | 0.658x | 3.726x |
| FP8 | 1 | decode | 64 | 4 | Q4 | 3 | 120.75 | 126.63 | 267.34 | 42.98 | 2.111x | 2.946x |
| FP8 | 1 | decode | 256 | 1 | Q1 | 3 | 427.70 | 429.85 | 273.10 | 100.88 | 0.635x | 4.261x |
| FP8 | 1 | decode | 256 | 4 | Q4 | 3 | 353.08 | 361.91 | 1015.45 | 124.55 | 2.806x | 2.906x |
| FP8 | 2 | decode | 1 | 1 | Q1 | 0 | 53.22 | 55.40 | 16.16 | 18.37 | 0.292x | 3.016x |
| FP8 | 2 | decode | 1 | 4 | Q1 | 0 | 46.92 | 48.93 | 18.10 | 18.16 | 0.370x | 2.694x |
| FP8 | 2 | decode | 8 | 1 | Q1 | 0 | 46.36 | 47.94 | 23.09 | 18.37 | 0.482x | 2.610x |
| FP8 | 2 | decode | 8 | 4 | Q1 | 3 | 47.51 | 49.60 | 41.10 | 18.44 | 0.829x | 2.690x |
| FP8 | 2 | decode | 64 | 1 | Q1 | 3 | 64.77 | 66.73 | 52.96 | 24.89 | 0.794x | 2.681x |
| FP8 | 2 | decode | 64 | 4 | Q4 | 0 | 69.37 | 75.13 | 151.00 | 34.72 | 2.010x | 2.164x |
| FP8 | 2 | decode | 256 | 1 | Q1 | 3 | 245.53 | 248.06 | 155.38 | 60.46 | 0.626x | 4.103x |
| FP8 | 2 | decode | 256 | 4 | Q4 | 3 | 185.85 | 194.12 | 518.27 | 71.64 | 2.670x | 2.710x |

BF16 prefill is 3.53--3.73x faster than Triton end to end. The large ratio
is not produced by a missing Triton launch: at TP1, flattened PrimTS is
2312.87 us versus Triton's 3297.16 us, while exact Q4 grouping reduces the
logical KV work from 13,968,880 selected tokens to a 4,008,112-token union and
reduces PrimTS attention to 850.42 us. A CUDA-graph-off audit reproduced
2311.27/850.41/3295.30 us for flattened PrimTS/Q4 PrimTS/Triton, within 0.4
percent of the captured numbers. The maximum BF16 output difference is
`3.91e-3`.

Across the 16 conservative BF16 decode rows, PrimTS is faster in 13, with a
1.117x geometric-mean end-to-end speedup; the range is 0.833--1.290x. FP8
prefill is 6.61--7.01x faster than the current Triton FP8 path, but that
Triton path is an experimental baseline rather than a tuned production FP8
kernel. FP8 decode is faster in only four of 16 policy-selected rows and has a
0.791x geometric mean, so there is no general FP8 speedup claim. The maximum
FP8 backend difference after the scaled-input correction is `4.34e-2`.

The large Q4/SWA decode ratios are not a launch-profile regression. These
synthetic adjacent routes are independent and overlap by only about 25
percent, so each Q4 union contains roughly 1,393--1,402 page-4 blocks, versus
about 513 pages of grouped SWA. This is 1.72--1.75x extra KV relative to SWA.
The earlier near-parity Q4 checkpoint used correlated captured routes. The
caller now owns the fixed group and therefore also owns any overlap-based
decision to use Q4; FlashInfer does not silently replace it with another group.

The FP8 audit found three real implementation hazards: the original benchmark
requested an unqualified FP8 output, each elected load warp replayed the whole
page-fragment loop and over-completed its TMA barrier, and generic encoded
page-4 direct/fused publication profiles could leave peers waiting. The
qualified path uses FP8 Q/K/V with FP16 output, partitions page fragments
exactly once across loader warps, and routes unsafe generic profiles through a
separate split reducer. Q4 Keeps remains direct because that profile is
qualified. These changes fix correctness and deadlock behavior; they do not
close the FP8 Q1 or sparse-versus-SWA performance gaps shown above.

#### Matched decode checkpoint

Decode is measured separately from the full-8K prefill table above. The input
context is 8K before sparse selection, batch is `1/8/64/256`, TP is
`1/2/4/8`, and the two causal boundary cases end in page-4 tail `0/3`. Every
number below is the worse ratio across the two tails, with 10 warmups, 100
CUDA-graph replays, and a 258-MiB L2 eviction before each replay. Rankings are
randomized causally per request and shared only by the adjacent MTP tokens of
that request.

SQ1 uses one independent sparse route and one contiguous 2K-equivalent route
per query token. All 32 cells meet the 1.20x attention and end-to-end target:

| TP | Worst PrimTS / contiguous | Worst E2E / contiguous | Worst Triton / contiguous |
|---:|---:|---:|---:|
| 1 | 1.088x | 1.138x | 1.304x |
| 2 | 1.063x | 1.096x | 1.357x |
| 4 | 1.109x | 1.144x | 1.350x |
| 8 | 1.112x | 1.141x | 1.397x |

SQ4 uses one exact Q4 union CTA per `(request, KV head)` and compares it with
one exact grouped causal SWA2K CTA with the same topology. PrimTS is grouped:
it loads the union K/V once and applies packed membership and causal masks for
four query tokens. The existing vLLM Triton
`_qsa_sparse_paged_gqa_splitk_kernel` remains a per-token sparse baseline and
processes the four top-k rows independently. These are forced-Q4 results, not
the mixed Q1/Q2/Q4 launch policy:

| TP | Batch | Q4 / grouped SWA | Q4 E2E / grouped SWA | Triton / grouped SWA | Q4 / Triton |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.665x | 0.798x | 0.733x | 0.907x |
| 1 | 8 | 0.689x | 0.873x | 1.277x | 0.547x |
| 1 | 64 | 1.155x | 1.247x | 2.505x | 0.463x |
| 1 | 256 | 1.035x | 1.079x | 2.445x | 0.423x |
| 2 | 1 | 0.664x | 0.801x | 0.581x | 1.142x |
| 2 | 8 | 0.655x | 0.818x | 0.873x | 0.750x |
| 2 | 64 | 1.034x | 1.155x | 2.244x | 0.464x |
| 2 | 256 | 1.037x | 1.115x | 2.641x | 0.393x |
| 4 | 1 | 1.362x | 1.562x | 0.468x | 3.019x |
| 4 | 8 | 1.332x | 1.538x | 0.737x | 1.819x |
| 4 | 64 | 1.135x | 1.313x | 1.942x | 0.589x |
| 4 | 256 | 1.056x | 1.141x | 2.715x | 0.389x |
| 8 | 1 | 1.386x | 1.583x | 0.558x | 2.484x |
| 8 | 8 | 1.326x | 1.556x | 0.932x | 1.460x |
| 8 | 64 | 1.145x | 1.301x | 2.233x | 0.513x |
| 8 | 256 | 1.051x | 1.130x | 3.153x | 0.333x |

At batch 64 and 256, grouped PrimTS is 1.70--3.00x faster than the per-token
Triton kernel. The small TP4/TP8 grids reverse that result because forced Q4
underfills the GPU. Group selection is now an explicit caller decision, so
that observation no longer causes an automatic fallback to Q1.

The framework supplies a phase-owned group size: prefill and mixed batches use
packed Q4, while uniform decode uses `G=MTP+1`. FlashInfer's
`validate_prims_ts_qsa_group_size` validates that exact choice against
CPU request boundaries and head geometry; it does not infer a different group
from batch size or SM count. The currently compiled decode widths are
Q1/Q2/Q4/Q5. Until Q3 is implemented, MTP2 safely uses fixed Q1. Invalid
grouped boundaries and TileQ64 overflow are rejected rather than silently
mixing rows from different requests.

The group must also fit as complete query-head rows in TileQ64:
`group_size * (Hq / Hkv) <= 64`. This capacity check and the supported grouping
set live in FlashInfer. FlashInfer then reuses the dense grouped-Q geometry to
select the smallest canonical TileQ containing the complete group, avoiding
unnecessary padding. Q4 therefore uses TileQ64/32/16 at TP1-2/TP4/TP8, while
Q2 uses TileQ32/16/8.

Every real Q2/Q4/Q5 group must contain adjacent rows from one request; an
inert CUDA-graph padding suffix must begin and end on the same group boundary.
Consequently ordinary SQ1 decode remains Q1 even at a large batch; validation
never groups unrelated requests.

For TP4 batch 64, grouping the SQ4 workload as two Q2 unions measures 1.106x
attention and 1.277x end-to-end versus matched grouped-Q2 SWA, improving on
forced Q4's 1.135x and 1.313x. Its roughly 12.8-microsecond metadata build is
the remaining reason end-to-end exceeds 1.20x. The caller may choose Q2 for
that shape; FlashInfer will not override the fixed group. Split-KV remains
automatic after grouping and is quantized from the post-group CTA grid toward
a service wave, bounded by useful KV work and an eight-split cap. For TP2
ratio-12 Q2/Q4/Q5, this selects S8 at BS16 and S4 at BS32. At batch 256 forced
Q4 meets the target for every TP;
at batch 64 its attention kernel meets the target for every TP, while
end-to-end still misses at TP1/TP4/TP8.

The fixed-Q5/MTP4 qualification on job 646850 uses real consecutive top-k
rows, BF16 D256, cold L2, CUDA graphs, and a production-stable 10,260-token
compact-KV capacity. The TP1 BS16/32 rows select S4/S2 and complete in
34.84/46.29 us including metadata, versus Triton's 65.52/106.65 us including
index expansion (1.881x/2.304x speedups). TP2 selects S8/S4 and completes in
30.65/35.59 us versus 45.21/65.58 us (1.475x/1.842x). Isolated metadata takes
10.65--12.28 us, while its incremental cost in the combined graph is
4.98--6.05 us because the prepared metadata-to-attention cache handoff stays
live. All rows match the independent reference within `9.8e-4`; the full
table is in `benchmarks/qsa/PR53896_MIGRATION.md`.

All SQ1 and grouped Q2/Q4 outputs differ from their independent-route
references by at most `9.8e-4`. Reported selected-KV bandwidth is a logical
byte rate and may exceed GB300's roughly 8-TB/s physical HBM limit because
adjacent query routes can reuse cache lines during one kernel execution.

#### Graph-stable fixed-bound checkpoint

The observed-union results above are an attention upper bound, not a valid
CUDA-graph launch policy. The benchmark now accepts
`--topk-dump-union-static-max-seq-len`, validates that the observed union fits,
and reports both the observed maximum and fixed `launch_KV`. Production Q2 and
Q4 captures use 4,104 and 8,208 tokens respectively.

FlashInfer `04abb0ee` stages full encoded locators one KV tile at a time for a
long fixed-bound route and retains only packed membership nibbles until
softmax. Short routes of at most eight CTA-local KV tiles retain the complete
locator window. This recovers graph-stable Q4 prefill at every TP. TP1/TP2
now use Q64/KV128 D256 Keeps; TP4/TP8 retain their grouped Swaps paths:

| TP | Attention (us) | Metadata + attention (us) | Contiguous (us) | Attention / contiguous | End-to-end / contiguous |
|---:|---:|---:|---:|---:|---:|
| 1 | 759.54 | 781.94 | 1191.75 | 0.637x | 0.656x |
| 2 | 386.74 | 409.91 | 595.26 | 0.650x | 0.689x |
| 4 | 405.22 | 432.02 | 570.31 | 0.711x | 0.758x |
| 8 | 377.83 | 403.90 | 568.71 | 0.664x | 0.710x |

The fair KV-tile comparison uses the same real top-k union, exact causal
membership, fixed 8,208-token bound, eight TMA issuers, split fanout, cold L2,
and CUDA graphs. KV128 lowers high-grid attention latency by 15.6 percent at
TP1 and 16.6 percent at TP2 relative to KV256. At eight Q4 groups with split
eight, KV128 measures 20.48/20.43 microseconds attention and 0.998x/1.003x
end-to-end versus contiguous for TP1/TP2; KV256 measures 26.86/26.56
microseconds and 1.244x/1.236x end-to-end. At 64 groups with split two, KV128
is 27.6/27.5 percent faster in attention. All cases match the independent
route reference within `9.8e-4`.

The fixed-bound decode crossover is therefore size dependent. Q4 now passes
from eight groups at TP1/TP2, at 64 groups for TP8, and at 256 groups for every
TP; Q2 passes the TP4 128-group case at `0.983x` attention and `1.178x`
end-to-end. Small TP4/TP8 grids remain on Q1. The previous TP1 eight-group
metadata gap was a KV256 attention-profile issue, not a reason to optimize the
metadata kernel first; KV128 closes it with the existing metadata path.

This fixed-bound kernel and route policy is now wired into the model-facing
owner as a whole-launch Q1/Q2/Q4 decision. Mixed launches are conservatively
kept on Q1; partitioning one heterogeneous batch into disjoint Q4, Q2, and Q1
output slices remains a future extension rather than a correctness dependency.

The current GB300 checkpoint (`FlashInfer 102b246`) at TP4 measures 3.9--4.1
microseconds for metadata. For one route, page-4 attention is 14.4 microseconds
at exactly 2048 tokens and 12.3 microseconds with a three-token tail. For eight
routes it is 15.5 and 14.4 microseconds respectively, versus 12.3 microseconds
for contiguous 2048 KV. Attention-only latency is therefore 1.00--1.26x the
contiguous reference; metadata plus attention is 1.17--1.50x.

### Large-batch scaling checkpoint

The same GB300/SM103 checkpoint was measured with 20 warmup iterations and 300
CUDA-graph replay iterations for TP `1, 2, 4, 8` and flattened route batches
`8, 32, 128, 512`. `tail=0` contains exactly 2048 selected tokens; `tail=3`
contains 2051 visible tokens. The reference is contiguous 2048-token PrimTS.
The table reports attention-only latency; metadata stayed between 4.08 and 4.16
microseconds across the complete matrix.

```shell
VLLM_TARGET_DEVICE=cpu PYTHONPATH=/workspace/qwen_next/flashinfer \
    /workspace/qwen_next/.tools/uv run --no-sync python \
    benchmarks/kernels/benchmark_qsa_prims_ts.py \
    --tp-sizes 1 2 4 8 --rows 8 32 128 512 \
    --qsa-max-seq-len 2051 --tail-lens 0 3 \
    --warmup-iterations 20 --iterations 300
```

| TP | Batch | QSA 2048 (us) | Ref (us) | Ratio | QSA 2051 (us) | Ref (us) | Ratio |
|---:|------:|--------------:|---------:|------:|--------------:|---------:|------:|
| 1 | 8 | 26.69 | 12.32 | 2.166x | 26.66 | 12.33 | 2.163x |
| 1 | 32 | 64.31 | 19.04 | 3.377x | 78.02 | 19.02 | 4.101x |
| 1 | 128 | 239.57 | 33.29 | 7.197x | 268.45 | 34.34 | 7.818x |
| 1 | 512 | 821.31 | 103.44 | 7.940x | 922.64 | 103.59 | 8.907x |
| 2 | 8 | 17.74 | 12.32 | 1.439x | 16.72 | 12.33 | 1.356x |
| 2 | 32 | 35.98 | 13.42 | 2.681x | 51.21 | 13.24 | 3.868x |
| 2 | 128 | 120.94 | 19.84 | 6.097x | 135.28 | 19.74 | 6.852x |
| 2 | 512 | 471.31 | 61.43 | 7.672x | 530.20 | 61.45 | 8.628x |
| 4 | 8 | 14.39 | 12.31 | 1.168x | 13.58 | 12.32 | 1.103x |
| 4 | 32 | 34.86 | 12.33 | 2.828x | 49.41 | 12.33 | 4.009x |
| 4 | 128 | 120.81 | 18.52 | 6.524x | 135.14 | 18.47 | 7.318x |
| 4 | 512 | 468.57 | 55.70 | 8.413x | 526.10 | 57.35 | 9.173x |
| 8 | 8 | 14.38 | 12.32 | 1.167x | 13.31 | 12.32 | 1.081x |
| 8 | 32 | 34.87 | 12.32 | 2.829x | 49.21 | 12.33 | 3.992x |
| 8 | 128 | 120.81 | 18.48 | 6.538x | 135.14 | 18.47 | 7.316x |
| 8 | 512 | 468.80 | 57.29 | 8.182x | 526.12 | 57.32 | 9.178x |

The page-4 route remains close to contiguous at batch 8 for TP4/TP8, but the
current large-grid fallback does not meet the contiguous-performance target.
Its gap grows to 6.1--7.8x at batch 128 and 7.7--9.2x at batch 512. TP4 and TP8
are nearly identical because both geometries use one local KV head and the
current tile shape does not benefit from TP8's lower query-head count. The next
optimization should therefore target multi-wave row scheduling and page-table
load amortization rather than the already-small metadata kernel.

The latency profile assigns one KV128 tile to each of 16 or 17 one-instance
split CTAs only when the complete grid fits one SM service wave. Larger
TP/MTP/prefill grids automatically retain the proven two-instance profile.
This keeps the optimized path numerically exact without weakening the
all-shape fallback.

All 41 FlashInfer page-4 cases pass on SM103, including every TP geometry and
encoded storage layout. The vLLM model regression suite reports 19 passed and
10 environment-dependent skips in the development container. The one-instance
17-way route also compiles for the SM100a target. Runtime validation on SM100
remains pending access to that hardware.

### Phase-shaped randomized-route checkpoint

The current harness (`vLLM 1597d03`, `FlashInfer fb6db29`) gives every query
token an independently randomized, causally valid top-k route and an
independently shuffled physical page table. SQ greater than one is flattened
into SQ=1 route tasks. Decode positions are the last SQ positions of the 8K
context, so SQ4 exercises compact lengths 2048 through 2051. Prefill uses all
positions 0 through 8191 and therefore exercises compact lengths 1 through
2051. Every row is checked against the Triton backend before timing; a failure
also runs an exact PyTorch oracle for the worst row.

The first four-loader direct profile reduced the correction group from four
warps to two and failed randomized routes at 128 or more rows. Keeping four
correction warps and using two loader warps is correct across TP1/2/4/8. The
following attention-only results use 10 warmups and 100 CUDA-graph replays on
GB300 `nvl72d090-T01`. Every decode result has maximum PrimTS/Triton difference
at most `9.8e-4`; prefill is at most `3.91e-3`.

| TP | BS | SQ | Flattened rows | PrimTS (us) | Triton (us) | Contiguous (us) | PrimTS / contiguous |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 1 | 16.46 | 12.37 | 12.39 | 1.329x |
| 1 | 1 | 4 | 4 | 18.73 | 18.52 | 12.38 | 1.512x |
| 1 | 8 | 1 | 8 | 26.68 | 20.58 | 14.42 | 1.851x |
| 1 | 8 | 4 | 32 | 79.94 | 37.05 | 22.59 | 3.539x |
| 1 | 64 | 1 | 64 | 69.64 | 68.57 | 45.73 | 1.523x |
| 1 | 64 | 4 | 256 | 284.72 | 176.46 | 82.02 | 3.471x |
| 1 | 256 | 1 | 256 | 262.78 | 201.88 | 163.70 | 1.605x |
| 1 | 256 | 4 | 1024 | 988.58 | 557.25 | 270.74 | 3.651x |
| 2 | 1 | 1 | 1 | 16.48 | 10.33 | 12.36 | 1.334x |
| 2 | 1 | 4 | 4 | 16.47 | 14.31 | 12.36 | 1.332x |
| 2 | 8 | 1 | 8 | 17.71 | 16.49 | 12.37 | 1.432x |
| 2 | 8 | 4 | 32 | 51.26 | 22.58 | 17.05 | 3.006x |
| 2 | 64 | 1 | 64 | 65.66 | 46.48 | 28.79 | 2.281x |
| 2 | 64 | 4 | 256 | 148.05 | 103.11 | 46.11 | 3.211x |
| 2 | 256 | 1 | 256 | 133.45 | 115.53 | 84.38 | 1.582x |
| 2 | 256 | 4 | 1024 | 501.60 | 327.16 | 141.18 | 3.553x |
| 4 | 1 | 1 | 1 | 16.39 | 10.36 | 12.39 | 1.323x |
| 4 | 1 | 4 | 4 | 16.45 | 12.37 | 12.36 | 1.331x |
| 4 | 8 | 1 | 8 | 16.49 | 14.42 | 12.37 | 1.333x |
| 4 | 8 | 4 | 32 | 49.26 | 20.57 | 16.43 | 2.998x |
| 4 | 64 | 1 | 64 | 65.17 | 43.01 | 27.68 | 2.354x |
| 4 | 64 | 4 | 256 | 145.57 | 94.57 | 45.19 | 3.221x |
| 4 | 256 | 1 | 256 | 131.31 | 107.84 | 83.69 | 1.569x |
| 4 | 256 | 4 | 1024 | 493.92 | 347.69 | 136.01 | 3.632x |
| 8 | 1 | 1 | 1 | 15.15 | 10.31 | 12.39 | 1.223x |
| 8 | 1 | 4 | 4 | 14.42 | 12.35 | 12.37 | 1.166x |
| 8 | 8 | 1 | 8 | 16.45 | 14.42 | 12.39 | 1.328x |
| 8 | 8 | 4 | 32 | 49.26 | 20.58 | 16.44 | 2.996x |
| 8 | 64 | 1 | 64 | 65.35 | 45.14 | 28.36 | 2.304x |
| 8 | 64 | 4 | 256 | 146.03 | 98.88 | 45.17 | 3.233x |
| 8 | 256 | 1 | 256 | 132.23 | 111.16 | 83.10 | 1.591x |
| 8 | 256 | 4 | 1024 | 493.85 | 367.26 | 135.14 | 3.654x |

| Prefill TP | BS | SQ=SKV | PrimTS (us) | Triton (us) | Contiguous (us) | PrimTS / contiguous |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 8192 | 6711.75 | 3445.23 | 1443.45 | 4.650x |
| 2 | 1 | 8192 | 3380.37 | 1741.88 | 717.54 | 4.711x |
| 4 | 1 | 8192 | 3310.91 | 1962.67 | 694.84 | 4.765x |
| 8 | 1 | 8192 | 3309.29 | 2753.78 | 704.60 | 4.697x |

The safe second TMA issuer cuts the TP4/128-row direct kernel from 136.68 to
76.58 microseconds, but the complete matrix shows a `3.0x`--`3.65x` SQ4
throughput gap and a `4.65x`--`4.77x` prefill gap. TMA issue parallelism by
itself is therefore insufficient. The earlier synthetic large-batch table
above remains a historical checkpoint; this randomized phase-shaped matrix is
the active performance contract.

Nsight Compute on TP4/BS64/SQ4 compares the same 256-CTA, 512-thread launch
geometry. Both kernels fit one 16-warp CTA per SM and achieve about 25 percent
occupancy. The absolute profiled durations include replay overhead, but the
counter comparison is direct:

| Metric | QSA page-4 | Contiguous KV |
|---|---:|---:|
| Compute throughput | 14.07% | 55.83% |
| Memory throughput | 20.94% | 43.14% |
| L1/TEX throughput | 14.08% | 59.43% |
| L2 throughput | 15.99% | 43.66% |
| L1 hit rate | 15.04% | 52.84% |
| L2 hit rate | 22.64% | 57.73% |
| Memory-pipe busy | 5.35% | 12.31% |

Random page-4 gathers leave the otherwise identical launch latency-bound and
unable to feed either TMA/MMA or the cache hierarchy. The next experiment
should improve CTA-local producer/consumer overlap or reduce the number of
fragment transactions; simply adding issuers cannot close the gap. The queued
`num_kv_insts=1` profile is one candidate, not the assumed solution.

As a route-locality upper-bound experiment, the harness can reorder the same
causally valid selected set with `--route-order logical` or
`--route-order physical`. The physical variant sorts by the final encoded
storage-page/subpage locator after applying the independently randomized vLLM
block table; its setup cost is intentionally excluded from attention timing.
On TP4/BS64/SQ4 (256 flattened saturated routes), top-k, logical, and physical
order measured 145.57, 145.54, and 145.57 microseconds respectively. Physical
sorting preserved the selected set and causal prefix but produced no measurable
attention gain. Route order alone is therefore not the missing cache/feed
optimization, and subsequent work should continue with the existing pipeline
and transaction-shape experiments before changing `num_kv_insts` policy.

### Four-issuer direct encoded-cache checkpoint

FlashInfer `84d2747` adds a producer-only fifth warpgroup to the non-split,
two-instance encoded page-4 profile. Its four load warps issue sixteen of the
64 page-fragment TMA transactions per D128 K/V stage each. The complete
four-warp correction group and the existing `num_kv_insts` selection policy are
unchanged. Compact physical page-4 storage retains its single elected-lane
producer; the wider topology is selected only when CSR entries encode subpage
locators.

The following full phase matrix uses vLLM `60c269a`, independently randomized
causally valid routes, 10 warmups, and 100 CUDA-graph replays on GB300
`nvl72d090-T01`. All decode cases have maximum PrimTS/Triton difference at most
`9.8e-4`.

| TP | BS | SQ | Flattened rows | PrimTS (us) | Triton (us) | Contiguous (us) | PrimTS / contiguous |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 1 | 16.46 | 12.35 | 12.39 | 1.329x |
| 1 | 1 | 4 | 4 | 17.82 | 18.49 | 12.37 | 1.440x |
| 1 | 8 | 1 | 8 | 26.69 | 20.78 | 14.41 | 1.852x |
| 1 | 8 | 4 | 32 | 79.98 | 37.01 | 22.59 | 3.541x |
| 1 | 64 | 1 | 64 | 49.49 | 68.67 | 45.50 | 1.088x |
| 1 | 64 | 4 | 256 | 172.03 | 176.30 | 82.04 | 2.097x |
| 1 | 256 | 1 | 256 | 187.17 | 202.24 | 163.63 | 1.144x |
| 1 | 256 | 4 | 1024 | 591.80 | 557.75 | 270.27 | 2.190x |
| 2 | 1 | 1 | 1 | 16.47 | 10.37 | 12.36 | 1.332x |
| 2 | 1 | 4 | 4 | 16.44 | 14.41 | 12.38 | 1.328x |
| 2 | 8 | 1 | 8 | 18.34 | 16.47 | 12.37 | 1.483x |
| 2 | 8 | 4 | 32 | 51.25 | 22.59 | 16.97 | 3.020x |
| 2 | 64 | 1 | 64 | 65.66 | 45.93 | 28.77 | 2.282x |
| 2 | 64 | 4 | 256 | 90.25 | 103.60 | 46.58 | 1.938x |
| 2 | 256 | 1 | 256 | 92.31 | 115.22 | 83.93 | 1.100x |
| 2 | 256 | 4 | 1024 | 301.20 | 327.74 | 140.96 | 2.137x |
| 4 | 1 | 1 | 1 | 14.66 | 10.49 | 12.38 | 1.184x |
| 4 | 1 | 4 | 4 | 16.46 | 12.39 | 12.37 | 1.331x |
| 4 | 8 | 1 | 8 | 16.45 | 14.43 | 12.37 | 1.329x |
| 4 | 8 | 4 | 32 | 49.28 | 20.55 | 16.46 | 2.994x |
| 4 | 64 | 1 | 64 | 65.10 | 43.05 | 28.03 | 2.322x |
| 4 | 64 | 4 | 256 | 88.13 | 94.45 | 45.12 | 1.953x |
| 4 | 256 | 1 | 256 | 91.53 | 107.98 | 83.55 | 1.095x |
| 4 | 256 | 4 | 1024 | 293.12 | 347.46 | 136.23 | 2.152x |
| 8 | 1 | 1 | 1 | 14.42 | 10.32 | 12.43 | 1.161x |
| 8 | 1 | 4 | 4 | 16.44 | 12.35 | 12.38 | 1.328x |
| 8 | 8 | 1 | 8 | 16.46 | 14.42 | 12.39 | 1.329x |
| 8 | 8 | 4 | 32 | 49.25 | 20.58 | 16.45 | 2.994x |
| 8 | 64 | 1 | 64 | 65.45 | 45.15 | 28.25 | 2.317x |
| 8 | 64 | 4 | 256 | 88.14 | 98.62 | 45.14 | 1.953x |
| 8 | 256 | 1 | 256 | 92.28 | 111.63 | 82.86 | 1.114x |
| 8 | 256 | 4 | 1024 | 293.13 | 367.34 | 135.33 | 2.166x |

| Prefill TP | BS | SQ=SKV | PrimTS (us) | Triton (us) | Contiguous (us) | PrimTS / contiguous |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 8192 | 4020.37 | 3445.32 | 1443.33 | 2.785x |
| 2 | 1 | 8192 | 2026.98 | 1742.35 | 717.96 | 2.823x |
| 4 | 1 | 8192 | 1957.27 | 1962.73 | 694.58 | 2.818x |
| 8 | 1 | 8192 | 1953.09 | 2755.93 | 705.47 | 2.768x |

Relative to the safe two-loader checkpoint, the large SQ4 gap falls from
`3.21x`--`3.65x` to `1.94x`--`2.19x`, and prefill falls from
`4.65x`--`4.77x` to `2.77x`--`2.82x`. Several saturated SQ1 shapes are already
within 20 percent of contiguous, but small MTP grids and the remaining SQ4 and
prefill paths are not. The next controlled experiment keeps two K/V
instructions and asks whether D64 head-dimension staging becomes useful once
four producer warps remove the earlier issue bottleneck. The separate
`num_kv_insts=1` idea remains deferred until these existing pipeline experiments
are complete.

### Full-D256 direct producer-stage checkpoint

FlashInfer `9565e1e` keeps `num_kv_insts=2` and the four-issuer topology, but
collapses the two D128 producer stages into one D256 SMEM stage for direct
encoded-cache Tile-Q=8 kernels. BMM1 consumes sixteen K16 slices from the one
stage. BMM2 issues two legal M128 PV instructions into adjacent TMEM column
panels; a single M256 CTA1 instruction is not legal. Split/reduction kernels
and Tile-Q=16 retain D128. The latter gate is required because the first TP1
Tile-Q=16 D256 prefill prototype raised a synchronous illegal-address error.

The following rerun uses the same independently randomized causal routes,
GB300 node, 10 warmups, and 100 CUDA-graph replays as the four-issuer table.
All decode rows differ from Triton by at most `9.8e-4`; prefill differs by at
most `3.91e-3`.

| TP | BS | SQ | Flattened rows | PrimTS (us) | Triton (us) | Contiguous (us) | PrimTS / contiguous |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 1 | 16.48 | 12.38 | 12.37 | 1.333x |
| 1 | 1 | 4 | 4 | 17.93 | 18.49 | 12.37 | 1.450x |
| 1 | 8 | 1 | 8 | 26.69 | 20.86 | 14.38 | 1.855x |
| 1 | 8 | 4 | 32 | 79.92 | 37.04 | 22.58 | 3.540x |
| 1 | 64 | 1 | 64 | 49.43 | 68.63 | 45.63 | 1.083x |
| 1 | 64 | 4 | 256 | 172.03 | 176.73 | 82.05 | 2.096x |
| 1 | 256 | 1 | 256 | 187.28 | 202.24 | 163.81 | 1.143x |
| 1 | 256 | 4 | 1024 | 591.88 | 557.55 | 270.25 | 2.190x |
| 2 | 1 | 1 | 1 | 16.47 | 10.33 | 12.36 | 1.332x |
| 2 | 1 | 4 | 4 | 16.56 | 14.42 | 12.37 | 1.339x |
| 2 | 8 | 1 | 8 | 18.50 | 16.47 | 12.40 | 1.493x |
| 2 | 8 | 4 | 32 | 51.23 | 22.58 | 17.25 | 2.970x |
| 2 | 64 | 1 | 64 | 65.66 | 46.00 | 28.83 | 2.278x |
| 2 | 64 | 4 | 256 | 90.17 | 103.81 | 46.50 | 1.939x |
| 2 | 256 | 1 | 256 | 92.31 | 115.16 | 84.01 | 1.099x |
| 2 | 256 | 4 | 1024 | 301.25 | 327.93 | 140.84 | 2.139x |
| 4 | 1 | 1 | 1 | 14.52 | 10.42 | 12.36 | 1.174x |
| 4 | 1 | 4 | 4 | 16.44 | 12.37 | 12.36 | 1.330x |
| 4 | 8 | 1 | 8 | 16.46 | 14.40 | 12.36 | 1.331x |
| 4 | 8 | 4 | 32 | 49.24 | 20.55 | 16.45 | 2.993x |
| 4 | 64 | 1 | 64 | 64.92 | 42.97 | 27.99 | 2.319x |
| 4 | 64 | 4 | 256 | 84.25 | 94.56 | 45.09 | 1.869x |
| 4 | 256 | 1 | 256 | 92.19 | 107.94 | 83.58 | 1.103x |
| 4 | 256 | 4 | 1024 | 286.50 | 347.09 | 136.41 | 2.100x |
| 8 | 1 | 1 | 1 | 16.48 | 10.32 | 12.37 | 1.332x |
| 8 | 1 | 4 | 4 | 14.58 | 12.36 | 12.37 | 1.179x |
| 8 | 8 | 1 | 8 | 16.45 | 14.40 | 12.36 | 1.331x |
| 8 | 8 | 4 | 32 | 49.26 | 20.56 | 16.43 | 2.998x |
| 8 | 64 | 1 | 64 | 65.25 | 45.11 | 28.50 | 2.290x |
| 8 | 64 | 4 | 256 | 84.48 | 98.68 | 45.29 | 1.865x |
| 8 | 256 | 1 | 256 | 93.02 | 111.63 | 83.17 | 1.118x |
| 8 | 256 | 4 | 1024 | 284.73 | 367.25 | 135.32 | 2.104x |

| Prefill TP | BS | SQ=SKV | PrimTS (us) | Triton (us) | Contiguous (us) | PrimTS / contiguous |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 8192 | 4020.24 | 3445.28 | 1443.18 | 2.786x |
| 2 | 1 | 8192 | 2027.04 | 1742.72 | 717.83 | 2.824x |
| 4 | 1 | 8192 | 1795.57 | 1963.16 | 694.48 | 2.585x |
| 8 | 1 | 8192 | 1793.30 | 2756.99 | 705.46 | 2.542x |

Relative to D128, the applicable TP4/TP8 prefill ratios fall from
`2.818x`/`2.768x` to `2.585x`/`2.542x`. Their SQ4 256-row ratios fall from
`1.953x`/`1.953x` to `1.869x`/`1.865x`, and 1024-row ratios fall from
`2.152x`/`2.166x` to `2.100x`/`2.104x`. TP1/TP2 correctly remain on D128.
The D256 producer stage is therefore retained, but the large SQ4 and prefill
cases are still outside the 20-percent target. The next experiment continues
on the two-instance path; `num_kv_insts=1` remains a separate optimization
idea rather than the active baseline.

### Eight-issuer current production matrix

The current checkpoint (`vLLM 6f5e4ff`, FlashInfer `d5c252f`) was rerun on
GB300 `nvl72d177-T17` with 10 warmups and 100 CUDA-graph replays. It uses the
production auto policy without forcing `num_kv_insts=1`; the saturated D128
and D256 routes below retain two K/V instructions and eight TMA issuer warps.
Top-k sets are independently randomized and causally valid, the physical page
table is randomized, selected blocks remain in top-k order, and the contiguous
baseline shares compact KV pages among query tokens in one request.

```shell
VLLM_TARGET_DEVICE=cpu PYTHONPATH=/workspace/qwen_next/flashinfer \
  .venv/bin/python benchmarks/kernels/benchmark_qsa_prims_ts.py \
  --phases decode --tp-sizes 1 2 4 8 \
  --decode-batch-sizes 1 8 64 256 --decode-seq-lens-q 1 4 \
  --warmup-iterations 10 --iterations 100
```

| TP | BS | SQ | Rows | PrimTS (us) | PrimTS e2e (us) | Triton (us) | Contiguous (us) | PrimTS / contiguous |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 1 | 16.47 | 18.54 | 12.36 | 12.39 | 1.329x |
| 1 | 1 | 4 | 4 | 16.47 | 19.47 | 18.50 | 12.35 | 1.334x |
| 1 | 8 | 1 | 8 | 26.69 | 26.70 | 22.58 | 14.43 | 1.850x |
| 1 | 8 | 4 | 32 | 79.97 | 82.04 | 37.04 | 22.60 | 3.538x |
| 1 | 64 | 1 | 64 | 49.41 | 52.31 | 68.86 | 46.02 | 1.074x |
| 1 | 64 | 4 | 256 | 139.18 | 141.82 | 177.65 | 82.13 | 1.695x |
| 1 | 256 | 1 | 256 | 175.66 | 178.75 | 201.92 | 163.76 | 1.073x |
| 1 | 256 | 4 | 1024 | 480.91 | 485.50 | 557.24 | 270.40 | 1.778x |
| 2 | 1 | 1 | 1 | 16.45 | 17.00 | 10.31 | 12.36 | 1.331x |
| 2 | 1 | 4 | 4 | 16.47 | 18.50 | 14.40 | 12.34 | 1.335x |
| 2 | 8 | 1 | 8 | 16.98 | 19.61 | 18.50 | 12.40 | 1.370x |
| 2 | 8 | 4 | 32 | 51.26 | 53.31 | 20.58 | 16.46 | 3.114x |
| 2 | 64 | 1 | 64 | 65.64 | 69.52 | 46.37 | 28.84 | 2.276x |
| 2 | 64 | 4 | 256 | 90.21 | 92.32 | 102.91 | 47.10 | 1.915x |
| 2 | 256 | 1 | 256 | 92.48 | 96.20 | 115.21 | 84.42 | 1.096x |
| 2 | 256 | 4 | 1024 | 248.34 | 252.57 | 327.35 | 141.22 | 1.759x |
| 4 | 1 | 1 | 1 | 16.47 | 18.53 | 10.35 | 12.39 | 1.329x |
| 4 | 1 | 4 | 4 | 16.47 | 18.53 | 12.39 | 12.38 | 1.331x |
| 4 | 8 | 1 | 8 | 16.45 | 18.52 | 14.40 | 12.36 | 1.331x |
| 4 | 8 | 4 | 32 | 41.04 | 43.13 | 20.55 | 16.46 | 2.494x |
| 4 | 64 | 1 | 64 | 39.97 | 43.08 | 42.93 | 27.88 | 1.434x |
| 4 | 64 | 4 | 256 | 83.59 | 86.35 | 94.41 | 45.42 | 1.840x |
| 4 | 256 | 1 | 256 | 92.13 | 95.21 | 107.89 | 84.16 | 1.095x |
| 4 | 256 | 4 | 1024 | 255.98 | 260.31 | 347.64 | 136.38 | 1.877x |
| 8 | 1 | 1 | 1 | 16.46 | 18.52 | 10.35 | 12.38 | 1.329x |
| 8 | 1 | 4 | 4 | 16.48 | 18.54 | 12.35 | 12.37 | 1.333x |
| 8 | 8 | 1 | 8 | 16.44 | 18.49 | 14.44 | 12.39 | 1.327x |
| 8 | 8 | 4 | 32 | 41.14 | 43.30 | 22.60 | 16.46 | 2.500x |
| 8 | 64 | 1 | 64 | 41.02 | 43.12 | 45.24 | 28.60 | 1.434x |
| 8 | 64 | 4 | 256 | 83.52 | 86.62 | 98.37 | 45.31 | 1.843x |
| 8 | 256 | 1 | 256 | 92.45 | 95.63 | 111.03 | 83.97 | 1.101x |
| 8 | 256 | 4 | 1024 | 254.30 | 258.93 | 367.49 | 135.67 | 1.874x |

The same checkpoint's BS1, SQ=SKV=8192 prefill results are:

| TP | Metadata (us) | PrimTS (us) | PrimTS e2e (us) | Triton (us) | Contiguous (us) | PrimTS / contiguous |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.37 | 2677.75 | 2689.74 | 3446.23 | 1445.03 | 1.853x |
| 2 | 12.36 | 1351.62 | 1360.93 | 1742.60 | 717.07 | 1.885x |
| 4 | 12.34 | 1151.71 | 1174.02 | 1961.94 | 693.62 | 1.660x |
| 8 | 12.40 | 1152.51 | 1173.63 | 2750.90 | 705.37 | 1.634x |

Maximum PrimTS/Triton error is `9.8e-4` for decode and `3.91e-3` for
prefill. Five of 32 request-shared decode shapes meet the 20-percent target for
both attention-only and metadata-plus-attention latency: TP1 BS64/SQ1 and
BS256/SQ1, plus TP2/4/8 BS256/SQ1. Prefill and every BS256/SQ4 shape remain
outside the strict target. PrimTS is nevertheless faster than Triton for every
prefill TP and for the large SQ4 routes.

Both retained multiwave topologies also cross-compile successfully with
`--gpu-arch sm_100a`: D128 uses Tile-Q16, two KV128 instructions, eight loader
warps, and 768 threads; D256 uses Tile-Q8, one full-head producer stage, two
KV128 instructions, eight loader warps, and 768 threads. Runtime validation is
still SM103-only because this allocation contains GB300 GPUs.

### Capturing real indexer routes

The synthetic benchmark can be checked against actual model selections with an
opt-in rank-zero dump. The disabled path adds only one Python boolean check in
the QSA indexer. A trigger file keeps model-load profiling and warmup forwards
out of the trace:

```shell
export VLLM_QSA_TOPK_DUMP_DIR=/workspace/qwen_next/real_topk_dumps
export VLLM_QSA_TOPK_DUMP_LAYER=3
export VLLM_QSA_TOPK_DUMP_MAX=2
export VLLM_QSA_TOPK_DUMP_TRIGGER=/workspace/qwen_next/real_topk.capture
```

Start vLLM with those variables, wait until the server reports application
startup, and only then create the trigger before submitting the measured
request. Each dump contains expanded selected token IDs, raw query positions,
request ownership, sequence lengths, and both the QSA-side and main-attention
block tables. It is written atomically and only by distributed rank zero.

Analyze one or more captures with:

```shell
python benchmarks/kernels/analyze_qsa_topk_dump.py \
  /workspace/qwen_next/real_topk_dumps/*.pt \
  --output /workspace/qwen_next/real_topk_summary.json
```

The analyzer reports adjacent-query top-k intersection and, for every
32-locator KV128 window, the number of immediately usable fixed physical page
pairs plus an optimistic greedy-pair count after local reordering. This makes
the real trace directly comparable with the randomized and shared-request
synthetic modes and the page8 coalescing upper bound.

The storage page size must come from the bound main-attention cache, not from
the startup `CacheConfig`. Hybrid KV-cache sizing can change it after model
construction. New dumps read `main_layer.kv_cache.shape[2]` at capture time;
for older dumps the analyzer accepts `--main-storage-page-size` as an explicit
override.

#### Layer-3 BrightDelta capture

A TP4 eager server using the local BF16 checkpoint received one 3,741-token
prompt after the trigger was created. Hybrid-cache scheduling split its
prefill into 3,136- and 605-row forwards, consuming both configured dump slots;
these are two prefill chunks, not a prefill/decode pair. The runtime main-cache
page size was 784 tokens. The older instrumented server had captured the
startup value 16, so analysis used `--main-storage-page-size 784`.

| Position range | Saturated adjacent pairs | Real top-k overlap | Independent-random overlap | Changed blocks / 512 | Greedy physical pairs / KV128 window |
|---:|---:|---:|---:|---:|---:|
| 2048--3135 | 1,088 | 0.935 mean, 0.938 median | 0.802 mean | 33.5 mean | 0.160 mean, 0 median, 0 p90 |
| 3136--3740 | 604 | 0.881 mean, 0.883 median | 0.597 mean | 61.1 mean | 0.095 mean, 0 median, 0 p90 |

After subtracting the overlap expected merely from selecting 512 entries out
of the visible causal prefix, the normalized retained correlation is 0.635 and
0.704 for the two chunks. Real adjacent queries are therefore much closer to
the shared-request synthetic bound than independent random routes are, though
they still replace roughly 33--61 blocks per token in this prompt.

This does not revive page8 coalescing. Fixed physical pairs average only 0.059
and 0.001 per 32-locator window; even optimistic within-window sorting averages
0.160 and 0.095 pairs, with both median and p90 equal to zero. The replay below
distinguishes passive cache locality from a route-fusion opportunity. A future
trace intended to reach decode must reserve more dump calls than the number of
chunked prefill forwards.

#### Captured-route replay

The benchmark can replay contiguous chunks from one real request while
densely remapping unused physical page IDs. The remap preserves page sharing,
page order, and whether neighboring physical IDs are adjacent. It does not
change logical selections or causal positions. This command replays the two
layer-3 chunks with the actual non-power-of-two storage page size:

```shell
VLLM_TARGET_DEVICE=cpu PYTHONPATH=/workspace/qwen_next/flashinfer \
  .venv/bin/python benchmarks/kernels/benchmark_qsa_prims_ts.py \
  --topk-dumps \
    /workspace/qwen_next/real_topk_dumps/qsa_topk_layer03_call0000_rank0.pt \
    /workspace/qwen_next/real_topk_dumps/qsa_topk_layer03_call0001_rank0.pt \
  --topk-dump-storage-page-size 784 --topk-dump-patterns captured \
  --tp-sizes 1 2 4 8 --warmup-iterations 20 --iterations 300
```

All four TP geometries match Triton within `3.91e-3`. The correct general
locator-divide path therefore also works at runtime page size 784, beyond the
power-of-two page sizes used by the earlier unit tests.

| TP | Metadata (us) | PrimTS (us) | PrimTS e2e (us) | Triton (us) | Contiguous (us) | PrimTS / contiguous |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.18 | 1099.14 | 1099.38 | 1562.67 | 559.22 | 1.965x |
| 2 | 6.18 | 551.28 | 560.29 | 784.41 | 282.07 | 1.954x |
| 4 | 6.63 | 474.63 | 482.46 | 861.90 | 269.66 | 1.760x |
| 8 | 7.03 | 477.37 | 485.37 | 1222.19 | 271.76 | 1.757x |

The real trace is faster than Triton at every TP, but remains well outside the
20-percent request-shared contiguous target. A matched TP4 control keeps the
same 3,741 positions, one-request page table, cache allocation, and timing
protocol while changing only the selected rankings:

| Ranking | Adjacent overlap | PrimTS (us) | Contiguous (us) | Ratio |
|---|---:|---:|---:|---:|
| captured | 0.915 | 474.13 | 269.51 | 1.759x |
| independent random | 0.729 | 474.17 | 269.75 | 1.758x |
| shared request | 1.000 | 474.89 | 269.52 | 1.762x |

Passive overlap has no measurable effect. This single request occupies only
five 784-token physical pages, so its K/V working set already receives L2
reuse; every independent route still issues the same fragmented TMA count.
Sorting or scheduling correlated CTAs together cannot close the gap by itself.

The overlap is useful only if adjacent Q rows share an actual producer. Exact
unions from the saturated captured rows give the following optimistic load
bound, excluding union construction and per-query membership masking:

| Position range | Q rows fused | Mean union blocks | Ideal page-fragment load reduction |
|---:|---:|---:|---:|
| 2048--3135 | 2 | 545.5 | 46.7% |
| 2048--3135 | 4 | 581.4 | 71.6% |
| 3136--3740 | 2 | 573.1 | 44.0% |
| 3136--3740 | 4 | 653.2 | 68.1% |

A meaningful next upper bound must therefore load a two- or four-query union
once into SMEM and apply a query-by-union membership mask before softmax. It
can derive that internal metadata from the existing per-route CSR rows, but it
cannot be modeled as mere page reordering. This route-fusion experiment stays
on the retained `num_insts_kv=2` baseline; `num_insts_kv=1` remains later.

#### Shared-producer upper bound

The first generic fixed-SQ union launch did not actually fuse the producer: at
TP4 it retained TileQ8, one query token per CTA, and one TMA issuer. After
selecting a true grouped two-instance profile, Q2 uses TileQ16 and Q4 uses
TileQ32; both use D128 staging, eight loader warps, and one direct CTA per
union group. This remains a timing-only lower bound because union pages absent
from an individual query's top-k are not masked yet.

On GB300 `nvl72d145-T01`, with 20 warmups and 300 CUDA-graph replays:

| Q rows | Groups | Mean / max union blocks | Ideal load reduction | Flattened PrimTS (us) | Shared producer (us) | Contiguous (us) | Shared / contiguous |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 847 | 555.4 / 631 | 45.8% | 291.29 | 184.44 | 174.28 | 1.058x |
| 4 | 423 | 605.9 / 726 | 70.4% | 291.50 | 112.59 | 174.34 | 0.646x |

Q2 is 1.58x faster than flattened execution and already within 5.8 percent of
the effective contiguous baseline. Q4 is 2.59x faster than flattened execution
and has enough headroom to pay for membership masking. The next experiment
keeps the existing external route representation, derives union membership
internally, and applies that mask alongside the causal-tail fast path. It does
not switch to `num_kv_insts=1`.

#### Exact packed-membership union

The GPU metadata adapter now consumes the unchanged per-token logical-index
matrix and emits fixed-capacity grouped CSR rows. One bitmap per source query
forms a sorted logical-page union. Each output entry packs
`(encoded_locator << 4) | membership`, where the low bit corresponding to a Q
row records whether that row selected the page. PrimTS strips the nibble for
TMA addressing and applies it to score registers before the existing grouped
causal-tail mask. The adapter accepts caller-owned work/output tensors and is
CUDA-graph replayable; no model-facing top-k argument is added.

The benchmark checks every live GPU locator and sequence length against an
independent CPU builder before timing, then compares attention against the
flattened QSA result. On the real layer-3 capture all cases have maximum error
`9.8e-4`. With 20 warmups and 300 graph replays:

| TP | Q rows | Metadata (us) | Union attention (us) | E2E (us) | Contiguous (us) | Attention / contiguous | E2E / contiguous |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 18.47 | 402.16 | 420.26 | 348.92 | 1.153x | 1.204x |
| 1 | 4 | 16.45 | 440.45 | 456.59 | 348.85 | 1.263x | 1.309x |
| 2 | 2 | 19.04 | 206.27 | 223.27 | 183.42 | 1.125x | 1.217x |
| 2 | 4 | 16.44 | 229.61 | 245.26 | 183.64 | 1.250x | 1.336x |
| 4 | 2 | 19.42 | 190.60 | 208.55 | 173.98 | 1.096x | 1.199x |
| 4 | 4 | 16.42 | 117.40 | 131.78 | 174.00 | 0.675x | 0.757x |
| 8 | 2 | 18.48 | 180.19 | 198.07 | 177.88 | 1.013x | 1.113x |
| 8 | 4 | 16.46 | 108.91 | 124.92 | 177.60 | 0.613x | 0.703x |

For Hq/Hkv=12 (TP1/TP2), TileQ32 holds only two query tokens; a Q4 group
therefore launches two CTAs and reloads the same union. The retained policy is
Q2 for TP1/TP2 and Q4 for TP4/TP8. This meets the attention-only target at all
TP sizes. The remaining end-to-end misses are the two Q2 metadata cases, by
about 1.6 and 3.2 microseconds. An isolated 512-lane builder specialization was
neutral while it retained per-CTA clearing and a barrier. The later full-8K
experiment combines tail splitting with stream-ordered workspace clearing and
weaker, sufficient atomic semantics. All attention profiles here retain
`num_insts_kv=2`.

#### Full 8K real-prefill checkpoint

A second triggered TP4 model run captured an exact 8192-token prefill. Hybrid
cache scheduling emitted two contiguous layer-3 chunks covering positions
`0--7839` and `7840--8191`; both record the runtime main-cache page size of 784.
The combined trace has mean adjacent saturated-route overlap 0.936.

The union replay can include causal prefix rows instead of discarding all rows
before the 2048-token selection budget saturates:

```shell
VLLM_TARGET_DEVICE=cpu PYTHONPATH=/workspace/qwen_next/flashinfer \
  .venv/bin/python benchmarks/kernels/benchmark_qsa_prims_ts.py \
  --topk-dumps \
    /workspace/qwen_next/real_topk_dumps_8k/qsa_topk_layer03_call0000_rank0.pt \
    /workspace/qwen_next/real_topk_dumps_8k/qsa_topk_layer03_call0001_rank0.pt \
  --topk-dump-patterns captured --tp-sizes 2 \
  --topk-dump-union-group-sizes 2 \
  --topk-dump-union-membership-modes masked \
  --topk-dump-union-row-scope full \
  --warmup-iterations 20 --iterations 300
```

For every prefix row, the independent union builder uses
`min(floor((position + 1) / 4), 512)` selected complete pages plus one optional
tail page. It checks the full GPU-generated locator, membership, and sequence
length metadata before timing. The full prefill contains 4096 Q2 groups with
472.6 mean and 561 maximum union blocks, giving a 47.3-percent ideal fragment
load reduction. Maximum union-versus-flattened output error is `1.95e-3`.

| Path | Selected KV tokens | Nominal KV traffic (GB) | Metadata (us) | Attention (us) | E2E (us) | Contiguous (us) | Attention / contiguous | E2E / contiguous |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Flattened PrimTS | 14,690,304 | 15.04 | 10.28 | 1401.81 | 1411.81 | 716.17 | 1.957x | 1.971x |
| Triton sparse | 14,690,304 | 15.04 | n/a | 1741.39 | 1741.39 | 716.17 | 2.432x | 2.432x |
| Exact Q2 union | 7,743,816 | 7.93 | 65.55 | 812.37 | 878.27 | 716.51 | 1.134x | 1.226x |

Exact Q2 producer sharing therefore closes TP2 full-prefill attention to within
13.4 percent of effective contiguous KV while retaining `num_insts_kv=2`.
Metadata remains a separately measured 65.55-microsecond cost and is the only
reason the combined path is 2.6 percentage points outside the 20-percent goal.
For TP2, one selected BF16 KV token represents 1024 nominal bytes: one local KV
head, 256 elements, two bytes per element, and both K and V. Repeated selections
can be served from L2, so dividing these logical byte counts by latency is not
achieved HBM bandwidth. Achieved bandwidth is reported separately from profiler
DRAM-byte counters and must remain below the GB300 hardware limit.

Nsight Compute profiles one warmed component at a time by combining
`--profile-from-start off` with `--profile-union-component`. The table below
uses the profiler's DRAM byte counters and its matching instrumented duration;
it does not mix counters with the faster CUDA-event timings above.

| Path | Grid | Nominal KV (GB) | NCU duration (ms) | DRAM read / write (MB) | Achieved HBM (TB/s) | DRAM peak | L2 traffic (GB) | L2 rate (TB/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Flattened PrimTS | 8192 | 15.04 | 2.167 | 75.70 / 20.84 | 0.0445 | 0.562% | 15.10 | 6.97 |
| Exact Q2 union | 4096 | 7.93 | 1.268 | 69.22 / 21.72 | 0.0717 | 0.904% | 21.07 | 16.61 |
| Contiguous | 8192 | 15.04 | 1.112 | 53.21 / 11.91 | 0.0586 | 0.739% | 15.11 | 13.59 |

All three paths use less than one percent of the roughly 8-TB/s GB300 HBM
limit in this warm-cache replay. The benchmark is therefore L2-resident, not
HBM-bandwidth-bound. Flattened and contiguous L2 traffic closely match their
15.04-GB nominal KV work, while the Q2 union produces 21.07 GB of L2 traffic
for 7.93 GB of unique nominal KV work, a 2.66x amplification. Reducing that
CTA-local reload amplification is a concrete reason to test the retained
`num_insts_kv=1` experiment; it should not be described as increasing HBM
bandwidth.

Replacing the grouped metadata packer's integer SWAR population count with
PTX `popc.b32` is a separately retained optimization. On the same TP2 full
prefill, stable 20-warmup/300-replay runs reduced metadata to 59.8--59.9
microseconds and end-to-end latency to 871.8--872.5 microseconds, or
1.218--1.219x the approximately 715.6-microsecond contiguous reference. This
removes about 5.7 microseconds without changing packed metadata or attention
results, but leaves roughly 13 microseconds to reach the end-to-end target.

#### Barrier-free tail builder and CTA-scoped atomics

The full-8K builder no longer lets the optional 513th page round its Triton
vector from 512 to 1024 lanes. It processes the 512 complete selected pages in
the vector path and issues one scalar load/atomic for a causal tail of one to
three tokens. This applies to both saturated rows and short causal prefixes:
the tail column is `complete_pages * 4`, so the model-facing 2051-wide index
layout is unchanged.

Bitmap clearing uses a measured hybrid. Grids below 4096 flattened rows clear
inside each producer CTA and avoid a separate fill launch. At 4096 rows and
above, one stream-ordered workspace fill removes the repeated CTA-local clear
and barrier. The global-fill path is also mandatory when a very long-context
bitmap has more words than the builder's selected-page vector can cover.
Every bitmap row has exactly one producer CTA, so its lane collisions require
atomicity only within that CTA. The atomics therefore use relaxed CTA scope.
The later packer is a separate kernel on the same CUDA stream, and the kernel
boundary orders its reads after all builder writes. Generated PTX contains
`atom.global.cta.relaxed.or.b32` for both SM103 and SM100. Offline compilation
also succeeds for the Q2 and Q4 builder and ordered-packer specializations on
both architectures.

Nsight Systems profiling of one warmed TP2 full-8K metadata invocation shows
where the improvement comes from:

| Metadata component | Native-popcount baseline (us) | Final builder (us) |
|---|---:|---:|
| Workspace fill | n/a | 1.920 |
| Bitmap builder | 37.568 | 20.704 |
| Ordered union packer | 24.384 | 24.384 |
| Total GPU kernels | 61.952 | 47.008 |

The stable end-to-end result uses the same real layer-3 trace, full causal row
range, 20 warmups, and 300 CUDA-graph replays. Two consecutive runs agree to
0.01 microseconds in metadata time:

| Path | Metadata (us) | Attention (us) | E2E (us) | Contiguous (us) | Attention / contiguous | E2E / contiguous | Max error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact Q2 union | 43.08 | 809.44 | 856.62 | 719.96 | 1.124x | 1.190x | 9.8e-4 |

This closes the TP2 real-8K prefill checkpoint for both attention-only and
metadata-plus-attention latency. Q4 metadata and attention remain numerically
correct on the same full causal trace. The separate flattened-SQ1 TP/batch and
tail gate is recorded next.

The later Q4/KV128 checkpoint (10 warmups, 100 cold-L2 graph replays) measures
the hybrid clear at 41.41 microseconds versus 43.24 microseconds with CTA-local
clearing throughout. Its profiled GPU components are 2.112 microseconds fill,
19.264 microseconds bitmap build, and 15.712 microseconds ordered pack. A
two-warp packer improved a 512-group microcase but regressed the full 2048-group
prefill to 47.64 microseconds. A two-warp builder measured 42.76 microseconds,
also behind the 41.41-microsecond four-warp result. The production builder and
packer therefore both retain four warps. A dense `tl.histogram` builder was
also rejected: its 64/512-group metadata times were 22.35/83.61 microseconds
versus 12.21/19.18 microseconds for the bitmap/atomic design.

### Cold-L2 flattened-SQ1 checkpoint

The flattened TP/batch gate now uses exact tail-zero and tail-three positions.
Every backend is a CUDA graph; a separate 258-MiB eviction graph runs on the
same stream before every sample and outside its CUDA-event interval. Routes
and physical page tables are independently randomized per request. The command
is:

```shell
VLLM_TARGET_DEVICE=cpu PYTHONPATH=/workspace/qwen_next/flashinfer \
  .venv/bin/python benchmarks/kernels/benchmark_qsa_prims_ts.py \
  --phases decode --tp-sizes 1 2 4 8 \
  --decode-batch-sizes 8 32 128 512 --decode-seq-lens-q 1 \
  --decode-end-tails 0 3 --warmup-iterations 10 --iterations 100
```

Each cell reports tail zero / tail three. Effective TB/s is the exact sum of
selected tokens times local BF16 K+V bytes divided by attention latency; it is
not substituted for profiler DRAM counters when routes share cache lines.

| TP | BS | Attention / contiguous | E2E / contiguous | Effective selected-KV TB/s |
|---:|---:|---:|---:|---:|
| 1 | 8 | 1.484x / 1.588x | 1.570x / 1.692x | 1.14 / 1.14 |
| 1 | 32 | 1.826x / 2.084x | 1.904x / 2.137x | 1.99 / 1.64 |
| 1 | 128 | 1.026x / 1.016x | 1.063x / 1.045x | 5.13 / 4.94 |
| 1 | 512 | 1.046x / 1.014x | 1.056x / 1.024x | 6.25 / 6.11 |
| 2 | 8 | 1.429x / 1.536x | 1.587x / 1.697x | 0.68 / 0.67 |
| 2 | 32 | 1.584x / 2.008x | 1.666x / 2.111x | 1.72 / 1.23 |
| 2 | 128 | 1.024x / 1.004x | 1.077x / 1.054x | 4.32 / 4.17 |
| 2 | 512 | 1.057x / 1.062x | 1.075x / 1.081x | 5.65 / 5.34 |
| 4 | 8 | 1.685x / 1.352x | 1.920x / 1.630x | 0.60 / 0.76 |
| 4 | 32 | 1.548x / 1.540x | 1.670x / 1.627x | 1.77 / 1.64 |
| 4 | 128 | 1.035x / 1.042x | 1.089x / 1.095x | 4.29 / 4.04 |
| 4 | 512 | 1.071x / 1.092x | 1.089x / 1.110x | 5.63 / 5.23 |
| 8 | 8 | 1.481x / 1.346x | 1.706x / 1.603x | 0.65 / 0.76 |
| 8 | 32 | 1.548x / 1.543x | 1.666x / 1.641x | 1.77 / 1.64 |
| 8 | 128 | 1.045x / 1.009x | 1.093x / 1.059x | 4.23 / 4.17 |
| 8 | 512 | 1.074x / 1.086x | 1.094x / 1.103x | 5.62 / 5.27 |

All BS128/512 rows pass the strict 1.20x attention and end-to-end gates for
both tails. BS8/32 remain launch/pipeline limited. The largest miss is
TP2/BS32 tail three: the extra 17th KV128 tile makes the two-way split
imbalanced, producing 2.008x attention and 2.111x end-to-end ratios. Every row
matches the Triton sparse backend within `9.8e-4`.

### Retained TileQ32 and physical-8K windowed baseline

The TP2 Q2 route contains 24 live rows: two query tokens times twelve local
query heads. The retained policy rounds this to the existing TileQ32 path. A
native TileQ24 experiment reduced a prior full-prefill attention result from
809.44 to 803.75 microseconds, only about 0.7 percent in a cross-run comparison.
It required special padded TMEM S/O traffic, staged-P packing, correction
reductions, final-output copying, and P-descriptor stepping. Because it did not
change K/V traffic, CTA count, staging depth, or occupancy, that small result
did not justify the extra production surface. The TileQ24 commit was reverted;
the standalone and FMHA code again use only the established TileQ8/16/32
layouts.

The performance target was also tightened. The contiguous control no longer
allocates only the compact 2051-token QSA extent. It exposes the full physical
8192-token K/V cache through ordinary page-128 CSR metadata and launches causal
attention with `window_left=2047`, so exactly 2048 tokens are visible to the
final flattened query. This keeps the physical address range at 8K while
matching the intended effective contiguous work.

On GB300 `nvl72d061-T09`, with 20 warmups, 300 interleaved CUDA-graph replays,
and a 258-MiB L2 eviction before every sample:

| Path | Selected KV tokens | Nominal logical KV (GB) | Latency (us) | Versus physical-8K/window-2K |
|---|---:|---:|---:|---:|
| Flattened QSA | 14,690,304 | 15.04 | 1155.28 | 1.555x |
| TileQ32 exact Q2 union | 7,743,816 | 7.93 | 819.49 | 1.103x |
| Union metadata + attention | 7,743,816 | 7.93 | 866.33 | 1.167x |
| Physical-8K causal-window-2K | 16,777,216 | 17.18 | 742.66 | 1.000x |

The exact union removes 47.3 percent of selected page fragments versus the
flattened routes (472.6 mean and 561 maximum union blocks across 4096 groups).
Both attention-only and metadata-inclusive latency are within the strict
20-percent target. Dividing the dynamic logical byte counts by latency gives
13.02 TB/s for flattened QSA and 9.68 TB/s for the union. These are logical
selected-KV rates, not achieved HBM bandwidth; reuse through L2 can make them
exceed the roughly 8-TB/s GB300 HBM limit. Hardware bandwidth claims still
require profiler DRAM-byte counters.

### TileQ32 full-D256 producer result

Q2 with Hq/Hkv=12 exactly fills TileQ32, so the shared-producer attention can
load one D256 K/V stage instead of two D128 stages without changing its public
vLLM inputs, packed membership, causal-tail handling, two K/V instructions, or
eight TMA issuer warps. PV consumes the full stage through two legal M128
instructions and writes adjacent Qx128 TMEM panels. The final-output STSM
layout now strides those panels by the physical M128 width; using the logical
D256 producer-stage width would place the second panel beyond the Qx256 shared
scratch tile.

A same-node A/B on GB300 `nvl72d175-T01` uses the full layer-3 8K TP2 trace,
4,096 Q2 groups, masked packed membership, the full causal row range, a
258-MiB L2 eviction before every sample, CUDA graphs, 10 warmups, and 100 timed
replays:

| Stage | Metadata (us) | Attention (us) | E2E (us) | Physical-8K/window-2K (us) | Attention / contiguous | E2E / contiguous | Max error |
|---|---:|---:|---:|---:|---:|---:|---:|
| D128 | 51.45 | 816.23 | 862.57 | 739.55 | 1.104x | 1.166x | 9.8e-4 |
| D256/M128x2 | 51.55 | 738.88 | 785.13 | 739.51 | 0.999x | 1.062x | 9.8e-4 |

D256 improves attention by 9.5 percent and E2E by 9.0 percent over the
same-node D128 run. Attention reaches parity with the effective contiguous
control, and the separate metadata kernel leaves only 6.2 percent E2E
overhead.

The complete 10/100 TP sweep keeps D256 only where Q2 exactly fills TileQ32;
the lower head ratios at TP4/TP8 retain the qualified D128 TileQ16/TileQ8
paths:

| TP | Stage | Metadata (us) | Attention (us) | E2E (us) | Contiguous (us) | Attention / contiguous | E2E / contiguous | Logical selected-KV (TB/s) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | D256/M128x2 | 51.37 | 1453.71 | 1499.67 | 1492.25 | 0.974x | 1.005x | 10.91 |
| 2 | D256/M128x2 | 52.01 | 738.76 | 784.77 | 739.41 | 0.999x | 1.061x | 10.73 |
| 4 | D128 | 51.57 | 749.84 | 796.08 | 703.36 | 1.066x | 1.132x | 10.58 |
| 8 | D128 | 51.77 | 709.40 | 754.58 | 704.22 | 1.007x | 1.072x | 11.18 |

All TP sizes remain inside the 1.20x attention and E2E target and match
flattened QSA within `9.8e-4`. The logical selected-KV rates count dynamically
selected bytes and are not HBM-bandwidth claims. The same D256 specialization
compiles for SM100a under the fixed 148-SM compile-only occupancy model;
runtime qualification remains pending access to an SM100 node.

### Q2 masking-overhead checkpoint

The following same-node control changes only union membership semantics and
the attention mask. It retains the D256/M128x2 TileQ32 kernel, cold-L2 eviction,
CUDA graphs, and 10/100 timing protocol:

```shell
VLLM_TARGET_DEVICE=cpu PYTHONPATH=.:../flashinfer .venv/bin/python \
  benchmarks/kernels/benchmark_qsa_prims_ts.py \
  --topk-dumps \
    ../real_topk_dumps_8k/qsa_topk_layer03_call0000_rank0.pt \
    ../real_topk_dumps_8k/qsa_topk_layer03_call0001_rank0.pt \
  --topk-dump-patterns captured --tp-sizes 2 \
  --topk-dump-union-group-sizes 2 \
  --topk-dump-union-membership-modes full masked \
  --topk-dump-union-row-scope full --topk-dump-union-only \
  --warmup-iterations 10 --iterations 100
```

| Membership | Attention mask | Attention (us) | Contiguous (us) | Attention / contiguous |
|---|---|---:|---:|---:|
| Every page is `0b11` | Dense control | 720.10 | 739.21 | 0.974x |
| Exact Q0/Q1 bits | Causal | 738.86 | 739.56 | 0.999x |

The exact path is 18.76 microseconds, or 2.6 percent, slower. This is only the
incremental dynamic-mask cost. The full-bit control still uses packed
locators, loads membership nibbles from the staged page-offset window, maps
TileQ32 score rows to Q0/Q1, and tests the selected bit; its `0b11` values
simply prevent the conditional `-FLT_MAX` assignments. The delta also includes
the causal boundary path. It does not measure the fixed membership-loop cost
shared by both kernels. The dense control processes 4,096 additional logical
tail tokens, one per group, rather than receiving an artificial work
advantage.

Complete KV128 tiles skip per-row causal work, so only the final boundary tile
needs token-level checks. Membership is evaluated across every union page and
is likely the larger part of the measured delta, but that attribution remains
an inference. An exact breakdown requires a compile-time control that decodes
the shifted packed locator while omitting membership application. The later
hybrid-clear Q4 checkpoint reduces metadata to 41.41 microseconds; masking
remains a low-single-digit attention optimization opportunity after metadata.

## End-to-end accuracy qualification

Set `VLLM_QSA_ATTENTION_BACKEND` to make model-level comparisons explicit:

- `triton` always uses `_qsa_sparse_paged_gqa_splitk_kernel`.
- `prims_ts` requires an SM100-family GPU and the FlashInfer encoded-page-4
  API, and fails during model construction if either prerequisite is absent.
- `auto` is the production default and selects PrimTS when both prerequisites
  are present.

This switch changes only the final sparse attention implementation. The QSA
indexer, causal top-k expansion, paged cache update, model weights, prompts,
sampling parameters, and vLLM scheduling interface remain shared. It exists so
an accuracy or performance run cannot silently compare two automatically
selected instances of the same backend.

The model-level qualification matrix is tracked separately from the standalone
kernel tolerance tests:

| Attention | KV cache | MTP proposals | Effective decode SQ | GSM8K | GPQA-Diamond | AIME26 |
|---|---|---:|---:|---:|---:|---:|
| Triton | BF16 | 0 | 1 | pending | pending | pending |
| Triton | BF16 | 3 | 4 | pending | pending | pending |
| Triton | FP8-E4M3 | 0 | 1 | pending | pending | pending |
| Triton | FP8-E4M3 | 3 | 4 | pending | pending | pending |
| PrimTS | BF16 | 0 | 1 | pending | pending | pending |
| PrimTS | BF16 | 3 | 4 | pending | pending | pending |
| PrimTS | FP8-E4M3 | 0 | 1 | pending | pending | pending |
| PrimTS S4/load4 | FP8-E4M3 | 3 | 4 | 1292/1319 | 359/396 (2 reps) | 88/90 (3 reps) |

The populated production row uses vLLM `e9369c8e7`, FlashInfer `2bb8d808`,
TP2, and the exact xhigh sampling configuration documented in the migration
runbook. All requests completed with no invalid predictions. GSM8K has no
truncations; GPQA has two across both repetitions; AIME has none. The differing
replicate counts are shown explicitly: the first two pooled AIME repetitions
score 30/30 and the third scores 28/30. See `PR53896_MIGRATION.md` for the
matched Triton table, token-length distributions, and artifact paths.

For the validated BrightDelta image on nodes where the configured Enroot proxy
is unreachable, exporting `no_proxy='*'` before the first Pyxis invocation
allows direct registry access. The source-under-test keeps the image's compiled
vLLM extensions and bind-mounts only `qsa_cache.py`, `indexer_qsa.py`,
`ops/qsa.py`, and `qsa.py`. FlashInfer must also use a selective overlay:
bind-mount the source `flashinfer/decode.py`, the source
`flashinfer/attention` package, and
`flashinfer/trace/templates/attention.py` over the installed package. Do not
put the complete FlashInfer checkout on `PYTHONPATH`: the source checkout and
validated image have different MoE wrapper/JIT symbol contracts, and replacing
the image's root `flashinfer` package prevents model startup. The selective
overlay preserves the image's installed MoE/GEMM stack while replacing the
PrimTS attention surface under test. Every result must record the vLLM and
FlashInfer commit IDs, image tag, checkpoint, cache dtype, MTP setting,
prompt/scorer settings, and raw per-item output.

FP8 model weights and an FP8 KV cache are different contracts. An FP8-weight
run with `--kv-cache-dtype auto` still has a BF16 KV cache and must not be
entered in an FP8-KV row. The model-facing FP8-KV implementation under test
uses vLLM's encoded-`uint8` cache allocation and `reshape_and_cache_flash` to
quantize BF16 K/V at insertion. Before sparse attention it views those bytes as
the platform E4M3 dtype, statically quantizes the BF16 query into a persistent
graph-stable FP8 buffer, and applies:

```text
bmm1_scale = head_dim^-0.5 * q_scale * k_scale
bmm2_scale = v_scale
```

Both Triton and PrimTS receive those runtime descales and write BF16 model
output. The local BF16 and FP8 checkpoints contain no Q/K/V cache-scale
weights, so vLLM's explicit FP8-cache default of `1.0` is the expected scale
for this study. Standalone qualification must cover FP8 Q/K/V to BF16 output
with non-unit BMM scales before an FP8-KV model result is accepted.

The initial SM100 FP8-KV model startup failed before loading weights because
the QSA implementation inherited `FlashAttentionImpl`'s dense-kernel
capability check; FA2 rejects FP8 KV on this device even though the selected
QSA Triton/PrimTS kernels support it. QSA never calls the inherited dense
attention kernel. The constructor now uses an unquantized placeholder only
while reusing the parent's metadata and DCP initialization, then restores the
requested FP8 cache dtype before QSA cache insertion, scale initialization, or
attention. The patched Triton/FP8/MTP0 server loaded at TP4 and a ten-item
GSM8K smoke scored 10/10 with no errors, invalid predictions, or truncations.
The smoke exercised encoded FP8 K/V insertion, BF16-to-FP8 query quantization,
the Triton sparse attention kernel with descales, and BF16 output.

The first Triton/BF16/MTP0 smoke on GB300 used ten GSM8K items. All ten
requests completed without a request or parser error, eight were correct, and
three reached the fixed 256-token output cap. This is a transport/kernel smoke,
not an aggregate accuracy result; the full 1,319-item row remains authoritative.

`benchmarks/qsa/qsa_reasoning_eval.py` is the dependency-light OpenAI
completions client used for all three tasks. Its fixed protocol is greedy
decoding with seed 42, GSM8K five-shot with 256 output tokens, GPQA-Diamond
zero-shot chain-of-thought with 2,048 output tokens, and AIME26 zero-shot
chain-of-thought with 4,096 output tokens. GPQA choices are shuffled
deterministically per question and seed; the source mirror stores the correct
choice first, so evaluating its original order would be invalid. The validated
source counts and SHA256 digests are:

| Dataset | Items | SHA256 |
|---|---:|---|
| GPQA-Diamond JSON | 198 | `7fad87f34562cf760b32e85f533eba4025b450b4e5039d30cc13fb179b65c739` |
| AIME26 JSONL | 30 | `52822957957a3f577d1e9706c36a66a8108a3f99b6aff424cfb72dff0094a9ee` |

A ten-item smoke run followed by a full run has the form:

```shell
python3 benchmarks/qsa/qsa_reasoning_eval.py \
  --task gsm8k --run-name triton-bf16-mtp0 \
  --num-questions 10 --max-concurrency 10 \
  --url http://nvl72d175-T08:8000/v1/completions \
  --metadata vllm=ff70b29 --metadata flashinfer=eb1e8f0b \
  --output qsa_accuracy/triton-bf16-mtp0/gsm8k-smoke.json

python3 benchmarks/qsa/qsa_reasoning_eval.py \
  --task gsm8k --run-name triton-bf16-mtp0 \
  --url http://nvl72d175-T08:8000/v1/completions \
  --metadata vllm=ff70b29 --metadata flashinfer=eb1e8f0b \
  --output qsa_accuracy/triton-bf16-mtp0/gsm8k.json
```

Each artifact includes dataset hashes, exact settings, labels, parsed
predictions, raw outputs, output hashes, token counts, finish reasons, and
request errors. Keep raw records when comparing modes: equal aggregate
accuracy can hide a position-specific Q2/Q4 failure.
