# QSA PrimTS integration

This integration keeps the QSA indexer's existing fixed-width token output and
adapts it to FlashInfer PrimTS native paged-KV metadata. It does not add a
kernel-specific top-k argument.

## Metadata route

Each query token is flattened into one independent `SQ=1` attention row. For
the model configuration (`token_topk=2048`, compression ratio and semantic page
size `4`), the indexer returns `[R, 2051]` logical token IDs:

- up to 512 complete selected groups, with four adjacent IDs per group;
- an optional causal tail of one to three IDs;
- `-1` padding after the live prefix.

`qsa_build_page4_paged_metadata` reads only column `4 * page_rank`, maps that
logical token through the request's existing vLLM block table, and writes the
encoded locator expected by FlashInfer:

```text
subpages_per_storage_page = storage_page_size / 4
locator = physical_page * subpages_per_storage_page + token_offset / 4
```

The generated tensors are:

```text
paged_kv_indptr: [R + 1]
paged_kv_indices: [R * 513]
seq_lens: [R]
```

CSR rows have fixed capacity 513, so `indptr[r] = r * 513`. `seq_lens[r]`
selects the live prefix: `min(floor((position + 1) / 4), 512) * 4` complete
tokens plus `(position + 1) % 4` tail tokens. This makes the same buffers usable
for decode, MTP, variable-length prefill, and CUDA graph replay without a host
synchronization or per-forward allocation.

The eventual PrimTS call uses `mask_type="causal"`. A flattened row's compact
query offset is `seq_lens[r] - 1`; therefore complete selected groups are fully
visible and only the one-to-three-token tail needs an element mask once the
2048-token selection budget is saturated. Earlier rows use the same causal path
and mask their shorter compact K/V extent.

CUDA-graph padding rows have logical position `-1`. The adapter gives them the
reserved PrimTS inert-row encoding, `seq_len=1` and locator `-1`. Its TMA
out-of-bounds K/V is zero, so the row produces exact zero output without a
second masking launch.

## Attention owner

On SM100-family GPUs, the QSA owner selects PrimTS when the installed
FlashInfer exposes encoded page-4 support. It keeps the logical vLLM cache view
`[physical_page, Hkv, storage_page_size, 2D]` and splits K/V on the final
dimension; no layout transpose or copy is needed. The adapter writes into
registered maximum-token buffers, and the owner passes their live slices to
the ordinary PrimTS arguments:

```text
(query, (k_cache, v_cache), workspace,
 paged_kv_indptr, paged_kv_indices, seq_lens)
```

Workspace is allocated lazily at the exact policy size and reused while the
semantic launch key is stable. Switching token count or physical page extent
re-zeros the buffer because PrimTS section offsets can change. Warmed CUDA
graph replays keep both the key and storage address stable. The previous Triton
sparse-attention path remains the fallback on other architectures or when the
page-4 API is absent.

## Validation status

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

Bitmap clearing is now a single stream-ordered workspace fill before the
builder rather than 8192 separate CTA-local clears and barriers. Every bitmap
row has exactly one producer CTA, so its lane collisions require atomicity only
within that CTA. The atomics therefore use relaxed CTA scope. The later packer
is a separate kernel on the same CUDA stream, and the kernel boundary orders
its reads after all builder writes. Generated PTX contains
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
correct on the same full causal trace. The requested flattened-SQ1 TP/batch and
tail matrix remains a separate completion gate.
