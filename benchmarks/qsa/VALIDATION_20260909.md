# QToken-KvBlock-Sparse-Attention integration validation

These are historical results from before the current-main rebase. They are
the comparison target, not qualification of the rewritten branch. New-main
local and end-to-end validation must be reported separately.

## Scope and sources

TP2 on GB300/SM103, Qwen3.8-Flash-Next revision `de4b8e4`, using
`vllm/vllm-openai:qwen38-flash-next` and CUTLASS DSL 4.7.1. The local
FlashInfer overlay preserves image GDN/MoE/TVM-FFI binaries. Both backends
use the same runtime. These results do not qualify SM100 runtime behavior.

FlashInfer `c7f73ca6` includes the membership-word padding fix. Its follow-up
`d138709e` changes public-contract docstrings only (AST equivalence checked).
Accuracy uses immutable vLLM `aa9b78851` sources. Subsequent vLLM commits `93cd306c7` and
`8a31f0db4` clean documentation, typing and benchmark tools; inference
kernels, grouping/split policy and graph ownership are unchanged. Final
affected tests and runner smoke checks pass after that cleanup.

The adapter uses packed G4 prefill and fixed `G=MTP+1` decode when request
boundaries permit, with G1 fallback/drafting. See the
[interface guide](PRIMS_TS_QSA_VLLM_INTERFACE.md) for
the public plan/run calls, dense page tables and workspace lifetime.

## Accuracy

TP2, FP8-E4M3 KV cache, MTP3. All six independent jobs completed; source and
result-manifest checks pass.
These are the unchanged evaluator scores:

| Task | Triton | PrimTS | Request errors T/P | Invalid answers T/P | Truncations T/P |
| --- | ---: | ---: | ---: | ---: | ---: |
| GSM8K | 1295/1319 (98.18%) | 1293/1319 (98.03%) | 0/0 | 0/0 | 0/0 |
| GPQA-Diamond, two reps | 363/396 (91.67%) | 363/396 (91.67%) | 0/0 | 0/1 | 0/3 |
| LongBench v2 cohort | 30/48 (62.50%) | 34/48 (70.83%) | 0/0 | 0/0 | 0/0 |

**This is not a blanket clean-accuracy pass.** PrimTS GPQA samples r1:165,
r2:80 and r2:128 reach the 131,072-token cap. Sample 80 has no parseable
choice. The evaluator extracts a correct choice from sample 165's unfinished
reasoning, so counting every truncation as incorrect changes PrimTS GPQA to
362/396 (91.41%), versus Triton's unchanged 363/396 (91.67%). Sample 128 is
incorrect on both backends; Triton stops after 118,650 tokens.

Paired prompt hashes match. PrimTS has 6 regressions/4 improvements on
GSM8K, 5/6 and 5/4 on the two GPQA repetitions, and 1/5 on LongBench.
Backend numerical/sampling differences can change long reasoning paths;
these results neither prove equivalence nor identify a kernel fault. No
CUDA error accompanies the three truncated requests. The unfinished-answer
and separately deferred startup-fault caveats remain visible.

Same-backend repetitions also vary: Triton scores 180 then 183 (11 changed
choices), PrimTS 181 then 182 (7 changed choices), despite identical seed-42
prompts. This contextualizes the small score differences; it does not clear
the three unfinished responses or prove statistical equivalence.

Sampler: temperature 0.6, top_p 0.95, top_k 20, seed 42, max_tokens 131072,
reasoning_effort xhigh, n=1, stream=false. GSM8K and GPQA concurrency 64;
LongBench concurrency 8. GPQA is 198 questions times two complete
repetitions with the same sampler, not best-of-two. LongBench is the existing
untruncated 48-sample cohort around 8K/16K/32K, not the full benchmark.
The model-length limit is 139264 for GSM8K/GPQA and 196608 for LongBench,
leaving room for the full generation budget without truncating its input.

Token lengths below are API counts, including reasoning in output. Each
cell is mean / median / nearest-rank P90 / min--max.

| Task / backend | Input tokens | Output tokens |
| --- | --- | --- |
| GSM8K / Triton | 789.0 / 785 / 820 / 751--916 | 490.2 / 358 / 738 / 122--14073 |
| GSM8K / PrimTS | 789.0 / 785 / 820 / 751--916 | 519.6 / 367 / 766 / 111--16324 |
| GPQA / Triton | 306.9 / 273.5 / 435 / 132--2835 | 13321.5 / 5470.5 / 38934 / 189--118650 |
| GPQA / PrimTS | 306.9 / 273.5 / 435 / 132--2835 | 13521.8 / 5538 / 40037 / 215--131072 |
| LongBench / Triton | 20655.1 / 16703.5 / 33856 / 10366--34370 | 2687.6 / 1385.5 / 6928 / 298--15284 |
| LongBench / PrimTS | 20655.1 / 16703.5 / 33856 / 10366--34370 | 2884.9 / 1368.5 / 9013 / 198--12881 |

The three 16-sample LongBench cohorts have mean input lengths 12596.6,
16504.6 and 32864.2 tokens. Their ranges are 10366--15674, 15698--17336 and
30753--34370; the "8K" cohort was not truncated to 8192 tokens.

## Pure-stage performance

Nsight Systems 2025.4.1, CUDA-graph node tracing, default workers and no
forced synchronization or sanitizers. These are warmed model traces, not
cold-L2 standalone results. Prefill measures the fourth request after three
warmups; prefix caching is disabled. Sparse metadata/attention stay eager
at the agreed piecewise graph boundary. Other eligible operations use
graphs. Decode uses full graphs and MTP3, with exactly 64 resident requests
and no prefill work in the measured windows.
Performance requests use exact-length synthetic prompts, temperature zero
and ignore_eos. They are separate from the reasoning-accuracy sampler above.

The sparse stage includes metadata/index expansion, attention and any
reduction. All-layer GPU wall excludes request queueing and API TTFT.
Speedup is Triton divided by PrimTS. Times are milliseconds; decode is per
scheduler iteration, including the target and three draft forwards, not per
accepted output token. These are single matched Nsight captures, not a
multi-run confidence interval or a serving-throughput claim.

| Stage | KV dtype | Input/context | Actual BS | Sparse Triton | Sparse PrimTS | Speedup | All-layer Triton | All-layer PrimTS | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefill | BF16 | 8192 | 1 | 21.639 | 7.341 | 2.948x | 159.489 | 139.201 | 1.146x |
| Prefill | BF16 | 16384 | 1 | 42.710 | 17.486 | 2.443x | 283.425 | 257.761 | 1.100x |
| Prefill | FP8 | 8192 | 1 | 39.087 | 7.264 | 5.381x | 170.560 | 138.886 | 1.228x |
| Prefill | FP8 | 16384 | 1 | 76.519 | 17.161 | 4.459x | 318.740 | 259.246 | 1.229x |
| Decode MTP3 | BF16 | 8192 | 64 | 1.321 | 0.841 | 1.572x | 27.991 | 27.418 | 1.021x |
| Decode MTP3 | BF16 | 16384 | 64 | 1.323 | 0.891 | 1.484x | 28.048 | 27.637 | 1.015x |
| Decode MTP3 | FP8 | 8192 | 64 | 1.912 | 0.847 | 2.258x | 28.485 | 27.410 | 1.039x |
| Decode MTP3 | FP8 | 16384 | 64 | 1.917 | 0.920 | 2.084x | 28.455 | 27.585 | 1.032x |

Sparse absolute savings match all-layer savings within 0.45 ms for three
prefill pairs. BF16 8K additionally saves 6.21 ms of rank-local all-reduce
wait, which is not sparse compute. Both-rank accounting confirms essentially
identical attention work; its Triton GPU envelopes differ by 1.41 ms. Other
prefill pairs have less than 0.1 ms rank-envelope spread. PrimTS metadata
totals 0.81--0.83 ms at 8K and 1.60--1.61 ms at 16K across twelve layers.
No host copy of union metadata appears. TensorMap updates are small.
The final table uses sparse-activity interval unions in both stages. This
removes approximately 4--5 microseconds of prefill metadata/attention overlap
from earlier component-sum summaries; the underlying traces are unchanged.

Every decode trace contains 33 complete scheduler iterations, four graph
launches and fifteen sparse calls per iteration (twelve target, three MTP
draft). Both ranks schedule 256 query tokens for 64 requests, with zero
context/prefill tokens. Graph-node counts match within each pair: 68,706
BF16 and 69,201 FP8 per rank. The largest full-window TP-rank wall spread
is 0.0504 ms. Client stream chunks cross the profiler endpoint acknowledgement
boundary; kernel/graph counts independently reconcile the 33-step denominator.

FP8 8K's 1.065-ms sparse saving matches the 1.074-ms all-layer saving
within 0.010 ms. Idle within the GPU step envelope is 3.595 versus 3.602 ms;
other kernels are 22.900 versus 22.880 ms. There is no additional PrimTS
host-gap penalty in this pair. Decode metadata is 0.117 ms versus Triton's
0.044-ms expansion, included in the comparison. Pipeline times use interval
unions so PDL overlap is not double-counted.
BF16 8K saves 0.481 ms in the sparse pipeline and 0.572 ms across all
layers. The additional 0.091 ms is mostly a 0.074-ms idle-gap reduction,
not extra sparse-kernel savings. Its other kernels differ by only 0.017 ms.

At 16K, FP8 saves 0.997 ms sparse versus 0.871 ms all-layer: PrimTS has
0.226 ms more idle but 0.106 ms less other-kernel time, plus small copy/overlap
differences. BF16 saves 0.431 ms sparse versus 0.411 ms all-layer: 0.261 ms
more other-kernel time is mostly offset by 0.239 ms less idle. Do not attribute
these residuals to attention. There is no consistent additional PrimTS
host-gap penalty across the four pairs; the single captures do not prove
zero host overhead or its statistical equivalence.

Decode component durations below are raw kernel sums in ms per step.
They must not be added to reconstruct the stage table: PDL allows overlap
between the core, reducer, and metadata, which that table removes by interval
union. Each component executes fifteen times per step.

| KV / context | Backend | Metadata / expansion | Attention core | Reduction |
| --- | --- | ---: | ---: | ---: |
| BF16 / 8K | Triton | 0.043 | 1.192 | 0.088 |
| BF16 / 8K | PrimTS | 0.115 | 0.692 | 0.047 |
| BF16 / 16K | Triton | 0.044 | 1.193 | 0.089 |
| BF16 / 16K | PrimTS | 0.129 | 0.729 | 0.047 |
| FP8 / 8K | Triton | 0.044 | 1.789 | 0.081 |
| FP8 / 8K | PrimTS | 0.117 | 0.697 | 0.045 |
| FP8 / 16K | Triton | 0.044 | 1.793 | 0.082 |
| FP8 / 16K | PrimTS | 0.130 | 0.759 | 0.045 |

Some traces include 18--19 ms/step of CPU event waits on a worker thread;
92--93% of that time overlaps GPU kernels. This also appears in Triton's
BF16 16K trace. Summed CPU synchronization durations are not additional
wall time and must not be added to the GPU envelope. Decode has no TensorMap
updates or host copy of the sparse union in capture.

The input columns are exact initial prompt lengths. During steady decode,
per-step mean live KV grows through roughly 8311--8429 tokens for 8K and
16625--16753 for 16K. These are ranges of per-step means, not individual
request minima/maxima. Backend samples need not accept identical draft
tokens; the measured scheduler work and route shape are matched.

Prefill max_model_len=139264, max_num_seqs=64, exact input-sized piecewise
captures, MTP0 and one generated token. Decode max_model_len=32768,
max_num_batched_tokens=8192, max_num_seqs=64, graph token sizes
4/8/16/32/64/128/256, GPU memory fraction 0.91, MTP3 and 2048 output tokens
reserved per request. Compile-prime reuse is keyed by source/configuration;
every timed decode server starts fresh after that prime.

### Decode residency

Requested BS256 is capacity-limited under this TP2 model configuration.
Both FP8 8K trials reach about 121 running requests with 99.4% cache usage;
the remaining requests wait. The client rejects these runs before capture
because early requests finish before all 256 reach the profiling barrier.
No BS256 latency is claimed. The fallback measurements use BS64 on both
backends, with no request preemption during capture.

vLLM's displayed equivalent token capacity cannot be scaled linearly for
this hybrid model. For each actual context, compute whole per-request
blocks for every cache group and sum them:

`resident_batch <= (num_pool_blocks - 1) // sum(ceil(group_request_bytes / group_page_bytes))`.

This reserves the null block and includes recurrent/speculative state and
attention-page rounding.
For the observed FP8 cache geometry, four recurrent groups use four blocks
each and the circular buffer uses one, giving `17 + ceil(context/3200)`
blocks/request. The BS256 attempt's 2436-block pool therefore fits only
`(2436-1)//20 = 121` requests at 8K, before reserving generated tokens.
The local profiling overlay audits the final scheduler configuration only
at startup; it changes no real allocation or runtime policy. Compilation is
warmed and the server restarted before measuring the final cache budget.
BF16 uses the same seventeen fixed blocks with 1600-token attention blocks.
All eight BS64 runs pass the exact group-aware bound including output reserve
and lookahead. This is a validated fallback, not the requested BS256 result.

## Memory and correctness gates

Compatible ordered layers share stable-address model-scoped arenas. Plans
and K/V bindings remain per-layer; unlike geometries and live split counters
remain disjoint. The measured sparse storage drops from 249.323 to
19.179 MiB per rank, a 92.31% saving. This does not include model weights,
KV cache or graph-private allocations. A separate approximately 47.68-GiB
cold-start PLE autotuning allocation occurs on both backends; warm/restart
avoids treating it as steady-state cache consumption.

That storage inventory uses max_model_len=32768, max_num_batched_tokens=8192,
max_num_seqs=32 and graph token counts 4/8/16/32/64/128. Twelve target sparse
layers plus one MTP layer previously retained 13 copies of 20,110,336 bytes;
they now share one set of eleven geometry-specific arenas. Larger capture
ladders have different totals; 19.179 MiB is not a universal workspace size.

Final affected gates pass: 88 vLLM reference/integration, 7 graph-routing,
2 MTP metadata, 141 FlashInfer metadata and 8 route-policy cases. Earlier
related kernel tests, the public example, and twelve shared-layer replays
under memcheck/initcheck also pass. The padding regression fails before the
fix. Q5 and packed/automatic-G1 runner smoke checks pass after tool cleanup.
Full-feature FlashInfer hooks and the relevant vLLM lint/type/Markdown hooks
pass; this is not an all-project test or all-files pre-commit claim.

## Limits

The intermittent full-model startup illegal address remains open and
explicitly deferred. It also reproduced with private per-layer storage;
shared-buffer ownership is not an established cause. Successful starts and
the independent membership-padding fix do not prove it resolved. Preserve
its failed-run evidence separately from successful qualification results.

Small-route standalone regressions remain, including BS1 and BF16 BS8/SQ4.
Do not claim all decode cases improve. Standalone tables retain their
cold-L2 CUDA-graph method and separate source fingerprints.

## Reproduction and retained evidence

Use [the validation guide](README.md) for the maintained runners.
The local campaign records retain raw responses, prompt/source hashes,
startup/failure logs, cache-group inventories, client counters, Nsight reports,
SQLite exports and both-rank analysis. They are intentionally not checked into
the product source tree:

- `qsa_accuracy/final-head-20260909/RESULTS.md`: six completed jobs, manifest
  verification, paired outcomes, strict truncation scores and token lengths.
- `qsa_nsys/final_head_job688094/STAGE_TABLE.md`: the eight paired stage points
  and full accounting; `decode-stage-audit.json` separates core/reducer sums
  and CPU-wait overlap. Requested-size failures remain under `decode/`;
  resident captures are under `decode-feasible/`.
- `qsa_bench/vllm_workspace_job688094/final-regressions/`: the final affected
  test gates, with hook and evaluator-compatibility logs in its parent.
- `qsa_bench/vllm_workspace_job684987/SUMMARY.md`: both canonical cold-L2
  standalone suites on FlashInfer c79e80ae / vLLM 53c7e370e; later changes
  are qualified separately, not relabeled as that measurement's sources.
- `qsa_bench/DEFERRED_ISSUES.md`: preserved startup and unfinished-answer
  evidence, including the limitations of successful isolated replays.

Accuracy used immutable source snapshots. Timed decode used FlashInfer
d138709e / vLLM 8a31f0db4 and the preserved pre-existing, uncommitted
23-line early-finish guard in `profile_steady_decode.py` (Git blob
`9bf84f75d556bb1e6e0466f19a6910a0971983c8`). That guard rejects incomplete
resident batches; it changes no inference kernel and is not included in this
documentation commit. Prefill began before the review-only typing cleanup;
its final FP8 8K case includes that equivalent cleanup. No runtime change was
made in response to these timings. The final follow-up adds documentation only.
