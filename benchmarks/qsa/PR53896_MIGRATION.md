# QSA PrimTS migration to vLLM PR 53896

## Revisions

- vLLM base: `13c80fb30ab835cbe387c01c4611970b7c3373e1`, the fetched
  `refs/pull/53896/head` on 2026-08-30.
- Source integration: local branch `qsa-prims-ts-pr53896`; the FP8 accuracy
  gate used implementation revision `8dd467f74`. Each production port commit
  carries its original `Cherry-picked-from` revision.
- FlashInfer kernel branch: `qsa-page4-prims-ts` at `8fa4396a`.
- Accuracy model: [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/main)
  revision `de4b8e4`, staged locally as `models/Qwen3.8-Flash-Next-de4b8e4`.
  The model config identifies the implementation as `qwen4_exp`, uses 24 Q
  heads, two KV heads, head dimension 256, indexer budget 2048, compression
  ratio four, and one hybrid MTP layer.

PR 53896 renamed the implementation from `qwen3_8_flash_next` to
`qwen4_exp`. The port therefore maps the local NVIDIA changes onto
`vllm/models/qwen4_exp/nvidia` and preserves the PR's newer combined-cache
layout, pre-indexer, MTP, PLE, and multimodal changes.

## Port structure

The migration is split into reviewable commits:

1. page-4 CSR metadata and PrimTS attention routing;
2. grouped Q2/Q4 union metadata and production routing;
3. KV128/TileQ64 policy, with the Q4 cap owned by FlashInfer;
4. explicit `auto`/`triton`/`prims_ts` backend selection;
5. BF16 and FP8-E4M3 model-facing cache paths.

The physical cache is a combined
`[page, Hkv, storage_page_size, 2 * head_dim]` tensor in PR 53896. Its common
transpose and final-dimension split produce native Triton's
`[page, storage_page_size, Hkv, head_dim]` K and V views. PrimTS applies one
additional zero-copy axis transpose to obtain
`[page, Hkv, storage_page_size, head_dim]`. This is the main layout adaptation
relative to the old checkout.

## Qualification order

1. Run native Triton with BF16 weights/cache and MTP disabled. This is the
   reference gate for GSM8K, GPQA-Diamond, and AIME26.
2. Run PrimTS with the identical BF16 MTP=0 protocol and compare per-item
   predictions and completion hashes against Triton.
3. Run a small, fixed native-Triton MTP=3 gate. If BF16 MTP=3 is still
   unstable on this PR, record the failure and skip the full MTP=3 matrix.
4. Repeat the passing MTP=0 protocol with FP8-E4M3 KV cache.
5. Resume decode performance work only after the accuracy gate passes.

Every server run must record the vLLM, FlashInfer, and model revisions; TP;
KV-cache dtype; MTP setting; backend; launch command; evaluator arguments; and
raw output directory. Fresh-server first-request smokes precede full runs.

## Current validation

- Production files pass Python bytecode compilation in the local vLLM
  environment.
- The complete QSA reference suite passes in the CUDA development container:
  `53 passed` in
  `tests/models/qwen4_exp/test_qsa_reference.py`. This covers the native
  Triton metadata/indexer/attention path, PR-53896 combined-cache adaptation,
  backend selection and PrimTS wrappers, and PrimTS page-4 Q1/Q2/Q4 metadata.
- PR 53896 added `num_reqs` to common attention metadata. The migrated
  metadata-builder fixtures now set the live request count explicitly, rather
  than treating CUDA-graph padding rows as live requests.
- Set `TRITON_CACHE_DIR` to a workspace-backed directory for CUDA tests and
  model runs. The container user's default Triton cache has a small disk quota
  and otherwise fails compilation before a kernel can run.
- The Qwen runtime image carries CUTLASS DSL 4.6.2, while the current PrimTS
  task-scheduling implementation requires `cutlass.experimental.cuda` from
  CUTLASS DSL 4.7. The runtime qualification uses 4.7.1 from a workspace-local
  wheel target and exposes only its `nvidia_cutlass_dsl/dsl_packages`
  directory. This avoids shadowing the image's NumPy, protobuf, CUDA Python,
  and other runtime dependencies.
- `benchmarks/qsa/runtime_overlay/sitecustomize.py` keeps the image's
  ABI-matched FlashInfer root, GDN, fused-MoE, cubins, and vLLM extensions. It
  overlays only the local FlashInfer attention package and attention trace
  templates, then exports the three page-4 PrimTS APIs through
  `flashinfer.decode`. Set `QSA_FLASHINFER_SOURCE` and
  `QSA_CUTLASS_DSL_PACKAGES` to enable these two narrow overlays.
- A cold-L2 eager decode smoke at TP=2, BS=1, SQ=1, pre-sparse KV=8192 ran the
  real compiled PrimTS kernel successfully. Its maximum difference from the
  reference was 0.00024. This is a runtime/import gate, not an accepted
  performance measurement (one warmup and one measured iteration).

## Native Triton BF16/MTP=0 reference

The first PR-53896 model-level reference completed on 2026-08-30 with vLLM
`cfe1c01dd`, FlashInfer `0.6.17` from the dedicated
`vllm/vllm-openai:qwen38-flash-next` image, model revision `de4b8e4`, TP=2,
BF16 weights/KV cache, and MTP disabled. The server used the explicit
`VLLM_QSA_ATTENTION_BACKEND=triton` route, greedy sampling, seed 42, and the
model's chat template. All truncations and invalid final-answer parses remain
in the denominator.

| Task | Score | Truncated | Request errors | Completion tokens |
|---|---:|---:|---:|---:|
| GSM8K, 5-shot | 1284/1319 (97.35%) | 12 | 0 | 652,236 |
| GPQA-Diamond | 146/198 (73.74%) | 48 | 0 | 1,426,628 |
| AIME 2026 | 17/30 (56.67%) | 13 | 0 | 314,804 |

Raw per-item outputs are under
`qsa_accuracy/pr53896/triton-bf16-mtp0/` in the workspace. The evaluator
records dataset hashes, full raw generations, parsed predictions, run
metadata, and elapsed time. The server log is
`server-official.log` in the same directory.

Historical figures in `QSA_PRIMS_TS.md` come from the prior model branch and
are not mixed with this reference. The next accepted comparison is PrimTS
BF16/MTP=0 with the identical model snapshot and evaluation protocol.

## PrimTS BF16/MTP=0 accuracy gate

The matching PrimTS run completed with vLLM `a17e08919`, FlashInfer
`8fa4396a`, CUTLASS DSL 4.7.1, model revision `de4b8e4`, and the same TP=2,
greedy chat, seed, prompts, and generation limits as the Triton reference.
The first full-model startup exposed and fixed the PR-53896 cache-axis
adaptation described above; the failed attempt is retained as
`server-official-r1-cache-layout-failure.log`.

| Task | Triton | PrimTS | PrimTS truncations | PrimTS errors |
|---|---:|---:|---:|---:|
| GSM8K, 5-shot | 1284/1319 (97.35%) | 1280/1319 (97.04%) | 12 | 0 |
| GPQA-Diamond | 146/198 (73.74%) | 148/198 (74.75%) | 46 | 0 |
| AIME 2026 | 17/30 (56.67%) | 21/30 (70.00%) | 9 | 0 |

The item-level audit found 18/8/6 prediction changes and 9/2/1 regressions
versus 5/4/5 improvements on GSM8K/GPQA/AIME, respectively. Raw generation
hashes changed for 975/157/22 items. This is expected autoregressive
sensitivity to a numerically different BF16 attention implementation rather
than evidence of a systematic accuracy drop: the three tasks have mixed
directions, the total correct count is two higher, total truncations fall from
73 to 67, and neither run has request errors. PrimTS BF16/MTP=0 therefore
passes the model-level accuracy gate.

The evaluator wall times are not accepted kernel-performance results. The two
servers exposed different usable KV-cache capacities, and the PrimTS smoke
triggered first-inference Triton/CuTeDSL compilation. Decode performance must
continue to use the standalone cold-L2 CUDA-graph protocol.

Raw PrimTS artifacts are under
`qsa_accuracy/pr53896/prims-ts-bf16-mtp0/` in the workspace. The next accuracy
step is FP8-E4M3 MTP=0, following the blocked MTP=3 gate below.

## Native Triton BF16/MTP=3 gate

The small native-Triton MTP=3 gate did not reach server readiness. During
CUDA-graph capture, the image's compiled FlashInfer extension rejected the
PR-53896 GDN call:

```text
_C::fused_gdn_decode_post_conv_mtp() expected at most 14 argument(s)
but received 15 argument(s)
```

This happens on the native Triton route before any QSA inference request, so
it is a baseline image/source ABI mismatch rather than a PrimTS QSA result.
The image exposes the older 14-argument fused-GDN op while PR 53896 invokes
the newer 15-argument interface. Per the qualification policy, the full BF16
MTP=3 matrix and matching PrimTS MTP=3 gate are skipped for now. The complete
failure is retained at
`qsa_accuracy/pr53896/triton-bf16-mtp3-gate/server-official.log`.

## Native Triton FP8-E4M3/MTP=0 reference

The matching native-Triton FP8 KV-cache reference completed with vLLM
`b4a79ab08`, image FlashInfer 0.6.17, model revision `de4b8e4`, TP=2, and
`--kv-cache-dtype fp8_e4m3`. Prompts, greedy seed, generation limits,
concurrency, and task order were unchanged from the BF16 reference.

| Task | BF16 Triton | FP8 Triton | FP8 truncations | FP8 errors |
|---|---:|---:|---:|---:|
| GSM8K, 5-shot | 1284/1319 (97.35%) | 1288/1319 (97.65%) | 8 | 0 |
| GPQA-Diamond | 146/198 (73.74%) | 151/198 (76.26%) | 44 | 0 |
| AIME 2026 | 17/30 (56.67%) | 19/30 (63.33%) | 11 | 0 |

Across the three tasks, FP8 Triton has nine more correct answers and ten fewer
truncations than BF16 Triton. This rules out an aggregate FP8 accuracy drop in
the native baseline, although the 30-item AIME result remains statistically
noisy. Raw artifacts and the server log are under
`qsa_accuracy/pr53896/triton-fp8-mtp0/`.

## PrimTS FP8/MTP=0 serving diagnosis

The full PrimTS FP8 accuracy row is not yet qualified. Two full GSM8K attempts
were stopped rather than recording partial scores: after an initially fast
64-request warmup, continuous batching repeatedly spent ten-second logging
windows producing zero or only a few tokens. The server did not report a CUDA
error, and completed requests remained numerically valid. Logs and incomplete
artifacts are retained under `qsa_accuracy/pr53896/prims-ts-fp8-mtp0/`.

The failure is not reproduced by a single attention launch. The new
`benchmark_qsa_capture.py` diagnostic replays saved model tensors, current
PR-53896 metadata, cold L2, and CUDA graphs. On GB300/TP=2 it measured:

| Captured workload | PrimTS | metadata + PrimTS | Triton | Result |
|---|---:|---:|---:|---|
| one row, 768 visible tokens | 64.07 us | 66.17 us | 13.87 us | exact saved PrimTS output |
| 64 real rows, positions 703--766 | 56.13 us | 57.66 us | 42.89 us | max difference 0.015625 vs Triton |
| 767-row real prefill prefix | 200.47 us | 201.56 us | 408.69 us | max difference 0.0625 vs Triton |
| one real row + 63 inert graph rows | 49.02 us | 50.97 us | 43.49 us | exact live-row output |

Thirty-two consecutive PrimTS launches captured in one CUDA graph were also
stable (28.11 us per warm-cache attention call). These controls rule out the
real top-k order, variable causal lengths, inert graph padding, and ordinary
single-device graph replay as the source of the multi-second stalls.

The serving evidence instead points to exact-batch specialization during
continuous mixed prefill/decode. PrimTS includes `batch_size` in its semantic
compile key. Startup compiled 51 capture sizes independently, taking 12.56
seconds per size on the first process. A 64-request warmup has one stable shape
and reached 1,318 generated tokens/s after first-use compilation. The full
1,319-request evaluation continuously replaces finished requests with prompt
chunks, producing shapes outside the decode capture list (which stops at 512).
Those shapes enter the same process-local CuTeDSL compile path; the JIT monitor
deduplicates warnings by kernel name, so later specializations do not produce
one warning per batch value. This explains the alternating zero-throughput
windows and short fast intervals and is the current best-supported root cause.

Two controls support that conclusion:

- Eager TP=2 finished 2,048 FP8 PrimTS tokens correctly but needed 64.72
  seconds (31.6 tokens/s), showing that falling off the compiled model path has
  the same throughput scale as the stalled intervals.
- A server restricted to one 64-row CUDA-graph bucket completed 24,395 tokens
  across 64 GSM8K requests in 71.71 seconds including first-use warmup. Its
  steady generation interval reached 2,430.5 tokens/s and had no stalls. The
  64-question diagnostic scored 46/64 with 18 max-512 truncations; it is a
  serving-performance control, not a replacement accuracy result.

The vLLM workspace owner now follows the standalone FlashInfer contract and
allocates uninitialized output-only scratch. It no longer records a complete
workspace memset whenever the semantic shape changes. This removes avoidable
graph/eager traffic but is not by itself claimed to solve exact-batch JIT.

The production fix should separate the compiled batch *capacity* from the live
row count: select a small set of PrimTS capacity buckets, pass the live count to
the kernel, and make rows in `[live, capacity)` inert without reading Q/K/V.
An alternative vLLM adapter can stage Q, metadata, and O through bucket-sized
buffers, but that adds copies and large per-layer buffers. Until dynamic
capacity is implemented, a safe integration fallback is native Triton for
unwarmed mixed-prefill shapes while retaining PrimTS for qualified decode
buckets. Full PrimTS FP8 GSM8K/GPQA/AIME remains blocked on this fix.

The first implementation uses the vLLM adapter option so it does not change
FlashInfer's existing public interface. Live row counts round up to bounded
power-of-two capacities. Each QSA owner keeps graph-stable Q/O and request
metadata staging buffers; padded rows receive position `-1`, request zero, and
zero Q, reusing the already-qualified inert-row metadata path. Exact bucket
sizes stay zero-copy. A three-live-row/four-capacity owner smoke passes on CPU.

A fresh TP=2 FP8 server on job 619973 validated the bucket reuse behavior. The
first 51-shape PrimTS capture pass fell from about 10 minutes 40 seconds before
the fix to 156 seconds; later capture passes reused the compiled buckets. A
256-question GSM8K refill control at concurrency 64 and max output 512
completed all requests with zero API errors: 96,401 completion tokens in
135.26 seconds, including first-use runtime compilation. Steady generation
windows recovered to 1.3--1.9k tokens/s instead of repeatedly remaining at
zero. The control artifact is
`qsa_accuracy/pr53896/prims-ts-fp8-mtp0/bucketed-refill256x512.json`.

The refill control is evidence for serving progress and is not used as an
accuracy result.

## PrimTS FP8-E4M3/MTP=0 accuracy gate

The full bucketed PrimTS matrix completed on the same TP=2 server with vLLM
`8dd467f74`, FlashInfer `8fa4396a`, CUTLASS DSL 4.7.1, model revision
`de4b8e4`, greedy sampling, and seed 42. Prompts, token limits, concurrency,
and task order match the native-Triton FP8 reference.

| Task | FP8 Triton | FP8 PrimTS | PrimTS truncations | PrimTS errors |
|---|---:|---:|---:|---:|
| GSM8K, 5-shot | 1288/1319 (97.65%) | 1287/1319 (97.57%) | 6 | 0 |
| GPQA-Diamond | 151/198 (76.26%) | 154/198 (77.78%) | 39 | 0 |
| AIME 2026 | 19/30 (63.33%) | 18/30 (60.00%) | 12 | 0 |

Across all three tasks, PrimTS has 1,459 correct answers versus 1,458 for
Triton, 57 truncations versus 63, and zero request errors in either run. The
item-level audit found 11/15/3 prediction changes and 5/5/2 regressions versus
4/8/1 improvements on GSM8K/GPQA/AIME, respectively. Raw generation hashes
changed for 1,020/164/20 items. The mixed directions and essentially identical
aggregate score are consistent with autoregressive sensitivity to the
numerically different attention implementation, not a systematic FP8 accuracy
drop. PrimTS FP8-E4M3/MTP=0 therefore passes the model-level accuracy gate.

The PrimTS evaluator wall times were 410.10/526.86/239.53 seconds for
GSM8K/GPQA/AIME. These include model serving and occupancy-dependent tails and
are not accepted kernel-performance measurements. Raw outputs are under
`qsa_accuracy/pr53896/prims-ts-fp8-mtp0/`; the matching server log is
`server-bucketed-r1.log` in that directory.
