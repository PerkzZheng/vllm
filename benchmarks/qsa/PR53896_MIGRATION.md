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

For new E2E accuracy and benchmark runs, use the following generation policy
for both Triton and PrimTS. Earlier greedy tables in this document remain
historical checkpoints and must not be mixed with results from this protocol:

```text
temperature=0.6
top_p=0.95
top_k=20
seed=42
max_tokens=131072
reasoning_effort="xhigh"
n=1
stream=false
```

The server context limit must cover the complete prompt plus the 131,072-token
generation allowance. Smoke tests may use a smaller token cap, but their
outputs are ABI/kernel gates rather than benchmark scores.

## LongBench v2 long-context gate

GSM8K, GPQA-Diamond, and AIME26 have short prompts for this model: their mean
rendered input lengths are about 789, 307, and 215 tokens, respectively. They
exercise long autoregressive decode, but they do not directly qualify QSA on a
long prefill. LongBench v2 is therefore an additional paired Triton/PrimTS
gate; it does not replace or interrupt the three-task matrix.

The gate uses the official THUDM LongBench v2 direct-answer prompt and answer
extractor semantics. The source repository is pinned locally at
`THUDM/LongBench@2e00731f8d0bff23dc4325161044d0ed8af94c1e`. The Hugging Face
raw JSON is pinned by SHA256
`15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2`;
its generated parquet equivalent is
`cee77345f57cd74e5ca43e643caef34115e4592f1d5a4051ffd164bdf3ab2c34`.
LongBench's published `short`/`medium`/`long` classes are word-count ranges
(`<32K`, `32K--128K`, and `>128K`), so they are not used as model sequence
lengths. Instead, `qsa_reasoning_eval.py` renders the official prompt, applies
the exact local Qwen chat template, and selects unique examples near 8,192,
16,384, and 32,768 input tokens.

Selection is deterministic at seed 42. The gate requests 16 complete,
unmodified examples nearest each target. LongBench v2 rejects source documents
below 8,192 words, and only seven natural Qwen chat prompts fall within 50
percent of 8,192 tokens. The nominal 8K bucket therefore uses the 16 shortest
eligible prompts within a two-times upper bound and reports their exact range;
the 16K and 32K buckets still choose the nearest unused examples. No context is
truncated or padded. The resulting 48-example manifest has these rendered
input lengths:

| Nominal target | Examples | Minimum | Mean | Median | Maximum |
|---:|---:|---:|---:|---:|---:|
| 8K | 16 | 10,366 | 12,596.6 | 12,729 | 15,674 |
| 16K | 16 | 15,698 | 16,504.6 | 16,703.5 | 17,336 |
| 32K | 16 | 30,753 | 32,864.2 | 33,527.5 | 34,370 |

The selector fails closed if a bucket has too few natural examples. A
prepare-only manifest freezes IDs, prompt hashes, labels, target buckets, and
exact tokenizer input lengths. Every backend and KV/MTP configuration consumes
that same manifest. API-reported prompt-token counts are retained per request
to detect any offline/server template drift. A second prepare-only pass using
the manifest reproduced all 48 prompt hashes and token counts.

The reviewed manifest is committed at
`benchmarks/qsa/manifests/longbench_v2_qwen38_8k16k32k_seed42.json` with
SHA256 `651d887459b8e1a86f2ee3d93f3b54bd4e3755411decdb51f32f0d56281939c4`.
Pass it through `--longbench-manifest` for every scored run.

```bash
python benchmarks/qsa/qsa_reasoning_eval.py \
  --task longbench-v2 \
  --prepare-only \
  --run-name longbench-v2-8k-16k-32k-seed42 \
  --longbench-dataset /workspace/qwen_next/datasets/longbench-v2/data.json \
  --tokenizer /workspace/qwen_next/models/Qwen3.8-Flash-Next-de4b8e4 \
  --output /workspace/qwen_next/qsa_accuracy/pr53896/longbench-v2/manifest.json
```

This is a kernel-correctness subset rather than an official 503-example
leaderboard score. Accuracy, invalid answers, request errors, per-example
Triton/PrimTS agreement, and input/output/total sequence distributions are
reported separately for each token bucket. Generation uses the same exact
sampling policy as the three-task matrix. The server must use
`--max-model-len 196608` or greater: the largest frozen prompt plus the
131,072-token generation allowance is 165,442 tokens. The three-task servers
used 139,264 and are not restarted or repurposed for the full LongBench run.

The existing 139,264-token servers were retained and used only for a
three-item smoke with `max_tokens=1024`, choosing one frozen item per bucket.
All three backends reported prompt-token counts exactly equal to the offline
manifest and completed the 32,713-token item correctly:

| Backend/cache/MTP | Prompts | Request errors | Correct | Invalid | Truncated |
|---|---:|---:|---:|---:|---:|
| Triton, FP8, MTP=3 | 3 | 0 | 2 | 0 | 1 |
| PrimTS, BF16, MTP=0 | 3 | 0 | 1 | 1 | 1 |
| PrimTS, FP8, MTP=0 | 3 | 0 | 2 | 1 | 1 |

The invalid answers are outputs that reached the deliberately reduced smoke
cap, not API or kernel failures. A diagnostic attempt to use the required
131,072-token cap on the old server was rejected before inference because the
10,366--34,370-token prompts exceed its remaining context allowance. Its 48
HTTP-400 responses are retained for diagnosis and excluded from accuracy.

## Published-table reproduction gate

PrimTS is paused until native Triton reproduces the supplied accuracy table.
The first exact-policy BF16/MTP=0 run on 2026-08-30 did not reproduce its
GSM8K row:

| Configuration | GSM8K score | Difference from supplied row | Errors | Invalid | Truncated |
|---|---:|---:|---:|---:|---:|
| supplied BF16 row, job 2780006 | 1273/1319 (96.51%) | -- | not available | not available | not available |
| local native Triton, BF16, MTP=0 | 1292/1319 (97.95%) | +19 answers (+1.44 pp) | 0 | 0 | 0 |

The local run used TP=2, model revision `de4b8e4`, vLLM/evaluator
`8a8015c3d`, FlashInfer `7eacf585`, and the exact sampling policy above. It
completed 660,178 output tokens in 402.82 seconds. All 1,319 generations
contain a canonical GSM8K `#### N` answer marker. The initial generic
last-number parser reported 1291/1319 because one response included numeric
text after its answer marker; strict `####` rescoring corrects that one item.
The evaluator now prioritizes the canonical marker.

Raw output is retained at
`qsa_accuracy/pr53896/triton-bf16-mtp0-sampling-v2/gsm8k.json`, with the
server and evaluator logs beside it. This result is not evidence of a Triton
accuracy problem--it is higher than the supplied row--but it is a failed
protocol-reproduction gate. The exact published prompt builder, scorer,
repetition convention, TP/batching topology, and job artifacts are not
available locally; the internal runbook repository cannot be fetched without
GitLab credentials. Do not start the PrimTS accuracy matrix from this baseline
until that protocol difference is resolved or an explicit paired-run protocol
is accepted.

## Exact-policy paired matrix in progress

The user accepted the local native-Triton rows as the paired reference and
requested the matching PrimTS matrix. The table below uses only the exact
sampling policy in this document. GSM8K has 1,319 examples; GPQA-Diamond is
reported over two 198-example repetitions; AIME26 is reported over two
30-example repetitions. These rows must not be mixed with the historical
greedy gates later in this document.

| Backend | KV cache | MTP | GSM8K | GPQA-Diamond, 2 reps | AIME26, 2 reps | Slurm job |
|---|---|---:|---:|---:|---:|---:|
| Triton | BF16 | 0 | 1292/1319 (97.95%) | 362/396 (91.41%) | 55/60 (91.67%) | 622875 |
| Triton | BF16 | 3 | 1294/1319 (98.10%) | 363/396 (91.67%) | 59/60 (98.33%) | 622875 |
| Triton | FP8-E4M3 | 0 | 1290/1319 (97.80%) | 368/396 (92.93%) | 60/60 (100.00%) | 622875 |
| Triton | FP8-E4M3 | 3 | 1290/1319 (97.80%) | 363/396 (91.67%) | 60/60 (100.00%) | 622875 |
| PrimTS | BF16 | 0 | 1288/1319 (97.65%) | 361/396 (91.16%) | 56/60 (93.33%) | 623062 |
| PrimTS | BF16 | 3 | 1293/1319 (98.03%) | 367/396 (92.68%) | 60/60 (100.00%) | 623063 |
| PrimTS | FP8-E4M3 | 0 | 1288/1319 (97.65%) | 357/396 (90.15%) | 60/60 (100.00%) | 623064 |
| PrimTS | FP8-E4M3 | 3 | pending | pending | pending | 623068 |

The completed BF16/MTP=3 AIME repetitions each scored 30/30. Across both
repetitions there were zero request errors, invalid parses, or truncations.
The FP8/MTP=3 allocation landed on a node without the private BrightDelta
image cache; its registry import returned HTTP 403. The allocation is retained
while a compatible local-image ABI path is qualified. This is an
infrastructure/runtime-image issue, not a QSA result.

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
step is FP8-E4M3 MTP=0, following the MTP=3 gate below.

## BF16/MTP=3 GDN ABI repair and backend gates

The original native-Triton MTP=3 gate did not reach server readiness. During
CUDA-graph capture, the image's old vLLM stable extension rejected the
PR-53896 GDN call:

```text
_C::fused_gdn_decode_post_conv_mtp() expected at most 14 argument(s)
but received 15 argument(s)
```

This occurred before any QSA inference request and was a vLLM source/binary
ABI mismatch, not a FlashInfer Python/JIT or PrimTS failure. The old
`vllm/_C_stable_libtorch.abi3.so` exposed the 14-argument operator. PR 53896
calls the current 15-argument schema, whose final arguments are:

```text
float scale, float norm_eps, str output_gate_activation
```

The stable extension was rebuilt from the PR-53896 source against the local
Torch 2.13/CUDA 13 runtime and its `sm_100f` family target, then staged at
`vllm/_C_stable_libtorch.abi3.so`. `sm_100f` is the Blackwell family binary
used on the SM103 GB300 node. Local recoverable artifacts are:

| Artifact | SHA256 |
|---|---|
| `.runtime/gdn-abi-backup/_C_stable_libtorch.14arg.abi3.so` | `fbcb725abbec191f0e013bf6d37ea55d157f9b8183b401abf7f1cf77712e4c94` |
| `.runtime/gdn-abi-backup/_C_stable_libtorch.15arg.torch2.13.abi3.so` | `6a82c126726ab81b44a8fbec9a5652af54abe177dbcd88e0075b224dc4c625ad` |
| `.runtime/gdn-abi-backup/_C_stable_libtorch.15arg.torch2.13.glibc235.abi3.so` | `9417fd16c7e94e220dcf2cae74244452794d058aca145d441012ea3238da2538` |

The build tree is `.runtime/vllm-pr53896-gdn-build`. A system-Torch 2.8 build
attempt is retained separately in `.runtime/vllm-pr53896-gdn-system-build`;
it cannot build this target because Torch 2.8 lacks the required
`torch/csrc/stable` headers and is not a valid runtime match.

The private BrightDelta registry returned HTTP 403 on two later nodes that did
not already cache the image. A compatible local CUDA 13/Torch 2.13 image was
therefore exported to `.runtime/images/arf-mr5-unit-20260807.sqsh`. That image
uses glibc 2.35, while the first 15-argument extension required glibc 2.38.
The third artifact above was rebuilt against the image's system Torch 2.13 and
glibc 2.35, then staged atomically as `vllm/_C_stable_libtorch.abi3.so`. Its
operator schema retains all 15 arguments, including `output_gate_activation`.
It passed real TP=2 FP8/MTP=3 startup, CUDA-graph capture, and repeated draft
inference on both native Triton and PrimTS. Because it targets the older glibc
with the same Torch stable ABI, it also remains usable by the newer runtime.

All 16 focused cases in `tests/kernels/mamba/test_gdn_fused_mtp.py` pass with
the rebuilt extension. They cover pure fused MTP=3 for both `silu` and
`sigmoid`, plus mixed prefill/decode fallback cases. One first pass hit the
container user's default Triton-cache quota; the affected case passed with
`TRITON_CACHE_DIR` redirected to the workspace.

The repaired extension also passes actual TP=2 BF16/MTP=3 serving on both QSA
backends:

| Backend | Result | Repeated-MTP evidence |
|---|---|---|
| native Triton | Server healthy; `17 + 25` returned `42`; a 64-token completion returned HTTP 200 | MTP metrics reported accepted draft tokens; the short request had mean acceptance length 3.00 |
| local PrimTS | Server healthy; the same prompt returned `42`; a 64-token `xhigh` reasoning request with temperature 0.6/top-p 0.95/top-k 20/seed 42 returned HTTP 200 | 45/57 drafted tokens accepted, 78.9% draft acceptance, mean acceptance length 3.37 |

Both gates used model revision `de4b8e4`, BF16 KV, MTP=3, TP=2,
`max_model_len=32768`, and `max_num_seqs=128`. Triton used
`VLLM_QSA_ATTENTION_BACKEND=triton`. PrimTS used
`VLLM_QSA_ATTENTION_BACKEND=prims_ts` with
`QSA_FLASHINFER_SOURCE=/workspace/qwen_next/flashinfer`; runtime warnings from
the PrimTS task scheduler and the page-4 metadata JIT confirm that the local
attention overlay executed rather than the Triton sparse-attention kernel.

PrimTS graph profiling initially left only 0.42 GiB for KV at
`gpu_memory_utilization=0.90`, below the 0.89-GiB minimum for a 32K request.
This was a capacity check after successful kernel profiling, not a CUDA or
GDN failure. Raising the setting to 0.91 allowed the identical 32K gate to
start. The accepted server reported 48.78 GiB available KV after its final
profile; production runs should pin an explicit measured KV-cache allocation
instead of depending on this profiler-sensitive percentage.

Raw evidence is retained at:

- `qsa_accuracy/pr53896/triton-bf16-mtp3-gate/server-abi15-brightdelta-r3.log`
- `qsa_accuracy/pr53896/triton-bf16-mtp3-gate/request-abi15-brightdelta-long.json`
- `qsa_accuracy/pr53896/prims-ts-bf16-mtp3-gate/server-abi15-brightdelta-r2.log`
- `qsa_accuracy/pr53896/prims-ts-bf16-mtp3-gate/request-abi15-brightdelta-long.json`

The original ABI failure remains at
`qsa_accuracy/pr53896/triton-bf16-mtp3-gate/server-official.log`. MTP=3 is no
longer skipped for ABI reasons. The full Triton/PrimTS accuracy matrix remains
to be rerun with the new 131,072-token sampling protocol above; the smokes in
this section are validation gates, not task scores.

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

## Standalone decode performance checkpoint

The accepted decode protocol uses one SM103 GB300 GPU, a 258-MiB L2 eviction
before every measured replay, CUDA graphs, 10 warmup iterations, 100 measured
iterations, randomized causal top-k routes, an 8K source context, and compact
KV lengths of 2048 or 2051. The TP=2 model server remains on GPUs 0/1; all
standalone measurements use GPU 2 and do not disturb serving. Times below are
microseconds and ranges cover tail lengths zero and three.

The BF16 Q1 checkpoint is unchanged by the FP8 work:

| TP | BS | PrimTS | Triton | contiguous SWA2K |
|---:|---:|---:|---:|---:|
| 1 | 1 | 32.71--32.82 | 17.80--18.03 | 26.26--26.39 |
| 1 | 8 | 61.21--61.61 | 28.53--28.70 | 28.57--28.62 |
| 1 | 64 | 92.18--92.60 | 79.89--80.19 | 61.08--61.13 |
| 1 | 256 | 244.02--245.65 | 211.48--212.18 | 182.02 |
| 2 | 1 | 31.11--31.73 | 16.30--16.63 | 25.57--25.90 |
| 2 | 8 | 39.14--40.85 | about 24.67 | 26.82--28.15 |
| 2 | 64 | 59.53--60.08 | about 55.68 | 39.56--39.90 |
| 2 | 256 | 161.20--162.63 | 124.95--126.32 | 103.40--103.64 |

The FP8 audit found that production Q1 used one loader warp, while a separate
multi-warp `SmemKvResource` path replayed all 32 page-fragment TMAs from every
issuer because `elect_sync()` elects one lane per warp. The production fix
partitions fragments by load-warp rank and selects four loaders for low-grid
Q1 when `batch_size * num_heads_kv < 32`. Each warp issues eight page TMAs;
every page is fetched exactly once. The sparse recurrence, split-two separate
reducer, and output arithmetic are unchanged.

| TP | BS | FP8 PrimTS before | FP8 PrimTS after | Triton | contiguous SWA2K | max diff |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 45.28--48.83 | 23.69--26.60 | 16.74--16.95 | 18.09--18.21 | 0.00067 |
| 1 | 8 | 42.32--45.61 | 24.65--27.22 | 31.66--32.03 | 18.50--18.52 | 0.00070 |
| 2 | 1 | 46.85--49.91 | 24.34--26.52 | 16.17--16.26 | about 18.14 | 0.00058 |
| 2 | 8 | 42.16--45.53 | 24.46--27.13 | 23.02--23.49 | 18.28--18.45 | 0.00062 |

This is a 1.68--1.93x Q1 kernel speedup. TP1/BS8 is now faster than Triton and
TP2/BS8 is within 15.5 percent, but BS1 remains 42--63 percent slower than
Triton. Against the raw 2K contiguous kernel, Q1 remains 30--47 percent over
target, so the low-grid decode performance task is improved but not closed.

SQ4 must be compared with the exact grouped Q4 union, not with four flattened
Q1 launches or an unadjusted shared SWA window. The table below reports the
measured sparse union and metadata-inclusive time, Triton, grouped SWA2K, and
the grouped-SWA time projected by the measured union/SWA KV-volume ratio. The
synthetic randomized routes have 1.69--1.75x non-shared KV beyond the exactly
shared SWA window. For FP8, the grouped Keeps SWA proxy writes FP16 because
that profile does not support BF16 output; FP16 and BF16 both have two-byte
output traffic. The sparse and Triton paths still write model-facing BF16.

| TP | BS | PrimTS union | metadata + union | Triton | grouped SWA2K | projected SWA | union/projected | e2e/projected |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 28.51--28.66 | 33.35--33.40 | 22.52--22.63 | 26.51--26.69 | 71.88--73.00 | 0.393--0.397 | 0.458--0.464 |
| 1 | 8 | 34.74--34.78 | 39.20--39.73 | 49.71--50.17 | 28.34--28.51 | 77.71--77.73 | 0.447 | 0.504--0.511 |
| 1 | 64 | 118.47--119.18 | 124.02--124.60 | 268.84--268.97 | 42.69--43.03 | 116.77--117.62 | 1.007--1.021 | 1.054--1.067 |
| 1 | 256 | 352.12--352.98 | 359.97--360.96 | 1015.48--1015.55 | 123.51--123.91 | 337.51--338.58 | 1.043 | 1.066--1.067 |
| 2 | 1 | 27.69--28.05 | 32.84--32.90 | 18.05--18.25 | 26.61--26.63 | 71.93--72.03 | 0.385--0.389 | 0.456--0.457 |
| 2 | 8 | 28.84--28.94 | 34.78--34.82 | 41.11--42.14 | 27.97--28.31 | 76.56--77.17 | 0.374--0.378 | 0.451--0.455 |
| 2 | 64 | 67.88--68.15 | 73.44--73.77 | 151.41--151.54 | 34.11--34.27 | 93.14--93.60 | 0.725--0.732 | 0.785--0.792 |
| 2 | 256 | 184.66--185.30 | 192.70--193.46 | 517.44--517.47 | 71.57--71.62 | 195.49--195.74 | 0.945--0.947 | 0.986--0.988 |

The fair SQ4 projection therefore meets the 20-percent target at every TP1/TP2
BS point: worst attention ratio is 1.043x and worst metadata-inclusive ratio
is 1.067x. PrimTS union is faster than Triton at BS8/64/256; Triton retains the
BS1 low-grid advantage.

Rejected experiments are kept out of production:

- Four-to-eight Q1 loader warps is correct but slower at 32.1--37.1
  microseconds; sixteen loader warps do not make forward progress in the task
  graph.
- BF16 physical KV64 produces a 0.153564 maximum error even after membership
  and tail-mask fixes, locating the incompatibility below masking in the
  score/PV schedule.
- FP8 virtual BLOCK_N32 improves low-grid timing to roughly 29--37
  microseconds but exceeds the accepted numerical tolerance on randomized
  routes and tails, so the production recurrence remains unchanged.

The targeted FlashInfer resolver tests pass (`8 passed`), both production Q1
matrices pass with unchanged error, and the complete TP1/TP2 SQ4 and grouped
union matrices pass. SM100 compiles are supported by the existing profiles;
runtime qualification still requires access to an SM100 node.
