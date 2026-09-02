# QSA PrimTS migration to vLLM PR 53896

## Revisions

- vLLM base: `13c80fb30ab835cbe387c01c4611970b7c3373e1`, the fetched
  `refs/pull/53896/head` on 2026-08-30.
- Source integration: local branch `qsa-prims-ts-pr53896`; the FP8 accuracy
  gate used implementation revision `8dd467f74`. Each production port commit
  carries its original `Cherry-picked-from` revision.
- Compact PrimTS metadata integration: vLLM revision `778d2bd87`.
- FlashInfer kernel branch: `qsa-page4-prims-ts` at `68fd2bf5`.
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

The first full paired gate completed on job 623471 with FP8-E4M3 KV cache,
MTP=3, TP=2, and the 196,608-token server limit. Triton and PrimTS ran on
separate GPU pairs of the same SM103 node and consumed the identical frozen
manifest. All 96 requests returned HTTP 200, no generation reached the
131,072-token cap, and every API prompt-token count exactly matched its
offline manifest value.

| Input bucket | Examples | Triton | PrimTS | Prediction changes | Regressions | Improvements |
|---:|---:|---:|---:|---:|---:|---:|
| nominal 8K | 16 | 8/16 (50.00%) | 8/16 (50.00%) | 2 | 1 | 1 |
| nominal 16K | 16 | 12/16 (75.00%) | 14/16 (87.50%) | 4 | 1 | 3 |
| nominal 32K | 16 | 12/16 (75.00%) | 12/16 (75.00%) | 1 | 0 | 0 |
| all | 48 | 32/48 (66.67%) | 34/48 (70.83%) | 7 | 2 | 4 |

All 48 raw output hashes differ, which is expected from autoregressive
sensitivity, but final-answer predictions agree on 41/48 examples. PrimTS has
two more correct answers overall and no evidence of a long-prefill accuracy
drop. Token distributions are reported below; `p50/p90/p99` are observed
order statistics over each 16-example bucket.

| Backend | Bucket | Input mean (range) | Output mean; p50/p90/p99 (range) | Total mean (range) |
|---|---:|---:|---:|---:|
| Triton | 8K | 12,596.6 (10,366--15,674) | 3,446.1; 1,815/9,673/9,701 (715--9,701) | 16,042.8 (11,201--23,314) |
| Triton | 16K | 16,504.6 (15,698--17,336) | 1,638.4; 966/4,375/5,273 (266--5,273) | 18,143.0 (16,509--22,222) |
| Triton | 32K | 32,864.2 (30,753--34,370) | 2,549.7; 1,745/5,718/6,569 (449--6,569) | 35,413.9 (31,613--39,962) |
| PrimTS | 8K | 12,596.6 (10,366--15,674) | 2,561.3; 2,854/5,354/6,045 (474--6,045) | 15,157.9 (11,614--18,995) |
| PrimTS | 16K | 16,504.6 (15,698--17,336) | 1,648.6; 914/4,659/4,801 (10--4,801) | 18,153.1 (16,280--21,955) |
| PrimTS | 32K | 32,864.2 (30,753--34,370) | 2,700.9; 1,857/6,619/12,727 (391--12,727) | 35,565.2 (31,838--44,910) |

Raw artifacts are under
`qsa_accuracy/pr53896/longbench-v2-paired-fp8-mtp3/`. The evaluator wall
times were 133.83 seconds for Triton and 177.55 seconds for PrimTS. These are
accuracy-run timings with different generated-token totals (122,148 versus
110,573), first-use JIT, and shared host activity, so they are not accepted
kernel-performance measurements.

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
| PrimTS | FP8-E4M3 | 3 | 1291/1319 (97.88%) | 367/396 (92.68%) | 59/60 (98.33%) | 623068, 623471, 624558 |

The completed BF16/MTP=3 AIME repetitions each scored 30/30. Across both
repetitions there were zero request errors, invalid parses, or truncations.
The final FP8/MTP=3 row also had zero request errors or invalid parses. Its
GSM8K run had no truncations. GPQA repetition one had no truncations and
scored 184/198; repetition two had one truncation and scored 183/198. AIME
repetition one scored 29/30 and repetition two scored 30/30, with no
truncations in either run.

The sole FP8/MTP=3 AIME miss was example index 2. It completed normally after
1,861 output tokens and produced a parsed answer of 56 instead of 62. Four
serial replays of exactly that item with the same sampling arguments and seed
42--one on job 623471 and three on job 624558--all produced the correct answer
62, with no request errors, invalid parses, or truncations. Their completion
lengths were 4,259, 6,004, 4,401, and 3,139 tokens. Outputs differed across
otherwise identical repetitions, so the original miss is not a reproducible
PrimTS kernel failure; it is consistent with scheduling/numerical sensitivity
in stochastic autoregressive sampling.
The four raw replay records are retained under
`qsa_accuracy/pr53896/prims-fp8-mtp3-sampling-v2/`.

The FP8/MTP=3 runtime used the portable local CUDA 13/Torch 2.13 image and the
locally rebuilt 15-argument GDN extension described below. Fresh-node caches
were copied to node-local storage. PrimTS servers must pass
`--no-enable-flashinfer-autotune`: without it, startup invokes the TRT-LLM
BF16 fused-MoE tactic profiler during CUDA-graph capture. On the cold node that
profiled 22 tactics per shape, took about 50 seconds per profile, and eventually
reported asynchronous illegal accesses. This failure occurred in fused-MoE
autotuning before QSA inference, rather than in the PrimTS attention kernel.
With autotuning disabled, TP=2 FP8/MTP=3 startup, graph capture, and all three
accuracy tasks completed normally. The two AIME repetitions used
`--max-cudagraph-capture-size 64`; generation and scoring arguments remained
identical to the matrix protocol.

## Current validation

- Production files pass Python bytecode compilation in the local vLLM
  environment.
- The complete QSA reference suite passes in the CUDA development container:
  `57 passed` in
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
  templates, then exports the native-CSR and unified QSA PrimTS APIs through
  `flashinfer.decode`. The unified exports include workspace sizing, compact
  metadata construction, and the high-level attention call; omitting these
  symbols silently disables the PrimTS backend in vLLM's optional capability
  resolver. Set `QSA_FLASHINFER_SOURCE` and
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

### Compact block-index metadata

The PrimTS route now retains the indexer's native `[rows, 512]` selected
four-token block IDs instead of materializing `[rows, 2051]` token IDs. Q1 and
grouped Q2/Q4 metadata kernels accept the compact blocks directly, derive the
zero-to-three-token causal tail from each logical query position, and still
produce the existing fixed-capacity page-4 CSR interface. Native Triton keeps
the expanded representation. This removes one standalone expansion launch
from the PrimTS serving path and reduces the persistent top-k output buffer by
four times.

The comparison below uses the accepted cold-L2, CUDA-graph, interleaved
10-warmup/100-iteration protocol on job 624558. `legacy` includes block
expansion followed by the same metadata builder; `compact` starts from the
native block output. Ranges cover causal tails zero and three. Times are
microseconds.

| Route | TP | BS | compact metadata | legacy metadata | measured saving |
|---|---:|---:|---:|---:|---:|
| Q1 | 1 | 1 | 7.92--8.17 | 9.94--10.20 | 2.02--2.03 |
| Q1 | 1 | 8 | 7.91--8.02 | 10.00--10.21 | 2.09--2.19 |
| Q1 | 1 | 64 | 8.13--8.42 | 10.41--10.70 | 1.99--2.57 |
| Q1 | 1 | 256 | 8.18--8.23 | 11.97--12.10 | 3.74--3.92 |
| Q1 | 2 | 1 | 8.06--8.14 | 10.02--10.20 | 1.96--2.06 |
| Q1 | 2 | 8 | 7.95--8.19 | 10.05--10.30 | 2.10--2.11 |
| Q1 | 2 | 64 | 8.14--8.15 | 10.23--10.87 | 2.09--2.72 |
| Q1 | 2 | 256 | 8.25--8.26 | 12.14--12.23 | 3.89--3.97 |
| Q4 union | 1 | 1 | 10.52--11.55 | 13.07--13.93 | 2.38--2.55 |
| Q4 union | 1 | 8 | 11.19--12.20 | 13.62--14.26 | 2.06--2.43 |
| Q4 union | 1 | 64 | 12.20--12.23 | 15.12--15.77 | 2.89--3.57 |
| Q4 union | 1 | 256 | 14.30--14.34 | 21.14--22.36 | 6.80--8.06 |
| Q4 union | 2 | 1 | 10.45--11.15 | 12.91--13.33 | 1.76--2.88 |
| Q4 union | 2 | 8 | 11.58--11.59 | 13.81--14.06 | 2.23--2.47 |
| Q4 union | 2 | 64 | 12.17--12.26 | 15.02--15.25 | 2.76--3.08 |
| Q4 union | 2 | 256 | 14.19--14.26 | 20.99--21.82 | 6.73--7.63 |

The compact builder is launch-bound at small row counts: reducing its vector
extent from the old nominal 1,024 lanes to 512 did not measurably change its
roughly eight-microsecond Q1 latency. The useful optimization is removing the
separate expansion launch and its scaling memory traffic. Attention remains
the dominant residual gap. Raw benchmark logs are retained at
`qsa_compact_metadata_bf16_decode_sq1_tp12_cold_graph_10w100i_job624558.log`
and
`qsa_compact_metadata_bf16_decode_q4_tp12_cold_graph_10w100i_job624558.log`
in the workspace root.

The BF16 Q1 checkpoint is superseded by the output-row-parallel FP32 reducer,
physical KV64 virtual-BLOCK_N16 route, and one-wave native-KV64 split policy.
The table uses the same cold-L2, CUDA-graph, interleaved 10-warmup/100-sample
protocol. Ranges cover tails zero and three; times are microseconds.

| TP | BS | PrimTS | Triton | contiguous SWA2K |
|---:|---:|---:|---:|---:|
| 1 | 1 | 18.64--18.70 | 17.49--18.11 | 26.05--26.34 |
| 1 | 8 | 20.78--22.55 | 28.67--28.72 | 28.58--28.69 |
| 1 | 64 | 68.21--69.53 | 79.90--80.98 | 61.02--61.09 |
| 1 | 256 | 208.73--211.92 | 211.36--212.11 | 181.88--181.99 |
| 2 | 1 | 18.39--18.41 | 16.18--16.56 | 26.39--26.41 |
| 2 | 8 | 19.38--20.14 | about 24.67 | 27.14--27.69 |
| 2 | 64 | 44.79--45.27 | 55.41--55.62 | 40.02--40.38 |
| 2 | 256 | 114.58--116.15 | 124.95--125.06 | 103.24--103.74 |

At BS64/256 the worst attention/contiguous ratio is `1.164x`; the worst
compact-metadata-inclusive ratio is `1.180x`. BS1/8 is faster than contiguous,
and PrimTS is faster than Triton except at BS1. All rows have maximum
difference at or below `0.00098`. The full vLLM QSA reference suite passes
(`57 passed`). The accepted raw log is
`qsa_bf16_q1_onewave_native_kv64_tp12_full_cold_graph_10w100i_job624558.log`.

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
target. This S2 checkpoint is superseded for the exact model-facing BF16-output
shape by the qualified S8 profile below.

The candidate low-grid FP8 Q1 route uses grouped Keeps Q64/KV128, one KV
instruction, eight static KV splits, four loader warps, and the standalone
one-CTA/eight-split reducer. It is deliberately narrow: ratio-12 GQA, D256,
BF16 output, an encoded page-4 route with a 2048--2051 compact KV extent, and
`batch_size * num_heads_kv < 32`. Generic FP8 Q1 and FP16-output routes retain
the S2 profile. The four loader warps partition the 32 page-fragment TMA loads;
they do not replay the same transactions.

The following table is retained as a historical pre-fix timing record. Its
S8 output was not correct, so neither its maximum differences nor its timing
ratios are an accepted result.

| TP | BS | PrimTS | compact metadata + PrimTS | Triton | contiguous SWA2K | pre-fix max diff |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 17.46--17.76 | 18.98--20.04 | 16.21--17.85 | 18.18--18.28 | 0.02930 |
| 1 | 8 | 18.73--19.02 | 20.50--20.58 | 31.10--31.86 | 18.42--18.55 | 0.03009 |
| 2 | 1 | 16.39--16.55 | 18.43--19.60 | 16.31--16.41 | 18.28--18.41 | 0.02398 |
| 2 | 8 | 18.41--18.42 | 20.43--20.44 | 22.97--23.44 | 18.40--18.44 | 0.02479 |

The pre-fix timing had a worst attention/contiguous ratio of `1.033x` and a
worst metadata-inclusive ratio of `1.113x`. The corrected kernel must repeat
the cold-L2 CUDA-graph measurement before the 20-percent target can be signed
off; that corrected checkpoint is recorded below. The diagnostic raw log is
`qsa_fp8_q1_kv128_keeps_s8_load4_tp12_bs18_cold_graph_10w100i_job624558.log`.

### Held-locator S8 correctness fix

Standalone token-identity sentinels exposed a loader bug hidden by the earlier
randomized aggregate comparison. For page-4/KV128, one K/V tile owns 32 page
locators. When a split owns at most eight K/V tiles, the page-table producer
stores the complete CTA-local locator window in shared memory. The vector
`page_ids()` consumer did not recognize that held-window layout. It instead
used `(tile_idx * 32) & 31`, which is always zero, so S4 and S8 replayed the
first KV128 tile in each CTA for every local tile iteration. S1 and S2 escaped
the defect because their 17- and 9-tile local spans use per-tile staging rather
than the at-most-eight-tile held window.

FlashInfer revision `68fd2bf5` passes the explicit CTA-local tile index to all
vector page-ID consumers and addresses held storage as
`stage_base + local_tile_idx * pages_per_tile`. The scalar locator path already
used this addressing rule. The standalone regression fixes the physical
Q64/KV128, D256, FP8-to-BF16 kernel while varying only S1/S2/S4/S8. It uses
random nonidentity page maps and causal lengths 2048/2049/2050/2051:

- With Q and K zero and V encoding logical token rank, every split and tail is
  exact after the fix. Before the fix, S4 and S8 had maximum errors of 3.0 and
  repeated their first local KV tile.
- Random inputs match a split-aware FP8-P1 oracle, including BF16 partial-O
  publication and LSE-weighted reduction, with maximum absolute error at or
  below `3.1e-5`.
- The S8 caller-workspace path passes CUDA-graph capture and replay, then
  passes a second replay after changing the live page-ID tensor in place.
- The automatic production policy passes TP1 and TP2 at BS1 and BS8. The
  broader affected-loader matrix passes all seven TP geometries across compact,
  packed-HND, and packed-NHD storage (21 cases), plus 32 Q4/KV128/KV256 tail,
  mask, batch, and issuer cases. The final focused policy/oracle rerun is
  13/13, and Ruff reports no lint errors.

Recent S8-enabled FP8/MTP=3 results of 1292/1319 GSM8K, 365/396 aggregate
GPQA-Diamond, and 57/60 aggregate AIME26 were produced by the faulty loader and
are invalidated. They are not part of the accepted paired table above. No e2e
accuracy conclusion should be drawn from them; rerun FP8/MTP=3 only after the
corrected standalone cold-L2 performance checkpoint is recorded.

### Corrected S8 performance and FP8/MTP=3 accuracy qualification

FlashInfer `68fd2bf5` and vLLM `17e214c57` were qualified after the held-
locator fix. The standalone benchmark uses cold L2 (258 MiB flush), CUDA graph
replay, interleaved measurements, and 10 warmup/100 measured iterations on
GB300 job 630633. Times are microseconds. `e2e` includes the compact Q1
metadata builder and attention.

| TP | BS | causal tail | PrimTS | e2e | Triton | contiguous SWA2K | PrimTS/contiguous | e2e/contiguous |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0 | 18.44 | 20.40 | 16.40 | 18.20 | 1.013x | 1.121x |
| 1 | 1 | 3 | 18.43 | 20.68 | 17.39 | 18.38 | 1.003x | 1.125x |
| 1 | 8 | 0 | 20.48 | 22.47 | 31.42 | 18.59 | 1.102x | 1.209x |
| 1 | 8 | 3 | 20.49 | 22.55 | 30.88 | 18.44 | 1.111x | 1.223x |
| 2 | 1 | 0 | 18.25 | 20.43 | 16.29 | 18.22 | 1.002x | 1.121x |
| 2 | 1 | 3 | 18.46 | 20.54 | 16.29 | 18.44 | 1.001x | 1.114x |
| 2 | 8 | 0 | 18.57 | 20.57 | 22.65 | 18.36 | 1.012x | 1.121x |
| 2 | 8 | 3 | 19.34 | 20.71 | 23.23 | 18.45 | 1.048x | 1.122x |

The corrected attention kernel is within 11.1 percent of contiguous at every
point and is faster than Triton at BS8. Metadata-inclusive TP1/BS8 is the only
remaining miss against the 20-percent target: 20.9--22.3 percent, or 0.9--2.3
percentage points beyond the limit. All TP1/BS1 and TP2 cases are within
11.4--12.5 percent. The compact Q1 builder measures roughly eight microseconds
in isolation but adds about two microseconds to this end-to-end graph. The raw
log is
`qsa_fp8_q1_s8_locatorfix_tp12_bs18_cold_graph_10w100i_job630633.log`.

The corrected end-to-end accuracy gate uses the local Qwen3.8-Flash-Next
`de4b8e4` weights, TP2, FP8-E4M3 KV cache, MTP=3, chat mode, and exactly
`temperature=0.6`, `top_p=0.95`, `top_k=20`, `seed=42`,
`max_tokens=131072`, `reasoning_effort=xhigh`, `n=1`, and `stream=false`.
Jobs 630633 and 631364 used isolated TP2 servers and independent cache
directories.

| Benchmark | Triton FP8/MTP3 | corrected PrimTS FP8/MTP3 | delta |
|---|---:|---:|---:|
| GSM8K | 1290/1319 (97.80%) | 1292/1319 (97.95%) | +2 |
| GPQA-Diamond, two repetitions | 363/396 (91.67%) | 362/396 (91.41%) | -1 |
| AIME26, two repetitions | 60/60 (100.00%) | 59/60 (98.33%) | -1 |

Both corrected GPQA repetitions score 181/198. AIME scores 30/30 and 29/30.
There are no request errors or invalid predictions in any primary run and no
GSM8K or GPQA truncations. The only AIME miss is item 11 in repetition two:
it reaches the 131,072-token cap. The same item is correct in corrected
PrimTS repetition one (43,912 tokens), both Triton repetitions (44,298 and
33,985), and both prior PrimTS repetitions. An isolated rerun on the corrected
kernel also returns the correct answer, 896, with a normal stop after 23,518
tokens. The primary table intentionally remains 59/60 rather than replacing
the prespecified full-run observation with the diagnostic rerun.

The paired item audit further shows stochastic sampling variation rather than
a coherent backend failure. Relative to the corresponding Triton repetition,
GSM8K has 6 regressions and 8 improvements, GPQA-r1 has 7 and 7, GPQA-r2 has
5 and 4, AIME-r1 has none, and AIME-r2 has only the diagnosed length outlier.
Output token distributions explain the long wall time: GSM8K has mean 509 and
p99 3,059 (range 127--13,281); GPQA repetitions have means 13.9K/13.7K,
p90 45.1K/38.3K, and maxima 93.5K/122.7K; AIME repetitions have means
17.5K/21.0K, p90 41.3K/40.9K, and maxima 67.5K/131.1K.

Primary artifacts are under
`qsa_accuracy/pr53896/prims-fp8-mtp3-s8-locatorfix-68fd2bf5-job630633`
and
`qsa_accuracy/pr53896/prims-fp8-mtp3-s8-locatorfix-68fd2bf5-job631364`.

### Restored-S8 post-workspace accuracy requalification

The later locator-fixed, restored-S8 integration was rerun on vLLM
`4f9b1f241`, model `de4b8e4`, TP2, FP8-E4M3 KV, and MTP=3 with the same
prescribed sampling configuration. All requests completed without errors,
invalid predictions, or truncations.

| Benchmark | restored S8 | FP8/MTP3 Triton reference | delta |
|---|---:|---:|---:|
| GSM8K | 1297/1319 (98.33%) | 1290/1319 (97.80%) | +7 |
| GPQA-Diamond, two repetitions | 365/396 (92.17%) | 363/396 (91.67%) | +2 |
| AIME26, two repetitions | 58/60 (96.67%) | 60/60 (100.00%) | -2 |

The AIME delta is concentrated entirely in problem ID 10: both restored-S8
runs answer 165, while the gold answer is 156. The dataset prompt has lost
prime marks in the perpendicular-line condition, making this item especially
sensitive to small numerical changes in the sampled reasoning trajectory.
This is nevertheless reproducibly route-sensitive under the exact control:

- current-PR Triton answers 156 in 2/2 isolated replays;
- locator-fixed S2 (`503783f3`) answers 156 in 2/2 isolated replays; and
- restored Q64/KV128 S8 answers 165 in 0/2 full-run observations.

The S2 model control uses `CUDA_LAUNCH_BLOCKING=1` to avoid its separate
cross-shape CUDA-graph ordering failure; it is an arithmetic/trajectory
control, not a deployable policy. The public unified-workspace CUDA-graph
equivalence test passes with metadata PDL both disabled and enabled, and the
vLLM model path still uses explicit metadata outputs plus a standalone
attention workspace. Therefore the item-10 result does not justify reverting
the workspace interface. Accuracy qualification remains open on the S8 route
pending a numerically closer profile or a graph-safe S2 fallback.

Artifacts are under
`qsa_accuracy/pr53896/primary-q1fix-s8-job633973`,
`qsa_accuracy/pr53896/primary-q1fix-s8-job635017`,
`qsa_accuracy/pr53896/primary-q1fix-s8-job635397`,
`qsa_accuracy/pr53896/current-triton-fp8-mtp3-job635397`, and
`qsa_accuracy/pr53896/s2-503-fp8-mtp3-job635017`.

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
- The initial BF16 physical-KV64 BLOCK_N16 experiment produced a 0.153564
  maximum error because per-tile staging multiplied a raw page-unit base by
  `pages_per_tile` a second time. Correct page-unit addressing and inert
  padding locators reduce the maximum difference below 0.001; physical KV64
  is now the qualified route.
- FP8 virtual BLOCK_N32 improves low-grid timing to roughly 29--37
  microseconds but exceeds the accepted numerical tolerance on randomized
  routes and tails, so the production recurrence remains unchanged.

The targeted FlashInfer resolver tests pass (`8 passed`), both production Q1
matrices pass with unchanged error, and the complete TP1/TP2 SQ4 and grouped
union matrices pass. SM100 compiles are supported by the existing profiles;
runtime qualification still requires access to an SM100 node.

## Rejected unified public QSA workspace checkpoint

FlashInfer now exposes a common two-call integration surface:

```text
get_prims_ts_qsa_workspace_size(...)
prims_ts_qsa_attention(...)
```

Frameworks continue to provide Q, K/V, compact `[R, 512]` block indices, the
block table, request mapping, query positions, caller-owned indptr/lengths, and
output. They allocate one byte workspace; the internal `qsa_page_indices`
array is no longer a second framework allocation or public attention input.
Query rank selects Q1 (`[B,Hq,D]`) or Q2/Q4 (`[B,2|4,Hq,D]`).

The arena uses lifetime-aware partial reuse. Its page-index prefix remains live
from metadata packing through attention. Only the scratch suffix is shared:
group bitmaps occupy it during metadata, then split-KV partials, statistics,
and counters occupy it during attention. The 768-byte attention control tail
(counters, fixed-Q placeholder, and sink scalar) is cleared by the metadata
kernel itself; output-only partial O/statistics are not cleared. Q1 assigns the
reset to CTA zero. Q2/Q4 distribute aliased words to the pack CTA that has
already consumed the corresponding bitmap rows, avoiding both a cross-CTA
race and a separate memset launch. This sequence is CUDA-graph capturable.

For 8K Q, 128K maximum KV capacity, top-k 512, and Q4, the allocation is
50,364,416 bytes (48.03125 MiB): 16,809,984 bytes of persistent page indices
plus a 32 MiB bitmap-dominated shared suffix. The focused SM103 validation has
20 passing FlashInfer tests, including PDL on/off, real Q4 equivalence with the
old two-step path, and CUDA graph replay. With the ABI-matched PR 53896 runtime
overlay and test dependencies, the vLLM owner/reference suite has 56 passing
tests.

At this historical checkpoint, the benchmark harness reported the high-level
unified call beside the explicit metadata+attention path. On GB300 with BF16, cold L2 (258 MiB
eviction), CUDA graphs, 10 warmups, and 100 interleaved samples, the unified
wrapper adds no material latency:

| TP | BS | explicit metadata + attention (us) | unified call (us) | delta (us) |
|---:|---:|---:|---:|---:|
| 1 | 1 | 20.42 | 20.47 | +0.05 |
| 1 | 8 | 23.17 | 23.37 | +0.20 |
| 2 | 1 | 20.35 | 20.56 | +0.21 |
| 2 | 8 | 20.51 | 20.68 | +0.17 |

A grouped-Q4 smoke matrix also places unified latency within measurement
noise of the explicit path (from 2.0 us faster to 0.8 us slower across TP1/TP2
and one/eight groups). All unified outputs are checked against the explicit
path before timing.

The follow-up PDL path connects metadata directly to attention instead of
only connecting the grouped bitmap builder to the union packer. Q1 signals
attention at mapper entry; Q2/Q4 form a bitmap -> pack -> attention chain.
Attention waits before its first CSR or aliased-control read. The generic
native-CSR API keeps this behavior opt-in, while the high-level QSA call
enables it automatically on supported devices. This remains CUDA-graph safe.

The same stable Q1 protocol shows a useful low-grid reduction, especially at
TP1/BS8:

| TP | BS | explicit metadata + attention (us) | unified PDL (us) | prior unified (us) | PDL gain (us) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 20.54 | 20.39 | 20.47 | 0.08 |
| 1 | 8 | 23.47 | 22.56 | 23.37 | 0.81 |
| 2 | 1 | 20.44 | 20.41 | 20.56 | 0.15 |
| 2 | 8 | 20.57 | 20.50 | 20.68 | 0.18 |

For grouped Q4, unified PDL remains neutral: unified versus explicit is
83.81/83.47 us and 86.02/86.08 us for TP1 one/eight groups, and 81.86/81.75
us and 84.51/84.20 us for TP2. One multi-configuration run stopped emitting
output in its final TP2/eight-group case, but an isolated 10-warmup,
100-sample rerun completed at 84.51 us, and the graph/reference tests did not
reproduce a deadlock.

Model-level FP8/MTP3 qualification subsequently rejected grouped PDL even
though these isolated timing and graph checks passed. On job 633335, sustained
Q4 replay first stopped accepting all three MTP draft positions, then produced
unbounded GSM8K reasoning without EOS. A controlled first-64 replay with all
PDL disabled scored 63/64 with zero errors/truncations and completed in 104.9
seconds. Retaining bitmap-to-pack PDL while disabling only pack-to-attention
still failed after 60/64 responses. Disabling both grouped dependencies
restored 63/64, zero errors/truncations, normal nonzero MTP acceptance, and
complete EOS termination in 131.6 seconds on the first replay, but a later
sustained replay still degraded. The isolation policy therefore stream-orders
Q1 as well as Q2/Q4, explicitly resets every attention control tail, and
retains the qualified S2 FP8 reducer instead of the experimental S8 promotion.
These changes pass the focused standalone tests but do not cure the
model-level failure.

Power-of-two route padding was then removed as an independent variable. With
19 requests continuously resident, every attention row was live and the route
shape remained compatible with Q2, yet MTP acceptance declined from about 57
percent to 17--21 percent. Thus inactive rows are not sufficient to explain
the regression. Exact-live routing is itself not a qualified CUDA-graph
contract, however: vLLM captures the outer graph at a bucket capacity while
the internal QSA view then changes with the live count. The owner is now back
on the pre-unified explicit metadata buffer plus attention workspace and the
original graph-stable power-of-two route buckets. Its reference suite passes
all 56 cases; the sustained FP8/MTP3 model-level gate is the remaining
qualification.

The integration revert in `9f49b7824` intentionally removed the vLLM unified
workspace wrappers, but the standalone benchmark still imported and timed
them. The benchmark now follows the production two-step contract again:
caller-owned metadata buffers plus the standalone attention workspace. The
historical unified measurements above remain useful as direct FlashInfer API
qualification, but they are no longer mislabeled as the active vLLM path.

## FP8 Q1 S4 accuracy route

The restored Q64/KV128 S8 route is graph-safe and fast, but it is not the
accuracy-qualified production choice. Under the exact FP8/MTP3 sampling
configuration, AIME26 problem ID 10 answers correctly in two of two Triton
replays and two of two locator-fixed S2 replays, but incorrectly in both S8
replays. Holding the workspace, metadata, Q64/KV128 Keeps profile, and normal
asynchronous graph capture fixed while changing only split fanout from eight
to four restores the correct answer in two of two identical-seed replays.
This isolates the trajectory change to the attention/reduction profile rather
than workspace ownership.

S4 also completes the normal TP2 model graph-capture sequence, including the
final two-token graph. The focused resolver, numerical-oracle, and graph suite
passes all 11 selected cases across TP1/TP2 and batch one/eight. The resolver
shows the relevant arithmetic boundary: S4 uses the compact four-part reducer,
whereas S8 uses the eight-part output-row reducer.

The accuracy choice has a measurable low-grid cost. The matched FP8, TP2,
BS1, SQ1 cold-L2 CUDA-graph comparison uses 10 warmups and 100 interleaved
samples:

| tail | route | attention (us) | metadata + attention (us) | contiguous (us) | attention / contiguous | end-to-end / contiguous |
|---:|:---|---:|---:|---:|---:|---:|
| 0 | S4 | 18.80 | 21.29 | 16.51 | 1.139x | 1.290x |
| 0 | S8 | 16.36 | 18.59 | 16.34 | 1.001x | 1.137x |
| 3 | S4 | 20.56 | 22.49 | 17.73 | 1.159x | 1.269x |
| 3 | S8 | 17.19 | 18.97 | 18.11 | 0.949x | 1.047x |

S4 remains within 20 percent for attention alone, but its metadata-inclusive
BS1 result remains outside the target. Accuracy takes precedence; the existing
SQ1 metadata-elision item below is now the first performance follow-up.

The broader FP8 SQ1 decode matrix uses the same cold-L2, CUDA-graph, 10-warmup,
100-sample protocol, independent causal top-k rows, and a 256-token storage
page. `PrimTS e2e` includes compact metadata; `Triton e2e` includes
`expand_qsa_block_indices_cuda`.

| TP | BS | tail | PrimTS e2e (us) | Triton e2e (us) | Triton / PrimTS | PrimTS / contiguous |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0 / 3 | 20.98 / 21.16 | 18.08 / 18.45 | 0.862x / 0.872x | 1.188x / 1.247x |
| 1 | 8 | 0 / 3 | 22.46 / 24.18 | 32.86 / 32.87 | 1.463x / 1.359x | 1.228x / 1.313x |
| 1 | 64 | 0 / 3 | 91.90 / 101.01 | 90.67 / 90.57 | 0.987x / 0.897x | 2.579x / 2.869x |
| 1 | 512 | 0 / 3 | 530.35 / 587.10 | 527.92 / 528.36 | 0.995x / 0.900x | 3.036x / 3.368x |
| 2 | 1 | 0 / 3 | 20.62 / 21.97 | 18.35 / 18.19 | 0.890x / 0.828x | 1.144x / 1.280x |
| 2 | 8 | 0 / 3 | 22.15 / 23.15 | 24.59 / 24.60 | 1.110x / 1.063x | 1.212x / 1.292x |
| 2 | 64 | 0 / 3 | 48.56 / 51.82 | 53.18 / 53.46 | 1.095x / 1.032x | 1.934x / 2.085x |
| 2 | 512 | 0 / 3 | 274.38 / 298.88 | 279.43 / 278.99 | 1.018x / 0.933x | 2.734x / 2.985x |

PrimTS is faster than fair Triton in seven of sixteen rows. Of the remaining
rows, only TP2/BS1/tail-three exceeds a 20-percent Triton regression
(21.97 versus 18.19 us, 20.8 percent). The misses are therefore bounded
enough to defer kernel and SQ1 metadata tuning as a follow-up after the
end-to-end gates. The much larger high-batch gap to the
contiguous SWA efficiency target remains a separate optimization TODO.
Raw output is in
`qsa_bench/s4-matrix-job636732/fp8-tp1tp2-bs1-8-64-512-sq1-tail03-10w100i.log`.

The matched SQ4/group-four matrix confirms that the grouped route's Triton
miss is isolated to BS1. The two values in each cell are tail zero / tail
three. `projected SWA` scales grouped contiguous SWA by the measured
non-shared-token ratio, so it is the fair efficiency target for a union that
must load more physical K/V than four identical sliding windows.

| TP | BS | metadata (us) | PrimTS e2e (us) | Triton e2e (us) | Triton / PrimTS | PrimTS / projected SWA |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 10.11 / 10.10 | 32.38 / 32.96 | 24.45 / 24.41 | 0.755x / 0.740x | 0.459x / 0.463x |
| 1 | 8 | 10.90 / 10.32 | 38.84 / 38.54 | 51.13 / 51.24 | 1.316x / 1.330x | 0.516x / 0.504x |
| 1 | 64 | 11.44 / 12.15 | 122.92 / 123.52 | 270.81 / 270.73 | 2.203x / 2.192x | 1.080x / 1.055x |
| 1 | 512 | 16.19 / 16.36 | 623.47 / 624.07 | 1786.35 / 1785.24 | 2.865x / 2.861x | 1.065x / 1.069x |
| 2 | 1 | 10.18 / 10.22 | 31.64 / 31.83 | 19.26 / 18.68 | 0.609x / 0.587x | 0.440x / 0.442x |
| 2 | 8 | 10.10 / 10.42 | 33.36 / 32.85 | 42.44 / 42.67 | 1.272x / 1.299x | 0.455x / 0.448x |
| 2 | 64 | 11.51 / 12.03 | 72.48 / 72.35 | 152.50 / 153.40 | 2.104x / 2.120x | 0.797x / 0.779x |
| 2 | 512 | 16.26 / 18.04 | 362.86 / 363.73 | 1026.36 / 1026.62 | 2.828x / 2.822x | 1.072x / 1.070x |

Outside BS1, grouped PrimTS is 1.27--2.87x faster than fair Triton and never
exceeds projected SWA by more than 8 percent at the high-batch points where
the projected control is meaningful. At BS1 the attention launch itself is only modestly
slower than grouped SWA, but the approximately 10-us metadata launch makes
PrimTS 32--70 percent slower than Triton. Keep BS1 metadata elision/fusion and
low-grid kernel tuning as TODO next; it does not block the current integration
or accuracy signoff. Raw output is in
`qsa_bench/s4-matrix-job636732/fp8-tp1tp2-bs1-8-64-512-sq4-group4-tail03-10w100i.log`.

### Workspace-lifetime requalification

The initial S4 qualification was reopened after one full FP8, TP2, MTP3
AIME26 repetition stopped at 28/30 requests with zero forward throughput.
Another run using the same kernel and shared explicit attention scratch
completed, making the failure intermittent. This path is not using
FlashInfer's public unified QSA workspace interface: vLLM writes metadata into
registered buffers and supplies a separate attention scratch allocation.
Reverting that public interface would therefore not be a valid A/B test.

Commit `951e539e2` had removed the prior semantic-launch-key reset and changed
the shared attention scratch allocation from `torch.zeros` to `torch.empty`.
The production owner now pools a distinct, zero-initialized scratch allocation
for each semantic graph key. The key covers device, flattened query-row count,
SQ, storage page size, input/output dtype, and required byte extent. This
prevents graphs with different section layouts from sharing stale control or
partial storage while keeping every captured address stable.

With vLLM `e9369c8e7`, FlashInfer `2bb8d808`, unchanged S4/load4, and the
per-key pool, the exact-sampling AIME26 gate completed 30/30 with zero errors,
invalid predictions, or truncations in 1,072.7 seconds and 551,838 completion
tokens. The result is under
`qsa_accuracy/pr53896/s4-load4-workspace-pool-fp8-mtp3-job635397`.
Job 636732 added two consecutive disjoint-GPU A/B repetitions. A diagnostic
runtime at `a99f54360` differed only by restoring one shared uninitialized
scratch allocation; its two runs both completed 30/30 with zero request
failures (515,704 and 563,027 completion tokens). The per-key pool also
completed both repetitions. Its first scored 30/30 with 511,422 completion
tokens; its second completed every request with no errors or truncations but
scored 28/30 with 496,795 completion tokens. The misses were AIME IDs 10 and
30 after normal EOS, so they are sampling-trajectory accuracy variation rather
than a workspace liveness failure. Combined with the first run above, the pool
has completed three sustained gates.

The two clean shared controls mean the historical stall is not deterministic
and this A/B does not prove workspace causality. Keep the per-key pool anyway:
it provides the correct graph-lifetime invariant by construction at negligible
capacity cost, while the production route has passed all three liveness gates.
Do not revert the independent public unified-workspace API. If the stall
recurs, record the blocked graph key and kernel role state before changing TMA
issuer count or assigning the failure to scratch ownership. Raw artifacts are
under `qsa_accuracy/pr53896/s4-load4-workspace-pool-fp8-mtp3-job636732` and
`qsa_accuracy/pr53896/s4-load4-shared-workspace-fp8-mtp3-job636732`.

### Full S4 production accuracy checkpoint

After the workspace A/B, the unchanged production S4/load4 route completed a
full FP8-E4M3/MTP3 gate on job 636732. The code-bearing revisions are vLLM
`e9369c8e7` and FlashInfer `2bb8d808`; later commits on both branches only add
the qualification notes. The model, TP2 configuration, and exact sampling
parameters match the corrected PR53896 Triton reference above.

| Benchmark | PR53896 Triton FP8/MTP3 | pooled S4/load4 FP8/MTP3 | observation |
|---|---:|---:|---|
| GSM8K | 1290/1319 (97.80%) | 1292/1319 (97.95%) | +2 correct; no truncations |
| GPQA-Diamond | 363/396 (91.67%), 2 reps | 359/396 (90.66%), 2 reps | -4 correct; 1 vs 2 truncations |
| AIME26 | 60/60 (100.00%), 2 reps | 88/90 (97.78%), 3 reps | first two S4 reps are 30/30; third is 28/30 |

Every S4 request completed. There are no request errors or invalid
predictions. GPQA repetitions score 179/198 and 180/198; the second exactly
matches the separately cited 180/198 FP8 reference. The first has two outputs
that reach the prescribed 131,072-token cap, while the second has none. AIME's
third-repetition misses are IDs 10 and 30 after normal EOS, not truncations or
liveness failures. This distribution does not show a systematic S4 accuracy
drop, but future arithmetic or metadata changes must rerun the same gate.

The observed token distributions also explain the wall time. GSM8K inputs
average 789 tokens (range 751--916); outputs average 519, with p95 1,112, p99
3,217, and range 122--15,492. GPQA inputs average 307 tokens (range 132--2,835).
Its two output distributions have means 14,228/12,581, p95
61,157/53,435, p99 98,284/70,888, and maxima 131,072/102,888. The primary
artifacts are
`qsa_accuracy/pr53896/s4-load4-workspace-pool-fp8-mtp3-job636732` and
`qsa_accuracy/pr53896/s4-load4-workspace-pool-fp8-mtp3-job636732-secondary`;
the first pooled AIME repetition remains under the corresponding job 635397
directory.

A corrected S4/load1 control also completed 30/30, but it misses the latency
target badly. The matched TP2/BS1/SQ1 cold-L2 CUDA-graph result is 24.57 us
attention and 26.61 us metadata-inclusive for tail zero, and 26.77/28.68 us
for tail three, versus 16.40/17.07 us contiguous. A prior run labeled load1
was actually load4 because a later resolver update overwrote the first
assignment; both assignments are fixed in the diagnostic worktree. Keep true
load1 as a safety control rather than the default.

## Remaining performance and integration signoff

Work proceeds in this order:

1. TODO next: tune the bounded decode misses against Triton. Start with SQ4
   BS1 metadata elision/fusion and the only greater-than-20-percent SQ1 row
   (TP2/BS1/tail three), then close the broader SQ1 gap to the contiguous SWA
   target. Also remove the standalone Q1 metadata launch. SQ1 has no grouped
   union or membership mask: retain a fixed 513-entry row
   stride, consume or emit the 512 encoded physical page-4 locators directly,
   reserve entry 512 for the zero-to-three-token causal tail, preinitialize
   the fixed CSR indptr, and derive the live compact length from the query
   position. CUDA-graph padding rows must keep locator -1 and length one.
   Prefer fusing logical-block to physical-locator translation into the
   indexer/top-k output. Keep the existing CSR-facing attention interface;
   Q2/Q4 continue to use grouped-union metadata.
2. Rerun the full PrimTS FP8-E4M3/MTP=3 end-to-end accuracy gate after any
   subsequent kernel or metadata changes.
3. Measure matched Triton/PrimTS end-to-end prefill and decode speedups with
   8K, 16K, 32K, and 64K natural input lengths. Decode uses MTP=3 and batch
   sizes 1, 8, 64, and 512; prefill and decode timings are reported
   separately.
4. Promote the compact metadata builders into a public FlashInfer integration
   API. The API must support Q1/Q2/Q4, variable query lengths, causal tails,
   caller-owned output/workspace buffers, no host synchronization or replay-
   time allocation, and stable capacities suitable for CUDA graph capture.
   The vLLM path becomes the reference integration, with standalone examples
   documenting the contract for other frameworks.
