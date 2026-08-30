# QSA PrimTS migration to vLLM PR 53896

## Revisions

- vLLM base: `13c80fb30ab835cbe387c01c4611970b7c3373e1`, the fetched
  `refs/pull/53896/head` on 2026-08-30.
- Source integration: local branch `qsa-prims-ts-integration` at
  `c4ba907`, with each production port commit carrying its original
  `Cherry-picked-from` revision.
- FlashInfer kernel branch: `qsa-page4-prims-ts` at `8fa4396a`.
- Accuracy model: `Qwen/Qwen3.8-Flash-Next` revision `de4b8e4`. The model
  config identifies the implementation as `qwen4_exp`, uses 24 Q heads, two
  KV heads, head dimension 256, indexer budget 2048, compression ratio four,
  and one hybrid MTP layer.

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
step is the small native-Triton MTP=3 gate; if it is stable, repeat that gate
with PrimTS before moving to FP8-E4M3 MTP=0.
