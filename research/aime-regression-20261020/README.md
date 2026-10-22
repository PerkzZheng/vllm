# Reproduce the AIME regression diagnostics

This is a portable export of the October 20–22 investigation, with actual numerical inputs and recorded outputs. It complements the [accuracy summary](../../benchmarks/prims_ts_sparse_mla_accuracy/README.md) and [remaining investigation](../../benchmarks/prims_ts_sparse_mla_accuracy/INVESTIGATION.md). The V4 Flash native FP8 AIME change from 24/30 to 21/30 is still unresolved. These artifacts do not establish an attention-kernel or TopK cause.

## Included data

| Data | Coverage |
| --- | --- |
| `fixtures.tar.xz` | Approximately 72.4 MiB compressed, 294.4 MiB extracted, 400 files |
| Attention inputs | All 60 real Q/KV snapshots: T=1/3/30, H64/D512, SWA-only/C4/C128, short/8k/32k contexts |
| Attention outputs | All 120 historical/current replay files: plain output, LSE, fused FP8 values, packed UE8M0 scales and repeat checks |
| WO-A outputs | All 120 token-major/group-major consumer outputs covering 60 layout pairs |
| Historical implementation | 97 byte-identical historical FlashInfer attention/vLLM adapter files, their hashes and original licenses; no compiled binaries |
| `evidence/` | Deidentified numerical comparisons, qualifications, factorial per-question metrics/token-divergence statistics, trajectory/repetition summaries and MoE binary/tactic audit reports |
| `profiles/` | All three exact 20-entry saved MoE profiles: historical initial, historical resume and current |
| `fixtures.json` | Archive hash, per-file SHA256, byte counts and original relative paths |
| `provenance.json` | Original/exported report hashes, path-normalization rules and source-script provenance |

The archive contains numeric tensors and short geometry/layer labels. It contains no model weights, benchmark questions, generated reasoning text, original prompt/completion token streams, SQLite checkpoints, dataset caches, credentials or server logs. Original workspace paths in exported reports are made relative; their original and exported hashes are separate. The three tactic profiles and all archived tensor/source files retain their original bytes. The original frozen research directory and queued job are unchanged.

## CPU-only verification

Clone the publication branch and set the checkout path. No GPU, model checkpoint, dataset download or Torch import is needed for these two checks. Python 3.12 and `uv` suffice.

```bash
export VLLM_ROOT=/work/vllm
export AIME_DIAGNOSTICS="$VLLM_ROOT/research/aime-regression-20261020"
export AIME_REPRO_ROOT=/work/aime-fixtures
export AIME_REPRO_RUN=/work/aime-new-run
uv venv --python 3.12 /work/aime-cpu/.venv
uv run --no-project --python /work/aime-cpu/.venv/bin/python python \
  "$AIME_DIAGNOSTICS/prepare_fixtures.py" "$AIME_REPRO_ROOT"
uv run --no-project --python /work/aime-cpu/.venv/bin/python python \
  "$AIME_DIAGNOSTICS/verify_recorded_results.py" "$AIME_REPRO_ROOT"
```

Extraction requires a fresh destination and verifies the archive and every file against `fixtures.json`. To verify an already extracted directory, use `prepare_fixtures.py "$AIME_REPRO_ROOT" --verify-only`. `verify_recorded_results.py` checks each PyTorch archive's tensor metadata and storage payloads directly through ZIP reads, excluding archive bookkeeping. Expected result: **60/60 historical/current attention pairs and 60/60 WO-A layout pairs have identical tensor metadata/storage bytes**. This checks the recorded evidence; it does not rerun GPU kernels.

## GPU environment

Use the exact ARM64 GB300 nightly and source setup in [RUNNING.md](../../benchmarks/prims_ts_sparse_mla_accuracy/RUNNING.md). The image digest is `sha256:c8539d9d8aab312168a241ea679697e0d0f92e05838ede8dd4eb4332ee904a85`; Torch is `2.13.0+cu130`, CUDA 13.0 and CUTLASS DSL 4.8.0. The evaluated FlashInfer pin is `6fdd402bf2d2d7ea18264e86bb3a8815c56c3b92`. This vLLM branch contains the evaluated `84cb2c03` integration plus reproduction files. `runtime.json` checks the relevant vLLM source bytes so documentation commits do not invalidate the guard.

TopK and attention replay require **no model weights or datasets**. For those paths, perform only the runtime/artifact setup from RUNNING.md, not its checkpoint/data download step. WO-A replay needs the official V4 Flash checkpoint; the full factorial also needs the pinned AIME dataset/harness. Download those through the existing `prepare.py` workflow. Dataset access conditions still apply; nothing gated is redistributed here.

Set a fresh output directory, keep it outside both the fixture directory and this Git directory, and source the shared environment:

```bash
export FLASHINFER_ROOT=/work/flashinfer
export SPARSE_MLA_EVAL_ROOT=/work/accuracy-data
export SPARSE_MLA_ARM_ROOT="$AIME_REPRO_RUN/component-cache"
export AIME_DIAGNOSTICS="$VLLM_ROOT/research/aime-regression-20261020"
source "$VLLM_ROOT/benchmarks/prims_ts_sparse_mla_accuracy/env.sh"
```

The component scripts expose exactly one assigned GB300 through `CUDA_VISIBLE_DEVICES`. Every invocation below is a fresh process. They verify package origins, relevant source hashes, FlashInfer revision, framework versions and hardware. Generated results never replace the bundled evidence. The portable scripts have CPU/static checks; they have not been used for a new GPU campaign. The original completed GPU checks are recorded in `evidence/`.

## Replay historical and current attention

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-project \
  --python "$SPARSE_MLA_EVAL_ROOT/.venv/bin/python" python \
  "$AIME_DIAGNOSTICS/replay_attention.py" --variant current
CUDA_VISIBLE_DEVICES=0 uv run --no-project \
  --python "$SPARSE_MLA_EVAL_ROOT/.venv/bin/python" python \
  "$AIME_DIAGNOSTICS/replay_attention.py" --variant historical
CUDA_VISIBLE_DEVICES=0 uv run --no-project \
  --python "$SPARSE_MLA_EVAL_ROOT/.venv/bin/python" python \
  "$AIME_DIAGNOSTICS/compare_replays.py"
```

New replay outputs go to `$AIME_REPRO_RUN/replays/{current,historical}`; the comparison goes to `$AIME_REPRO_RUN/replay-comparison.json`. Output directories refuse overwrite. The historical control loads the 97 verified old source files, while all non-attention FlashInfer/model/consumer code remains current. Its bridge copies old token-major logical FP8 values into the current group-major output; it shares scale words without dequantization or re-quantization. The replay restores the original physical KV row IDs, order, holes, duplicate multiplicity, lengths, Q/KV values, sinks, effective scales and RoPE inputs.

The comparator imports the **unchanged FP64 oracle from the pinned FlashInfer sparse MLA test**, preserves its numerical bound, and separately checks old/current plain/LSE/fused/scale byte equality and capture/current equality. It exits 2 on a failed bound, finite/repeat check or expected equality. The recorded result is 60/60 equal with zero bound violations. This is a component accuracy replay through 32k contexts, not a timing benchmark or a full sampled generation.

## Reproduce the WO-A layout comparison

After downloading the pinned V4 Flash checkpoint, run the four independent workers. They use the bundled identical fused activations, actual processed model weights and the production DeepGEMM WO-A consumer, with an identity WO-B boundary. Model weights are downloaded separately, never stored in this archive.

```bash
for layout in token group; do
  for shard in 0 1; do
    CUDA_VISIBLE_DEVICES=0 uv run --no-project \
      --python "$SPARSE_MLA_EVAL_ROOT/.venv/bin/python" python \
      "$AIME_DIAGNOSTICS/consumer_worker.py" --layout "$layout" --shard "$shard"
  done
done
uv run --no-project --python "$SPARSE_MLA_EVAL_ROOT/.venv/bin/python" python \
  "$AIME_DIAGNOSTICS/compare_consumers.py"
```

Workers retain the original FP64 consumer bound, rtol 0.008 / atol 0.001, plus eager repeat, CUDA Graph/eager and graph repeat checks. Expected result: 60/60 layout pairs equal; 120 layout fixtures pass. New outputs/reports are under `$AIME_REPRO_RUN/consumer`. `compare_consumers.py` is a CPU tensor comparison. Arbitrary FP32 output-scale consumption is outside this validated production UE8M0 path.

## Run the pending TopK diagnostics

These tests isolate actual nightly provider behavior; they do not load the model. They retain K512, FP32 physical pitch 262,144, compressed INT32 lengths `[B,1]`, rows 1/3/30, and the **raw context** as the host maximum-sequence argument. The public auto path must actually resolve to the cooperative backend. Each worker checks the loaded `_C_stable_libtorch.abi3.so` SHA256 `d34b72362b9f1395bf7d1db637f861f64847ba465d7ed359a47110379e021f8e`, schema and dispatcher provenance; source-to-binary equivalence is not assumed.

```bash
for provider in auto raw; do
  for scores in unique ties; do
    CUDA_VISIBLE_DEVICES=0 uv run --no-project \
      --python "$SPARSE_MLA_EVAL_ROOT/.venv/bin/python" python \
      "$AIME_DIAGNOSTICS/topk_order.py" --provider "$provider" --scores "$scores"
  done
done
CUDA_VISIBLE_DEVICES=0 uv run --no-project \
  --python "$SPARSE_MLA_EVAL_ROOT/.venv/bin/python" python \
  "$AIME_DIAGNOSTICS/replay_orders.py"
CUDA_VISIBLE_DEVICES=0 uv run --no-project \
  --python "$SPARSE_MLA_EVAL_ROOT/.venv/bin/python" python \
  "$AIME_DIAGNOSTICS/topk_narrow.py" --output-dir "$AIME_REPRO_RUN/narrow"
```

The ordering suite uses raw contexts 2,048/2,051/2,052/2,054/2,060 and 128 eager plus 128 graph repeats per geometry. It records ordered-output variants separately from sorted selected-set variants. Wide unique scores require exact set membership; true cutoff ties allow any correct set without requiring a stable order. If an actual unique-score run exhibits two orders of the same set, `replay_orders.py` applies only those rank permutations to two existing T1/T3 C4 snapshots. If no such pair is observed, it records that replay was not attempted. This measures attention sensitivity to order, not AIME causality.

The separate narrow supplement uses distinct FP32 scores `1 + i * 2**-20`, N512/513/515, ascending/descending layouts and 32 eager plus 32 graph repeats. It distinguishes invalid/duplicate indices from valid-but-wrong membership, retains all cases and intentionally exits **2** if any unique-score membership oracle fails. The pinned CUDA source has a coarse-bin truncation counterexample, but **the actual nightly result remains unmeasured as of publication**. Do not interpret an unsorted correct set as a TopK bug, a supplement failure as failure of the independently completed ordering suite, or a source-level counterexample as proof of the AIME cause.

## Repeat the four-arm AIME diagnostic

Use four disjoint GB300 GPUs, the official V4 Flash checkpoint, prepared pinned local datasets/harness and fresh output/cache directories. The helper scripts start one TP1 server per arm, load the selected saved MoE profile at natural warmup, verify all 20 choices with no fallback/re-tuning, and wait until all four health/tactic gates pass before generating. They run the original 30 AIME2026 questions at seed 1234, temperature/top-p 1, high reasoning, context 1,048,576 and generation cap 393,216. `--return-token-ids` adds only response instrumentation; normal accuracy-runner requests keep their previous default.

```bash
# Use a NEW AIME_REPRO_RUN for an independent factorial, not an old result/cache.
bash "$AIME_DIAGNOSTICS/run_factorial_arm.sh" historical historical-initial-global 0 8101 &
repro_a=$!
bash "$AIME_DIAGNOSTICS/run_factorial_arm.sh" historical current-per-model 1 8102 &
repro_b=$!
bash "$AIME_DIAGNOSTICS/run_factorial_arm.sh" current historical-initial-global 2 8103 &
repro_c=$!
bash "$AIME_DIAGNOSTICS/run_factorial_arm.sh" current current-per-model 3 8104 &
repro_d=$!
repro_failed=0
for repro_pid in "$repro_a" "$repro_b" "$repro_c" "$repro_d"; do
  wait "$repro_pid" || repro_failed=1
done
test "$repro_failed" = 0
```

Each arm writes raw responses, grades, token IDs, server logs and tactic receipts only into the new external `$AIME_REPRO_RUN/factorial` directory. On exit it stops its own server. No scheduler allocation or cancellation is performed. The retained historical resume profile is available for a separately named future experiment; it is deliberately not selected by this completed-stage factorial.

Recorded outcomes were 22/30, 23/30, 22/30 and 21/30 for the four arms in the order above. The original 120-response audit and divergence statistics are in `evidence/paired-repeats/`. A new run need not reproduce those exact trajectories or scores: this is one seed per cell, fresh AIME-only history differs from the official GSM8K→GPQA→AIME history, and asynchronous batches and TopK order were not forced identical. All capped/empty outcomes must remain included. Exact original token streams are not redistributed; old official token IDs were unavailable in the first place. For new causal captures, retain the locally generated IDs and record actual producer logits/selected index sets/order near context 2,052.

## Provenance and limits

Export validation passed: all 400 extracted file hashes/sizes, 60 recorded attention pairs and 60 recorded WO-A pairs were checked on CPU without Torch. The complete TopK ordering/narrow numerical loops match their original abstract syntax after normalizing equivalent accelerator synchronization calls. Default evaluation generation arguments remain unchanged; six legacy/default/opt-in token-ID resume combinations were checked. Independent read-only review passed the export/provenance records, runtime/provider guards, original numerical bounds and failure propagation. This is validation of the portable export and recorded evidence, not a new GPU run.

The original GPU diagnostics completed before this export: 60 attention pairs and 60 consumer-layout pairs were equal; the four-arm diagnostic produced 120 audited responses. The exported launchers preserve numerical algorithms, seeds, geometry and tolerances while relocating paths and replacing equivalent synchronization/cache calls with the repository's `torch.accelerator` convention. Source checks, output ownership and overwrite guards are adapted for an ordinary clone rather than the original Slurm workspace. No product kernel or original artifact was modified. Portable GPU reruns remain unqualified until someone executes them in the pinned environment.

The small MoE reports reproduce the retained audit evidence and contain all three tuning profiles' differences. Native binaries/ELFs and the CUDA disassembly tools are not bundled; independently repeating that binary audit requires the original providers/builds. Report hashes prove export provenance, not a new binary or model-accuracy result. Gated inputs and raw generations must be obtained/generated locally. The pending original TopK job remains independent of this publication.
