# Run the accuracy evaluation

## What this reproduces

These scripts reproduce the evaluated model/task inputs, serving flags, sampling and grading on the pinned nightly. They keep external data and outputs out of Git. They do not guarantee identical generations: dynamic batching, atomic TopK ordering, JIT/autotuning and long reasoning trajectories can vary. The new portable launch path has offline validation only; the completed GPU results used the original campaign launchers described in the summary.

Run inside an **ARM64 GB300** allocation, with one separate arm directory per model/backend/KV/seed. The evaluated tensor-parallel sizes are V4 Flash 1, V4 Pro 4, V4.1 Flash 2 and GLM 5.3 Flash 2. An allocation may contain extra GPUs; pass only the assigned GPU IDs. Do not overlap servers on the same GPUs. The original cluster's normal QoS required at least four allocated GPUs. Scheduler submission is site-specific; the scripts do not allocate or cancel jobs.

The exact image is:

```text
vllm/vllm-openai@sha256:c8539d9d8aab312168a241ea679697e0d0f92e05838ede8dd4eb4332ee904a85
```

Use this digest with Docker/Pyxis/Enroot, overriding the image's default serving entrypoint with a shell. Mount the integration checkout, a FlashInfer checkout, and a writable external work directory at stable absolute paths. Replacing it with a moving `nightly` tag invalidates the environment comparison. The image's native libraries were built from vLLM `2eb491e18b1ce5b8a25e4c88f32dbb5d028d525c`.

## Prepare the environment and pinned dependencies

The following paths are examples **inside the container**. Set them to the actual mounts. Run setup outside the vLLM checkout so Python initially finds the image's original packages. Do not reuse a running campaign's checkout or environment for setup.

```bash
export VLLM_ROOT=/work/vllm
export FLASHINFER_ROOT=/work/flashinfer
export SPARSE_MLA_EVAL_ROOT=/work/accuracy-data
export ACCURACY_SCRIPTS="$VLLM_ROOT/benchmarks/prims_ts_sparse_mla_accuracy"
export UV_CACHE_DIR="$SPARSE_MLA_EVAL_ROOT/cache/uv"
mkdir -p "$SPARSE_MLA_EVAL_ROOT"

# Checkout this publication branch in VLLM_ROOT before entering the container.
# It contains the tested 84cb2c03 integration and the evaluation package.
# The pinned image does not include git. Install it as root in the container,
# or provide a site-managed git executable on PATH before continuing.
apt-get update
apt-get install -y git
git clone https://github.com/PerkzZheng/flashinfer.git "$FLASHINFER_ROOT"
git -C "$FLASHINFER_ROOT" fetch origin 6fdd402bf2d2d7ea18264e86bb3a8815c56c3b92
git -C "$FLASHINFER_ROOT" checkout --detach 6fdd402bf2d2d7ea18264e86bb3a8815c56c3b92
git -C "$FLASHINFER_ROOT" submodule update --init 3rdparty/cutlass 3rdparty/cccl 3rdparty/spdlog

cd /tmp
unset PYTHONPATH
uv venv --system-site-packages --python 3.12 "$SPARSE_MLA_EVAL_ROOT/.venv"
uv pip install --python "$SPARSE_MLA_EVAL_ROOT/.venv/bin/python" --no-deps nvidia-cutlass-dsl==4.8.0
uv run --no-project --python "$SPARSE_MLA_EVAL_ROOT/.venv/bin/python" python \
  "$ACCURACY_SCRIPTS/prepare_runtime.py" \
  --vllm-root "$VLLM_ROOT" --flashinfer-root "$FLASHINFER_ROOT"
```

`prepare_runtime.py` copies only missing runtime/generated files from the installed base nightly, including vendored DeepGEMM, without replacing tracked integration Python. It restores FlashInfer generated header links and build metadata. Copied-file hashes go to `runtime-artifacts.json` outside Git. If sparse checkout hides tracked Python, the helper fails; materialize `vllm/` first. This reuses the exact matching ABI, not libraries from a newer image. Keep the generated files untracked.

Stage the pinned harness and datasets. GPQA Diamond requires accepted Hugging Face access terms and an authorized read token through `HF_TOKEN` or the standard token file. Do not put credentials in scripts or command arguments. Model snapshots require substantial disk space.

```bash
uv run --no-project --python "$SPARSE_MLA_EVAL_ROOT/.venv/bin/python" python \
  "$ACCURACY_SCRIPTS/prepare.py" --model v4-flash
# Add --model v4-pro --model v41-flash --model glm53-flash as needed.
# With no --model, download only datasets and prepare tasks.

uv venv --system-site-packages --python 3.12 "$SPARSE_MLA_EVAL_ROOT/.venv-eval"
uv pip install --python "$SPARSE_MLA_EVAL_ROOT/.venv-eval/bin/python" \
  -e "$SPARSE_MLA_EVAL_ROOT/lm-evaluation-harness[api]"
```

Dataset/model/harness revisions and all model-dependent options are in `protocol.json`. The generated YAML files reuse the pinned harness task definitions, point to local data, and retain the AIME leading-zero grader. `--tasks-only` regenerates YAML using existing local data without downloading weights/datasets. The harness must be at the pinned revision with clean tracked source.

The serving environment should report Torch `2.13.0+cu130`, CUDA `13.0`, CUTLASS DSL `4.8.0`, and GB300. If dependency resolution changes the serving stack, recreate it from the pinned image rather than treating the changed stack as a comparable run. Record package versions, `nvidia-smi`, both Git heads and the image digest with every new campaign.

## Start one native arm

For V4 Flash native FP8:

```bash
export SPARSE_MLA_ARM_ROOT=/work/runs/v4-flash/prims-fp8/seed1234
source "$ACCURACY_SCRIPTS/env.sh"
uv run --no-project --python "$SPARSE_MLA_EVAL_ROOT/.venv/bin/python" python - <<'PY'
import importlib.metadata
import json
import os
from pathlib import Path

import flashinfer
import torch
import vllm

assert Path(vllm.__file__).resolve().is_relative_to(Path(os.environ['VLLM_ROOT']).resolve())
assert Path(flashinfer.__file__).resolve().is_relative_to(Path(os.environ['FLASHINFER_ROOT']).resolve())
assert torch.__version__ == '2.13.0+cu130' and torch.version.cuda == '13.0'
assert importlib.metadata.version('nvidia-cutlass-dsl') == '4.8.0'
assert 'GB300' in torch.cuda.get_device_name(0)
print(json.dumps(dict(vllm=vllm.__file__, flashinfer=flashinfer.__file__,
                     torch=torch.__version__, gpu=torch.cuda.get_device_name(0))))
PY
uv run --no-project --python "$SPARSE_MLA_EVAL_ROOT/.venv/bin/python" python \
  "$ACCURACY_SCRIPTS/serve.py" --model v4-flash --backend prims \
  --kv-cache fp8 --gpus 0 --port 8001 \
  > "$SPARSE_MLA_ARM_ROOT/server.log" 2>&1
```

The server command blocks until it exits. In another shell with the same exports, wait for `curl --fail http://127.0.0.1:8001/health` to succeed and inspect startup logs for the resolved backend/cache format. Pro can require up to 180 minutes for startup/JIT/graph warmup. A health timeout is an infrastructure/runtime failure, not an incorrect model answer. `serve.py --print-command` validates model revision/geometry and prints arguments without starting a server.

The server uses maximum context 1,048,576, high reasoning, dtype BF16 for the model, 32 sequences, 4,096 batched tokens and the original default graph compilation mode. `env.sh` sets the worker RPC budget to 1,800 seconds to tolerate the observed long JIT pauses; this does not repair missing warmup variants.

| Model | Alias | Native backend | GPUs passed to serve.py |
| --- | --- | --- | --- |
| DeepSeek V4 Flash | `v4-flash` | `PRIMS_TS_MLA_SPARSE_DSV4` | `0` |
| DeepSeek V4 Pro | `v4-pro` | `PRIMS_TS_MLA_SPARSE_DSV4` | `0,1,2,3` |
| DeepSeek V4.1 Flash | `v41-flash` | `PRIMS_TS_MLA_SPARSE_DSV41` | `0,1` |
| GLM 5.3 Flash | `glm53-flash` | `PRIMS_TS_MLA_SPARSE` | `0,1` |

Use `--kv-cache bfloat16` for native BF16. Both prefill and decode use the same native sparse MLA integration. DeepSeek output inverse RoPE plus FP8 quantization is fused on the production UE8M0 path; output block size/layout follows the model's WO-A consumer. The model's packed query rows, separate KV index buffers and already folded Q/KV descales are supplied by the adapter.

## Evaluate, resume and summarize

In the client shell:

```bash
source "$ACCURACY_SCRIPTS/env.sh"
uv run --no-project --python "$SPARSE_MLA_EVAL_ROOT/.venv-eval/bin/python" python \
  "$ACCURACY_SCRIPTS/run_suite.py" --model v4-flash \
  --url http://127.0.0.1:8001/v1/chat/completions \
  --output "$SPARSE_MLA_ARM_ROOT/results"
uv run --no-project --python "$SPARSE_MLA_EVAL_ROOT/.venv-eval/bin/python" python \
  "$ACCURACY_SCRIPTS/summarize.py" "$SPARSE_MLA_ARM_ROOT/results" \
  > "$SPARSE_MLA_ARM_ROOT/summary.jsonl"
```

The default task order is GSM8K → GPQA → AIME2026, seed 1234. All prompts/choice ordering/few-shot selection use fixed seeds 1234. The model's top-p and generation cap come from `protocol.json`; temperature 1.0 and high reasoning are retained. No task stop string truncates reasoning. `--tasks aime2026` is useful for a diagnostic but changes the server history and is not the original complete campaign.

For a short smoke check before a full pass:

```bash
uv run --no-project --python "$SPARSE_MLA_EVAL_ROOT/.venv-eval/bin/python" python \
  "$ACCURACY_SCRIPTS/evaluate.py" --model v4-flash --task gsm8k \
  --top-p 1.0 --max-tokens 393216 --concurrency 1 --limit 1 \
  --output "$SPARSE_MLA_ARM_ROOT/smoke"
```

Smoke results belong in their own directory. Every shard retains `arguments.json`, `generations.sqlite`, complete raw `responses.jsonl`, harness samples/grades in `results.json`, and `run.log`. Results are written atomically. Repeating the suite command resumes only that arm's own fixed-configuration SQLite responses and regrades them. Changed generation configuration is rejected. Do not point a different backend, revision or seed at the same output directory; server identity is not inferred from a URL. Archive server/version receipts and create a fresh arm directory for each change. A resumed original run is not an independent repeat.

`summarize.py` counts only closed graded shards, rejects overlapping question IDs or duplicate API IDs within each task, and reports partial coverage, length stops and mean completion length. It is an aggregate convenience tool, not the historical campaign's independent raw-input/freshness audit. Preserve raw records for that audit. Grading keeps all capped, empty-final and extraction-sensitive outcomes; use flexible extraction as primary for GSM8K/GPQA and retain strict diagnostics.

For additional generation seeds, run `run_suite.py --seed 1235` in a fresh arm directory and repeat the same seed for every compared backend. Keep task seeds fixed. A fresh server/cache profile is advisable for a controlled repeat. Compare per-question correctness with matching prompt/target hashes and report improved/regressed counts and paired confidence intervals. Evaluate each backend once per seed and estimate mean pass@1; selecting the best answer across seeds would be a different metric.

## Default backend comparisons

Stop the previous server before reusing its GPUs or port, and use a new arm directory/cache. Omit an explicit backend selector by passing `--backend default`:

```bash
uv run --no-project --python "$SPARSE_MLA_EVAL_ROOT/.venv/bin/python" python \
  "$ACCURACY_SCRIPTS/serve.py" --model v4-flash --backend default \
  --kv-cache fp8 --gpus 0 --port 8001
```

Use the same client command and task protocol with that default arm's result directory. V4 Pro also uses `default/fp8`; V4.1 uses `default/auto` and must be labeled FP4 compressed/MXFP8 SWA; GLM supports default BF16 and default FP8 separately. The launcher rejects unsupported DeepSeek default BF16 cells and misleading V4.1 default/fp8 labeling. Do not replace unsupported defaults with TRTLLM: the requested comparison is each model's true default. Record automatic backend selection and actual cache format from startup logs.

## Boundaries of this package

This directory contains the portable full-accuracy workflow and aggregate historical evidence. It does not bundle gated datasets, reasoning traces, model weights, nightly binaries or the large frozen historical snapshot used by the original independent audit. Existing local model/data snapshots can be reused only after checking the revisions in `protocol.json`. The detailed AIME replay and TopK probes remain frozen in their original research directories, identified in [INVESTIGATION.md](INVESTIGATION.md); they are not model accuracy tests or claimed product fixes.
