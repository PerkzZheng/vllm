#!/bin/bash
set -euo pipefail
repro_scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repro_attention="${1:?historical or current}"
repro_profile="${2:?historical-initial-global or current-per-model}"
repro_gpu="${3:?assigned GPU ID}"
repro_port="${4:?free local port}"
case "$repro_attention" in historical|current) ;; *) exit 2 ;; esac
case "$repro_profile" in historical-initial-global|current-per-model) ;; *) exit 2 ;; esac
repro_group="${AIME_REPRO_RUN:?}/factorial"
export SPARSE_MLA_ARM_ROOT="$repro_group/attention-${repro_attention}__moe-${repro_profile}"
test ! -e "$SPARSE_MLA_ARM_ROOT"
source "${VLLM_ROOT:?}/benchmarks/prims_ts_sparse_mla_accuracy/env.sh"
cp "$repro_scripts/profiles/$repro_profile.json" "$SPARSE_MLA_ARM_ROOT/selected-moe-cache.json"
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="$SPARSE_MLA_ARM_ROOT/autotune"
export PYTHONPATH="$repro_scripts/factorial-hooks:$repro_scripts:$PYTHONPATH"
export AIME_FACTORIAL_ATTENTION="$repro_attention"
export CUDA_VISIBLE_DEVICES="$repro_gpu"
export MAX_JOBS=8 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
repro_deadline=$(( $(date +%s) + 3600 ))
setsid env AIME_FACTORIAL_SERVER=1 uv run --no-project \
    --python "$SPARSE_MLA_EVAL_ROOT/.venv/bin/python" python \
    "$VLLM_ROOT/benchmarks/prims_ts_sparse_mla_accuracy/serve.py" \
    --model v4-flash --backend prims --kv-cache fp8 \
    --gpus "$repro_gpu" --port "$repro_port" \
    > "$SPARSE_MLA_ARM_ROOT/server.log" 2>&1 &
repro_pid=$!
trap 'kill -TERM -- -"$repro_pid" 2>/dev/null || true' EXIT
repro_ready=0
while [ "$(date +%s)" -lt "$repro_deadline" ]; do
    if curl --silent --fail "http://127.0.0.1:$repro_port/health" > /dev/null; then
        repro_ready=1
        break
    fi
    kill -0 "$repro_pid"
    sleep 5
done
test "$repro_ready" = 1
uv run --no-project --python "$SPARSE_MLA_EVAL_ROOT/.venv/bin/python" python -c \
    'import json,sys; assert json.load(open(sys.argv[1]))["verified"]' \
    "$SPARSE_MLA_ARM_ROOT/tactic-control.json"
touch "$SPARSE_MLA_ARM_ROOT/ready"
repro_all_ready=0
while [ "$(date +%s)" -lt "$repro_deadline" ]; do
    repro_count=$(find "$repro_group" -maxdepth 2 -name ready | wc -l)
    if [ "$repro_count" -eq 4 ]; then repro_all_ready=1; break; fi
    sleep 2
done
test "$repro_all_ready" = 1
uv run --no-project --python "$SPARSE_MLA_EVAL_ROOT/.venv-eval/bin/python" python \
    "$VLLM_ROOT/benchmarks/prims_ts_sparse_mla_accuracy/evaluate.py" \
    --model v4-flash --task aime2026 --seed 1234 --top-p 1 \
    --max-tokens 393216 --concurrency 32 --return-token-ids \
    --start 0 --limit 30 --url "http://127.0.0.1:$repro_port/v1/chat/completions" \
    --output "$SPARSE_MLA_ARM_ROOT/results" \
    > "$SPARSE_MLA_ARM_ROOT/evaluate.log" 2>&1
