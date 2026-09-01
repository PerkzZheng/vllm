#!/usr/bin/env bash
set -euo pipefail

qsa_workspace=${QSA_WORKSPACE:-/workspace}
qsa_repo=${qsa_workspace}/vllm-pr53896-qsa
qsa_flashinfer=${qsa_workspace}/flashinfer
qsa_model=${qsa_workspace}/models/Qwen3.8-Flash-Next-de4b8e4
qsa_runtime_overlay=${qsa_repo}/benchmarks/qsa/runtime_overlay
qsa_cutlass_packages=${qsa_workspace}/.runtime/cutlass-dsl-4.7.1/nvidia_cutlass_dsl/dsl_packages
qsa_cache_tag=${QSA_CACHE_TAG:-post-pdl-fp8-mtp3}
qsa_port=${QSA_PORT:-8000}

usage() {
  echo "usage: $0 server | eval TASK RUN_NAME OUTPUT_JSON" >&2
  exit 2
}

if [[ $# -lt 1 ]]; then
  usage
fi

case "$1" in
  server)
    export PYTHONPATH=${qsa_runtime_overlay}:${qsa_repo}
    export QSA_FLASHINFER_SOURCE=${qsa_flashinfer}
    export QSA_CUTLASS_DSL_PACKAGES=${qsa_cutlass_packages}
    export VLLM_QSA_ATTENTION_BACKEND=prims_ts
    export TRITON_CACHE_DIR=${qsa_workspace}/.triton-cache/${qsa_cache_tag}
    export VLLM_CACHE_ROOT=${qsa_workspace}/.vllm-cache/${qsa_cache_tag}
    export PYTHONNOUSERSITE=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
    mkdir -p "${TRITON_CACHE_DIR}" "${VLLM_CACHE_ROOT}"
    cd "${qsa_repo}"
    exec python -m vllm.entrypoints.openai.api_server \
      --model "${qsa_model}" \
      --served-model-name Qwen/Qwen3.8-Flash-Next \
      --reasoning-parser qwen3 \
      --tensor-parallel-size 2 \
      --disable-custom-all-reduce \
      --gpu-memory-utilization 0.91 \
      --kv-cache-dtype fp8_e4m3 \
      --enable-prefix-caching \
      --max-model-len 139264 \
      --max-num-seqs 64 \
      --cudagraph-capture-sizes 2 4 8 16 24 32 40 48 56 64 \
      --no-enable-flashinfer-autotune \
      --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
      --host 0.0.0.0 \
      --port "${qsa_port}"
    ;;
  eval)
    if [[ $# -ne 4 ]]; then
      usage
    fi
    qsa_task=$2
    qsa_run_name=$3
    qsa_output=$4
    qsa_job=${SLURM_JOB_ID:-unknown}
    cd "${qsa_repo}"
    exec python benchmarks/qsa/qsa_reasoning_eval.py \
      --task "${qsa_task}" \
      --url "http://127.0.0.1:${qsa_port}" \
      --model Qwen/Qwen3.8-Flash-Next \
      --run-name "${qsa_run_name}" \
      --temperature 0.6 \
      --top-p 0.95 \
      --top-k 20 \
      --seed 42 \
      --max-tokens 131072 \
      --reasoning-effort xhigh \
      --n 1 \
      --max-concurrency 64 \
      --metadata vllm=a9b9c9d19 \
      --metadata flashinfer=bd3863b8 \
      --metadata model=de4b8e4 \
      --metadata backend=prims_ts \
      --metadata kv_cache=fp8_e4m3 \
      --metadata mtp=3 \
      --metadata tp=2 \
      --metadata "job=${qsa_job}" \
      --output "${qsa_output}"
    ;;
  *)
    usage
    ;;
esac
