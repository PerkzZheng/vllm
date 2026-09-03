#!/usr/bin/env bash
set -euo pipefail

qsa_workspace=${QSA_WORKSPACE:-/workspace}
qsa_repo=${QSA_REPO:-${qsa_workspace}/vllm-pr53896-qsa}
qsa_flashinfer=${QSA_FLASHINFER_SOURCE:-${qsa_workspace}/flashinfer}
qsa_model=${QSA_MODEL:-${qsa_workspace}/models/Qwen3.8-Flash-Next-de4b8e4}
qsa_model_name=${QSA_MODEL_NAME:-Qwen/Qwen3.8-Flash-Next}
qsa_runtime_overlay=${QSA_RUNTIME_OVERLAY:-${qsa_repo}/benchmarks/qsa/runtime_overlay}
qsa_cutlass_packages=${QSA_CUTLASS_DSL_PACKAGES:-${qsa_workspace}/.runtime/cutlass-dsl-4.7.1/nvidia_cutlass_dsl/dsl_packages}
qsa_backend=${QSA_BACKEND:-prims_ts}
qsa_kv_cache=${QSA_KV_CACHE:-bf16}
qsa_mtp_tokens=${QSA_MTP_TOKENS:-0}
qsa_tp=${QSA_TP:-2}
qsa_port=${QSA_PORT:-8000}
qsa_cache_tag=${QSA_CACHE_TAG:-accuracy-${qsa_backend}-${qsa_kv_cache}-mtp${qsa_mtp_tokens}}

usage() {
  echo "usage: $0 server | eval TASK RUN_NAME OUTPUT_JSON" >&2
  exit 2
}

if [[ $# -lt 1 ]]; then
  usage
fi

case "$qsa_kv_cache" in
  bf16)
    qsa_kv_args=()
    ;;
  fp8|fp8_e4m3)
    qsa_kv_cache=fp8_e4m3
    qsa_kv_args=(--kv-cache-dtype fp8_e4m3)
    ;;
  *)
    echo "QSA_KV_CACHE must be bf16, fp8, or fp8_e4m3" >&2
    exit 2
    ;;
esac

if ! [[ $qsa_mtp_tokens =~ ^[0-9]+$ ]]; then
  echo "QSA_MTP_TOKENS must be a nonnegative integer" >&2
  exit 2
fi

qsa_mtp_args=()
if ((qsa_mtp_tokens > 0)); then
  qsa_mtp_args=(
    --speculative-config
    "{\"method\":\"mtp\",\"num_speculative_tokens\":${qsa_mtp_tokens}}"
  )
fi

case "$1" in
  server)
    export PYTHONPATH=${qsa_runtime_overlay}:${qsa_repo}${PYTHONPATH:+:${PYTHONPATH}}
    export QSA_FLASHINFER_SOURCE=${qsa_flashinfer}
    export QSA_CUTLASS_DSL_PACKAGES=${qsa_cutlass_packages}
    export VLLM_QSA_ATTENTION_BACKEND=${qsa_backend}
    export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${qsa_workspace}/.triton-cache/${qsa_cache_tag}}
    export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-${qsa_workspace}/.vllm-cache/${qsa_cache_tag}}
    export PYTHONNOUSERSITE=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}
    mkdir -p "${TRITON_CACHE_DIR}" "${VLLM_CACHE_ROOT}"
    cd "${qsa_repo}"
    exec python3 -m vllm.entrypoints.openai.api_server \
      --model "${qsa_model}" \
      --served-model-name "${qsa_model_name}" \
      --reasoning-parser qwen3 \
      --tensor-parallel-size "${qsa_tp}" \
      --disable-custom-all-reduce \
      --gpu-memory-utilization "${QSA_GPU_MEMORY_UTILIZATION:-0.91}" \
      "${qsa_kv_args[@]}" \
      --enable-prefix-caching \
      --max-model-len "${QSA_MAX_MODEL_LEN:-139264}" \
      --max-num-seqs "${QSA_MAX_NUM_SEQS:-64}" \
      --cudagraph-capture-sizes 2 4 8 16 24 32 40 48 56 64 \
      --no-enable-flashinfer-autotune \
      "${qsa_mtp_args[@]}" \
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
    exec python3 benchmarks/qsa/qsa_reasoning_eval.py \
      --task "${qsa_task}" \
      --url "http://127.0.0.1:${qsa_port}" \
      --model "${qsa_model_name}" \
      --run-name "${qsa_run_name}" \
      --temperature 0.6 \
      --top-p 0.95 \
      --top-k 20 \
      --seed 42 \
      --max-tokens 131072 \
      --reasoning-effort xhigh \
      --n 1 \
      --max-concurrency "${QSA_MAX_CONCURRENCY:-64}" \
      --metadata "vllm=${QSA_VLLM_REV:-unknown}" \
      --metadata "flashinfer=${QSA_FLASHINFER_REV:-unknown}" \
      --metadata "model=${QSA_MODEL_REV:-unknown}" \
      --metadata "backend=${qsa_backend}" \
      --metadata "kv_cache=${qsa_kv_cache}" \
      --metadata "mtp=${qsa_mtp_tokens}" \
      --metadata "tp=${qsa_tp}" \
      --metadata "job=${qsa_job}" \
      --output "${qsa_output}"
    ;;
  *)
    usage
    ;;
esac
