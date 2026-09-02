#!/usr/bin/env bash
set -euo pipefail

qsa_workspace=${QSA_WORKSPACE:-/workspace}
qsa_repo=${QSA_REPO:-${qsa_workspace}/vllm-pr53896-qsa}
qsa_flashinfer=${QSA_FLASHINFER_SOURCE:-${qsa_workspace}/flashinfer}
qsa_model=${QSA_MODEL:-${qsa_workspace}/models/Qwen3.8-Flash-Next-de4b8e4}
qsa_runtime_overlay=${qsa_repo}/benchmarks/qsa/runtime_overlay
qsa_cutlass_packages=${qsa_workspace}/.runtime/cutlass-dsl-4.7.1/nvidia_cutlass_dsl/dsl_packages
qsa_python=${QSA_PYTHON:-python3}
qsa_port=${QSA_PORT:-8000}
qsa_seed=${QSA_SEED:-42}
qsa_output_root=${QSA_OUTPUT_ROOT:-${qsa_workspace}/qsa_e2e_perf}
qsa_decode_output_len=${QSA_DECODE_OUTPUT_LEN:-128}
qsa_tp_size=${QSA_TP_SIZE:-2}
qsa_gpu_memory_utilization=${QSA_GPU_MEMORY_UTILIZATION:-0.91}
qsa_max_num_batched_tokens=${QSA_MAX_NUM_BATCHED_TOKENS:-8192}
qsa_max_num_seqs=${QSA_MAX_NUM_SEQS:-64}
qsa_cudagraph_capture_sizes=${QSA_CUDAGRAPH_CAPTURE_SIZES:-2 4 8 16 24 32 40 48 56 64}
qsa_enable_cutedsl_warmup=${QSA_ENABLE_CUTEDSL_WARMUP:-true}

usage() {
  cat >&2 <<'EOF'
usage:
  run_e2e_perf.sh server triton|prims_ts
  run_e2e_perf.sh prefill triton|prims_ts INPUT_LEN
  run_e2e_perf.sh decode-independent triton|prims_ts INPUT_LEN BATCH_SIZE

Environment overrides:
  QSA_PORT, QSA_OUTPUT_ROOT, QSA_CACHE_TAG, QSA_PYTHON,
  QSA_PREFILL_PROMPTS, QSA_DECODE_OUTPUT_LEN, QSA_TP_SIZE,
  QSA_GPU_MEMORY_UTILIZATION, QSA_MAX_NUM_BATCHED_TOKENS,
  QSA_MAX_NUM_SEQS, QSA_CUDAGRAPH_CAPTURE_SIZES,
  QSA_ENABLE_CUTEDSL_WARMUP (default: true)
EOF
  exit 2
}

if [[ $# -lt 2 ]]; then
  usage
fi

action=$1
backend=$2
if [[ ${backend} != triton && ${backend} != prims_ts ]]; then
  usage
fi

cache_tag=${QSA_CACHE_TAG:-e2e-perf-${backend}}
result_dir=${qsa_output_root}/${cache_tag}
mkdir -p "${result_dir}"

# Use the PR 53896 Python sources for both the server and benchmark client,
# while retaining the dedicated image's compiled vLLM extensions and
# dependency ABI.  Keep this environment identical across every action so a
# caller's active venv cannot silently supply a partial package stack.
export PYTHONPATH=${qsa_runtime_overlay}:${qsa_repo}
export QSA_USE_IMAGE_DSL_STACK=1
export QSA_CUTLASS_DSL_PACKAGES=${qsa_cutlass_packages}
export PYTHONNOUSERSITE=1

common_bench_args=(
  --backend openai
  --base-url "http://127.0.0.1:${qsa_port}"
  --endpoint /v1/completions
  --model Qwen/Qwen3.8-Flash-Next
  --tokenizer "${qsa_model}"
  --request-rate inf
  --temperature 0
  --ignore-eos
  --ready-check-timeout-sec 1800
  --percentile-metrics ttft,tpot,itl,e2el
  --metric-percentiles 50,90,99
)

run_bench() {
  "${qsa_python}" -m vllm.entrypoints.cli.main bench serve "$@"
}

case "${action}" in
  server)
    if [[ $# -ne 2 ]]; then
      usage
    fi
    if [[ ${backend} == prims_ts ]]; then
      # Keep the image's TVM-FFI ABI while overriding only CUTLASS Python/DSL
      # code needed by the local PrimTS kernels.
      export QSA_FLASHINFER_SOURCE=${qsa_flashinfer}
    else
      # Do not import the experimental FlashInfer overlay for the Triton
      # baseline, but keep the same image ABI and requested CUTLASS 4.7 layer.
      unset QSA_FLASHINFER_SOURCE
    fi
    export VLLM_QSA_ATTENTION_BACKEND=${backend}
    export FLASHINFER_DISABLE_VERSION_CHECK=1
    export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${qsa_workspace}/.triton-cache/${cache_tag}}
    export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-${qsa_workspace}/.vllm-cache/${cache_tag}}
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
    mkdir -p "${TRITON_CACHE_DIR}" "${VLLM_CACHE_ROOT}"
    read -r -a cudagraph_capture_sizes <<<"${qsa_cudagraph_capture_sizes}"
    cd "${qsa_repo}"
    exec "${qsa_python}" -m vllm.entrypoints.openai.api_server \
      --model "${qsa_model}" \
      --served-model-name Qwen/Qwen3.8-Flash-Next \
      --reasoning-parser qwen3 \
      --tensor-parallel-size "${qsa_tp_size}" \
      --disable-custom-all-reduce \
      --gpu-memory-utilization "${qsa_gpu_memory_utilization}" \
      --kv-cache-dtype fp8_e4m3 \
      --enable-prefix-caching \
      --max-model-len 139264 \
      --max-num-batched-tokens "${qsa_max_num_batched_tokens}" \
      --max-num-seqs "${qsa_max_num_seqs}" \
      --cudagraph-capture-sizes "${cudagraph_capture_sizes[@]}" \
      --kernel-config "{\"enable_cutedsl_warmup\":${qsa_enable_cutedsl_warmup}}" \
      --no-enable-flashinfer-autotune \
      --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
      --host 0.0.0.0 \
      --port "${qsa_port}"
    ;;
  prefill)
    if [[ $# -ne 3 ]]; then
      usage
    fi
    input_len=$3
    measured_prompts=${QSA_PREFILL_PROMPTS:-3}
    warm_seed=$((qsa_seed - 1))
    # Use a distinct prompt to compile and warm this input-length path without
    # turning the measured prefill into a prefix-cache hit.
    run_bench \
      "${common_bench_args[@]}" \
      --dataset-name random \
      --random-input-len "${input_len}" \
      --random-output-len 1 \
      --random-range-ratio 0 \
      --num-prompts 1 \
      --max-concurrency 1 \
      --seed "${warm_seed}"
    run_bench \
      "${common_bench_args[@]}" \
      --dataset-name random \
      --random-input-len "${input_len}" \
      --random-output-len 1 \
      --random-range-ratio 0 \
      --num-prompts "${measured_prompts}" \
      --max-concurrency 1 \
      --seed "${qsa_seed}" \
      --save-result \
      --result-dir "${result_dir}" \
      --result-filename "${backend}-fp8-mtp3-tp${qsa_tp_size}-prefill-input${input_len}-bs1.json"
    ;;
  decode-independent)
    if [[ $# -ne 4 ]]; then
      usage
    fi
    input_len=$3
    batch_size=$4
    if (( batch_size <= 8 )); then
      measured_prompts=$((batch_size * 2))
    else
      measured_prompts=${batch_size}
    fi
    warm_seed=$((qsa_seed - 1))
    # MTP3 disables cross-request prefix reuse for this hybrid Mamba/QSA model,
    # so use independent contexts and report combinations that exceed physical
    # KV capacity as infeasible rather than silently relying on shared-prefix
    # accounting.
    run_bench \
      "${common_bench_args[@]}" \
      --dataset-name random \
      --random-input-len "${input_len}" \
      --random-output-len "${qsa_decode_output_len}" \
      --random-range-ratio 0 \
      --num-prompts "${batch_size}" \
      --max-concurrency "${batch_size}" \
      --seed "${warm_seed}"
    run_bench \
      "${common_bench_args[@]}" \
      --dataset-name random \
      --random-input-len "${input_len}" \
      --random-output-len "${qsa_decode_output_len}" \
      --random-range-ratio 0 \
      --num-prompts "${measured_prompts}" \
      --max-concurrency "${batch_size}" \
      --seed "${qsa_seed}" \
      --save-result \
      --result-dir "${result_dir}" \
      --result-filename "${backend}-fp8-mtp3-tp${qsa_tp_size}-decode-independent-input${input_len}-bs${batch_size}.json"
    ;;
  *)
    usage
    ;;
esac
