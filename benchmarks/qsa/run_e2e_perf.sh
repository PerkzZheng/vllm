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
qsa_kv_cache_dtype=${QSA_KV_CACHE_DTYPE:-fp8_e4m3}
qsa_mtp_tokens=${QSA_MTP_TOKENS:-3}
qsa_max_model_len=${QSA_MAX_MODEL_LEN:-139264}
qsa_profile=${QSA_PROFILE:-false}
qsa_enable_prefix_caching=${QSA_ENABLE_PREFIX_CACHING:-true}
qsa_ready_check_timeout_sec=${QSA_READY_CHECK_TIMEOUT_SEC:-1800}

case "${qsa_kv_cache_dtype}" in
  auto)
    qsa_dtype_label=bf16
    ;;
  fp8 | fp8_e4m3)
    qsa_dtype_label=fp8
    ;;
  *)
    echo "unsupported QSA_KV_CACHE_DTYPE: ${qsa_kv_cache_dtype}" >&2
    exit 2
    ;;
esac

if ! [[ ${qsa_mtp_tokens} =~ ^[0-9]+$ ]]; then
  echo "QSA_MTP_TOKENS must be a non-negative integer" >&2
  exit 2
fi
if ! [[ ${qsa_ready_check_timeout_sec} =~ ^[0-9]+$ ]]; then
  echo "QSA_READY_CHECK_TIMEOUT_SEC must be a non-negative integer" >&2
  exit 2
fi
if [[ ${qsa_profile} != true && ${qsa_profile} != false ]] ||
  [[ ${qsa_enable_prefix_caching} != true && ${qsa_enable_prefix_caching} != false ]]; then
  echo "QSA_PROFILE and QSA_ENABLE_PREFIX_CACHING must be true or false" >&2
  exit 2
fi

usage() {
  cat >&2 <<'EOF'
usage:
  run_e2e_perf.sh server triton|prims_ts
  run_e2e_perf.sh prefill triton|prims_ts INPUT_LEN
  run_e2e_perf.sh decode-independent triton|prims_ts INPUT_LEN BATCH_SIZE

Environment overrides:
  QSA_PORT, QSA_OUTPUT_ROOT, QSA_CACHE_TAG, QSA_PYTHON,
  QSA_CACHE_ROOT,
  QSA_PREFILL_PROMPTS, QSA_PREFILL_WARMUP_PROMPTS,
  QSA_DECODE_OUTPUT_LEN, QSA_TP_SIZE,
  QSA_DECODE_WARMUP_PROMPTS,
  QSA_GPU_MEMORY_UTILIZATION, QSA_MAX_NUM_BATCHED_TOKENS,
  QSA_MAX_NUM_SEQS, QSA_CUDAGRAPH_CAPTURE_SIZES,
  QSA_KV_CACHE_DTYPE (auto or fp8_e4m3), QSA_MTP_TOKENS,
  QSA_MAX_MODEL_LEN, QSA_PROFILE (default: false),
  QSA_READY_CHECK_TIMEOUT_SEC (default: 1800),
  QSA_ENABLE_PREFIX_CACHING (default: true),
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
qsa_cache_root=${QSA_CACHE_ROOT:-${qsa_workspace}/.cache/${cache_tag}}
mkdir -p "${result_dir}"

# Use the PR 53896 Python sources for both the server and benchmark client,
# while retaining the dedicated image's compiled vLLM extensions and
# dependency ABI.  Keep this environment identical across every action so a
# caller's active venv cannot silently supply a partial package stack.
export PYTHONPATH=${qsa_runtime_overlay}:${qsa_repo}
export QSA_USE_IMAGE_DSL_STACK=1
export QSA_CUTLASS_DSL_PACKAGES=${qsa_cutlass_packages}
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME=${qsa_cache_root}/xdg
export HF_HOME=${qsa_cache_root}/huggingface
export TORCH_HOME=${qsa_cache_root}/torch
export CUDA_CACHE_PATH=${qsa_cache_root}/cuda
# FlashInfer does not use XDG_CACHE_HOME.  It appends .cache/flashinfer to
# FLASHINFER_WORKSPACE_BASE, so pin the base separately to avoid the host-home
# bind that Pyxis exposes as /root.
export FLASHINFER_WORKSPACE_BASE=${qsa_cache_root}/flashinfer-workspace
mkdir -p \
  "${XDG_CACHE_HOME}" \
  "${HF_HOME}" \
  "${TORCH_HOME}" \
  "${CUDA_CACHE_PATH}" \
  "${FLASHINFER_WORKSPACE_BASE}"

common_bench_args=(
  --backend openai
  --base-url "http://127.0.0.1:${qsa_port}"
  --endpoint /v1/completions
  --model Qwen/Qwen3.8-Flash-Next
  --tokenizer "${qsa_model}"
  --request-rate inf
  --temperature 0
  --ignore-eos
  --ready-check-timeout-sec "${qsa_ready_check_timeout_sec}"
  --percentile-metrics ttft,tpot,itl,e2el
  --metric-percentiles 50,90,99
)
profile_args=()
if [[ ${qsa_profile} == true ]]; then
  profile_args+=(--profile)
fi

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
    server_args=(
      --model "${qsa_model}"
      --served-model-name Qwen/Qwen3.8-Flash-Next
      --reasoning-parser qwen3
      --tensor-parallel-size "${qsa_tp_size}"
      --disable-custom-all-reduce
      --gpu-memory-utilization "${qsa_gpu_memory_utilization}"
      --kv-cache-dtype "${qsa_kv_cache_dtype}"
      --max-model-len "${qsa_max_model_len}"
      --max-num-batched-tokens "${qsa_max_num_batched_tokens}"
      --max-num-seqs "${qsa_max_num_seqs}"
      --cudagraph-capture-sizes "${cudagraph_capture_sizes[@]}"
      --kernel-config "{\"enable_cutedsl_warmup\":${qsa_enable_cutedsl_warmup}}"
      --no-enable-flashinfer-autotune
      --host 0.0.0.0
      --port "${qsa_port}"
    )
    if [[ ${qsa_enable_prefix_caching} == true ]]; then
      server_args+=(--enable-prefix-caching)
    else
      server_args+=(--no-enable-prefix-caching)
    fi
    if (( qsa_mtp_tokens > 0 )); then
      server_args+=(
        --speculative-config
        "{\"method\":\"mtp\",\"num_speculative_tokens\":${qsa_mtp_tokens}}"
      )
    fi
    if [[ ${qsa_profile} == true ]]; then
      server_args+=(
        --profiler-config
        '{"profiler":"cuda","detailed_trace_annotation":true}'
      )
    fi
    cd "${qsa_repo}"
    exec "${qsa_python}" -m vllm.entrypoints.openai.api_server "${server_args[@]}"
    ;;
  prefill)
    if [[ $# -ne 3 ]]; then
      usage
    fi
    input_len=$3
    measured_prompts=${QSA_PREFILL_PROMPTS:-3}
    warmup_prompts=${QSA_PREFILL_WARMUP_PROMPTS:-1}
    if ! [[ ${warmup_prompts} =~ ^[0-9]+$ ]]; then
      echo "QSA_PREFILL_WARMUP_PROMPTS must be a non-negative integer" >&2
      exit 2
    fi
    # vLLM starts profiling after its built-in warm-up phase. When the server
    # health endpoint was checked separately, set QSA_READY_CHECK_TIMEOUT_SEC=0;
    # three warm-ups then make the captured benchmark request exactly request
    # four. Prefix caching must remain disabled so repeated prompts still run
    # complete prefill.
    run_bench \
      "${common_bench_args[@]}" \
      "${profile_args[@]}" \
      --dataset-name random \
      --random-input-len "${input_len}" \
      --random-output-len 1 \
      --random-range-ratio 0 \
      --num-warmups "${warmup_prompts}" \
      --num-prompts "${measured_prompts}" \
      --max-concurrency 1 \
      --seed "${qsa_seed}" \
      --save-result \
      --result-dir "${result_dir}" \
      --result-filename "${backend}-${qsa_dtype_label}-mtp${qsa_mtp_tokens}-tp${qsa_tp_size}-prefill-input${input_len}-bs1.json"
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
    warmup_prompts=${QSA_DECODE_WARMUP_PROMPTS:-${measured_prompts}}
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
      --num-prompts "${warmup_prompts}" \
      --max-concurrency "${batch_size}" \
      --seed "${warm_seed}"
    run_bench \
      "${common_bench_args[@]}" \
      "${profile_args[@]}" \
      --dataset-name random \
      --random-input-len "${input_len}" \
      --random-output-len "${qsa_decode_output_len}" \
      --random-range-ratio 0 \
      --num-prompts "${measured_prompts}" \
      --max-concurrency "${batch_size}" \
      --seed "${qsa_seed}" \
      --save-result \
      --result-dir "${result_dir}" \
      --result-filename "${backend}-${qsa_dtype_label}-mtp${qsa_mtp_tokens}-tp${qsa_tp_size}-decode-independent-input${input_len}-bs${batch_size}.json"
    ;;
  *)
    usage
    ;;
esac
