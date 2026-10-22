#!/bin/bash
# Source once per arm with a unique cache directory and result directory.
: "${SPARSE_MLA_EVAL_ROOT:?Set an external directory for datasets, models and evaluator}"
: "${VLLM_ROOT:?Set this integration checkout}"
: "${FLASHINFER_ROOT:?Set the pinned FlashInfer checkout}"
: "${SPARSE_MLA_ARM_ROOT:?Set a unique writable directory per model/backend/cache/seed}"
export UV_CACHE_DIR="$SPARSE_MLA_EVAL_ROOT/cache/uv"
export XDG_CACHE_HOME="$SPARSE_MLA_ARM_ROOT/cache/general"
export VLLM_CACHE_ROOT="$SPARSE_MLA_ARM_ROOT/cache/vllm"
export VLLM_CONFIG_ROOT="$SPARSE_MLA_ARM_ROOT/cache/vllm-config"
export VLLM_NO_USAGE_STATS=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export FLASHINFER_WORKSPACE_BASE="$SPARSE_MLA_ARM_ROOT/cache/flashinfer"
export FLASHINFER_CUBIN_DIR="$SPARSE_MLA_ARM_ROOT/cache/flashinfer-cubins"
export DG_JIT_CACHE_DIR="$SPARSE_MLA_ARM_ROOT/cache/deep-gemm"
export CUTE_DSL_CACHE_DIR="$SPARSE_MLA_ARM_ROOT/cache/cute-dsl"
export CUTE_DSL_ARCH=sm_103a
export TILELANG_CACHE_DIR="$SPARSE_MLA_ARM_ROOT/cache/tilelang"
export TRITON_CACHE_DIR="$SPARSE_MLA_ARM_ROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$SPARSE_MLA_ARM_ROOT/cache/inductor"
export TORCH_EXTENSIONS_DIR="$SPARSE_MLA_ARM_ROOT/cache/torch-extensions"
export TORCH_HOME="$SPARSE_MLA_ARM_ROOT/cache/torch"
export CUDA_CACHE_PATH="$SPARSE_MLA_ARM_ROOT/cache/cuda"
export HF_HUB_CACHE="$SPARSE_MLA_EVAL_ROOT/cache/hub"
export HF_XET_CACHE="$SPARSE_MLA_EVAL_ROOT/cache/xet"
export HF_DATASETS_CACHE="$SPARSE_MLA_EVAL_ROOT/cache/datasets"
export HF_MODULES_CACHE="$SPARSE_MLA_EVAL_ROOT/cache/hf-modules"
export HF_ASSETS_CACHE="$SPARSE_MLA_EVAL_ROOT/cache/assets"
export PYTHONPATH="$VLLM_ROOT:$FLASHINFER_ROOT:$SPARSE_MLA_EVAL_ROOT/lm-evaluation-harness"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$SPARSE_MLA_ARM_ROOT" "$FLASHINFER_CUBIN_DIR" "$DG_JIT_CACHE_DIR" \
    "$CUTE_DSL_CACHE_DIR" "$XDG_CACHE_HOME/torch/kernels" "$UV_CACHE_DIR"
