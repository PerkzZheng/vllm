# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 Prims-TS attention preserving compressed KV/index sharing."""

from vllm.models.deepseek_v4.nvidia.prims_ts_sparse import PrimsTSAttentionMixin
from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
    DeepseekSparseSWAFlashInferBackend,
    DeepseekV4FlashInferMLAAttention,
    DeepseekV4FlashInferMLASparseBackend,
)
from vllm.v1.attention.backends.mla.sparse_utils import align_blocks_to_rows


class DeepseekV41PrimsTSBackend(DeepseekV4FlashInferMLASparseBackend):
    supported_kv_cache_dtypes = ["auto", "bfloat16", "fp8", "fp8_e4m3"]
    customize_spec = staticmethod(align_blocks_to_rows)

    @staticmethod
    def get_name() -> str:
        return "PRIMS_TS_MLA_SPARSE_DSV41"

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        return capability.major == 10


class DeepseekV41PrimsTSBackendSWA(DeepseekSparseSWAFlashInferBackend):
    customize_spec = staticmethod(align_blocks_to_rows)


class DeepseekV41PrimsTSAttention(
    PrimsTSAttentionMixin, DeepseekV4FlashInferMLAAttention
):
    backend_cls = DeepseekV41PrimsTSBackend
    swa_backend_cls = DeepseekV41PrimsTSBackendSWA
