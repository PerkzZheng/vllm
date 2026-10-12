# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native NoPE sparse MLA for GLM, using the same kernel for prefill/decode."""

import torch

from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseImpl,
    FlashInferMLASparseTRTLLMBackend,
)
from vllm.v1.attention.backends.mla.prims_ts_utils import sparse_mla


class PrimsTSMLASparseBackend(FlashInferMLASparseTRTLLMBackend):
    supported_dtypes = [torch.bfloat16]
    supported_kv_cache_dtypes = ["auto", "bfloat16", "fp8", "fp8_e4m3"]

    @staticmethod
    def get_name() -> str:
        return "PRIMS_TS_MLA_SPARSE"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512]

    @staticmethod
    def get_impl_cls():
        return PrimsTSMLASparseImpl


class PrimsTSMLASparseImpl(FlashInferMLASparseImpl):
    lse_base_on_e = True

    def _prepare_mqa_kernel(self, layer, device):
        self.bmm1_scale = self.scale
        self.bmm2_scale = 1.0
        if self.kv_cache_dtype.startswith("fp8"):
            self.bmm1_scale *= layer._q_scale_float * layer._k_scale_float
            self.bmm2_scale *= layer._k_scale_float

    def _run_mqa_kernel(self, q, kv_cache, topk_indices, seq_lens):
        assert self.bmm1_scale is not None and self.bmm2_scale is not None
        return sparse_mla(
            q,
            kv_cache.view(q.dtype),
            topk_indices,
            seq_lens,
            bmm1_scale=self.bmm1_scale,
            bmm2_scale=self.bmm2_scale,
            return_lse=self.need_to_return_lse_for_decode,
        )
