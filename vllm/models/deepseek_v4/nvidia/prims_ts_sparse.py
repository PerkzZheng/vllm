# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prims-TS attention using the existing native-cache DeepSeek producers."""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    DeepseekSparseSWAFlashInferBackend,
    DeepseekV4FlashInferMLAAttention,
    DeepseekV4FlashInferMLASparseBackend,
)
from vllm.v1.attention.backends.mla.prims_ts_utils import (
    quantized_mla_output,
    sparse_mla,
)
from vllm.v1.attention.backends.mla.sparse_utils import align_blocks_to_rows

logger = init_logger(__name__)


class DeepseekV4PrimsTSBackend(DeepseekV4FlashInferMLASparseBackend):
    supported_kv_cache_dtypes = ["auto", "bfloat16", "fp8", "fp8_e4m3"]
    # Storage-row indices require a whole number of native rows between pages.
    customize_spec = staticmethod(align_blocks_to_rows)

    @staticmethod
    def get_name() -> str:
        return "PRIMS_TS_MLA_SPARSE_DSV4"

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        return capability.major == 10


class DeepseekV4PrimsTSBackendSWA(DeepseekSparseSWAFlashInferBackend):
    customize_spec = staticmethod(align_blocks_to_rows)


class PrimsTSAttentionMixin(torch.nn.Module):
    """Share the attention call, preserving V4/V4.1 cache and index production."""

    def __init__(self, vllm_config, *args, **kwargs):
        if vllm_config.attention_config.use_non_causal:
            raise NotImplementedError("Prims-TS DeepSeek currently requires causal MLA")
        super().__init__(vllm_config, *args, **kwargs)
        self._fuse_rope_quant = (
            self.wo_a.weight.dtype == torch.float8_e4m3fn
            and self.n_local_heads == self.padded_heads
            and (self.nope_head_dim, self.rope_head_dim) == (448, 64)
            and self._einsum_recipe[2] in (32, 128)
        )
        if self._fuse_rope_quant:
            logger.info_once(
                "Prims-TS sparse MLA fuses output inverse RoPE and FP8 "
                "quantization with block size %d.",
                self._einsum_recipe[2],
            )

    def _alloc_attn_out(self, num_tokens, hidden_states):
        if self._fuse_rope_quant:
            return quantized_mla_output(
                num_tokens,
                self.padded_heads,
                self.n_local_groups,
                self._einsum_recipe[2],
                hidden_states.device,
            )
        return torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )

    def _forward(
        self,
        q,
        kv_cache,
        swa_k_cache,
        swa_metadata,
        attn_metadata,
        swa_only,
        output,
        positions=None,
    ):
        rows = swa_metadata.num_decode_tokens + swa_metadata.num_prefill_tokens
        if rows == 0:
            return
        compressed, _, indices, lengths = self._build_sparse_index_metadata(
            kv_cache=kv_cache,
            swa_k_cache=swa_k_cache,
            swa_metadata=swa_metadata,
            attn_metadata=attn_metadata,
            swa_only=swa_only,
        )
        swa_width = swa_metadata.decode_swa_width
        if swa_metadata.prefill_left_visible is not None:
            swa_width += getattr(self, "max_image_tokens", 0)
        q = q[:rows]
        if q.shape[1] < self.padded_heads:
            padded = q.new_zeros((rows, self.padded_heads, q.shape[-1]))
            padded[:, : q.shape[1]] = q
            q = padded
        fp8 = q.dtype == torch.float8_e4m3fn
        quantized = isinstance(output, QuantizedActivation)
        sparse_mla(
            q,
            swa_k_cache,
            indices[:, :swa_width],
            lengths.clamp(max=swa_width),
            extra_kv_cache=compressed if not swa_only else None,
            extra_indices=indices[:, swa_width:] if not swa_only else None,
            extra_lengths=(lengths - swa_width).clamp(min=0) if not swa_only else None,
            bmm1_scale=self._flashinfer_fp8_bmm1_scale if fp8 else self.scale,
            bmm2_scale=self._flashinfer_fp8_bmm2_scale if fp8 else 1.0,
            sinks=self.attn_sink,
            out=output if quantized else output[:rows],
            output_quant_block_size=self._einsum_recipe[2] if quantized else None,
            token_positions=positions[:rows] if quantized else None,
            cos_sin_cache=self.rotary_emb.cos_sin_cache if quantized else None,
        )


class DeepseekV4PrimsTSAttention(
    PrimsTSAttentionMixin, DeepseekV4FlashInferMLAAttention
):
    backend_cls = DeepseekV4PrimsTSBackend
    swa_backend_cls = DeepseekV4PrimsTSBackendSWA
