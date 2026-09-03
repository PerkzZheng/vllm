# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA QSA owner with Triton kernels."""

from __future__ import annotations

import os
from typing import ClassVar, cast

import torch
from torch import nn

from vllm import _custom_ops as custom_ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding, get_rope
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    canonicalize_singleton_dim_strides,
    direct_register_custom_op,
    is_quantized_kv_cache,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

from ..common.qsa_cache import QSAForwardMetadata
from . import model
from .indexer_qsa import QSAIndexer

_QSA_BACKEND_ENV = "VLLM_QSA_ATTENTION_BACKEND"
_QSA_DIAGNOSTIC_NVTX = os.environ.get("VLLM_QSA_DIAGNOSTIC_NVTX", "0") == "1"


def _qsa_nvtx_push(name: str) -> None:
    if _QSA_DIAGNOSTIC_NVTX:
        torch.cuda.nvtx.range_push(name)


def _qsa_nvtx_pop() -> None:
    if _QSA_DIAGNOSTIC_NVTX:
        torch.cuda.nvtx.range_pop()


def _resolve_qsa_prims_ts_backend(*, capable: bool, available: bool) -> bool:
    """Resolve the QSA attention implementation selected for this process.

    ``auto`` preserves the production default. The explicit modes make
    end-to-end accuracy and performance A/B tests reproducible without
    changing the model or attention-backend interface.
    """

    backend = os.environ.get(_QSA_BACKEND_ENV, "auto").strip().lower()
    if backend not in ("auto", "triton", "prims_ts"):
        raise ValueError(
            f"{_QSA_BACKEND_ENV} must be auto, triton, or prims_ts; got {backend!r}"
        )
    if backend == "triton":
        return False
    supported = capable and available
    if backend == "prims_ts" and not supported:
        raise RuntimeError(
            "VLLM_QSA_ATTENTION_BACKEND=prims_ts requires an SM100-family GPU "
            "and the FlashInfer page-4 PrimTS API"
        )
    return supported


class Qwen4ExpQSAMetadataBuilder(FlashAttentionMetadataBuilder):
    """Flash metadata supporting uniform decode and target-verify graphs."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class Qwen4ExpQSAFlashAttentionBackend(FlashAttentionBackend):
    """FullAttentionSpec backend used by the merged QSA owner."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_name() -> str:
        return "QWEN4_EXP_QSA_TRITON"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # QSA consumes manager pages directly and does not use FA4 paged attention.
        return [MultipleOf(16)]

    @staticmethod
    def get_impl_cls() -> type[Qwen4ExpQSAFlashAttentionImpl]:
        return Qwen4ExpQSAFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[Qwen4ExpQSAMetadataBuilder]:
        return Qwen4ExpQSAMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
    """Run paged sparse GQA with the QSA Triton kernel."""

    supports_dcp: bool = False
    supports_pcp: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        # Reuse FlashAttention's metadata/DCP initialization, but do not apply
        # its dense-kernel FP8 capability gate. QSA never calls the inherited
        # dense attention kernel: it owns FP8 query quantization, cache update,
        # descales, and sparse attention end to end below.
        base_kv_cache_dtype = (
            "auto" if is_quantized_kv_cache(kv_cache_dtype) else kv_cache_dtype
        )
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=base_kv_cache_dtype,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            sinks=sinks,
        )
        self.kv_cache_dtype = kv_cache_dtype
        if not is_flash_attn_varlen_func_available():
            raise NotImplementedError("Qwen4Exp QSA requires FlashAttention")
        if self.dcp_world_size != 1:
            raise NotImplementedError(
                "Qwen4Exp QSA does not support decode context parallelism"
            )
        if self.kv_cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
            raise NotImplementedError(
                "Qwen4Exp QSA supports BF16 and FP8-E4M3 KV caches"
            )
        self.supports_quant_query_input = False
        from .ops.qsa import has_qsa_prims_ts_attention

        self.use_qsa_prims_ts = _resolve_qsa_prims_ts_backend(
            capable=current_platform.is_device_capability_family(100),
            available=has_qsa_prims_ts_attention(),
        )

    def _get_qsa_prims_ts_workspace(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        block_table: torch.Tensor,
        block_topk: int,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        from .ops.qsa import qsa_prims_ts_combined_workspace_size

        seq_len_q = query.shape[1] if query.ndim == 4 else 1
        workspace_key = (
            query.device,
            query.shape[0],
            seq_len_q,
            tuple(key_cache.shape),
            key_cache.data_ptr(),
            block_table.shape[1],
            block_topk,
            query.dtype,
            out_dtype,
        )
        workspace_pool = getattr(layer, "_qsa_prims_ts_workspace_pool", None)
        if workspace_pool is None:
            workspace_pool = {}
            layer._qsa_prims_ts_workspace_pool = workspace_pool
        workspace = workspace_pool.get(workspace_key)
        if workspace is None:
            required_bytes = qsa_prims_ts_combined_workspace_size(
                query,
                key_cache,
                block_table,
                block_topk,
                out_dtype=out_dtype,
            )
            # CUDA graphs replay without re-entering this Python owner. Give
            # every semantic graph key and KV-cache generation its own
            # allocation so throwaway graph-memory profiling storage cannot
            # survive into the real capture. Metadata fully initializes its
            # outputs and bitmap scratch; split-KV counters are initialized
            # once by the prepared FlashInfer plan.
            workspace = torch.empty(
                required_bytes,
                dtype=torch.uint8,
                device=query.device,
            )
            workspace_pool[workspace_key] = workspace
        # Retain the last allocation under the historical attribute for
        # diagnostics and integrations that inspect it.
        layer._qsa_prims_ts_workspace = workspace
        return workspace

    def _get_qsa_prims_ts_plan(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_indices: torch.Tensor,
        block_table: torch.Tensor,
        token_to_req: torch.Tensor,
        logical_positions: torch.Tensor,
        workspace: torch.Tensor,
        out: torch.Tensor,
        bmm1_scale: float,
        bmm2_scale: float,
    ) -> object:
        """Return a validated plan for one graph-stable QSA semantic key."""

        from .ops.qsa import qsa_prims_ts_prepare_attention

        plan_key = (
            query.device,
            tuple(query.shape),
            query.stride(),
            query.dtype,
            tuple(out.shape),
            out.stride(),
            out.dtype,
            tuple(key_cache.shape),
            key_cache.stride(),
            key_cache.data_ptr(),
            value_cache.data_ptr(),
            tuple(block_indices.shape),
            block_indices.stride(),
            tuple(block_table.shape),
            block_table.stride(),
            workspace.data_ptr(),
            float(bmm1_scale),
            float(bmm2_scale),
        )
        plan_pool = getattr(layer, "_qsa_prims_ts_plan_pool", None)
        if plan_pool is None:
            plan_pool = {}
            layer._qsa_prims_ts_plan_pool = plan_pool
        plan = plan_pool.get(plan_key)
        if plan is None:
            plan = qsa_prims_ts_prepare_attention(
                query,
                key_cache,
                value_cache,
                block_indices,
                block_table,
                token_to_req,
                logical_positions,
                workspace,
                out,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
            )
            plan_pool[plan_key] = plan
        return plan

    def forward_qsa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        token_to_req: torch.Tensor,
        logical_positions: torch.Tensor,
        query_start_loc_cpu: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QSA does not support fused output quantization")
        if self.alibi_slopes is not None or self.sinks is not None:
            raise NotImplementedError("QSA does not support ALiBi or attention sinks")
        if self.sliding_window != (-1, -1):
            raise NotImplementedError("QSA does not support sliding-window attention")

        num_tokens = attn_metadata.num_actual_tokens
        output.zero_()
        if num_tokens == 0:
            return output

        topk_buffer = getattr(layer, "topk_indices_buffer", None)
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        logical_indices = topk_buffer[:num_tokens]
        indices_are_blocks = bool(getattr(layer, "qsa_indices_are_blocks", False))
        token_to_req = token_to_req[:num_tokens]
        logical_positions = logical_positions[:num_tokens]
        if query.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA requires BF16 Q/output")

        query_for_attention = query[:num_tokens]
        bmm1_scale = self.scale
        bmm2_scale = 1.0
        fp8_query_buffer: torch.Tensor | None = None
        if is_quantized_kv_cache(self.kv_cache_dtype):
            if kv_cache.dtype != torch.uint8:
                raise ValueError("FP8 QSA cache storage must use encoded uint8 bytes")
            kv_cache = kv_cache.view(current_platform.fp8_dtype())
            fp8_query_buffer = getattr(layer, "_qsa_fp8_query_buffer", None)
            if fp8_query_buffer is None or fp8_query_buffer.shape[0] < num_tokens:
                raise RuntimeError("QSA owner did not provide its FP8 query buffer")
            query_for_attention = fp8_query_buffer[:num_tokens]
            custom_ops.scaled_fp8_quant(
                query[:num_tokens].view(num_tokens, -1),
                scale=layer._q_scale,
                output=query_for_attention.view(num_tokens, -1),
            )
            bmm1_scale *= layer._q_scale_float * layer._k_scale_float
            bmm2_scale = layer._v_scale_float
        elif kv_cache.dtype != torch.bfloat16:
            raise ValueError("BF16 QSA cache storage must use BF16")

        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)

        if self.use_qsa_prims_ts:
            from .ops.qsa import (
                qsa_prims_ts_group_size,
                qsa_prims_ts_run_prepared,
            )

            if not indices_are_blocks:
                raise RuntimeError(
                    "PrimTS QSA requires the indexer compact page-four block output"
                )

            # PR 53896 stores the combined cache as [P,Hkv,N,2D]. The common
            # transpose/split above gives Triton's [P,N,Hkv,D] views; PrimTS
            # consumes HND pages, so recover [P,Hkv,N,D] without a copy.
            prims_key_cache = canonicalize_singleton_dim_strides(
                key_cache.transpose(1, 2)
            )
            prims_value_cache = canonicalize_singleton_dim_strides(
                value_cache.transpose(1, 2)
            )
            _qsa_nvtx_push("qsa_prims_route_setup")
            # Piecewise prefill executes eagerly at the attention boundary;
            # route exactly the live rows. Full CUDA graphs already capture a
            # fixed token extent and do not need a second staging policy here.
            prims_output = output[:num_tokens]
            group_size = qsa_prims_ts_group_size(
                query_for_attention,
                prims_key_cache,
                query_start_loc_cpu,
                group_size=int(getattr(layer, "qsa_prims_ts_group_size", 1)),
            )
            route_rows = num_tokens // group_size
            _qsa_nvtx_pop()

            _qsa_nvtx_push("qsa_prims_plan_lookup")
            # The grouped union bitmap is indexed by logical page-four ID.
            # Bound it by the largest live context, not by the model's full
            # block-table allocation.  For example, an 8K prefill needs only
            # three 3,136-token storage pages instead of all 45 pages reserved
            # for a 139K context.  Full CUDA-graph capture remains safe because
            # its attention metadata deliberately sets max_seq_len to the
            # model-length upper bound.
            active_storage_pages = min(
                attn_metadata.block_table.shape[1],
                (
                    attn_metadata.max_seq_len
                    + prims_key_cache.shape[2]
                    - 1
                )
                // prims_key_cache.shape[2],
            )
            metadata_block_table = attn_metadata.block_table[
                :, :active_storage_pages
            ]
            if group_size == 1:
                route_query = query_for_attention
                route_output = prims_output
            else:
                route_query = query_for_attention.view(
                    route_rows,
                    group_size,
                    query.shape[1],
                    query.shape[2],
                )
                route_output = prims_output.view_as(route_query)
            workspace = self._get_qsa_prims_ts_workspace(
                layer,
                route_query,
                prims_key_cache,
                metadata_block_table,
                logical_indices.shape[1],
                route_output.dtype,
            )
            plan = self._get_qsa_prims_ts_plan(
                layer,
                route_query,
                prims_key_cache,
                prims_value_cache,
                logical_indices,
                metadata_block_table,
                token_to_req,
                logical_positions,
                workspace,
                route_output,
                bmm1_scale,
                bmm2_scale,
            )
            _qsa_nvtx_pop()

            _qsa_nvtx_push("qsa_prims_metadata_attention_launch")
            qsa_prims_ts_run_prepared(
                plan,
                route_query,
                logical_indices,
                metadata_block_table,
                token_to_req,
                logical_positions,
                route_output,
            )
            _qsa_nvtx_pop()
            return output

        from .ops.qsa import qsa_sparse_paged_attention

        _qsa_nvtx_push("qsa_triton_attention_launch")
        qsa_sparse_paged_attention(
            query_for_attention,
            key_cache,
            value_cache,
            logical_indices,
            attn_metadata.block_table,
            token_to_req,
            output[:num_tokens],
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
        )
        _qsa_nvtx_pop()
        return output


class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
    """Merged Qwen full-attention owner with a QSA index side branch."""

    supports_dcp = False

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        """Bind one cache generation and discard plans bound to its predecessor.

        vLLM first binds a minimal cache while profiling CUDA-graph memory,
        tears it down, and later binds the real cache. Prepared PrimTS plans
        retain K/V tensor maps, and workspaces allocated during the profiling
        capture belong to a throwaway graph pool. Neither may cross this
        rebinding boundary.
        """

        workspace_pool = getattr(self, "_qsa_prims_ts_workspace_pool", None)
        if workspace_pool is not None:
            workspace_pool.clear()
        plan_pool = getattr(self, "_qsa_prims_ts_plan_pool", None)
        if plan_pool is not None:
            plan_pool.clear()
        self._qsa_prims_ts_workspace = None
        self.kv_cache = kv_cache

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("Qwen4Exp QSA requires a paged KV cache")
        if model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA currently requires BF16")
        if cache_config.cache_dtype not in (
            "auto",
            "bfloat16",
            "fp8",
            "fp8_e4m3",
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA supports BF16 and FP8-E4M3 KV caches"
            )
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support KV quantization")
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA does not support context parallelism"
            )
        if not getattr(config, "is_causal", True):
            raise NotImplementedError("Qwen4Exp QSA requires causal decoder attention")

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support dual-chunk RoPE")
        # Qwen4Exp full-attention checkpoints always pack a sigmoid output
        # gate next to Q, even when an inherited config default says otherwise.
        self.attn_output_gate = True

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=model.without_modelopt_fp4(quant_config),
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        mm_config = model_config.multimodal_config
        text_only = mm_config is None or mm_config.language_model_only
        mrope_section = getattr(self.rotary_emb, "mrope_section", None)
        supports_mrope = bool(
            type(self.rotary_emb) is MRotaryEmbedding
            and mrope_section
            and len(mrope_section) == 3
            and sum(mrope_section) == self.rotary_emb.rotary_dim // 2
            and getattr(self.rotary_emb, "mrope_interleaved", False)
        )
        supports_dtype = getattr(self.rotary_emb, "dtype", None) in (
            torch.float16,
            torch.bfloat16,
        )
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and current_platform.is_cuda()
            and supports_dtype
            and (text_only or supports_mrope)
        )

        self.layer_name = f"{prefix}.attn"
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = cache_config.cache_dtype
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        if self.kv_cache_torch_dtype not in (torch.bfloat16, torch.uint8):
            raise NotImplementedError(
                "Qwen4Exp QSA requires BF16 or encoded FP8 cache storage"
            )
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        set_default_quant_scales(self, register_buffer=True)

        self.attn_backend = Qwen4ExpQSAFlashAttentionBackend
        self.impl = Qwen4ExpQSAFlashAttentionImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            None,
            None,
            self.kv_cache_dtype,
            None,
            AttentionType.DECODER,
            None,
        )
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.qsa_indices_are_blocks = self.impl.use_qsa_prims_ts
        configured_group_size = getattr(config, "qsa_query_group_size", None)
        if configured_group_size is None:
            # The user-selected MTP width is a fixed semantic Q group, not a
            # performance heuristic. Unsupported widths stay on Q1.
            configured_group_size = vllm_config.uniform_decode_query_len
            if configured_group_size not in (1, 2, 4, 5):
                configured_group_size = 1
        elif configured_group_size not in (1, 2, 4, 5):
            raise ValueError(
                "qsa_query_group_size must be one of 1, 2, 4, or 5; got "
                f"{configured_group_size}"
            )
        self.qsa_prims_ts_group_size = int(configured_group_size)
        selection_width = (
            self.indexer.block_topk
            if self.qsa_indices_are_blocks
            else self.indexer.output_width
        )
        self.register_buffer(
            "topk_indices_buffer",
            torch.empty(
                max_tokens,
                selection_width,
                dtype=torch.int32,
            ),
            persistent=False,
        )
        if is_quantized_kv_cache(self.kv_cache_dtype):
            self.register_buffer(
                "_qsa_fp8_query_buffer",
                torch.zeros(
                    max_tokens,
                    self.num_heads,
                    self.head_dim,
                    dtype=current_platform.fp8_dtype(),
                ),
                persistent=False,
            )
        self._qsa_prims_ts_workspace: torch.Tensor | None = None
        self._qsa_prims_ts_workspace_pool: dict[tuple[object, ...], torch.Tensor] = {}
        self._qsa_prims_ts_plan_pool: dict[tuple[object, ...], object] = {}

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        """Finalize host descales used by the model-facing FP8 QSA path."""

        self.impl.process_weights_after_loading(act_dtype)
        for name in ("q", "k", "v"):
            scale = float(getattr(self, f"_{name}_scale").item())
            setattr(self, f"_{name}_scale_float", scale)
        self._k_scale_cpu.fill_(self._k_scale_float)
        self._v_scale_cpu.fill_(self._v_scale_float)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    def _run_qsa(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            output.zero_()
            return
        main_metadata = cast(FlashAttentionMetadata, metadata[self.layer_name])
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")

        num_tokens = main_metadata.num_actual_tokens
        side_metadata = cast(
            QSAForwardMetadata,
            metadata[self.indexer.raw_key_cache.prefix],
        )
        if side_metadata.num_actual_tokens != num_tokens:
            raise RuntimeError("QSA main and side metadata token counts disagree")
        selected = self.indexer(
            hidden_states,
            positions,
            self.topk_indices_buffer[:num_tokens],
            compact_blocks=self.qsa_indices_are_blocks,
        )
        if selected.shape != (
            num_tokens,
            self.topk_indices_buffer.shape[1],
        ):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        impl.do_kv_cache_update(
            self,
            key,
            value,
            self.kv_cache,
            main_metadata.slot_mapping,
        )
        impl.forward_qsa(
            self,
            query,
            key,
            value,
            self.kv_cache,
            main_metadata,
            output,
            token_to_req=side_metadata.token_to_req,
            logical_positions=side_metadata.logical_positions,
            query_start_loc_cpu=side_metadata.query_start_loc_cpu,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        num_tokens = hidden_states.shape[0]
        query = q.view(num_tokens, self.num_heads, self.head_dim)
        key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        attn_output = torch.empty_like(query)
        encoded_layer_name = _encode_layer_name(self.layer_name)
        if current_platform.opaque_attention_op():
            torch.ops.vllm.qwen4_exp_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                attn_output,
                encoded_layer_name,
            )
        else:
            qwen4_exp_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                attn_output,
                encoded_layer_name,
            )
        flat_output = attn_output.view(num_tokens, -1)
        if gate is not None:
            flat_output = flat_output * torch.sigmoid(gate)
        output, _ = self.o_proj(flat_output)
        return output


def qwen4_exp_qsa_with_output(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Run the complete QSA state/update/attend transaction."""

    layer_name = _resolve_layer_name(layer_name)
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen4ExpQSAAttention):
        raise TypeError(f"{layer_name} is not a Qwen4Exp QSA owner")
    layer._run_qsa(
        hidden_states,
        positions,
        query,
        key,
        value,
        output,
    )


def qwen4_exp_qsa_with_output_fake(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    del hidden_states, positions, query, key, value, output, layer_name


direct_register_custom_op(
    op_name="qwen4_exp_qsa_with_output",
    op_func=qwen4_exp_qsa_with_output,
    mutates_args=["output"],
    fake_impl=qwen4_exp_qsa_with_output_fake,
)


__all__ = [
    "QSAIndexer",
    "Qwen4ExpQSAAttention",
    "Qwen4ExpQSAFlashAttentionBackend",
    "Qwen4ExpQSAFlashAttentionImpl",
    "qwen4_exp_qsa_with_output",
]
