# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA QSA attention owner."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, cast

import torch
from torch import nn

from vllm import _custom_ops as custom_ops
from vllm import envs
from vllm.config import CUDAGraphMode, VllmConfig
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

from ..common.qsa_cache import QSA_PRIMS_TS_GROUP_SIZES, QSAForwardMetadata
from .indexer_qsa import QSAIndexer

_QSA_PREFILL_GROUP_SIZE = 4
_QSA_PRIMS_TS_HEAD_SIZE = 256
_QSA_PRIMS_TS_SPARSE_BLOCK_SIZE = 4
_QSA_PRIMS_TS_TILE_Q = 64
_QSA_PRIMS_TS_DEVICE_CAPABILITIES = (100, 103)
_QSA_PRIMS_TS_EAGER_PLAN_CACHE_CAPACITY = 4


@dataclass(frozen=True)
class _QSAPrimsTSPreparedState:
    key: tuple[object, ...]
    workspace: torch.Tensor
    plan: object


def _supports_qsa_prims_ts_geometry(
    *,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    sparse_block_size: int,
    max_group_size: int,
) -> bool:
    """Return whether every configured QSA route fits the PrimTS kernel."""

    return (
        head_size == _QSA_PRIMS_TS_HEAD_SIZE
        and sparse_block_size == _QSA_PRIMS_TS_SPARSE_BLOCK_SIZE
        and num_kv_heads > 0
        and num_heads % num_kv_heads == 0
        and max_group_size in QSA_PRIMS_TS_GROUP_SIZES
        and max_group_size * (num_heads // num_kv_heads) <= _QSA_PRIMS_TS_TILE_Q
    )


def _supports_qsa_prims_ts_device() -> bool:
    """Match the exact architectures implemented by the FlashInfer runtime."""

    return any(
        current_platform.is_device_capability(capability)
        for capability in _QSA_PRIMS_TS_DEVICE_CAPABILITIES
    )


def _has_qsa_prims_ts_attention() -> bool:
    """Probe the optional FlashInfer API only when backend resolution needs it."""

    from .ops.qsa import has_qsa_prims_ts_attention

    return has_qsa_prims_ts_attention()


def _resolve_qsa_prims_ts_backend(
    *, capable: bool, availability_probe: Callable[[], bool]
) -> bool:
    """Resolve the QSA attention implementation selected for this process.

    ``auto`` preserves the production default. The explicit modes make
    end-to-end accuracy and performance A/B tests reproducible without
    changing the model or attention-backend interface.
    """

    backend = envs.VLLM_QSA_ATTENTION_BACKEND
    if backend == "triton":
        return False
    available = availability_probe() if capable else False
    supported = capable and available
    if backend == "prims_ts" and not supported:
        raise RuntimeError(
            "VLLM_QSA_ATTENTION_BACKEND=prims_ts requires an SM100 or SM103 GPU, "
            "the FlashInfer sparse-block PrimTS QSA API, and a supported QSA "
            "geometry (head size 256, sparse block size 4, and grouped heads "
            "within TileQ64)"
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
        return "QWEN4_EXP_QSA"

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
    """Run paged sparse GQA with Triton or FlashInfer PrimTS."""

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
        *,
        qsa_sparse_block_size: int = _QSA_PRIMS_TS_SPARSE_BLOCK_SIZE,
        qsa_max_group_size: int = _QSA_PREFILL_GROUP_SIZE,
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
        geometry_supported = _supports_qsa_prims_ts_geometry(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            sparse_block_size=qsa_sparse_block_size,
            max_group_size=qsa_max_group_size,
        )
        self.use_qsa_prims_ts = _resolve_qsa_prims_ts_backend(
            capable=_supports_qsa_prims_ts_device() and geometry_supported,
            availability_probe=_has_qsa_prims_ts_attention,
        )

    def _get_qsa_prims_ts_prepared_state(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_indices: torch.Tensor,
        block_table: torch.Tensor,
        token_to_req: torch.Tensor,
        logical_positions: torch.Tensor,
        out: torch.Tensor,
        bmm1_scale: float,
        bmm2_scale: float,
        qo_indptr_cpu: torch.Tensor | None,
        qo_topology: tuple[int, ...] | None,
        group_size: int,
        sparse_block_size: int,
        *,
        persistent: bool = False,
    ) -> _QSAPrimsTSPreparedState:
        """Return workspace and plan for one graph-stable QSA geometry."""

        from .ops.qsa import (
            qsa_prims_ts_combined_workspace_size,
            qsa_prims_ts_prepare_attention,
        )

        state_key = (
            query.device,
            tuple(query.shape),
            query.stride(),
            query.dtype,
            tuple(out.shape),
            out.stride(),
            out.dtype,
            tuple(key_cache.shape),
            key_cache.stride(),
            key_cache.dtype,
            key_cache.data_ptr(),
            tuple(value_cache.shape),
            value_cache.stride(),
            value_cache.dtype,
            value_cache.data_ptr(),
            tuple(block_indices.shape),
            block_indices.stride(),
            block_indices.dtype,
            tuple(block_table.shape),
            block_table.stride(),
            block_table.dtype,
            tuple(token_to_req.shape),
            token_to_req.stride(),
            token_to_req.dtype,
            tuple(logical_positions.shape),
            logical_positions.stride(),
            logical_positions.dtype,
            float(bmm1_scale),
            float(bmm2_scale),
            group_size,
            sparse_block_size,
            qo_topology,
        )
        if persistent:
            state = layer._qsa_prims_ts_graph_states.get(state_key)
            if state is not None:
                return state
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "QSA PrimTS plans must be prepared during CUDA-graph warmup"
                )
        else:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "eager QSA PrimTS state cannot be used during CUDA-graph capture"
                )
            state = layer._qsa_prims_ts_eager_states.get(state_key)
            if state is not None:
                layer._qsa_prims_ts_eager_states.move_to_end(state_key)
                return state

        qo_indptr = (
            None
            if qo_indptr_cpu is None
            else qo_indptr_cpu.to(query.device, non_blocking=True)
        )
        required_bytes = qsa_prims_ts_combined_workspace_size(
            query,
            key_cache,
            block_table,
            block_indices.shape[1],
            out_dtype=out.dtype,
            qo_indptr=qo_indptr_cpu,
            max_seq_len_q=group_size if qo_indptr_cpu is not None else None,
            sparse_block_size=sparse_block_size,
        )
        if persistent:
            workspace = torch.empty(
                required_bytes,
                dtype=torch.uint8,
                device=query.device,
            )
        else:
            eager_workspace = layer._qsa_prims_ts_eager_workspace
            if (
                eager_workspace is None
                or eager_workspace.device != query.device
                or eager_workspace.numel() < required_bytes
            ):
                # Every eager plan binds typed views into this arena. Drop those
                # plans and the old arena before growing so none retain stale
                # workspace storage. Qwen4Exp rejects DBO/microbatching, and
                # ordinary eager launches are sequential on the current stream.
                layer._qsa_prims_ts_eager_states.clear()
                layer._qsa_prims_ts_eager_workspace = None
                del eager_workspace
                eager_workspace = torch.empty(
                    required_bytes,
                    dtype=torch.uint8,
                    device=query.device,
                )
                layer._qsa_prims_ts_eager_workspace = eager_workspace
            workspace = eager_workspace
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
            qo_indptr=qo_indptr,
            max_seq_len_q=group_size if qo_indptr is not None else None,
            sparse_block_size=sparse_block_size,
        )
        state = _QSAPrimsTSPreparedState(state_key, workspace, plan)
        if persistent:
            layer._qsa_prims_ts_graph_states[state_key] = state
        else:
            eager_states = layer._qsa_prims_ts_eager_states
            eager_states[state_key] = state
            eager_states.move_to_end(state_key)
            while len(eager_states) > _QSA_PRIMS_TS_EAGER_PLAN_CACHE_CAPACITY:
                eager_states.popitem(last=False)
        return state

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
        query_start_offsets: tuple[int, ...] | None = None,
        has_prefill: bool = True,
        uniform_decode_query_len: int | None = None,
        persistent_plan: bool = False,
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
                qsa_prims_ts_qo_indptr,
                qsa_prims_ts_run_prepared,
            )

            if not indices_are_blocks:
                raise RuntimeError(
                    "PrimTS QSA requires compact sparse-block indexer output"
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
            prims_output = output[:num_tokens]
            route_query_start_offsets = query_start_offsets
            if has_prefill:
                # Prefill always uses packed Q. Fast drafting metadata may omit
                # CPU request boundaries, in which case Q1 is the only
                # request-independent grouping.
                if route_query_start_offsets is None:
                    group_size = 1
                    route_query_start_offsets = (0, num_tokens)
                else:
                    group_size = _QSA_PREFILL_GROUP_SIZE
                use_fixed_layout = False
            elif query_start_offsets is None:
                # Missing CPU boundaries cannot prove multi-token request
                # ownership. Keep those launches request-independent.
                group_size = 1
                use_fixed_layout = True
            elif layer.qsa_prims_ts_decode_group_size == 1:
                group_size = 1
                use_fixed_layout = True
            elif uniform_decode_query_len == layer.qsa_prims_ts_decode_group_size:
                # Uniform target verification contributes MTP + 1 adjacent
                # rows per live request. Fixed routing is legal only after the
                # shared metadata builder proves those exact CPU boundaries.
                assert uniform_decode_query_len is not None
                group_size = uniform_decode_query_len
                use_fixed_layout = True
            else:
                # Irregular decode still groups within each request, never
                # across a boundary that happens to make total_q divisible.
                group_size = layer.qsa_prims_ts_decode_group_size
                use_fixed_layout = False

            route_qo_indptr_cpu: torch.Tensor | None = None
            if use_fixed_layout:
                if num_tokens % group_size:
                    raise RuntimeError(
                        "fixed QSA decode rows must be divisible by the query "
                        f"group size ({num_tokens=} {group_size=})"
                    )
                num_query_groups = num_tokens // group_size
                route_query = query_for_attention.view(
                    num_query_groups,
                    1,
                    group_size,
                    query_for_attention.shape[1],
                    query_for_attention.shape[2],
                )
                route_output = prims_output.view_as(route_query)
            else:
                assert route_query_start_offsets is not None
                route_qo_indptr_cpu = qsa_prims_ts_qo_indptr(
                    route_query_start_offsets,
                    num_tokens,
                    group_size,
                )
                route_query = query_for_attention
                route_output = prims_output

            # Eager metadata is bounded by the largest live context. Full
            # CUDA graphs instead retain the allocated width so warmup,
            # capture, and replay share one stable prepared-plan geometry.
            if persistent_plan:
                metadata_block_table = attn_metadata.block_table
            else:
                active_storage_pages = min(
                    attn_metadata.block_table.shape[1],
                    (attn_metadata.max_seq_len + prims_key_cache.shape[2] - 1)
                    // prims_key_cache.shape[2],
                )
                metadata_block_table = attn_metadata.block_table[
                    :, :active_storage_pages
                ]
            sparse_block_size = int(layer.indexer.compress_ratio)
            state = self._get_qsa_prims_ts_prepared_state(
                layer,
                route_query,
                prims_key_cache,
                prims_value_cache,
                logical_indices,
                metadata_block_table,
                token_to_req,
                logical_positions,
                route_output,
                bmm1_scale,
                bmm2_scale,
                route_qo_indptr_cpu,
                route_query_start_offsets if route_qo_indptr_cpu is not None else None,
                group_size,
                sparse_block_size,
                persistent=persistent_plan,
            )

            qsa_prims_ts_run_prepared(
                state.plan,
                route_query,
                logical_indices,
                metadata_block_table,
                token_to_req,
                logical_positions,
                route_output,
            )
            return output

        from .ops.qsa import qsa_sparse_paged_attention

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
        return output


class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
    """Merged Qwen full-attention owner with a QSA index side branch."""

    supports_dcp = False
    _qsa_prims_ts_graph_states: dict[tuple[object, ...], _QSAPrimsTSPreparedState]
    _qsa_prims_ts_eager_states: OrderedDict[
        tuple[object, ...], _QSAPrimsTSPreparedState
    ]
    _qsa_prims_ts_eager_workspace: torch.Tensor | None

    def _clear_qsa_prims_ts_prepared_storage(self) -> None:
        self._qsa_prims_ts_graph_states.clear()
        self._qsa_prims_ts_eager_states.clear()
        self._qsa_prims_ts_eager_workspace = None

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        """Bind one cache generation and discard plans bound to its predecessor.

        vLLM first binds a minimal cache while profiling CUDA-graph memory,
        tears it down, and later binds the real cache. Prepared PrimTS plans
        retain K/V tensor maps, and workspaces allocated during the profiling
        capture belong to a throwaway graph pool. Neither may cross this
        rebinding boundary.
        """

        self._clear_qsa_prims_ts_prepared_storage()
        self.kv_cache = kv_cache

    def unbind_kv_cache(self) -> None:
        """Release the KV cache and every prepared object that refers to it."""
        self._clear_qsa_prims_ts_prepared_storage()
        self.kv_cache = torch.tensor([])

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
            quant_config=(
                None
                if quant_config is not None
                and quant_config.get_name() == "modelopt_fp4"
                else quant_config
            ),
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

        decode_group_size = vllm_config.uniform_decode_query_len
        # TODO: add a Q3 kernel configuration instead of falling back to Q1
        # when a framework configures MTP=2.
        if decode_group_size not in QSA_PRIMS_TS_GROUP_SIZES:
            decode_group_size = 1
        self.qsa_prims_ts_decode_group_size = int(decode_group_size)
        sparse_block_size = int(config.indexer_compress_ratio)

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
            qsa_sparse_block_size=sparse_block_size,
            qsa_max_group_size=max(
                _QSA_PREFILL_GROUP_SIZE,
                self.qsa_prims_ts_decode_group_size,
            ),
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
        self._qsa_prims_ts_graph_states = {}
        self._qsa_prims_ts_eager_states = OrderedDict()
        self._qsa_prims_ts_eager_workspace = None

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
        forward_context = get_forward_context()
        metadata = forward_context.attn_metadata
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
            query_start_offsets=side_metadata.query_start_offsets,
            has_prefill=side_metadata.has_prefill,
            uniform_decode_query_len=side_metadata.uniform_decode_query_len,
            persistent_plan=(
                forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
                or side_metadata.prepare_cudagraph_plan
            ),
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
