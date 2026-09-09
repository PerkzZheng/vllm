# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""One-shot real QSA top-k capture layered over the normal runtime overlay.

Use only for benchmark-data collection. Set ``QSA_TOPK_CAPTURE_PATH`` and put
this directory first on ``PYTHONPATH`` (``run_e2e_perf.sh`` accepts it through
``QSA_RUNTIME_OVERLAY``). The first eager, single-request QSA call containing
at least ``QSA_TOPK_CAPTURE_MIN_ROWS`` rows is saved on rank zero. CUDA-graph
warmup calls are ignored by the row threshold and capture-state check.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import runpy
import sys
from pathlib import Path

_REPO = Path(os.environ.get("QSA_REPO", "/workspace/vllm-pr53896-qsa"))
runpy.run_path(
    str(_REPO / "benchmarks" / "qsa" / "runtime_overlay" / "sitecustomize.py"),
    run_name="_qsa_runtime_overlay",
)

# The runtime overlay must configure imports before Torch is initialized.
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

_CAPTURED = False
_TARGET_MODULE = "vllm.models.qwen4_exp.nvidia.qsa"


def _ranked_path(path: Path, rank: int) -> Path:
    suffix = path.suffix or ".pt"
    stem = path.name[: -len(suffix)] if path.suffix else path.name
    return path.with_name(f"{stem}-rank{rank}{suffix}")


def _install_capture(module) -> None:
    original_forward_qsa = module.Qwen4ExpQSAFlashAttentionImpl.forward_qsa

    def capture_forward_qsa(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        token_to_req,
        logical_positions,
        *args,
        **kwargs,
    ):
        global _CAPTURED

        result = original_forward_qsa(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            token_to_req,
            logical_positions,
            *args,
            **kwargs,
        )
        destination = os.environ.get("QSA_TOPK_CAPTURE_PATH")
        if not destination or _CAPTURED:
            return result
        num_tokens = int(attn_metadata.num_actual_tokens)
        min_rows = int(os.environ.get("QSA_TOPK_CAPTURE_MIN_ROWS", "4096"))
        if num_tokens < min_rows or torch.cuda.is_current_stream_capturing():
            return result
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            _CAPTURED = True
            return result

        active_requests = torch.unique(token_to_req[:num_tokens]).cpu()
        if active_requests.numel() != 1 or int(active_requests[0]) < 0:
            return result
        request = int(active_requests[0])
        topk = layer.topk_indices_buffer[:num_tokens].detach().cpu()
        payload = {
            "token_topk": 2048,
            "compress_ratio": 4,
            "main_storage_page_size": int(kv_cache.shape[2]),
            "token_to_req": torch.zeros(num_tokens, dtype=torch.int32),
            "logical_positions": logical_positions[:num_tokens].detach().cpu(),
            "main_block_table": (
                attn_metadata.block_table[request : request + 1].detach().cpu()
            ),
        }
        if bool(getattr(layer, "qsa_indices_are_blocks", False)):
            payload["selected_block_indices"] = topk
        else:
            # The indexer's final column is a count, not a logical token ID.
            payload["selected_token_indices"] = topk[:, :-1]
        path = _ranked_path(Path(destination), rank)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        print(f"[QSA top-k capture] saved {num_tokens} rows to {path}", flush=True)
        _CAPTURED = True
        return result

    module.Qwen4ExpQSAFlashAttentionImpl.forward_qsa = capture_forward_qsa


class _CaptureLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader) -> None:
        self.wrapped = wrapped

    def create_module(self, spec):
        create = getattr(self.wrapped, "create_module", None)
        return None if create is None else create(spec)

    def exec_module(self, module) -> None:
        self.wrapped.exec_module(module)
        _install_capture(module)


class _CaptureFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _CaptureLoader(spec.loader)
        return spec


sys.meta_path.insert(0, _CaptureFinder())
