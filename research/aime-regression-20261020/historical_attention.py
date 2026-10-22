# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Historical MLA with a current-base/model/consumer boundary for offline replay.

Call only in a fresh current-FlashInfer process. The producer writes historical
token-major FP8 scratch, then copies logical values to current group-major output;
scales and numerical values are never converted by this bridge.
"""

import functools
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path


@functools.cache
def activate_old_attention():
    import flashinfer

    attention = importlib.import_module("flashinfer.attention")

    root = Path(os.environ["AIME_REPRO_ROOT"]).resolve() / "source-controls"
    manifest = json.loads((root / "manifest.json").read_text())
    for relative, info in manifest["files"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != info["sha256"]:
            raise RuntimeError(f"Historical control source changed: {relative}")
    expected = (
        Path(os.environ["FLASHINFER_ROOT"]) / "flashinfer/__init__.py"
    ).resolve()
    if Path(flashinfer.__file__).resolve() != expected:
        raise RuntimeError(
            "The historical-attention control requires the recorded current FI base"
        )
    if any(
        name == "flashinfer.attention.prims_ts"
        or name.startswith("flashinfer.attention.prims_ts.")
        for name in sys.modules
    ):
        raise RuntimeError(
            "Start a fresh process before activating historical attention"
        )
    attention.__path__.insert(0, str(root / "old-attention/attention"))
    package = importlib.import_module("flashinfer.attention.prims_ts")
    if Path(package.__file__).resolve() != (
        root / "old-attention/attention/prims_ts/__init__.py"
    ):
        raise RuntimeError("Historical prims_ts package origin mismatch")
    name = "_aime_735e1f4f_metadata_utility"
    spec = importlib.util.spec_from_file_location(name, root / "old_prims_ts_utils.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def activate():
    """Activate the old attention package and return its layout-compatible call."""
    activate_old_attention()
    return old_sparse_mla


def old_sparse_mla(
    q,
    kv_cache,
    indices,
    lengths,
    *,
    extra_kv_cache=None,
    extra_indices=None,
    extra_lengths=None,
    bmm1_scale,
    bmm2_scale=1.0,
    sinks=None,
    out=None,
    return_lse=False,
    output_quant_block_size=None,
    token_positions=None,
    cos_sin_cache=None,
):
    import torch

    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation

    utility = activate_old_attention()
    quantized = isinstance(out, QuantizedActivation)
    producer_out = out
    if quantized:
        producer_out = replace(
            out,
            data=torch.empty(
                out.data.shape, dtype=out.data.dtype, device=out.data.device
            ),
        )
    values, lse = utility.sparse_mla(
        q,
        kv_cache,
        indices,
        lengths,
        extra_kv_cache=extra_kv_cache,
        extra_indices=extra_indices,
        extra_lengths=extra_lengths,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        sinks=sinks,
        out=producer_out,
        return_lse=return_lse,
        output_quant_block_size=output_quant_block_size,
        token_positions=token_positions,
        cos_sin_cache=cos_sin_cache,
    )
    if quantized:
        out.data[: q.shape[0]].copy_(values)
        values = out.data[: q.shape[0]]
    return values, lse
