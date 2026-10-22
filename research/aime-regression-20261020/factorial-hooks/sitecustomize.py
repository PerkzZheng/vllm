# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install the attention-only control before the model imports its utility."""

import importlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

if os.environ.get("AIME_FACTORIAL_SERVER") == "1":
    try:
        from common import check_sources

        check_sources()
        arm = Path(os.environ["SPARSE_MLA_ARM_ROOT"])
        variant = os.environ["AIME_FACTORIAL_ATTENTION"]
        assert variant in ("historical", "current")
        assert not any(
            name.startswith("vllm.models.deepseek_v4") for name in sys.modules
        ), "Model imported before the attention control"
        assert not any(
            name == "flashinfer.attention.prims_ts"
            or name.startswith("flashinfer.attention.prims_ts.")
            for name in sys.modules
        ), "Attention imported before the attention control"
        if variant == "historical":
            controls = Path(__file__).resolve().parents[1]
            sys.path.insert(0, str(controls))
            from historical_attention import activate

            function = activate()
        else:
            importlib.import_module("flashinfer.attention.prims_ts")

            function = None
        import torch

        from vllm.v1.attention.backends.mla import prims_ts_utils

        assert torch.__version__ == "2.13.0+cu130"
        assert torch.version.cuda == "13.0"
        assert importlib.metadata.version("nvidia-cutlass-dsl") == "4.8.0"
        expected_utility = (
            Path(os.environ["VLLM_ROOT"])
            / "vllm/v1/attention/backends/mla/prims_ts_utils.py"
        ).resolve()
        assert Path(prims_ts_utils.__file__).resolve() == expected_utility
        if function is not None:
            prims_ts_utils.sparse_mla = function
        import flashinfer.attention.prims_ts as attention

        if variant == "current":
            expected_attention = (
                Path(os.environ["FLASHINFER_ROOT"])
                / "flashinfer/attention/prims_ts/__init__.py"
            ).resolve()
            assert Path(attention.__file__).resolve() == expected_attention
        record = {
            "pid": os.getpid(),
            "variant": variant,
            "attention_origin": attention.__file__,
            "utility_origin": prims_ts_utils.__file__,
            "call_module": prims_ts_utils.sparse_mla.__module__,
            "consumer_allocator_module": prims_ts_utils.quantized_mla_output.__module__,
            "model_not_yet_imported": not any(
                n.startswith("vllm.models.deepseek_v4") for n in sys.modules
            ),
        }
        assert record["model_not_yet_imported"]
        from cache_control import install

        install(arm)
        (arm / "hook-origins").mkdir(exist_ok=True)
        (arm / "hook-origins" / f"{os.getpid()}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
    except Exception:  # noqa: BLE001 - control failure must abort startup.
        # sitecustomize exceptions otherwise only warn and silently keep running.
        import traceback

        traceback.print_exc()
        os._exit(91)
