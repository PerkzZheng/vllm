# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared path and pinned-runtime checks for the exported diagnostics."""

import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
DATA = Path(os.environ["AIME_REPRO_ROOT"]).resolve()
RUN = Path(os.environ["AIME_REPRO_RUN"]).resolve()
VLLM_ROOT = Path(os.environ["VLLM_ROOT"]).resolve()


def check_sources():
    import flashinfer

    import vllm

    expected = json.loads((PACKAGE / "runtime.json").read_text())
    flashinfer_root = Path(os.environ["FLASHINFER_ROOT"]).resolve()
    assert Path(vllm.__file__).resolve() == VLLM_ROOT / "vllm/__init__.py"
    assert (
        Path(flashinfer.__file__).resolve()
        == flashinfer_root / "flashinfer/__init__.py"
    )
    for name, digest in expected["vllm"].items():
        assert hashlib.sha256((VLLM_ROOT / name).read_bytes()).hexdigest() == digest, (
            name
        )
    revision = subprocess.check_output(
        ["git", "-C", str(flashinfer_root), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == expected["flashinfer_revision"], revision
    subprocess.run(
        [
            "git",
            "-C",
            str(flashinfer_root),
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "flashinfer/attention/prims_ts",
            "tests/attention/test_attention_ts_sparse_mla.py",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(VLLM_ROOT),
            "diff",
            "--exit-code",
            expected["vllm_revision"],
            "--",
            "vllm",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return expected


def verify_fixtures():
    from prepare_fixtures import verify

    verify(DATA, json.loads((PACKAGE / "fixtures.json").read_text()))


def check_runtime(torch):
    expected = check_sources()
    assert torch.__version__ == "2.13.0+cu130" and torch.version.cuda == "13.0"
    assert importlib.metadata.version("nvidia-cutlass-dsl") == "4.8.0"
    assert torch.accelerator.device_count() == 1
    assert "GB300" in torch.cuda.get_device_name(0)
    assert not RUN.is_relative_to(DATA), "Keep newly generated results outside fixtures"
    assert not RUN.is_relative_to(PACKAGE), "Keep generated results outside Git"
    RUN.mkdir(parents=True, exist_ok=True)
    return expected
