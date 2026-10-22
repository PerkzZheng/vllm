# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse only missing native/generated artifacts from the exact base nightly."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((Path(__file__).parent / "protocol.json").read_text())
    if os.environ.get("VLLM_BUILD_COMMIT") != protocol["vllm_base"]:
        parser.error(
            "Run inside the exact pinned nightly; native ABI reuse requires its base"
        )
    # Run before adding the checkout to PYTHONPATH, without importing torch.
    package = Path(importlib.util.find_spec("vllm").origin).parent
    target = args.vllm_root.resolve()
    if package.is_relative_to(target):
        parser.error("Unset PYTHONPATH and run outside the vLLM checkout first")
    tracked = set(
        subprocess.check_output(
            ["git", "-C", str(target), "ls-files"], text=True
        ).splitlines()
    )
    copied = {}
    for source in package.rglob("*"):
        if not source.is_file() or "__pycache__" in source.parts:
            continue
        relative = source.relative_to(package.parent)
        destination = target / relative
        if destination.exists():
            continue
        if str(relative) in tracked:
            raise RuntimeError(f"Tracked source hidden by sparse checkout: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
    flashinfer = args.flashinfer_root.resolve()
    installed = Path(importlib.util.find_spec("flashinfer").origin).parent
    build_meta = flashinfer / "flashinfer/_build_meta.py"
    if not build_meta.exists():
        shutil.copyfile(installed / "_build_meta.py", build_meta)
    data = flashinfer / "flashinfer/data"
    data.mkdir(exist_ok=True)
    for name, relative in {
        "csrc": "csrc",
        "include": "include",
        "cutlass": "3rdparty/cutlass",
        "cccl": "3rdparty/cccl",
        "spdlog": "3rdparty/spdlog",
    }.items():
        source = flashinfer / relative
        if not source.is_dir():
            raise RuntimeError(f"Initialize FlashInfer submodules: {source}")
        link = data / name
        if not link.exists():
            link.symlink_to(source, target_is_directory=True)
        if link.resolve() != source:
            raise RuntimeError(f"Different generated-header link: {link}")
    receipt = dict(
        base=protocol["vllm_base"], installed_package=str(package), copied=copied
    )
    (Path(os.environ["SPARSE_MLA_EVAL_ROOT"]) / "runtime-artifacts.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(f"Prepared {len(copied)} missing artifacts; tracked Python sources unchanged")


if __name__ == "__main__":
    main()
