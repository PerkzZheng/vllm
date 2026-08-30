"""Overlay local QSA PrimTS Python code onto the Qwen runtime image.

The dedicated Qwen image supplies ABI-matched vLLM extensions and FlashInfer
GDN/MoE modules.  Replacing the complete FlashInfer package would hide those
image-only modules, so this startup hook overlays only the experimental
attention package and its trace templates.

Set ``QSA_FLASHINFER_SOURCE`` to the root of a FlashInfer source checkout and
put this directory first on ``PYTHONPATH``.  If the image's CUTLASS DSL is too
old for PrimTS, set ``QSA_CUTLASS_DSL_PACKAGES`` to the newer wheel's
``nvidia_cutlass_dsl/dsl_packages`` directory.  The hook is intentionally a
no-op for either overlay when its variable is unset.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType


_IMAGE_VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")


def _overlay_cutlass_dsl(packages: Path) -> None:
    """Prefer one wheel's DSL package without shadowing its other dependencies."""

    if not (packages / "cutlass" / "experimental").is_dir():
        raise ImportError(
            "QSA_CUTLASS_DSL_PACKAGES must contain cutlass/experimental"
        )
    packages_path = str(packages)
    if packages_path in sys.path:
        sys.path.remove(packages_path)
    # Installed CUTLASS wheels use a .pth file that inserts their own package
    # directory at index zero.  sitecustomize runs afterward, so restore the
    # explicitly requested runtime layer here.
    sys.path.insert(0, packages_path)


def _extend_vllm_for_image_extensions() -> None:
    import vllm

    if _IMAGE_VLLM.is_dir():
        image_path = str(_IMAGE_VLLM)
        if image_path not in vllm.__path__:
            vllm.__path__.append(image_path)


def _load_replacement(name: str, path: Path) -> ModuleType:
    """Execute ``path`` as ``name`` and update its parent-package binding."""

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create an import spec for {name} from {path}")
    previous = sys.modules.get(name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    parent_name, child_name = name.rsplit(".", 1)
    setattr(importlib.import_module(parent_name), child_name, module)
    return module


def _overlay_qsa_flashinfer(source_root: Path) -> None:
    package_root = source_root / "flashinfer"
    attention_root = package_root / "attention"
    trace_attention = package_root / "trace" / "templates" / "attention.py"
    if not attention_root.is_dir() or not trace_attention.is_file():
        raise ImportError(
            "QSA_FLASHINFER_SOURCE must name a FlashInfer checkout containing "
            "flashinfer/attention and flashinfer/trace/templates/attention.py"
        )

    # Import the image package first.  In particular, keep its root package,
    # JIT/cubin loader, GDN, and fused-MoE modules intact.
    import flashinfer.attention as attention_package
    import flashinfer.decode as public_decode

    _load_replacement(
        "flashinfer.trace.templates.attention",
        trace_attention,
    )

    local_attention_path = str(attention_root)
    if local_attention_path not in attention_package.__path__:
        attention_package.__path__.insert(0, local_attention_path)

    prims_decode = importlib.import_module("flashinfer.attention.prims_ts.decode")
    for name in (
        "get_prims_ts_batch_decode_workspace_size",
        "get_prims_ts_qsa_group_size",
        "prims_ts_batch_decode_with_kv_cache",
    ):
        setattr(public_decode, name, getattr(prims_decode, name))


if packages := os.environ.get("QSA_CUTLASS_DSL_PACKAGES"):
    _overlay_cutlass_dsl(Path(packages).resolve())
_extend_vllm_for_image_extensions()
if source := os.environ.get("QSA_FLASHINFER_SOURCE"):
    _overlay_qsa_flashinfer(Path(source).resolve())
