"""Overlay local QToken-KvBlock-Sparse-Attention onto the Qwen runtime image.

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
import sys
from pathlib import Path
from types import ModuleType

_IMAGE_VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
_IMAGE_SITE_PACKAGES = _IMAGE_VLLM.parent
_IMAGE_CUTLASS_DSL_PACKAGES = (
    _IMAGE_SITE_PACKAGES / "nvidia_cutlass_dsl" / "dsl_packages"
)


def _prefer_image_dsl_stack() -> None:
    """Pin the image's ABI-matched TVM-FFI and CUTLASS packages."""

    if not _IMAGE_CUTLASS_DSL_PACKAGES.is_dir():
        raise ImportError("the image CUTLASS DSL package directory is missing")

    # CUTLASS wheels inject their dsl_packages path via .pth files.  Keep only
    # the image's path so a venv wheel cannot silently shadow the image ABI.
    sys.path[:] = [
        path
        for path in sys.path
        if not path.endswith("/nvidia_cutlass_dsl/dsl_packages")
    ]
    sys.path.insert(0, str(_IMAGE_CUTLASS_DSL_PACKAGES))

    # Import TVM-FFI while the image site-packages directory has priority, then
    # restore normal package ordering.  sys.modules pins its Python package and
    # native runtime for subsequent imports without shadowing unrelated venv
    # dependencies such as Triton.
    original_path = list(sys.path)
    sys.path.insert(0, str(_IMAGE_SITE_PACKAGES))
    try:
        tvm_ffi = importlib.import_module("tvm_ffi")
    finally:
        sys.path[:] = original_path
    if (
        not Path(tvm_ffi.__file__)
        .resolve()
        .is_relative_to(_IMAGE_SITE_PACKAGES.resolve())
    ):
        raise ImportError(
            "QSA_USE_IMAGE_DSL_STACK must be set before TVM-FFI is imported"
        )


def _overlay_cutlass_dsl(packages: Path) -> None:
    """Prefer one wheel's DSL package without shadowing its other dependencies."""

    if not (packages / "cutlass" / "experimental").is_dir():
        raise ImportError("QSA_CUTLASS_DSL_PACKAGES must contain cutlass/experimental")
    packages_path = str(packages)
    wheel_root = str(packages.parent.parent)
    if wheel_root in sys.path:
        sys.path.remove(wheel_root)
    # Make the matching nvidia_cutlass_dsl/{cu12,cu13} runtime libraries
    # visible in addition to its Python DSL package.
    sys.path.insert(0, wheel_root)
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


def _overlay_q_token_kv_block_sparse_flashinfer(source_root: Path) -> None:
    package_root = source_root / "flashinfer"
    attention_root = package_root / "attention"
    jit_metadata = package_root / "jit" / "q_token_kv_block_sparse_metadata.py"
    trace_template = package_root / "trace" / "template.py"
    trace_attention = package_root / "trace" / "templates" / "attention.py"
    if (
        not attention_root.is_dir()
        or not jit_metadata.is_file()
        or not trace_template.is_file()
        or not trace_attention.is_file()
    ):
        raise ImportError(
            "QSA_FLASHINFER_SOURCE must name a FlashInfer checkout containing "
            "the PrimTS attention, JIT metadata, and trace schema sources"
        )

    # Import the image package first.  In particular, keep its root package,
    # JIT/cubin loader, GDN, and fused-MoE modules intact.
    import flashinfer.attention as attention_package
    import flashinfer.decode as public_decode
    import flashinfer.jit as jit_package

    # The local attention schema uses newer TraceTemplate axis features (for
    # example fixed-value Const axes), so its template implementation and the
    # schema must be overlaid as one compatible pair.
    _load_replacement("flashinfer.trace.template", trace_template)
    _load_replacement("flashinfer.trace.templates.attention", trace_attention)

    # Keep the image's ABI-matched JIT runtime, but load the sparse metadata spec
    # from the same checkout as the attention code. The spec resolves its CUDA
    # and header inputs relative to its own file, so it does not depend on the
    # image wheel already containing this new kernel.
    metadata_jit = _load_replacement(
        "flashinfer.jit.q_token_kv_block_sparse_metadata", jit_metadata
    )
    metadata_jit_name = "gen_prims_ts_q_token_kv_block_sparse_metadata_module"
    setattr(jit_package, metadata_jit_name, getattr(metadata_jit, metadata_jit_name))

    local_attention_path = str(attention_root)
    if local_attention_path not in attention_package.__path__:
        attention_package.__path__.insert(0, local_attention_path)

    prims_decode = importlib.import_module("flashinfer.attention.prims_ts.decode")
    for name in (
        "get_prims_ts_batch_decode_workspace_size",
        "validate_q_token_kv_block_sparse_group_size",
        "make_q_token_kv_block_sparse_qo_indptr",
        "suggest_q_token_kv_block_sparse_group_size",
        "prepare_prims_ts_batch_decode_with_kv_cache",
        "prims_ts_batch_decode_with_kv_cache",
    ):
        setattr(public_decode, name, getattr(prims_decode, name))

    sparse_metadata = importlib.import_module(
        "flashinfer.attention.prims_ts.q_token_kv_block_sparse_metadata"
    )
    for name in (
        "QTokenKvBlockSparsePagedTSWrapper",
        "get_q_token_kv_block_sparse_workspace_size",
        "q_token_kv_block_sparse_attention_with_paged_kv_cache",
    ):
        setattr(public_decode, name, getattr(sparse_metadata, name))


if os.environ.get("QSA_USE_IMAGE_DSL_STACK") == "1":
    _prefer_image_dsl_stack()
if packages := os.environ.get("QSA_CUTLASS_DSL_PACKAGES"):
    _overlay_cutlass_dsl(Path(packages).resolve())
_extend_vllm_for_image_extensions()
if source := os.environ.get("QSA_FLASHINFER_SOURCE"):
    _overlay_q_token_kv_block_sparse_flashinfer(Path(source).resolve())
