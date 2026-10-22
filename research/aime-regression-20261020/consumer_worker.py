# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare one activation storage layout through production grouped W_o_a."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import traceback
from pathlib import Path
from types import SimpleNamespace

from common import DATA, RUN, VLLM_ROOT, check_runtime, verify_fixtures

ROOT = RUN / "consumer"
PROBE = DATA
SOURCE = VLLM_ROOT
MODEL = (
    Path(os.environ["SPARSE_MLA_EVAL_ROOT"]) / "models/deepseek-ai--DeepSeek-V4-Flash"
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=("token", "group"), required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    destination = ROOT / "results" / f"{args.layout}-{args.shard}"
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {
        "snapshots": sorted(
            str(p.relative_to(DATA)) for p in (DATA / "snapshots").glob("*/layer-*.pt")
        )
    }

    import torch
    from safetensors import safe_open

    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        _upcast_e8m0_to_fp32,
        deepgemm_post_process_fp8_weight_block,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic128Sym,
    )
    from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
        compute_fp8_einsum_recipe,
        deep_gemm_fp8_o_proj,
    )
    from vllm.utils.deep_gemm import _import_deep_gemm

    pinned_runtime = check_runtime(torch)
    verify_fixtures()
    assert (MODEL / "EVAL_REVISION").read_text().strip() == pinned_runtime[
        "model_revision"
    ]
    selected_dg = _import_deep_gemm()
    expected_dg = SOURCE / "vllm/third_party/deep_gemm"
    assert selected_dg is not None
    assert (
        Path(selected_dg.__file__).resolve() == (expected_dg / "__init__.py").resolve()
    )
    dg_binary = (expected_dg / "_C_extension.abi3.so").resolve()
    assert dg_binary in {Path(x).resolve() for x in torch.ops.loaded_libraries}
    assert sha(dg_binary) == pinned_runtime["deep_gemm_binary_sha256"]
    torch.backends.cuda.matmul.allow_tf32 = False
    recipe, aligned = compute_fp8_einsum_recipe(128)
    assert recipe == (1, 1, 128) and aligned
    config = json.loads((MODEL / "config.json").read_text())
    groups, rank = config["o_groups"], config["o_lora_rank"]
    heads = config["num_attention_heads"]
    weight_map = json.loads((MODEL / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    runtime = {
        "job": os.environ.get("SLURM_JOB_ID"),
        "layout": args.layout,
        "shard": args.shard,
        "gpu": torch.cuda.get_device_name(0),
        "visible_device": os.environ["CUDA_VISIBLE_DEVICES"],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "recipe": recipe,
        "fixtures_manifest_sha256": sha(Path(__file__).parent / "fixtures.json"),
        "deep_gemm": {
            "module": selected_dg.__file__,
            "loaded_binary": str(dg_binary),
            "binary_sha256": sha(dg_binary),
        },
        "weights": {},
    }

    def tensor_sha(tensor):
        raw = tensor.detach().cpu().contiguous().view(torch.uint8)
        return hashlib.sha256(raw.numpy().tobytes()).hexdigest()

    weights = {}

    def load_weight(prefix):
        if prefix in weights:
            return weights[prefix]
        # Captured self.prefix is the actual model attention prefix, not call index.
        assert prefix.startswith("model.layers.")
        name = prefix.removeprefix("model.") + ".wo_a"
        assert name + ".weight" in weight_map, name
        tensors = []
        for suffix in (".weight", ".scale"):
            with safe_open(
                MODEL / weight_map[name + suffix], framework="pt", device="cpu"
            ) as handle:
                tensors.append(handle.get_tensor(name + suffix))
        raw_weight, raw_scale = tensors
        block = tuple(config["quantization_config"]["weight_block_size"])
        assert raw_weight.shape == (groups * rank, heads // groups * 512)
        assert raw_scale.dtype in (torch.uint8, torch.float8_e8m0fnu)
        assert raw_scale.shape == (
            raw_weight.shape[0] // block[0],
            raw_weight.shape[1] // block[1],
        )
        numeric = _upcast_e8m0_to_fp32(raw_scale).double().to("cuda")
        reference = raw_weight.to("cuda").double() * numeric.repeat_interleave(
            block[0], 0
        ).repeat_interleave(block[1], 1)
        reference = reference.view(groups, rank, -1)
        weight, scale = deepgemm_post_process_fp8_weight_block(
            raw_weight.to("cuda"),
            raw_scale.to("cuda"),
            block,
            use_e8m0=True,
            is_bmm=True,
            bmm_batch_size=groups,
        )
        assert tensor_sha(weight) == tensor_sha(raw_weight)
        runtime["weights"][prefix] = {
            "name": name,
            "weight_file": weight_map[name + ".weight"],
            "scale_file": weight_map[name + ".scale"],
            "raw_weight_sha256": tensor_sha(raw_weight),
            "raw_scale_sha256": tensor_sha(raw_scale),
            "post_weight_sha256": tensor_sha(weight),
            "post_scale_sha256": tensor_sha(scale),
            "weight_shape": list(weight.shape),
            "scale_shape": list(scale.shape),
            "scale_stride": list(scale.stride()),
            "quant_block": block,
        }
        weights[prefix] = SimpleNamespace(weight=weight, weight_scale=scale), reference
        return weights[prefix]

    records = []
    paths = [PROBE / rel for rel in manifest["snapshots"]][args.shard :: 2]
    assert len(paths) == 30
    stage = "setup"
    try:
        for path in paths:
            snapshot = torch.load(path, map_location="cpu", weights_only=True)
            replay_path = (
                PROBE
                / "replays/current"
                / (f"{snapshot['label']}-layer-{snapshot['call']:02d}.pt")
            )
            replay = torch.load(replay_path, map_location="cpu", weights_only=True)
            source_data, source_scales = replay["fused"], replay["scales"]
            rows, count, channels = source_data.shape
            assert count == groups and channels == heads // groups * 512
            assert rows in (1, 3, 30)
            assert torch.equal(
                source_data.contiguous().view(torch.uint8),
                snapshot["result"].contiguous().view(torch.uint8),
            )
            assert torch.equal(source_scales, snapshot["out_scale"])
            shape = (rows, groups, channels)
            if args.layout == "token":
                data = torch.empty(shape, device="cuda", dtype=source_data.dtype)
            else:
                data = torch.empty(
                    (groups, rows, channels), device="cuda", dtype=source_data.dtype
                ).transpose(0, 1)
            data.view(torch.uint8).copy_(source_data.view(torch.uint8))
            pitch = (rows + 3) // 4 * 4
            scale = torch.full(
                (groups, channels // 512, pitch),
                0x7F7F7F7F,
                device="cuda",
                dtype=torch.int32,
            ).permute(2, 0, 1)[:rows]
            scale.copy_(source_scales)
            assert tensor_sha(data) == tensor_sha(source_data)
            assert tensor_sha(scale) == tensor_sha(source_scales)
            assert data.stride() == (
                (groups * channels, channels, 1)
                if args.layout == "token"
                else (channels, rows * channels, 1)
            )
            assert scale.stride() == (1, channels // 512 * pitch, pitch)
            activation = QuantizedActivation(
                data, scale, torch.bfloat16, data.shape, kFp8Dynamic128Sym
            )
            wo_a, reference_weight = load_weight(snapshot["layer_prefix"])
            positions = snapshot["token_positions"].to("cuda")
            rope = snapshot["cos_sin_cache"].to("cuda")

            def consume(
                activation=activation,
                positions=positions,
                rope=rope,
                wo_a=wo_a,
                rows=rows,
            ):
                return deep_gemm_fp8_o_proj(
                    activation,
                    positions,
                    rope,
                    wo_a,
                    lambda z: z,
                    n_groups=groups,
                    heads_per_group=heads // groups,
                    nope_dim=448,
                    rope_dim=64,
                    o_lora_rank=rank,
                    einsum_recipe=recipe,
                    tma_aligned_scales=aligned,
                ).view(rows, groups, rank)

            stage = "eager"
            output = consume()
            torch.accelerator.synchronize()
            first = output.clone()
            again = consume()
            torch.accelerator.synchronize()
            repeat_equal = torch.equal(first, again)
            exponents = torch.stack(
                [(scale.to(torch.int64) >> (8 * i)) & 255 for i in range(4)], -1
            ).flatten(-2)
            assert bool((exponents < 255).all())
            decoded = (
                data.double().unflatten(-1, (-1, 128))
                * torch.exp2(exponents.double() - 127)[..., None]
            )
            expected = torch.einsum(
                "tgc,grc->tgr", decoded.flatten(-2), reference_weight
            )
            error = (first.double() - expected).abs()
            # Same production-consumer tolerance as prior qualification; FP64 oracle.
            oracle_pass = bool((error <= 0.001 + 0.008 * expected.abs()).all())
            record = {
                "fixture": str(path.relative_to(PROBE)),
                "layer_prefix": snapshot["layer_prefix"],
                "rows": rows,
                "input_sha256": tensor_sha(data),
                "scale_sha256": tensor_sha(scale),
                "data_stride": list(data.stride()),
                "scale_stride": list(scale.stride()),
                "repeat_equal": repeat_equal,
                "oracle_pass": oracle_pass,
                "max_abs": error.max().item(),
                "relative_rmse": (
                    error.square().mean().sqrt() / expected.square().mean().sqrt()
                ).item(),
                "eager": "passed",
                "graph": "not_attempted",
            }
            records.append(record)
            torch.save(first.cpu(), destination / (replay_path.stem + ".pt"))
            stage = "graph"
            try:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_output = consume()
                graph.replay()
                torch.accelerator.synchronize()
                graph_first = graph_output.clone()
                graph.replay()
                torch.accelerator.synchronize()
                record["graph"] = "passed"
                record["graph_equal_eager"] = torch.equal(first, graph_output)
                record["graph_repeat_equal"] = torch.equal(graph_first, graph_output)
                record["graph_sha256"] = tensor_sha(graph_output)
            except Exception:
                record["graph"] = "runtime_rejected"
                record["graph_error"] = traceback.format_exc()
                raise
            assert tensor_sha(data) == record["input_sha256"]
            assert tensor_sha(scale) == record["scale_sha256"]
            print(json.dumps(record), flush=True)
    except Exception:  # noqa: BLE001 - preserve CUDA rejection without reuse.
        runtime["failure"] = {
            "stage": stage,
            "fixture": str(path.relative_to(PROBE)),
            "classification": "runtime_rejected_or_setup_error",
            "traceback": traceback.format_exc(),
        }
    finally:
        runtime["records"] = records
        runtime["planned_fixtures"] = [str(p.relative_to(PROBE)) for p in paths]
        runtime["completed"] = len(records) == 30 and "failure" not in runtime
        (destination / "report.json").write_text(json.dumps(runtime, indent=2) + "\n")
    if not runtime["completed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
