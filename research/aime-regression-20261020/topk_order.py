# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Repeat the actual nightly TopK provider on fixed boundary logits."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from common import RUN, VLLM_ROOT, check_runtime

ROOT = RUN / "topk"
SOURCE = VLLM_ROOT
CAPACITY = 1048576 // 4
REPEATS = 128


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def preflight(torch):
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    return check_runtime(torch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("auto", "raw"), required=True)
    parser.add_argument("--scores", choices=("unique", "ties"), required=True)
    args = parser.parse_args()
    import torch

    from vllm.model_executor.layers import indexer_topk
    from vllm.v1.worker.workspace import init_workspace_manager

    manifest = preflight(torch)
    destination = ROOT / "results" / f"{args.provider}-{args.scores}.json"
    assert not destination.exists(), "Do not overwrite a completed diagnostic"
    init_workspace_manager(torch.device("cuda"))
    selector = indexer_topk.SparseIndexerTopk("auto")
    workspace = torch.empty(1024 * 1024, device="cuda", dtype=torch.uint8)
    loaded = sorted(
        str(Path(p).resolve())
        for p in torch.ops.loaded_libraries
        if Path(p).is_file() and "vllm" in str(Path(p).resolve())
    )
    binary = SOURCE / "vllm/_C_stable_libtorch.abi3.so"
    assert str(binary.resolve()) in loaded
    assert sha(binary) == manifest["provider_binary_sha256"]
    receipt = {
        "provider": args.provider,
        "scores": args.scores,
        "job": os.environ.get("SLURM_JOB_ID"),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "dispatcher_module": indexer_topk.__file__,
        "schema": str(torch.ops._C.cooperative_topk.default._schema),
        "dispatcher_table": torch._C._dispatch_dump_table("_C::cooperative_topk"),
        "loaded_vllm_libraries": {p: sha(Path(p)) for p in loaded},
        "actual_provider_binary": str(binary.resolve()),
        "actual_provider_binary_sha256": sha(binary),
        "capacity": CAPACITY,
        "repeats_per_mode": REPEATS,
        "source_to_binary_mapping_claimed": False,
        "cases": [],
        "same_set_order_pairs": [],
    }
    for rows in (1, 3, 30):
        for context in (2048, 2051, 2052, 2054, 2060):
            n = context // 4
            generator = torch.Generator().manual_seed(20261021 + rows * 100 + n)
            values = []
            for _ in range(rows):
                scores = torch.arange(n, dtype=torch.float32) / 1024 + 1
                if args.scores == "ties" and n > 512:
                    # 508 strictly larger values; choose4 from the cutoff tie.
                    scores[: n - 508] = 0.25
                values.append(scores[torch.randperm(n, generator=generator)])
            valid_cpu = torch.stack(values)
            logits = torch.full((rows, CAPACITY), -float("inf"), device="cuda")
            logits[:, :n].copy_(valid_cpu)
            seq_lens = torch.full((rows, 1), n, dtype=torch.int32, device="cuda")
            output = torch.full((rows, 512), -1, dtype=torch.int32, device="cuda")
            assert logits.stride() == (CAPACITY, 1)
            assert seq_lens.stride() == (1, 1)
            assert output.stride() == (512, 1)
            backend = selector.resolve_backend(logits, 512, rows)
            assert backend == "cooperative", backend

            def call(logits=logits, seq_lens=seq_lens, output=output, context=context):
                if args.provider == "auto":
                    selector(logits, seq_lens, 1, output, 512, context)
                else:
                    torch.ops._C.cooperative_topk(
                        logits, seq_lens, output, workspace, 512, context
                    )

            initial_logits = logits.clone()
            initial_lengths = seq_lens.clone()
            for _ in range(3):
                call()
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call()
            for mode in ("eager", "graph"):
                bank = torch.empty(
                    (REPEATS, rows, 512), dtype=torch.int32, device="cuda"
                )
                for repeat in range(REPEATS):
                    call() if mode == "eager" else graph.replay()
                    bank[repeat].copy_(output)
                observed = bank.cpu()
                assert torch.equal(logits, initial_logits)
                assert torch.equal(seq_lens, initial_lengths)
                ordered_hashes, set_hashes = set(), set()
                first_by_set = {}
                example = None
                for indices in observed:
                    canonical = indices.sort(dim=1).values
                    assert bool(((canonical >= 0) & (canonical < n)).all())
                    assert bool((canonical[:, 1:] > canonical[:, :-1]).all())
                    for row in range(rows):
                        selected = set(indices[row].tolist())
                        scores = valid_cpu[row].tolist()
                        cutoff = sorted(scores, reverse=True)[511]
                        strictly_greater = {
                            i for i, v in enumerate(scores) if v > cutoff
                        }
                        assert strictly_greater <= selected
                        assert all(scores[i] >= cutoff for i in selected)
                        if args.scores == "unique":
                            golden = set(
                                sorted(range(n), key=lambda i: scores[i], reverse=True)[
                                    :512
                                ]
                            )
                            assert selected == golden
                    ordered_hash, set_hash = digest(indices), digest(canonical)
                    ordered_hashes.add(ordered_hash)
                    set_hashes.add(set_hash)
                    if set_hash in first_by_set:
                        first = first_by_set[set_hash]
                        if example is None and not torch.equal(indices, first):
                            example = [first.tolist(), indices.tolist()]
                    else:
                        first_by_set[set_hash] = indices.clone()
                case = {
                    "rows": rows,
                    "context": context,
                    "valid_candidates": n,
                    "mode": mode,
                    "resolved_backend": backend,
                    "logits_shape": list(logits.shape),
                    "logits_stride": list(logits.stride()),
                    "seq_lens_shape": list(seq_lens.shape),
                    "seq_lens": [n] * rows,
                    "max_seq_len_argument": context,
                    "next_n": 1,
                    "output_shape": list(output.shape),
                    "output_stride": list(output.stride()),
                    "valid_logits_sha256": digest(valid_cpu),
                    "valid_logits_fp32": valid_cpu.tolist(),
                    "tail": "negative_infinity",
                    "input_unchanged": True,
                    "ordered_variants": len(ordered_hashes),
                    "set_variants": len(set_hashes),
                    "ordered_sha256": sorted(ordered_hashes),
                    "set_sha256": sorted(set_hashes),
                    "cpu_set_oracle_passed": True,
                }
                receipt["cases"].append(case)
                if example is not None:
                    receipt["same_set_order_pairs"].append(
                        {
                            "rows": rows,
                            "context": context,
                            "mode": mode,
                            "indices": example,
                        }
                    )
                print(
                    json.dumps(
                        {
                            k: case[k]
                            for k in (
                                "rows",
                                "context",
                                "mode",
                                "ordered_variants",
                                "set_variants",
                            )
                        }
                    ),
                    flush=True,
                )
            del graph, logits, initial_logits, bank
    receipt["passed"] = True
    destination.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
