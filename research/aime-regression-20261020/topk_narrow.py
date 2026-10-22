# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual-provider TopK membership check for distinct coarse-bin collisions."""

import argparse
import hashlib
import json
import os
import struct
import traceback
from pathlib import Path

from common import RUN, VLLM_ROOT, check_runtime

SOURCE = VLLM_ROOT
CAPACITY = 1048576 // 4
REPEATS = 32


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coarse_bin(value):
    bits = struct.unpack("<H", struct.pack("<e", value))[0]
    key = (~bits & 65535) if bits & 32768 else (bits | 32768)
    return key >> 4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    assert output_dir.is_relative_to(RUN)
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt_path = output_dir / "report.json"

    # Same frozen package paths, image and provider as the original probe.
    import torch

    from vllm.model_executor.layers import indexer_topk
    from vllm.v1.worker.workspace import init_workspace_manager

    manifest = check_runtime(torch)
    assert torch.__version__ == "2.13.0+cu130" and torch.version.cuda == "13.0"
    assert torch.accelerator.device_count() == 1
    assert "GB300" in torch.cuda.get_device_name(0)
    expected_dispatch = SOURCE / "vllm/model_executor/layers/indexer_topk.py"
    assert Path(indexer_topk.__file__).resolve() == expected_dispatch.resolve()
    binary = SOURCE / "vllm/_C_stable_libtorch.abi3.so"
    loaded = {str(Path(p).resolve()) for p in torch.ops.loaded_libraries}
    assert str(binary.resolve()) in loaded
    assert sha(binary) == manifest["provider_binary_sha256"]
    init_workspace_manager(torch.device("cuda"))
    selector = indexer_topk.SparseIndexerTopk("auto")
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
    report = {
        "completed": False,
        "job": os.environ.get("SLURM_JOB_ID"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "provider_binary": str(binary.resolve()),
        "provider_binary_sha256": sha(binary),
        "dispatcher_module": indexer_topk.__file__,
        "schema": str(torch.ops._C.cooperative_topk.default._schema),
        "dispatcher_table": torch._C._dispatch_dump_table("_C::cooperative_topk"),
        "source_to_binary_mapping_claimed": False,
        "repeats_per_mode": REPEATS,
        "cases": [],
    }
    try:
        for rows in (1, 3, 30):
            for context in (2048, 2052, 2060):
                n = context // 4
                for pattern in ("ascending", "descending"):
                    valid = 1.0 + torch.arange(n, dtype=torch.float32) * 2**-20
                    if pattern == "descending":
                        valid = valid.flip(0)
                    values = valid.tolist()
                    assert len(set(values)) == n
                    assert len({coarse_bin(value) for value in values}) == 1
                    golden = sorted(
                        sorted(range(n), key=lambda i: values[i], reverse=True)[:512]
                    )
                    expected = torch.tensor(golden, dtype=torch.int32)
                    logits = torch.full((rows, CAPACITY), -float("inf"), device="cuda")
                    logits[:, :n].copy_(valid.expand(rows, n))
                    lengths = torch.full((rows, 1), n, device="cuda", dtype=torch.int32)
                    output = torch.full(
                        (rows, 512), -1, device="cuda", dtype=torch.int32
                    )
                    initial = logits.clone()
                    assert selector.resolve_backend(logits, 512, rows) == "cooperative"
                    for provider in ("auto", "raw"):

                        def call(
                            provider=provider,
                            logits=logits,
                            lengths=lengths,
                            output=output,
                            context=context,
                        ):
                            if provider == "auto":
                                selector(logits, lengths, 1, output, 512, context)
                            else:
                                torch.ops._C.cooperative_topk(
                                    logits, lengths, output, workspace, 512, context
                                )

                        for _ in range(3):
                            call()
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
                            canonical = observed.sort(dim=-1).values
                            bounded = ((observed >= 0) & (observed < n)).all(-1)
                            distinct = (canonical[..., 1:] > canonical[..., :-1]).all(
                                -1
                            )
                            correct = (canonical == expected[None, None, :]).all(-1)
                            incorrect = ~(bounded & distinct & correct)
                            examples = incorrect.nonzero()
                            case = {
                                "rows": rows,
                                "context": context,
                                "valid_candidates": n,
                                "max_seq_len_argument": context,
                                "pattern": pattern,
                                "provider": provider,
                                "resolved_backend": "cooperative",
                                "mode": mode,
                                "logits_shape": list(logits.shape),
                                "logits_stride": list(logits.stride()),
                                "unique_fp32": True,
                                "coarse_bin": coarse_bin(values[0]),
                                "fp32_logits": values,
                                "golden_sorted_indices": golden,
                                "tested_rows": REPEATS * rows,
                                "incorrect_rows": int(incorrect.sum()),
                                "out_of_bounds_rows": int((~bounded).sum()),
                                "duplicate_rows": int((~distinct).sum()),
                                "golden_membership_mismatch_rows": int(
                                    (~correct).sum()
                                ),
                                "valid_but_wrong_set_rows": int(
                                    (bounded & distinct & ~correct).sum()
                                ),
                                "ordered_variants": len(
                                    {
                                        hashlib.sha256(
                                            t.contiguous().numpy().tobytes()
                                        ).hexdigest()
                                        for t in observed
                                    }
                                ),
                                "set_variants": len(
                                    {
                                        hashlib.sha256(
                                            t.contiguous().numpy().tobytes()
                                        ).hexdigest()
                                        for t in canonical
                                    }
                                ),
                            }
                            if len(examples):
                                repeat, row = examples[0].tolist()
                                indices = observed[repeat, row].tolist()
                                selected = set(indices)
                                case["first_failure"] = {
                                    "repeat": repeat,
                                    "row": row,
                                    "bounded": bool(bounded[repeat, row]),
                                    "distinct": bool(distinct[repeat, row]),
                                    "golden_membership": bool(correct[repeat, row]),
                                    "indices": indices,
                                    "missing_golden_indices": sorted(
                                        set(golden) - selected
                                    ),
                                    "unexpected_indices": sorted(
                                        selected - set(golden)
                                    ),
                                }
                            assert torch.equal(initial, logits)
                            assert bool((lengths == n).all())
                            case["inputs_unchanged"] = True
                            report["cases"].append(case)
                            receipt_path.write_text(json.dumps(report, indent=2) + "\n")
                            print(
                                json.dumps(
                                    {
                                        k: case[k]
                                        for k in (
                                            "rows",
                                            "context",
                                            "pattern",
                                            "provider",
                                            "mode",
                                            "tested_rows",
                                            "incorrect_rows",
                                            "ordered_variants",
                                            "set_variants",
                                        )
                                    }
                                ),
                                flush=True,
                            )
                        del graph
                    del logits, initial
        report["completed"] = True
        report["all_set_oracles_passed"] = not any(
            case["incorrect_rows"] for case in report["cases"]
        )
    except Exception:
        report["runtime_error"] = traceback.format_exc()
        raise
    finally:
        receipt_path.write_text(json.dumps(report, indent=2) + "\n")
    # Completion and correctness are distinct; retain every case before failing.
    if not report["all_set_oracles_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
