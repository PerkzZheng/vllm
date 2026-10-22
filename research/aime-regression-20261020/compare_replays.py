# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare matched replays and unchanged test-suite FP64 attention oracle."""

import importlib.util
import json
import os
from pathlib import Path

import torch
from common import check_runtime, verify_fixtures

check_runtime(torch)
verify_fixtures()
ROOT = Path(os.environ["AIME_REPRO_ROOT"])
RUN = Path(os.environ["AIME_REPRO_RUN"])
SOURCE = Path(os.environ["FLASHINFER_ROOT"])
spec = importlib.util.spec_from_file_location(
    "frozen_sparse_mla_oracle",
    SOURCE / "tests/attention/test_attention_ts_sparse_mla.py",
)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)
torch.backends.cuda.matmul.allow_tf32 = False


def dequant(value, scale, block):
    exponent = torch.stack([(scale >> offset) & 255 for offset in (0, 8, 16, 24)], -1)
    numeric = torch.exp2(exponent.flatten(-2).double() - 127)
    return (value.double().unflatten(-1, (-1, block)) * numeric[..., None]).flatten(-2)


def difference(actual, expected):
    actual, expected = actual.double(), expected.double()
    delta = actual - expected
    return {
        "max_abs": delta.abs().max().item(),
        "relative_rmse": (delta.square().mean() / expected.square().mean())
        .sqrt()
        .item(),
        "finite": bool(torch.isfinite(actual).all()),
    }


reports = []
for path in sorted((ROOT / "snapshots").glob("*/layer-*.pt")):
    record = torch.load(path, weights_only=True, map_location="cpu")
    # Compact only the independent oracle's gathered tensor. Kernel replay
    # retains original physical indices; oracle arithmetic ignores layout.
    pools, indices = [], []
    for name, index_name in (("primary", "indices"), ("extra", "extra_indices")):
        pool = record[name]
        if pool is None:
            pools.append(None)
            indices.append(None)
            continue
        raw = record[index_name]
        mapped = torch.searchsorted(pool["row_ids"], raw.clamp(min=0).long())
        mapped[raw < 0] = -1
        # Columns outside lengths can carry arbitrary positive padding. They
        # are masked by the independent reference and cannot index out of range.
        length = record["lengths" if name == "primary" else "extra_lengths"]
        valid = torch.arange(raw.shape[-1])[None, :] < length[:, None]
        mapped[~valid] = -1
        indices.append(mapped.to(device="cuda", dtype=torch.int32))
        values = pool["values"].to("cuda")
        if not values.shape[0]:
            values = torch.zeros((1, 512), dtype=values.dtype, device="cuda")
        # Convert before the frozen oracle's indexing, which has no Float8
        # advanced-indexing kernel in some Torch builds. Values remain exact.
        pools.append(values.double().unsqueeze(1))
    q = record["q"].to("cuda")
    sinks = record["sinks"].to("cuda") if record["sinks"] is not None else None
    reference, lse, bound = oracle._reference(
        q,
        pools[0],
        pools[1],
        indices[0],
        indices[1],
        lengths=record["lengths"].to("cuda"),
        extra_lengths=record["extra_lengths"].to("cuda")
        if record["extra_lengths"] is not None
        else None,
        softmax_scale=record["bmm1_scale"],
        output_scale=record["bmm2_scale"],
        sinks=sinks,
    )
    reference, bound, lse = reference.cpu(), bound.cpu(), lse.cpu()
    rotated = oracle._inverse_output_rope(
        reference, record["token_positions"], record["cos_sin_cache"]
    )
    report = {"label": record["label"], "layer": record["call"], "q": list(q.shape)}
    arms = {}
    for variant in ("current", "historical"):
        replay = torch.load(
            RUN / "replays" / variant / f"{record['label']}-{path.name}",
            map_location="cpu",
            weights_only=True,
        )
        arms[variant] = replay
        fused = dequant(
            replay["fused"], replay["scales"], record["output_quant_block_size"]
        )
        report[variant] = {
            "plain": difference(replay["plain"], reference),
            "bound_violations": int(
                ((replay["plain"].double() - reference).abs() > bound).sum()
            ),
            "lse": difference(replay["lse"], lse),
            "fused": difference(fused.reshape_as(rotated), rotated),
            "plain_repeat_equal": replay["plain_repeat_equal"],
        }
    report["new_vs_old_plain"] = difference(
        arms["current"]["plain"], arms["historical"]["plain"]
    )
    report["new_vs_old_fused"] = difference(
        dequant(
            arms["current"]["fused"],
            arms["current"]["scales"],
            record["output_quant_block_size"],
        ),
        dequant(
            arms["historical"]["fused"],
            arms["historical"]["scales"],
            record["output_quant_block_size"],
        ),
    )
    report["capture_vs_current_bytes_equal"] = torch.equal(
        record["result"].view(torch.uint8), arms["current"]["fused"].view(torch.uint8)
    )
    report["capture_vs_current_scales_equal"] = torch.equal(
        record["out_scale"], arms["current"]["scales"]
    )
    report["historical_current_bitwise"] = {
        field: torch.equal(
            arms["current"][field].contiguous().view(torch.uint8),
            arms["historical"][field].contiguous().view(torch.uint8),
        )
        for field in ("plain", "lse", "fused", "scales")
    }
    reports.append(report)
    print(json.dumps(report), flush=True)
(RUN / "replay-comparison.json").write_text(json.dumps(reports, indent=2) + "\n")

passed = len(reports) == 60 and all(
    item[variant]["bound_violations"] == 0
    and item[variant]["plain_repeat_equal"]
    and all(item[variant][name]["finite"] for name in ("plain", "fused", "lse"))
    for item in reports
    for variant in ("current", "historical")
)
passed = passed and all(
    all(item["historical_current_bitwise"].values())
    and item["capture_vs_current_bytes_equal"]
    and item["capture_vs_current_scales_equal"]
    for item in reports
)
raise SystemExit(0 if passed else 2)
