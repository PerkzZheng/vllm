# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregate four isolated layout workers without changing either input."""

import json
from pathlib import Path

import torch
from common import RUN

ROOT = RUN / "consumer"
reports = {
    (layout, shard): json.loads(
        (ROOT / "results" / f"{layout}-{shard}/report.json").read_text()
    )
    for layout in ("token", "group")
    for shard in (0, 1)
}
weight_receipts = {}
for report in reports.values():
    for prefix, receipt in report["weights"].items():
        if prefix in weight_receipts:
            assert weight_receipts[prefix] == receipt, prefix
        else:
            weight_receipts[prefix] = receipt
records = {
    layout: {
        r["fixture"]: r for shard in (0, 1) for r in reports[layout, shard]["records"]
    }
    for layout in ("token", "group")
}
pairs = []
for fixture in sorted(records["token"].keys() & records["group"].keys()):
    old, new = records["token"][fixture], records["group"][fixture]
    assert old["input_sha256"] == new["input_sha256"]
    assert old["scale_sha256"] == new["scale_sha256"]
    assert old["scale_stride"] == new["scale_stride"]
    shard = next(
        s
        for s in (0, 1)
        if fixture in {r["fixture"] for r in reports["token", s]["records"]}
    )
    stem = fixture.split("/")[1] + "-" + Path(fixture).stem + ".pt"
    left = torch.load(
        ROOT / "results" / f"token-{shard}" / stem,
        map_location="cpu",
        weights_only=True,
    )
    right = torch.load(
        ROOT / "results" / f"group-{shard}" / stem,
        map_location="cpu",
        weights_only=True,
    )
    pairs.append(
        {
            "fixture": fixture,
            "rows": old["rows"],
            "equal": torch.equal(left, right),
            "max_abs": (left.float() - right.float()).abs().max().item(),
        }
    )
passed = (
    all(r["completed"] for r in reports.values())
    and len(pairs) == 60
    and all(r["equal"] for r in pairs)
    and all(
        r["oracle_pass"]
        and r["repeat_equal"]
        and r.get("graph_equal_eager")
        and r.get("graph_repeat_equal")
        for layout in records.values()
        for r in layout.values()
    )
)
summary = {
    "passed": passed,
    "paired_count": len(pairs),
    "weight_receipts_identical": True,
    "weights": weight_receipts,
    "pairs": pairs,
    "workers": {
        f"{layout}-{shard}": {
            "completed": report["completed"],
            "records": len(report["records"]),
            "failure": report.get("failure"),
        }
        for (layout, shard), report in reports.items()
    },
}
(ROOT / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary), flush=True)
raise SystemExit(0 if passed else 2)
