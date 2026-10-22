# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure an observed same-set selector permutation on saved real attention."""

import json

from common import DATA, verify_fixtures
from topk_order import ROOT, digest, preflight


def main():
    import torch

    preflight(torch)
    verify_fixtures()
    from vllm.v1.attention.backends.mla.prims_ts_utils import (
        quantized_mla_output,
        sparse_mla,
    )

    pairs = []
    for path in sorted((ROOT / "results").glob("*-unique.json")):
        result = json.loads(path.read_text())
        assert result["passed"]
        pairs.extend((path.name, p) for p in result["same_set_order_pairs"])
    report = {"observed_same_set_order_pairs": len(pairs), "replays": []}
    if not pairs:
        report.update(status="not_attempted_no_observed_same_set_order_variation")
        (ROOT / "attention-order-replay.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        return
    source_report, pair = pairs[0]
    # The rank of each selected logical ID maps either observed order onto
    # exactly the same512 real KV rows; no synthetic logit becomes model data.
    row = next(
        i for i, (a, b) in enumerate(zip(*pair["indices"], strict=True)) if a != b
    )
    a, b = (torch.tensor(x[row], dtype=torch.int64) for x in pair["indices"])
    assert torch.equal(a.sort().values, b.sort().values)
    assert not torch.equal(a, b)
    permutations = [a.argsort().argsort(), b.argsort().argsort()]
    report["selector_pair"] = {
        "report": source_report,
        "rows": pair["rows"],
        "context": pair["context"],
        "mode": pair["mode"],
        "row": row,
        "permutations": [p.tolist() for p in permutations],
    }

    def pool(saved):
        ids = saved["row_ids"].long().to("cuda")
        extent = int(ids.max()) + 1
        result = torch.zeros(
            (extent, 1, 512), dtype=saved["values"].dtype, device="cuda"
        )
        result.view(torch.uint8)[ids, 0] = saved["values"].view(torch.uint8).to("cuda")
        return result

    for relative in (
        "case-08-tokens-8192/layer-02.pt",
        "batch3-tokens-8192/layer-02.pt",
    ):
        path = DATA / "snapshots" / relative
        saved = torch.load(path, map_location="cpu", weights_only=True)
        ci = saved["extra_indices"]
        cl = saved["extra_lengths"]
        if ci is None or not bool((cl == 512).all()):
            report["replays"].append(
                {"snapshot": relative, "status": "unsupported_not512livecompressedrows"}
            )
            continue
        assert ci.shape[1] >= 512
        assert bool((ci[:, :512] >= 0).all())
        assert all(row.unique().numel() == 512 for row in ci[:, :512])
        q = saved["q"].to("cuda")
        primary, extra = pool(saved["primary"]), pool(saved["extra"])
        tensors = {
            k: saved[k].to("cuda") if saved[k] is not None else None
            for k in (
                "indices",
                "lengths",
                "extra_lengths",
                "sinks",
                "token_positions",
                "cos_sin_cache",
            )
        }
        kwargs = {
            "extra_kv_cache": extra,
            "extra_lengths": tensors["extra_lengths"],
            "bmm1_scale": saved["bmm1_scale"],
            "bmm2_scale": saved["bmm2_scale"],
            "sinks": tensors["sinks"],
        }
        outputs = []
        for name, permutation in (
            ("original", None),
            ("order_a", permutations[0]),
            ("order_b", permutations[1]),
        ):
            ordered = ci.clone()
            if permutation is not None:
                ordered[:, :512] = ci[:, :512][:, permutation]
            assert torch.equal(
                ordered[:, :512].sort(dim=1).values, ci[:, :512].sort(dim=1).values
            )
            kwargs["extra_indices"] = ordered.to("cuda")
            plain, lse = sparse_mla(
                q,
                primary,
                tensors["indices"],
                tensors["lengths"],
                return_lse=True,
                **kwargs,
            )
            plain, lse = plain.clone(), lse.clone()
            again, _ = sparse_mla(
                q,
                primary,
                tensors["indices"],
                tensors["lengths"],
                return_lse=True,
                **kwargs,
            )
            assert torch.equal(plain, again), "Fixed-order attention itself changed"
            quant = quantized_mla_output(
                q.shape[0],
                q.shape[1],
                saved["output_groups"],
                saved["output_quant_block_size"],
                q.device,
            )
            sparse_mla(
                q,
                primary,
                tensors["indices"],
                tensors["lengths"],
                out=quant,
                output_quant_block_size=saved["output_quant_block_size"],
                token_positions=tensors["token_positions"],
                cos_sin_cache=tensors["cos_sin_cache"],
                **kwargs,
            )
            outputs.append(
                {
                    "name": name,
                    "plain": plain.cpu(),
                    "lse": lse.cpu(),
                    "fused_bytes": quant.data.view(torch.uint8).cpu(),
                    "scales": quant.scale.cpu(),
                    "indices": ordered,
                }
            )
        x, y = outputs[1:]
        delta = x["plain"].double() - y["plain"].double()
        row = {
            "snapshot": relative,
            "status": "completed",
            "q_shape": list(q.shape),
            "kv_values_unchanged": True,
            "selected_set_unchanged": True,
            "primary_indices_sha256": digest(saved["indices"]),
            "compressed_lengths_sha256": digest(cl),
            "per_order_fixed_input_repeat_equal": True,
            "plain_bitwise_equal": torch.equal(x["plain"], y["plain"]),
            "plain_max_abs_difference": float(delta.abs().max()),
            "plain_relative_l2_difference": float(
                delta.norm() / x["plain"].double().norm().clamp_min(1e-300)
            ),
            "lse_max_abs_difference": float((x["lse"] - y["lse"]).abs().max()),
            "fused_different_bytes": int((x["fused_bytes"] != y["fused_bytes"]).sum()),
            "fused_total_bytes": x["fused_bytes"].numel(),
            "different_scale_elements": int((x["scales"] != y["scales"]).sum()),
        }
        torch.save(
            outputs, ROOT / "results" / (relative.replace("/", "-") + "-orders.pt")
        )
        report["replays"].append(row)
        print(json.dumps(row), flush=True)
    report["status"] = "completed"
    report["causality_claimed"] = False
    (ROOT / "attention-order-replay.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
