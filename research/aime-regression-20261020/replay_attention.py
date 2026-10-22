# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay captured model inputs in a fresh process for one attention revision."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("current", "historical"), required=True)
    parser.add_argument(
        "--root", type=Path, default=Path(os.environ["AIME_REPRO_ROOT"])
    )
    args = parser.parse_args()
    import torch
    from common import check_runtime, verify_fixtures

    check_runtime(torch)
    verify_fixtures()

    if args.variant == "historical":
        from historical_attention import activate

        sparse_mla = activate()
    else:
        from vllm.v1.attention.backends.mla.prims_ts_utils import sparse_mla
    from vllm.v1.attention.backends.mla.prims_ts_utils import quantized_mla_output

    paths = sorted((args.root / "snapshots").glob("*/layer-*.pt"))
    assert paths, "No real model inputs were captured"
    destination = Path(os.environ["AIME_REPRO_RUN"]) / "replays" / args.variant
    destination.mkdir(parents=True, exist_ok=False)

    def pool(record):
        if record is None:
            return None
        # Keep physical row IDs and their contiguity. The unreferenced suffix
        # of the original global cache cannot affect these requests.
        ids = record["row_ids"].to("cuda")
        extent = max(1, int(ids.max()) + 1) if ids.numel() else 1
        result = torch.zeros(
            (extent, 1, 512), device="cuda", dtype=record["values"].dtype
        )
        result.view(torch.uint8)[ids, 0] = record["values"].view(torch.uint8).to("cuda")
        return result

    for path in paths:
        record = torch.load(path, map_location="cpu", weights_only=True)
        q = record["q"].to("cuda")
        assert record["output_shape"][0] == q.shape[0], (
            "Probe expects unpadded eager output"
        )
        primary, extra = pool(record["primary"]), pool(record["extra"])
        tensors = {
            key: record[key].to("cuda") if record[key] is not None else None
            for key in (
                "indices",
                "lengths",
                "extra_indices",
                "extra_lengths",
                "sinks",
                "token_positions",
                "cos_sin_cache",
            )
        }
        kwargs = {
            "extra_kv_cache": extra,
            "extra_indices": tensors["extra_indices"],
            "extra_lengths": tensors["extra_lengths"],
            "bmm1_scale": record["bmm1_scale"],
            "bmm2_scale": record["bmm2_scale"],
            "sinks": tensors["sinks"],
        }
        plain, lse = sparse_mla(
            q,
            primary,
            tensors["indices"],
            tensors["lengths"],
            return_lse=True,
            **kwargs,
        )
        plain = plain.clone()
        lse = lse.clone()
        again, _ = sparse_mla(
            q,
            primary,
            tensors["indices"],
            tensors["lengths"],
            return_lse=True,
            **kwargs,
        )
        quantized = quantized_mla_output(
            q.shape[0],
            q.shape[1],
            record["output_groups"],
            record["output_quant_block_size"],
            q.device,
        )
        sparse_mla(
            q,
            primary,
            tensors["indices"],
            tensors["lengths"],
            out=quantized,
            output_quant_block_size=record["output_quant_block_size"],
            token_positions=tensors["token_positions"],
            cos_sin_cache=tensors["cos_sin_cache"],
            **kwargs,
        )
        replay = {
            "plain": plain.cpu(),
            "lse": lse.cpu(),
            "plain_repeat_equal": torch.equal(plain, again),
            "fused": quantized.data.cpu(),
            "scales": quantized.scale.cpu(),
        }
        torch.save(replay, destination / f"{record['label']}-{path.name}")
        print(
            json.dumps(
                {
                    "variant": args.variant,
                    "label": record["label"],
                    "layer": record["call"],
                    "plain_repeat_equal": replay["plain_repeat_equal"],
                }
            ),
            flush=True,
        )
        del q, primary, extra, tensors, kwargs, plain, again, lse, quantized
        torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
