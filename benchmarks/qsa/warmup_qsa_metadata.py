#!/usr/bin/env python3
"""Populate Triton's cache for QSA page-4 metadata adapters.

vLLM multiprocessing workers may be unable to spawn a compiler subprocess in
some container runtimes.  Run this helper once in an ordinary GPU container
with the same source mounts, Triton version, architecture, and cache directory
as the server.  The server workers can then load the cached cubins directly.
"""

from __future__ import annotations

import argparse

import torch

from vllm.models.qwen3_8_flash_next.nvidia.ops.qsa import (
    qsa_build_page4_grouped_paged_metadata,
    qsa_build_page4_paged_metadata,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-topk", type=int, default=2048)
    parser.add_argument("--storage-page-size", type=int, default=1600)
    parser.add_argument("--page-table-width", type=int, default=21)
    return parser.parse_args()


def _inputs(
    rows: int,
    *,
    token_topk: int,
    storage_page_size: int,
    page_table_width: int,
    position_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    output_width = token_topk + 3
    logical_indices = torch.arange(
        output_width,
        dtype=torch.int32,
        device=device,
    ).repeat(rows, 1)
    block_table = torch.zeros(
        (1, page_table_width),
        dtype=torch.int32,
        device=device,
    )
    token_to_req = torch.zeros(rows, dtype=torch.int32, device=device)
    logical_positions = torch.full(
        (rows,),
        min(output_width - 1, storage_page_size * page_table_width - 1),
        dtype=position_dtype,
        device=device,
    )
    return logical_indices, block_table, token_to_req, logical_positions


def main() -> None:
    args = _parse_args()
    if args.token_topk <= 0 or args.token_topk % 4:
        raise ValueError("--token-topk must be positive and divisible by four")
    if args.storage_page_size <= 0 or args.storage_page_size % 4:
        raise ValueError("--storage-page-size must be positive and divisible by four")
    if args.page_table_width <= 0:
        raise ValueError("--page-table-width must be positive")

    for position_dtype in (torch.int32, torch.int64):
        inputs = _inputs(
            1,
            token_topk=args.token_topk,
            storage_page_size=args.storage_page_size,
            page_table_width=args.page_table_width,
            position_dtype=position_dtype,
        )
        qsa_build_page4_paged_metadata(
            *inputs,
            args.storage_page_size,
        )
        for group_size in (2, 4):
            inputs = _inputs(
                group_size,
                token_topk=args.token_topk,
                storage_page_size=args.storage_page_size,
                page_table_width=args.page_table_width,
                position_dtype=position_dtype,
            )
            qsa_build_page4_grouped_paged_metadata(
                *inputs,
                args.storage_page_size,
                group_size,
            )
    torch.cuda.synchronize()
    print(
        "QSA metadata cache warmup complete: "
        f"token_topk={args.token_topk}, "
        f"storage_page_size={args.storage_page_size}, "
        f"page_table_width={args.page_table_width}"
    )


if __name__ == "__main__":
    main()
