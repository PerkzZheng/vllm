# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check recorded tensor storage equality on CPU without importing Torch."""

import argparse
import zipfile
from pathlib import Path


def tensor_payloads(path):
    with zipfile.ZipFile(path) as archive:
        return {
            name.split("/", 1)[1]: archive.read(name)
            for name in archive.namelist()
            if name.endswith("/data.pkl") or "/data/" in name
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", type=Path)
    args = parser.parse_args()
    attention = sorted((args.fixtures / "replays/current").glob("*.pt"))
    assert len(attention) == 60
    for current in attention:
        old = args.fixtures / "replays/historical" / current.name
        assert tensor_payloads(current) == tensor_payloads(old), current.name
    pairs = 0
    for shard in (0, 1):
        token_root = args.fixtures / "consumer-layout/results" / f"token-{shard}"
        group_root = args.fixtures / "consumer-layout/results" / f"group-{shard}"
        token_files = sorted(token_root.glob("*.pt"))
        assert len(token_files) == 30
        for token in token_files:
            assert tensor_payloads(token) == tensor_payloads(group_root / token.name)
            pairs += 1
    print(
        f"PASS: {len(attention)} attention pairs and {pairs} WO-A pairs "
        "have identical tensor metadata/storage bytes"
    )


if __name__ == "__main__":
    main()
