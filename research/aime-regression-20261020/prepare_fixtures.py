# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extract and verify all numerical fixtures and historical sources without Torch."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent


def verify(root, manifest):
    for name, expected in manifest["files"].items():
        path = root / name
        assert path.is_file() and not path.is_symlink(), name
        assert path.stat().st_size == expected["bytes"], name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"], name
    print(f"Verified {len(manifest['files'])} numerical/source files")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((PACKAGE / "fixtures.json").read_text())
    destination = args.destination.resolve()
    if args.verify_only:
        verify(destination, manifest)
        return
    assert not destination.is_relative_to(PACKAGE), "Extract outside Git"
    assert not destination.exists(), "Use a fresh destination or --verify-only"
    archive = PACKAGE / "fixtures.tar.xz"
    assert (
        hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["archive_sha256"]
    )
    destination.mkdir(parents=True)
    seen = set()
    with tarfile.open(archive, "r:xz") as stream:
        for member in stream:
            name = member.name
            assert member.isfile() and name in manifest["files"] and name not in seen
            path = (destination / name).resolve()
            assert path.is_relative_to(destination), name
            assert member.size == manifest["files"][name]["bytes"], name
            content = stream.extractfile(member).read()
            assert (
                hashlib.sha256(content).hexdigest() == manifest["files"][name]["sha256"]
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            seen.add(name)
    assert seen == set(manifest["files"])
    verify(destination, manifest)


if __name__ == "__main__":
    main()
