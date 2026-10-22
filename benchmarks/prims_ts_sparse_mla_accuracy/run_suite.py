# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the original task order in resumable shards against a ready server."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / "protocol.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=PROTOCOL["models"], required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8001/v1/chat/completions")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("gsm8k", "gpqa", "aime2026"),
        default=("gsm8k", "gpqa", "aime2026"),
    )
    args = parser.parse_args()
    model = PROTOCOL["models"][args.model]
    totals = {"gsm8k": 1319, "gpqa": 198, "aime2026": 30}
    for task in args.tasks:
        for start in range(0, totals[task], 32):
            output = args.output.resolve() / task / str(args.seed) / f"{start:04d}"
            # evaluate.py verifies the saved arguments even on a resume. Its
            # SQLite checkpoint reuses only responses in this exact output dir.
            output.mkdir(parents=True, exist_ok=True)
            log = output / "run.log"
            if log.exists():
                log.rename(output / f"run-previous-{time.time_ns()}.log")
            command = [
                sys.executable,
                str(ROOT / "evaluate.py"),
                "--model",
                args.model,
                "--url",
                args.url,
                "--task",
                task,
                "--seed",
                str(args.seed),
                "--top-p",
                str(model["top_p"]),
                "--max-tokens",
                str(model["max_tokens"]),
                "--concurrency",
                str(PROTOCOL["concurrency"]),
                "--start",
                str(start),
                "--limit",
                str(min(32, totals[task] - start)),
                "--output",
                str(output),
            ]
            end = start + min(32, totals[task] - start)
            print(f"{task} seed={args.seed} rows={start}:{end}", flush=True)
            with log.open("w") as stream:
                subprocess.run(
                    command, check=True, stdout=stream, stderr=subprocess.STDOUT
                )


if __name__ == "__main__":
    main()
