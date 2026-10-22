# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize one arm/seed; reject duplicate questions and incomplete shard records."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", type=Path, help="One arm directory passed to run_suite.py"
    )
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    for task, total in (("gsm8k", 1319), ("gpqa", 198), ("aime2026", 30)):
        selected_filter = "none" if task == "aime2026" else "flexible-extract"
        samples, responses = {}, {}
        for path in sorted(
            (args.output / task / str(args.seed)).glob("*/results.json")
        ):
            result = json.loads(path.read_text())
            shard = [
                s
                for s in result["samples"]["paired_" + task]
                if s["filter"] == selected_filter
            ]
            raw = [
                json.loads(line)
                for line in (path.parent / "responses.jsonl").read_text().splitlines()
            ]
            if len(raw) != len(shard):
                raise RuntimeError(f"Raw/result count mismatch: {path}")
            for sample, response in zip(shard, raw):
                doc_id = sample["doc_id"]
                response_id = response["id"]
                if doc_id in samples or response_id in responses:
                    raise RuntimeError(f"Duplicate question or API response: {path}")
                samples[doc_id] = sample["exact_match"]
                responses[response_id] = response
        covered = set(samples)
        if not covered <= set(range(total)):
            raise RuntimeError(f"Unexpected {task} question IDs")
        count = len(samples)
        tokens = [r["usage"]["completion_tokens"] for r in responses.values()]
        print(
            json.dumps(
                dict(
                    task=task,
                    correct=sum(samples.values()),
                    evaluated=count,
                    total=total,
                    complete=covered == set(range(total)),
                    accuracy=sum(samples.values()) / count if count else None,
                    length_stops=sum(
                        r["choices"][0]["finish_reason"] == "length"
                        for r in responses.values()
                    ),
                    mean_completion_tokens=sum(tokens) / count if count else None,
                )
            )
        )


if __name__ == "__main__":
    main()
