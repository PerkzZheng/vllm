#!/usr/bin/env python3
"""Summarize QSA reasoning artifacts and audit item-level backend changes."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

TASKS = ("gsm8k", "gpqa-diamond", "aime26", "longbench-v2")
MIN_QUALIFIED_MAX_TOKENS = {
    "gsm8k": 4096,
    "gpqa-diamond": 16384,
    "aime26": 16384,
    "longbench-v2": 128,
}


def _load_results(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    results: dict[str, dict[str, dict[str, Any]]] = {}
    dataset_hashes: dict[str, object] = {}
    for path in sorted(root.glob("*/*.json")):
        if "smoke" in path.stem:
            continue
        result = json.loads(path.read_text())
        # Raw-completion artifacts are retained for diagnosis but are not part
        # of the qualified accuracy matrix.
        if result.get("api_mode") != "chat":
            continue
        run_name = str(result["run_name"])
        if "smoke" in run_name:
            continue
        task = str(result["task"])
        if task not in TASKS:
            raise ValueError(f"unsupported task {task!r} in {path}")
        if result["max_tokens"] < MIN_QUALIFIED_MAX_TOKENS[task]:
            continue
        if task in results.setdefault(run_name, {}):
            raise ValueError(f"duplicate {run_name}/{task} result: {path}")
        digest = result["dataset_sha256"]
        if task in dataset_hashes and dataset_hashes[task] != digest:
            raise ValueError(f"dataset hash changed for {task}: {path}")
        dataset_hashes[task] = digest
        result["_path"] = str(path)
        results[run_name][task] = result
    return results


def _cell(result: dict[str, Any] | None) -> str:
    if result is None:
        return "pending"
    return (
        f"{result['correct']}/{result['num_questions']} "
        f"({100 * result['accuracy']:.2f}%; "
        f"trunc={result['truncated']}, err={result['errors']}, "
        f"invalid={result['invalid_predictions']})"
    )


def _print_matrix(results: dict[str, dict[str, dict[str, Any]]]) -> None:
    print("| Run | GSM8K | GPQA-Diamond | AIME26 | LongBench v2 |")
    print("|---|---:|---:|---:|---:|")
    for run_name, run_results in sorted(results.items()):
        cells = " | ".join(_cell(run_results.get(task)) for task in TASKS)
        print(f"| {run_name} | {cells} |")


def _records_by_id(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = {str(record["id"]): record for record in result["records"]}
    if len(records) != len(result["records"]):
        raise ValueError(f"duplicate item ID in {result['_path']}")
    return records


def _audit_pair(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, int]:
    reference_records = _records_by_id(reference)
    candidate_records = _records_by_id(candidate)
    compared_ids = reference_records.keys() & candidate_records.keys()
    if not compared_ids:
        return {
            "items": 0,
            "prediction_changes": 0,
            "output_hash_changes": 0,
            "regressions": 0,
            "improvements": 0,
        }
    prediction_changes = 0
    output_hash_changes = 0
    regressions = 0
    improvements = 0
    for item_id in compared_ids:
        candidate_record = candidate_records[item_id]
        reference_record = reference_records[item_id]
        prediction_changes += (
            reference_record["prediction"] != candidate_record["prediction"]
        )
        output_hash_changes += (
            reference_record["output_sha256"]
            != candidate_record["output_sha256"]
        )
        regressions += reference_record["correct"] and not candidate_record["correct"]
        improvements += (
            not reference_record["correct"] and candidate_record["correct"]
        )
    return {
        "items": len(compared_ids),
        "prediction_changes": prediction_changes,
        "output_hash_changes": output_hash_changes,
        "regressions": regressions,
        "improvements": improvements,
    }


def _print_audit(
    results: dict[str, dict[str, dict[str, Any]]], reference_run: str
) -> None:
    reference = results.get(reference_run)
    if reference is None:
        raise ValueError(f"reference run not found: {reference_run}")
    print()
    print(f"Item-level audit versus `{reference_run}`:")
    print()
    print(
        "| Candidate | Task | Items | Prediction changes | Output changes | "
        "Regressions | Improvements |"
    )
    print("|---|---|---:|---:|---:|---:|---:|")
    for run_name, run_results in sorted(results.items()):
        if run_name == reference_run:
            continue
        for task in TASKS:
            if task not in reference or task not in run_results:
                continue
            audit = _audit_pair(reference[task], run_results[task])
            if audit["items"] == 0:
                continue
            print(
                f"| {run_name} | {task} | {audit['items']} | "
                f"{audit['prediction_changes']} | "
                f"{audit['output_hash_changes']} | {audit['regressions']} | "
                f"{audit['improvements']} |"
            )


def _percentile(values: list[int], quantile: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def _distribution(values: list[int]) -> str:
    if not values:
        return "unavailable"
    return (
        f"mean={statistics.fmean(values):.1f}, "
        f"p50={_percentile(values, 0.5)}, "
        f"p90={_percentile(values, 0.9)}, "
        f"p99={_percentile(values, 0.99)}, "
        f"range={min(values)}--{max(values)}"
    )


def _length_groups(
    task: str, result: dict[str, Any]
) -> list[tuple[str, list[dict[str, Any]]]]:
    if task != "longbench-v2":
        return [(task, result["records"])]
    targets = sorted(
        {
            record["target_input_tokens"]
            for record in result["records"]
            if record.get("target_input_tokens") is not None
        }
    )
    return [
        (
            f"{task}/{target // 1024}K",
            [
                record
                for record in result["records"]
                if record.get("target_input_tokens") == target
            ],
        )
        for target in targets
    ]


def _print_lengths(results: dict[str, dict[str, dict[str, Any]]]) -> None:
    print()
    print("Token-length distributions:")
    print()
    print("| Run | Task/bucket | N | Input tokens | Output tokens | Total tokens |")
    print("|---|---|---:|---|---|---|")
    for run_name, run_results in sorted(results.items()):
        for task, result in sorted(run_results.items()):
            for label, records in _length_groups(task, result):
                inputs = [
                    int(record.get("api_prompt_tokens") or record.get("input_tokens"))
                    for record in records
                    if record.get("api_prompt_tokens") or record.get("input_tokens")
                ]
                outputs = [
                    int(record["completion_tokens"])
                    for record in records
                    if record.get("error") is None
                    and record.get("completion_tokens") is not None
                ]
                totals = [
                    int(record.get("api_prompt_tokens") or record.get("input_tokens"))
                    + int(record["completion_tokens"])
                    for record in records
                    if (
                        record.get("api_prompt_tokens")
                        or record.get("input_tokens")
                    )
                    and record.get("error") is None
                    and record.get("completion_tokens") is not None
                ]
                print(
                    f"| {run_name} | {label} | {len(records)} | "
                    f"{_distribution(inputs)} | {_distribution(outputs)} | "
                    f"{_distribution(totals)} |"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("qsa_accuracy"))
    parser.add_argument("--reference-run", default="triton-bf16-mtp0")
    args = parser.parse_args()

    results = _load_results(args.root)
    if not results:
        parser.error(f"no full result artifacts found under {args.root}")
    _print_matrix(results)
    _print_audit(results, args.reference_run)
    _print_lengths(results)


if __name__ == "__main__":
    main()
