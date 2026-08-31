#!/usr/bin/env python3
"""Reproducible reasoning accuracy checks for QSA backend A/B runs."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import aiohttp

GSM8K_ROOT = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data"
)
GPQA_DIAMOND_URL = (
    "https://huggingface.co/datasets/SunriserFuture/EducationQ/"
    "resolve/main/gpqa_diamond.json?download=true"
)
AIME26_URL = (
    "https://huggingface.co/datasets/math-ai/aime26/"
    "resolve/main/aime2026.jsonl?download=true"
)
LETTERS = "ABCD"
LONGBENCH_V2_PROMPT = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: \"The correct answer is (insert answer here)\"."""


def _normalize_api_url(url: str, api_mode: str) -> str:
    """Accept either an OpenAI server root or an explicit endpoint."""
    endpoint = "chat/completions" if api_mode == "chat" else "completions"
    path = urlsplit(url).path.rstrip("/")
    if path in ("", "/v1"):
        prefix = url.rstrip("/")
        if path != "/v1":
            prefix += "/v1"
        return f"{prefix}/{endpoint}"
    return url


@dataclass(frozen=True)
class Example:
    example_id: str
    question: str
    prompt: str
    label: str
    choices: tuple[str, ...] = ()
    input_tokens: int | None = None
    target_input_tokens: int | None = None
    domain: str | None = None
    sub_domain: str | None = None
    difficulty: str | None = None
    length_class: str | None = None


def _download(url: str) -> tuple[bytes, str]:
    request = Request(url, headers={"User-Agent": "qsa-reasoning-eval/1"})
    with urlopen(request, timeout=120) as response:
        data = response.read()
    return data, hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_number(text: str) -> str | None:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return numbers[-1] if numbers else None


def _extract_gsm8k(text: str) -> str | None:
    # The canonical GSM8K answer marker is more reliable than the last number
    # in a reasoning response. Some chat templates can append text after the
    # final ``#### N`` answer.
    marked = re.findall(r"####\s*(-?\d+(?:\.\d+)?)", text.replace(",", ""))
    return marked[-1] if marked else _extract_number(text)


def _extract_choice(text: str) -> str | None:
    patterns = (
        r"(?i)(?:final\s+)?answer\s*(?:is|:)\s*\(?([A-D])\)?",
        r"(?i)\boption\s+([A-D])\b",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].upper()
    tail = text.strip()[-80:]
    matches = re.findall(r"(?:^|\s)\(?([A-D])\)?(?:[.!]?\s*$)", tail)
    return matches[-1].upper() if matches else None


def _extract_aime(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\s*\{\s*(-?\d+)\s*\}", text)
    if boxed:
        return str(int(boxed[-1]))
    answers = re.findall(
        r"(?i)(?:final\s+)?answer\s*(?:is|:)\s*(-?\d+)", text
    )
    if answers:
        return str(int(answers[-1]))
    number = _extract_number(text)
    return str(int(float(number))) if number is not None else None


def _load_gsm8k(num_fewshot: int) -> tuple[list[Example], dict[str, str]]:
    train_bytes, train_sha = _download(f"{GSM8K_ROOT}/train.jsonl")
    test_bytes, test_sha = _download(f"{GSM8K_ROOT}/test.jsonl")
    train = [json.loads(line) for line in train_bytes.splitlines()]
    test = [json.loads(line) for line in test_bytes.splitlines()]
    prefix = "".join(
        f"Question: {row['question']}\nAnswer: {row['answer']}\n\n"
        for row in train[:num_fewshot]
    )
    examples = [
        Example(
            example_id=str(index),
            question=row["question"],
            prompt=prefix + f"Question: {row['question']}\nAnswer:",
            label=_extract_number(row["answer"]) or "",
        )
        for index, row in enumerate(test)
    ]
    return examples, {"train": train_sha, "test": test_sha}


def _load_gpqa(seed: int) -> tuple[list[Example], dict[str, str]]:
    data, digest = _download(GPQA_DIAMOND_URL)
    rows = json.loads(data)
    examples: list[Example] = []
    for index, row in enumerate(rows):
        order = list(range(4))
        random.Random(f"{seed}:{row['id']}").shuffle(order)
        choices = tuple(row["options"][choice] for choice in order)
        correct_original = int(row.get("answer_index", LETTERS.index(row["answer"])))
        label = LETTERS[order.index(correct_original)]
        rendered = "\n".join(
            f"({letter}) {choice}" for letter, choice in zip(LETTERS, choices)
        )
        prompt = (
            "Answer the following multiple-choice question. Think step by step, "
            "then end with `Answer: X`, where X is A, B, C, or D.\n\n"
            f"Question: {row['question']}\n\nChoices:\n{rendered}\n\nAnswer:"
        )
        examples.append(
            Example(
                example_id=str(row.get("id", index)),
                question=row["question"],
                prompt=prompt,
                label=label,
                choices=choices,
            )
        )
    return examples, {"gpqa_diamond": digest}


def _load_aime26() -> tuple[list[Example], dict[str, str]]:
    data, digest = _download(AIME26_URL)
    rows = [json.loads(line) for line in data.splitlines()]
    examples = [
        Example(
            example_id=str(row["id"]),
            question=row["problem"],
            prompt=(
                "Solve the following problem. Show your reasoning, then end with "
                "`Answer: N`, where N is the integer from 0 through 999.\n\n"
                f"Problem: {row['problem']}\n\nSolution:"
            ),
            label=str(int(row["answer"])),
        )
        for row in rows
    ]
    return examples, {"aime2026": digest}


def _read_json_array(path: Path) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    with path.open() as source:
        buffer = ""
        position = 0
        started = False
        while True:
            if position:
                buffer = buffer[position:]
                position = 0
            chunk = source.read(1024 * 1024)
            buffer += chunk
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if not started:
                    if position == len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError("LongBench v2 JSON must contain an array")
                    started = True
                    position += 1
                    continue
                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                if position == len(buffer):
                    break
                if buffer[position] == "]":
                    return
                try:
                    row, position = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    break
                yield row
            if not chunk:
                raise ValueError("incomplete LongBench v2 JSON array")


def _read_longbench_v2(path: Path) -> list[dict[str, Any]] | Iterator[dict[str, Any]]:
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError(
                "LongBench v2 parquet input requires pyarrow"
            ) from error
        return pq.read_table(path).to_pylist()

    return _read_json_array(path)


def _within_word_scan_limit(text: str, limit: int) -> bool:
    words = 0
    for _ in re.finditer(r"\S+", text):
        words += 1
        if words > limit:
            return False
    return True


def _render_longbench_v2(row: dict[str, Any]) -> str:
    replacements = {
        "$DOC$": row["context"].strip(),
        "$Q$": row["question"].strip(),
        "$C_A$": row["choice_A"].strip(),
        "$C_B$": row["choice_B"].strip(),
        "$C_C$": row["choice_C"].strip(),
        "$C_D$": row["choice_D"].strip(),
    }
    prompt = LONGBENCH_V2_PROMPT
    for marker, value in replacements.items():
        prompt = prompt.replace(marker, value)
    return prompt


def _chat_input_tokens(tokenizer: Any, prompt: str, reasoning_effort: str) -> int:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        reasoning_effort=reasoning_effort,
    )
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if hasattr(encoded, "shape"):
        return int(encoded.shape[-1])
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return len(encoded)


def _select_longbench_v2(
    examples: list[Example],
    targets: list[int],
    samples_per_target: int,
    tolerance: float,
    seed: int,
) -> list[Example]:
    selected: list[Example] = []
    used: set[str] = set()
    for target in targets:
        candidates = [
            example
            for example in examples
            if example.example_id not in used
            and example.input_tokens is not None
            and abs(example.input_tokens - target) <= target * tolerance
        ]
        candidates.sort(
            key=lambda example: (
                abs((example.input_tokens or 0) - target),
                hashlib.sha256(
                    f"{seed}:{target}:{example.example_id}".encode()
                ).hexdigest(),
            )
        )
        if len(candidates) < samples_per_target:
            raise ValueError(
                f"LongBench v2 target {target} has only {len(candidates)} examples "
                f"within {tolerance:.1%}; requested {samples_per_target}"
            )
        for example in candidates[:samples_per_target]:
            selected.append(replace(example, target_input_tokens=target))
            used.add(example.example_id)
    return selected


def _load_longbench_v2(
    args: argparse.Namespace,
) -> tuple[list[Example], dict[str, str]]:
    if args.longbench_dataset is None:
        raise ValueError("--longbench-dataset is required for LongBench v2")
    if args.tokenizer is None:
        raise ValueError("--tokenizer is required for LongBench v2")

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("LongBench v2 selection requires transformers") from error

    rows = _read_longbench_v2(args.longbench_dataset)
    manifest_records: dict[str, dict[str, Any]] = {}
    requested_ids = args.longbench_ids
    if args.longbench_manifest is not None:
        manifest = json.loads(args.longbench_manifest.read_text())
        requested_ids = [record["id"] for record in manifest["records"]]
        manifest_records = {record["id"]: record for record in manifest["records"]}
    if requested_ids:
        requested_id_set = set(requested_ids)
        row_by_id = {
            str(row["_id"]): row
            for row in rows
            if str(row["_id"]) in requested_id_set
        }
        missing = [
            example_id
            for example_id in requested_ids
            if example_id not in row_by_id
        ]
        if missing:
            raise ValueError(f"unknown LongBench v2 IDs: {missing}")
        rows = [row_by_id[example_id] for example_id in requested_ids]

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
    )
    examples = []
    for row in rows:
        if not requested_ids:
            max_candidate_tokens = int(
                max(args.longbench_target_tokens)
                * (1 + args.longbench_tolerance)
            )
            if row["length"] == "long" or not _within_word_scan_limit(
                row["context"], int(max_candidate_tokens * 1.25)
            ):
                continue
        prompt = _render_longbench_v2(row)
        input_tokens = _chat_input_tokens(
            tokenizer, prompt, args.reasoning_effort
        )
        if not requested_ids:
            keep = any(
                abs(input_tokens - target)
                <= target * args.longbench_tolerance
                for target in args.longbench_target_tokens
            )
            if not keep:
                continue
        target_input_tokens = (
            manifest_records[str(row["_id"])].get("target_input_tokens")
            if manifest_records
            else None
        )
        examples.append(
            Example(
                example_id=str(row["_id"]),
                question=row["question"],
                prompt=prompt,
                label=row["answer"].strip().upper(),
                choices=tuple(row[f"choice_{letter}"] for letter in LETTERS),
                input_tokens=input_tokens,
                target_input_tokens=target_input_tokens,
                domain=row["domain"],
                sub_domain=row["sub_domain"],
                difficulty=row["difficulty"],
                length_class=row["length"],
            )
        )
        if manifest_records:
            expected = manifest_records[str(row["_id"])]
            actual_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
            if actual_sha256 != expected["prompt_sha256"]:
                raise ValueError(
                    f"LongBench v2 prompt changed for {row['_id']}: "
                    f"{actual_sha256} != {expected['prompt_sha256']}"
                )
            if input_tokens != expected["input_tokens"]:
                raise ValueError(
                    f"LongBench v2 token count changed for {row['_id']}: "
                    f"{input_tokens} != {expected['input_tokens']}"
                )

    if not requested_ids:
        examples = _select_longbench_v2(
            examples,
            args.longbench_target_tokens,
            args.longbench_samples_per_target,
            args.longbench_tolerance,
            args.seed,
        )
    return examples, {"longbench_v2": _sha256_file(args.longbench_dataset)}


def _selection_records(examples: list[Example]) -> list[dict[str, Any]]:
    return [
        {
            "id": example.example_id,
            "prompt_sha256": hashlib.sha256(example.prompt.encode()).hexdigest(),
            "input_tokens": example.input_tokens,
            "target_input_tokens": example.target_input_tokens,
            "domain": example.domain,
            "sub_domain": example.sub_domain,
            "difficulty": example.difficulty,
            "length_class": example.length_class,
            "label": example.label,
        }
        for example in examples
    ]


def _input_token_stats(examples: list[Example]) -> dict[str, float] | None:
    values = [
        example.input_tokens
        for example in examples
        if example.input_tokens is not None
    ]
    if not values:
        return None
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def _prepare_selection(args: argparse.Namespace) -> dict[str, Any]:
    examples, dataset_sha256 = _load_examples(args)
    selected = examples[args.start_index : args.start_index + args.num_questions]
    return {
        "run_name": args.run_name,
        "task": args.task,
        "dataset_sha256": dataset_sha256,
        "tokenizer": args.tokenizer,
        "reasoning_effort": args.reasoning_effort,
        "seed": args.seed,
        "num_questions": len(selected),
        "input_token_stats": _input_token_stats(selected),
        "records": _selection_records(selected),
    }


def _load_examples(
    args: argparse.Namespace,
) -> tuple[list[Example], dict[str, str]]:
    if args.task == "gsm8k":
        return _load_gsm8k(args.num_fewshot)
    if args.task == "gpqa-diamond":
        return _load_gpqa(args.seed)
    if args.task == "aime26":
        return _load_aime26()
    if args.task == "longbench-v2":
        return _load_longbench_v2(args)
    raise AssertionError(f"unsupported task: {args.task}")


def _prediction(task: str, output: str) -> str | None:
    if task in ("gpqa-diamond", "longbench-v2"):
        return _extract_choice(output)
    if task == "aime26":
        return _extract_aime(output)
    return _extract_gsm8k(output)


async def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    examples, dataset_sha256 = _load_examples(args)
    if args.indices is None:
        selected = examples[args.start_index : args.start_index + args.num_questions]
    else:
        invalid = [index for index in args.indices if not 0 <= index < len(examples)]
        if invalid:
            raise ValueError(f"example indices are out of range: {invalid}")
        if len(set(args.indices)) != len(args.indices):
            raise ValueError("example indices must be unique")
        selected = [examples[index] for index in args.indices]
    outputs = [""] * len(selected)
    errors: list[str | None] = [None] * len(selected)
    completion_tokens = [0] * len(selected)
    prompt_tokens = [0] * len(selected)
    finish_reasons: list[str | None] = [None] * len(selected)
    semaphore = asyncio.Semaphore(args.max_concurrency)
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)

    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def run_one(index: int) -> None:
            payload: dict[str, Any] = {
                "model": args.model,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_tokens": args.max_tokens,
                "seed": args.seed,
                "n": args.n,
                "stream": False,
            }
            if args.api_mode == "chat":
                payload["messages"] = [
                    {"role": "user", "content": selected[index].prompt}
                ]
                payload["reasoning_effort"] = args.reasoning_effort
            else:
                payload["prompt"] = selected[index].prompt
            if args.api_mode == "completions" and args.task == "gsm8k":
                payload["stop"] = ["Question", "Assistant:", "<|separator|>"]
            for attempt in range(args.retries + 1):
                try:
                    async with semaphore, session.post(
                        args.url, json=payload
                    ) as response:
                        body = await response.text()
                        if response.status >= 400:
                            raise RuntimeError(
                                f"HTTP {response.status}: {body[:2000]}"
                            )
                        response.raise_for_status()
                    result = json.loads(body)
                    choice = result["choices"][0]
                    if args.api_mode == "chat":
                        message = choice["message"]
                        reasoning = (
                            message.get("reasoning_content")
                            or message.get("reasoning")
                            or ""
                        )
                        content = message.get("content") or ""
                        outputs[index] = "\n".join(
                            part for part in (reasoning, content) if part
                        )
                    else:
                        outputs[index] = choice["text"]
                    finish_reasons[index] = choice.get("finish_reason")
                    completion_tokens[index] = result.get("usage", {}).get(
                        "completion_tokens", 0
                    )
                    prompt_tokens[index] = result.get("usage", {}).get(
                        "prompt_tokens", 0
                    )
                    return
                except Exception as error:
                    if attempt == args.retries:
                        errors[index] = f"{type(error).__name__}: {error}"
                    else:
                        await asyncio.sleep(1 << attempt)

        started = time.perf_counter()
        await asyncio.gather(*(run_one(index) for index in range(len(selected))))
        elapsed = time.perf_counter() - started

    predictions = [_prediction(args.task, output) for output in outputs]
    records = []
    for example, prediction, output, error, tokens, input_tokens, finish_reason in zip(
        selected,
        predictions,
        outputs,
        errors,
        completion_tokens,
        prompt_tokens,
        finish_reasons,
    ):
        records.append(
            {
                "id": example.example_id,
                "question": example.question,
                "choices": list(example.choices),
                "label": example.label,
                "prediction": prediction,
                "correct": prediction == example.label,
                "output": output,
                "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                "prompt_sha256": hashlib.sha256(
                    example.prompt.encode()
                ).hexdigest(),
                "input_tokens": example.input_tokens,
                "api_prompt_tokens": input_tokens,
                "target_input_tokens": example.target_input_tokens,
                "domain": example.domain,
                "sub_domain": example.sub_domain,
                "difficulty": example.difficulty,
                "length_class": example.length_class,
                "completion_tokens": tokens,
                "finish_reason": finish_reason,
                "error": error,
            }
        )

    correct = sum(record["correct"] for record in records)
    completed = sum(error is None for error in errors)
    metadata = dict(item.split("=", 1) for item in args.metadata)
    return {
        "run_name": args.run_name,
        "task": args.task,
        "model": args.model,
        "url": args.url,
        "api_mode": args.api_mode,
        "dataset_sha256": dataset_sha256,
        "start_index": args.start_index,
        "indices": args.indices,
        "num_questions": len(selected),
        "num_fewshot": args.num_fewshot if args.task == "gsm8k" else 0,
        "max_tokens": args.max_tokens,
        "max_concurrency": args.max_concurrency,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "reasoning_effort": args.reasoning_effort,
        "n": args.n,
        "stream": False,
        "accuracy": correct / len(records) if records else 0.0,
        "correct": correct,
        "completed": completed,
        "errors": len(records) - completed,
        "invalid_predictions": sum(prediction is None for prediction in predictions),
        "truncated": sum(reason == "length" for reason in finish_reasons),
        "elapsed_seconds": elapsed,
        "questions_per_second": len(records) / elapsed if elapsed else 0.0,
        "completion_tokens": sum(completion_tokens),
        "prompt_tokens": sum(prompt_tokens),
        "input_token_stats": _input_token_stats(selected),
        "metadata": metadata,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=("gsm8k", "gpqa-diamond", "aime26", "longbench-v2"),
        required=True,
    )
    parser.add_argument("--url")
    parser.add_argument(
        "--api-mode",
        choices=("chat", "completions"),
        default="chat",
        help="Use chat templates by default; raw completions are diagnostic only",
    )
    parser.add_argument("--model", default="brightdelta-180b-bf16")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--num-questions", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        help="Evaluate explicit zero-based dataset rows instead of one range",
    )
    parser.add_argument("--num-fewshot", type=int, default=5)
    parser.add_argument("--longbench-dataset", type=Path)
    parser.add_argument("--tokenizer")
    parser.add_argument("--longbench-manifest", type=Path)
    parser.add_argument(
        "--longbench-target-tokens",
        type=int,
        nargs="+",
        default=[8192, 16384, 32768],
    )
    parser.add_argument("--longbench-samples-per-target", type=int, default=16)
    parser.add_argument("--longbench-tolerance", type=float, default=1.0)
    parser.add_argument("--longbench-ids", nargs="+")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--request-timeout", type=float, default=7200)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--n", type=int, choices=(1,), default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    defaults = {
        "gsm8k": (1319, 131072),
        "gpqa-diamond": (198, 131072),
        "aime26": (30, 131072),
        "longbench-v2": (0, 131072),
    }
    default_questions, default_tokens = defaults[args.task]
    if args.num_questions is None:
        if args.task != "longbench-v2":
            args.num_questions = default_questions
        elif args.longbench_manifest is not None:
            manifest = json.loads(args.longbench_manifest.read_text())
            args.num_questions = len(manifest["records"])
        elif args.longbench_ids:
            args.num_questions = len(args.longbench_ids)
        else:
            args.num_questions = (
                len(args.longbench_target_tokens)
                * args.longbench_samples_per_target
            )
    if args.max_tokens is None:
        args.max_tokens = default_tokens
    if args.url is None:
        args.url = "http://127.0.0.1:8000"
    args.url = _normalize_api_url(args.url, args.api_mode)
    for item in args.metadata:
        if "=" not in item:
            parser.error(f"--metadata requires KEY=VALUE, got {item!r}")
    if args.longbench_ids and args.longbench_manifest is not None:
        parser.error("--longbench-ids and --longbench-manifest are mutually exclusive")
    if not 0 <= args.longbench_tolerance <= 1:
        parser.error("--longbench-tolerance must be in [0, 1]")

    result = (
        _prepare_selection(args)
        if args.prepare_only
        else asyncio.run(_evaluate(args))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    summary = {key: value for key, value in result.items() if key != "records"}
    print(json.dumps(summary, indent=2))
    print(f"Detailed results: {args.output}")


if __name__ == "__main__":
    main()
