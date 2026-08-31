#!/usr/bin/env python3
"""Reproducible reasoning accuracy checks for QSA backend A/B runs."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
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


def _download(url: str) -> tuple[bytes, str]:
    request = Request(url, headers={"User-Agent": "qsa-reasoning-eval/1"})
    with urlopen(request, timeout=120) as response:
        data = response.read()
    return data, hashlib.sha256(data).hexdigest()


def _extract_number(text: str) -> str | None:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return numbers[-1] if numbers else None


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


def _load_examples(
    task: str, num_fewshot: int, seed: int
) -> tuple[list[Example], dict[str, str]]:
    if task == "gsm8k":
        return _load_gsm8k(num_fewshot)
    if task == "gpqa-diamond":
        return _load_gpqa(seed)
    if task == "aime26":
        return _load_aime26()
    raise AssertionError(f"unsupported task: {task}")


def _prediction(task: str, output: str) -> str | None:
    if task == "gpqa-diamond":
        return _extract_choice(output)
    if task == "aime26":
        return _extract_aime(output)
    return _extract_number(output)


async def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    examples, dataset_sha256 = _load_examples(
        args.task, args.num_fewshot, args.seed
    )
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
    for example, prediction, output, error, tokens, finish_reason in zip(
        selected,
        predictions,
        outputs,
        errors,
        completion_tokens,
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
        "metadata": metadata,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=("gsm8k", "gpqa-diamond", "aime26"),
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
    }
    default_questions, default_tokens = defaults[args.task]
    if args.num_questions is None:
        args.num_questions = default_questions
    if args.max_tokens is None:
        args.max_tokens = default_tokens
    if args.url is None:
        args.url = "http://127.0.0.1:8000"
    args.url = _normalize_api_url(args.url, args.api_mode)
    for item in args.metadata:
        if "=" not in item:
            parser.error(f"--metadata requires KEY=VALUE, got {item!r}")

    result = asyncio.run(_evaluate(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    summary = {key: value for key, value in result.items() if key != "records"}
    print(json.dumps(summary, indent=2))
    print(f"Detailed results: {args.output}")


if __name__ == "__main__":
    main()
