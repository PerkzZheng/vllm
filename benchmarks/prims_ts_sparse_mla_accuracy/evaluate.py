# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired chat evaluation using pinned lm-eval prompts and answer graders."""

import argparse
import copy
import json
import os
from pathlib import Path

from lm_eval import evaluator
from lm_eval.api.model import CacheHook, hash_args
from lm_eval.models.openai_completions import LocalChatCompletion
from lm_eval.tasks import TaskManager
from sqlitedict import SqliteDict

ROOT = Path(os.environ["SPARSE_MLA_EVAL_ROOT"]).resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--url", default="http://127.0.0.1:8001/v1/chat/completions")
parser.add_argument("--task", choices=["gpqa", "gsm8k", "aime2026"], required=True)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--top-p", type=float, required=True)
parser.add_argument("--max-tokens", type=int, required=True)
parser.add_argument("--concurrency", type=int, default=4)
parser.add_argument("--return-token-ids", action="store_true")
parser.add_argument("--limit", type=int)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.limit is not None and (args.limit <= 0 or args.start < 0):
    parser.error("--start must be nonnegative and --limit positive")
args.output.mkdir(parents=True, exist_ok=True)
if args.start and args.limit is None:
    parser.error("--start requires --limit")
arguments = json.dumps(vars(args), default=str, indent=2)
arguments_path = args.output / "arguments.json"
if arguments_path.exists():
    previous, current = json.loads(arguments_path.read_text()), json.loads(arguments)
    previous.setdefault("return_token_ids", False)
    if any(
        previous.get(key) != value
        for key, value in current.items()
        if key not in ("url", "output")
    ):
        parser.error("output directory contains a different evaluation configuration")
# A relocated checkpoint or a different server address does not change its
# prompts, seed, sampling parameters, or request hashes.
arguments_path.write_text(arguments)


class ResumeHook(CacheHook):
    def __init__(self, model, database):
        super().__init__(None)
        self.model, self.database = model, database

    def add_partial(self, attr, request, answer):
        if attr == "generate_until":
            self.database[hash_args(attr, request)] = {
                "answer": answer,
                "response": self.model.last_response,
            }


class RecordedChat(LocalChatCompletion):
    def generate_until(self, requests):
        # Resume one fixed-seed run, never reuse samples across independent repeats.
        with SqliteDict(str(args.output / "generations.sqlite"), autocommit=True) as db:
            keys = [hash_args("generate_until", request.args) for request in requests]
            self.set_cache_hook(ResumeHook(self, db))
            missing = [request for request, key in zip(requests, keys) if key not in db]
            print(
                f"Resuming {len(requests) - len(missing)} requests; "
                f"generating {len(missing)}",
                flush=True,
            )
            if missing:
                super().generate_until(missing)
            db.commit()
            records = [db[key] for key in keys]
            with (args.output / "responses.jsonl").open("w") as stream:
                for record in records:
                    stream.write(json.dumps(record["response"]) + "\n")
            return [record["answer"] for record in records]

    def parse_generations(self, outputs, **kwargs):
        # Retain usage/finish_reason as well as final answers; truncation must
        # remain visible even if answer extraction happens to find a number.
        responses = outputs if isinstance(outputs, list) else [outputs]
        assert len(responses) == 1  # local-chat-completions uses API batch size one.
        self.last_response = responses[0]
        with (args.output / "responses.jsonl").open("a") as stream:
            for response in responses:
                stream.write(json.dumps(response) + "\n")
        # A reasoning-only truncated completion is an empty final answer.
        normalized = copy.deepcopy(responses)
        for response in normalized:
            for choice in response["choices"]:
                choice["message"]["content"] = choice["message"]["content"] or ""
        return super().parse_generations(normalized, **kwargs)


model = RecordedChat(
    model=args.model,
    base_url=args.url,
    tokenizer_backend=None,
    tokenized_requests=False,
    num_concurrent=args.concurrency,
    seed=args.seed,
    max_length=1048576,
    max_gen_toks=args.max_tokens,
    timeout=86400,
    max_retries=1,
    eos_string="",
)
results = evaluator.simple_evaluate(
    model=model,
    tasks=["paired_" + args.task],
    task_manager=TaskManager(
        include_path=str(ROOT / "eval_tasks"), include_defaults=False
    ),
    apply_chat_template=True,
    fewshot_as_multiturn=False,
    # Hold the task/choice order fixed across repeats and implementations.
    random_seed=1234,
    numpy_random_seed=1234,
    torch_random_seed=1234,
    fewshot_random_seed=1234,
    samples={"paired_" + args.task: list(range(args.start, args.start + args.limit))}
    if args.limit is not None
    else None,
    log_samples=True,
    gen_kwargs={
        "do_sample": True,
        "temperature": 1.0,
        "top_p": args.top_p,
        "max_gen_toks": args.max_tokens,
        "reasoning_effort": "high",
        "chat_template_kwargs": {"reasoning_effort": "high"},
        "until": [],
        **({"return_token_ids": True} if args.return_token_ids else {}),
    },
    bootstrap_iters=1000,
)
temporary = args.output / "results.json.tmp"
temporary.write_text(json.dumps(results, default=str, indent=2))
temporary.replace(args.output / "results.json")
print(json.dumps(results["results"], indent=2))
