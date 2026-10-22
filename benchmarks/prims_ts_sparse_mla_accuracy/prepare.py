# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage pinned datasets, selected model weights and the original task definitions."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
PROTOCOL = json.loads((SCRIPTS / "protocol.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=PROTOCOL["models"], action="append")
    parser.add_argument("--tasks-only", action="store_true")
    args = parser.parse_args()
    root = Path(os.environ["SPARSE_MLA_EVAL_ROOT"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    harness = root / "lm-evaluation-harness"
    if not harness.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/EleutherAI/lm-evaluation-harness.git",
                str(harness),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(harness), "checkout", "--detach", PROTOCOL["harness"]],
            check=True,
        )
    head = subprocess.check_output(
        ["git", "-C", str(harness), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != PROTOCOL["harness"]:
        raise RuntimeError("Existing harness has a different revision")
    subprocess.run(
        ["git", "-C", str(harness), "diff", "--exit-code", "HEAD"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    if not args.tasks_only:
        from vllm.transformers_utils.repo_utils import hf_api

        for item in PROTOCOL["datasets"].values():
            destination = root / "datasets" / item["repo"].replace("/", "--")
            hf_api().snapshot_download(
                item["repo"],
                repo_type="dataset",
                revision=item["revision"],
                local_dir=destination,
                allow_patterns=item["patterns"],
            )
            (destination / "EVAL_REVISION").write_text(item["revision"] + "\n")
        for alias in args.model or []:
            item = PROTOCOL["models"][alias]
            destination = root / "models" / item["repo"].replace("/", "--")
            hf_api().snapshot_download(
                item["repo"],
                revision=item["revision"],
                local_dir=destination,
                allow_patterns=[
                    "*.safetensors",
                    "*.json",
                    "*.model",
                    "*.txt",
                    "*.tiktoken",
                    "*.jinja",
                    "*.py",
                    "*.yaml",
                    "*.yml",
                    "README.md",
                    "LICENSE*",
                ],
                max_workers=8,
            )
            (destination / "EVAL_REVISION").write_text(item["revision"] + "\n")
    tasks = root / "eval_tasks"
    tasks.mkdir(exist_ok=True)
    definitions = {
        "gsm8k": {
            "include": "gsm8k/gsm8k-cot.yaml",
            "format": "parquet",
            "split": "test",
            "file": "openai--gsm8k/main/test-00000-of-00001.parquet",
        },
        "gpqa": {
            "include": "gpqa/cot_zeroshot/_gpqa_cot_zeroshot_yaml",
            "format": "csv",
            "split": "train",
            "file": "Idavidrein--gpqa/gpqa_diamond.csv",
        },
        "aime2026": {
            "include": "aime/aime25.yaml",
            "format": "parquet",
            "split": "test",
            "file": "MathArena--aime_2026/data/train-00000-of-00001.parquet",
        },
    }
    for name, item in definitions.items():
        # JSON strings are valid YAML scalars, including paths containing spaces.
        include = json.dumps(str(harness / "lm_eval/tasks" / item["include"]))
        data = json.dumps(str(root / "datasets" / item["file"]))
        text = (
            f"include: {include}\ntask: paired_{name}\n"
            f"dataset_path: {item['format']}\ndataset_name: null\n"
            f"dataset_kwargs:\n  data_files:\n    {item['split']}: {data}\n"
        )
        if name == "aime2026":
            prompt = (
                "Question: {{problem}}\nPut your final answer in \\boxed{}.\nAnswer:"
            )
            text += f"doc_to_text: {json.dumps(prompt)}\n"
            text += "process_results: !function utils.score_aime2026\n"
        (tasks / f"paired_{name}.yaml").write_text(text)
    shutil.copyfile(SCRIPTS / "aime_utils.py", tasks / "utils.py")
    print(f"Prepared pinned tasks in {tasks}")


if __name__ == "__main__":
    main()
