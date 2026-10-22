# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serve one pinned checkpoint with the campaign's model and sampling defaults."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / "protocol.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=PROTOCOL["models"], required=True)
    parser.add_argument("--backend", choices=("default", "prims"), required=True)
    parser.add_argument(
        "--kv-cache", choices=("bfloat16", "fp8", "auto"), required=True
    )
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    model = PROTOCOL["models"][args.model]
    root = Path(os.environ["SPARSE_MLA_EVAL_ROOT"]).resolve()
    checkpoint = root / "models" / model["repo"].replace("/", "--")
    if (checkpoint / "EVAL_REVISION").read_text().strip() != model["revision"]:
        parser.error("Checkpoint revision differs from protocol.json")
    if len(args.gpus.split(",")) != model["tp"]:
        parser.error(f"This model's tested tensor parallel size is {model['tp']}")
    if args.model != "glm53-flash" and args.backend == "default":
        if args.kv_cache == "bfloat16":
            parser.error("The pinned default DeepSeek backends do not support BF16 KV")
        if args.model == "v41-flash" and args.kv_cache != "auto":
            parser.error("Use auto: V4.1 default actually uses FP4 compressed KV")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "8"
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(checkpoint),
        "--served-model-name",
        args.model,
        "--trust-remote-code",
        "--reasoning-parser",
        model["reasoning"],
        "--default-chat-template-kwargs.reasoning_effort=high",
        "--max-model-len",
        str(PROTOCOL["max_model_len"]),
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        args.kv_cache,
        "--max-num-seqs",
        str(PROTOCOL["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(PROTOCOL["max_num_batched_tokens"]),
        "--tensor-parallel-size",
        str(model["tp"]),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    if model["tokenizer"]:
        command += ["--tokenizer-mode", model["tokenizer"]]
    if args.backend == "prims":
        command += ["--attention-backend", model["backend"]]
    print(
        json.dumps({"model_revision": model["revision"], "command": command}),
        flush=True,
    )
    if not args.print_command:
        os.execv(command[0], command)


if __name__ == "__main__":
    main()
