# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AIME's integer convention accepts leading zeroes in a boxed answer."""

import regex as re
from lm_eval.tasks.aime.utils import (
    last_boxed_only_string,
    process_results,
    remove_boxed,
)


def score_aime2026(doc, responses):
    boxed = last_boxed_only_string(responses[0])
    if boxed is not None:
        answer = remove_boxed(boxed).strip()
        if re.fullmatch(r"[0-9]{1,3}", answer):
            return {"exact_match": int(int(answer) == int(doc["answer"]))}
    return process_results(doc, responses)
