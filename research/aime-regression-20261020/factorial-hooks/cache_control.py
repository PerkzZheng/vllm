# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Seed and verify saved MoE tactics at the natural production warmup call."""

import functools
import hashlib
import json
import shutil

OP = "flashinfer::trtllm_fp4_block_scale_moe"


def install(arm):
    # Importing this module does not construct AutoTuner or query CUDA metadata.
    from vllm.model_executor.warmup import kernel_warmup

    original = kernel_warmup.flashinfer_autotune

    @functools.wraps(original)
    def controlled(runner, *, skip_attn=False):
        from flashinfer.autotuner import AutoTuner

        from vllm.distributed.parallel_state import get_world_group

        assert get_world_group().world_size == 1
        assert runner.vllm_config.parallel_config.data_parallel_rank == 0
        fixture = arm / "selected-moe-cache.json"
        expected = json.loads(fixture.read_text())
        expected_entries = {k: v for k, v in expected.items() if not k.startswith("_")}
        assert len(expected_entries) == 20
        assert all(OP in key for key in expected_entries)
        target = kernel_warmup.resolve_flashinfer_autotune_file(runner)
        target = target.with_name(f"{target.stem}_dp0_rank0{target.suffix}")
        assert target.resolve().is_relative_to((arm / "autotune").resolve())
        assert not target.exists(), "A factorial arm must start from its sole fixture"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fixture, target)
        fixture_sha = hashlib.sha256(fixture.read_bytes()).hexdigest()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == fixture_sha
        tuner = AutoTuner.get()
        assert tuner.load_configs(str(target)), "Saved cache rejected; do not retune"
        receipts = arm / "tactic-receipts"
        receipts.mkdir(exist_ok=True)
        before = receipts / "before.json"
        after = receipts / "after.json"
        tuner.save_configs(str(before))
        before_stats = (
            tuner.stats.tuned_op_successful_configs.get(OP, 0),
            len(tuner.stats.failed_profiling_count.get(OP, set())),
        )
        original_search = tuner.search_cache
        observed = {}

        @functools.wraps(original_search)
        def checked_search(
            custom_op,
            runners,
            input_shapes,
            tuning_config,
            inputs=None,
            *,
            require_profiling_policy=False,
        ):
            result = original_search(
                custom_op,
                runners,
                input_shapes,
                tuning_config,
                inputs=inputs,
                require_profiling_policy=require_profiling_policy,
            )
            if custom_op == OP:
                hit, runner_id, tactic, _ = result
                assert hit, "MoE cache miss; do not fall back or retune"
                selected = runners[runner_id]
                key = AutoTuner._get_cache_key(
                    custom_op,
                    selected,
                    input_shapes,
                    tuning_config,
                    selected.get_cache_key_extras(inputs) if inputs is not None else (),
                ).file_key
                expected_choice = expected_entries[key]
                choice = [selected.__class__.__name__, json.loads(json.dumps(tactic))]
                assert choice == expected_choice[:2], "Selected MoE tactic changed"
                if key not in observed:
                    observed[key] = choice
                    temporary = receipts / "observed.json.tmp"
                    temporary.write_text(json.dumps(observed, indent=2) + "\n")
                    temporary.replace(receipts / "observed.json")
            return result

        tuner.search_cache = checked_search

        def verify(path):
            actual = json.loads(path.read_text())
            assert actual["_metadata"] == expected["_metadata"]
            actual_moe = {k: v for k, v in actual.items() if OP in k}
            assert actual_moe == expected_entries, "MoE keys/tactics/policies changed"

        verify(before)
        original(runner, skip_attn=skip_attn)
        after_stats = (
            tuner.stats.tuned_op_successful_configs.get(OP, 0),
            len(tuner.stats.failed_profiling_count.get(OP, set())),
        )
        assert after_stats == before_stats, "Unexpected MoE profiling during warmup"
        assert observed.keys() == expected_entries.keys(), (
            "Incomplete MoE choice coverage"
        )
        tuner.save_configs(str(after))
        verify(after)
        verify(target)
        record = {
            "verified": True,
            "fixture_sha256": fixture_sha,
            "resolved_cache": str(target),
            "entries": 20,
            "metadata_and_tactics_unchanged": True,
            "moe_profiles_before": before_stats,
            "moe_profiles_after": after_stats,
            "observed_entries": len(observed),
            "runtime_cache_guard_active": True,
            "before_sha256": hashlib.sha256(before.read_bytes()).hexdigest(),
            "after_sha256": hashlib.sha256(after.read_bytes()).hexdigest(),
        }
        (arm / "tactic-control.json").write_text(json.dumps(record, indent=2) + "\n")

    kernel_warmup.flashinfer_autotune = controlled
