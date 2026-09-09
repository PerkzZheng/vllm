# PrimTS integration on vLLM PR 53896

This branch builds on `13c80fb30ab835cbe387c01c4611970b7c3373e1`, the
PR 53896 head fetched on 2026-08-30. The model implementation is
`vllm/models/qwen4_exp/nvidia`. The model used for evaluation is
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
revision `de4b8e4`.

## Integration

- Preserve PR 53896's indexer, MTP, PLE, and combined-cache layout.
- Use `VLLM_QSA_ATTENTION_BACKEND=triton|prims_ts|auto` for backend selection.
- PrimTS consumes compact block IDs and dense physical page tables.
- Use packed G4 prefill and proven fixed `G=MTP+1` decode, with Q1 fallback.
- One prepared FlashInfer plan owns metadata and disjoint attention scratch.
  Compatible ordered layers share storage; plans and K/V maps stay per-layer.
- Support BF16 and FP8-E4M3 cache, with BF16 model Q/output and live descales.

The [interface guide](PRIMS_TS_QSA_VLLM_INTERFACE.md) contains the public
FlashInfer calls, tensor shapes, vLLM phase policy, and CUDA-graph lifetime.

## Runtime environment

The evaluation scripts use the dedicated
`vllm/vllm-openai:qwen38-flash-next` image and the repository's
`.venv/bin/python`. The optional runtime overlay keeps image GDN/MoE/TVM-FFI
binaries while loading the local FlashInfer attention API and CUTLASS DSL 4.7.1.
Both backends use the same environment.

Keep caches and results on the persistent workspace. For capacity measurements,
warm compilation and restart the server before allocating its final KV cache.
A cold PLE autotuning allocation can otherwise reduce that cache on either
backend.

## Reproducible validation

- `tests/models/qwen4_exp/test_qsa_reference.py`: kernel/adapter correctness,
  fixed/packed routes, padding, shared workspaces, and graph replay.
- `benchmarks/kernels/benchmark_q_token_kv_block_sparse_ts_suites.py`:
  manifest-driven standalone comparisons including both backends' metadata.
- `run_reasoning_accuracy.sh` and `qsa_reasoning_eval.py`: matched evaluation.
- `run_e2e_perf.sh` and `profile_steady_decode.py`: warmed pure-stage profiling.

Record source/model revisions, GPU/runtime versions, sampler, input hashes,
actual batch size, errors, and truncations. Compare cold-L2 CUDA-graph
standalone runs separately from warmed model-stage traces. Report absolute
latencies as well as speedups; do not treat serving TTFT as pure prefill time.

Historical experiments remain in local campaign records and Git history.
They are not current API requirements or final-head qualification.
