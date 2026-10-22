# Prims-TS sparse MLA: accuracy results and investigation

The latest integration completed **24/24 native accuracy cells** on GB300: four models × BF16/FP8 KV × three tasks. The main unresolved signal is **V4 Flash native FP8 on AIME 2026: 24/30 historically → 21/30 currently**. This is an observed score drop; its cause has not been established. Passing component checks and confidence intervals do not establish end-to-end equivalence.

This branch contains the tested vLLM integration plus this reproduction package. The accuracy measurements apply to vLLM `84cb2c036be25c76a58947b9ebc036f8e8f76bb0` with FlashInfer `6fdd402bf2d2d7ea18264e86bb3a8815c56c3b92` from [FlashInfer PR #5434](https://github.com/flashinfer-ai/flashinfer/pull/5434). The publication adds documentation and evaluation scripts without changing that integration. It stays on the evaluated nightly base; it is not a rebase to public vLLM main.

- [Run the accuracy suite](RUNNING.md): environment, checkpoint/data pins, serving commands, resume behavior and result accounting.
- [Remaining investigation](INVESTIGATION.md): completed checks, exact unresolved hypotheses and the next experiments.
- [Full task-by-task comparison](COMPARISON.md): paired changes and uncertainty against historical native and true-default runs.
- [Machine-readable scores](results.json) and [protocol pins](protocol.json).

## Accuracy results

Scores are percentages. GSM8K has 1,319 questions, GPQA Diamond 198, and AIME 2026 30. “Old” is the completed historical campaign; “current” is the fresh October 19–20 campaign. Default scores are historical references, not rerun defaults at the new source revision.

| Model | Backend / actual KV format | GSM8K | GPQA Diamond | AIME 2026 |
| --- | --- | ---: | ---: | ---: |
| V4 Flash | Default / packed block-scaled FP8 | 96.59 | 69.70 | 73.33 |
| V4 Flash | Old native / BF16 | 96.97 | 69.19 | 73.33 |
| V4 Flash | Current native / BF16 | 97.12 | 68.69 | 76.67 |
| V4 Flash | Old native / FP8 E4M3 | 97.42 | 70.20 | 80.00 |
| V4 Flash | Current native / FP8 E4M3 | 97.19 | 70.20 | 70.00 |
| V4 Pro | Default / packed block-scaled FP8 | 97.12 | 70.20 | 96.67 |
| V4 Pro | Old native / BF16 | 96.59 | 74.24 | 96.67 |
| V4 Pro | Current native / BF16 | 96.59 | 72.73 | 96.67 |
| V4 Pro | Old native / FP8 E4M3 | 96.82 | 73.23 | 93.33 |
| V4 Pro | Current native / FP8 E4M3 | 96.97 | 77.27 | 93.33 |
| V4.1 Flash | Default / FP4 compressed, MXFP8 SWA | 95.38 | 89.90 | 93.33 |
| V4.1 Flash | Old native / BF16 | 95.98 | 90.40 | 96.67 |
| V4.1 Flash | Current native / BF16 | 95.68 | 87.88 | 96.67 |
| V4.1 Flash | Old native / FP8 E4M3 | 95.68 | 87.37 | 100.00 |
| V4.1 Flash | Current native / FP8 E4M3 | 95.83 | 88.38 | 93.33 |
| GLM 5.3 Flash | Default / BF16 | 96.13 | 77.78 | 73.33 |
| GLM 5.3 Flash | Old native / BF16 | 96.89 | 79.80 | 80.00 |
| GLM 5.3 Flash | Current native / BF16 | 96.82 | 79.29 | 76.67 |
| GLM 5.3 Flash | Default / FP8 | 95.75 | 84.34 | 83.33 |
| GLM 5.3 Flash | Old native / FP8 E4M3 | 95.91 | 83.84 | 76.67 |
| GLM 5.3 Flash | Current native / FP8 E4M3 | 95.83 | 83.33 | 73.33 |

Matching default BF16 cells are **unsupported** for the pinned DeepSeek backends. V4.1's default uses FP4 compressed KV even if passed `--kv-cache-dtype fp8`; it is therefore reported with its actual format. No alternative default backend was substituted. Default-versus-native differences include cache quantization and other backend behavior, and cannot isolate attention arithmetic.

The current campaign independently audited 12,376 fresh responses across all 24 cells, with zero integrity failures and 24,512 primary/strict grade records reproduced. Historical response checkpoints did not complete current cells. All capped and empty-final outcomes remain included. The historical campaign completed 39 supported task cells.

## Evaluation contract

- Official, revision-pinned weights retain their original weight quantization. All runs used the same ARM64 nightly container, GB300, Torch `2.13.0+cu130`, CUDA 13.0, CUTLASS DSL 4.8.0, configured context 1,048,576 and high reasoning.
- Temperature 1.0; generation seed and task/choice/few-shot seeds 1234; one full generation pass. V4 Flash/Pro: top-p 1.0, cap 393,216. V4.1: top-p 0.95, cap 262,144. GLM: top-p 0.95, chosen cap 163,840. V4.1 high maps to reasoning effort 50, not its published effort-100 leaderboard protocol.
- Pinned lm-evaluation-harness supplies GSM8K eight-shot CoT and GPQA zero-shot CoT prompts and flexible-extraction primary graders. Strict extraction is retained as a diagnostic. AIME uses its boxed-answer grader with leading-zero integer equivalence and the 2026 dataset, both exams. There are no AIME 2024/2025 results in this campaign.
- Client concurrency 32; server maximum 32 sequences and 4,096 batched tokens. Most tasks used resumable 32-question shards. Pro continuations used larger contiguous remaining ranges to avoid idle tail requests; their prompt/request identities were separately verified. Arrival order and dynamic batches can still differ.
- Configuring a million-token context is not a million-token retrieval-quality test. The small AIME cohort and single generation seed cannot establish universal equivalence or causality.

## Source revisions

| Role | Revision |
| --- | --- |
| Pinned nightly vLLM base | `2eb491e18b1ce5b8a25e4c88f32dbb5d028d525c` |
| Historical vLLM integration | `b285cbfb6cd0bc17e5c957a3c8f57c1d7598448d` |
| Current evaluated vLLM integration | `84cb2c036be25c76a58947b9ebc036f8e8f76bb0` |
| Historical FlashInfer, V4 Flash | `735e1f4f33f8d112d47e0ab38adc5ab7b9272abd` |
| Historical FlashInfer, other models | `33f55648f59d951373ff738c4975fbeb8d887460` |
| Current evaluated FlashInfer | `6fdd402bf2d2d7ea18264e86bb3a8815c56c3b92` |
| lm-evaluation-harness | `d6de81643928d653435c431bae19945d41d32520` |

## Fixed findings and remaining work

Integration found and repaired DSL 4.8 compilation issues: the scheduler's `WorkTileInfo` type identity, a split-reducer variable scope collision, and a staged fixed sparse-loop condition. The vLLM adapter aligns cache allocations to complete native rows, supplies separate per-pool indices/lengths and effective scales, and preserves the group-major FP8 output/UE8M0 scale contract consumed by WO-A. Those fixes are included in the evaluated pins. They are not evidence that the later AIME drop is resolved.

Completed same-input tests found all 60 old/current attention replays bitwise identical and all 60 old/current WO-A layout pairs bitwise identical. A separate 120-response attention × MoE-profile diagnostic did not isolate one consistent cause. Token-level divergence starts at context 2,052, where C4 compressed candidates first exceed top-k 512. The next target is the actual nightly TopK provider's selected sets and ordering, followed by real-model boundary captures and matched repeated full evaluations. See [INVESTIGATION.md](INVESTIGATION.md) for evidence and limits.

As of publication, TopK diagnostic job `1377619` is pending scheduler priority on batch/normal QoS. Its GPU behavior has not been measured; no selector bug or causal kernel regression is claimed. No numerical reference, tolerance, official response or official score was changed by the investigation.

## Evidence and publication scope

Aggregate score provenance hashes are in `results.json`; detailed paired counts are in `COMPARISON.md`. Large raw responses, SQLite checkpoints, model weights, benchmark dataset contents, credentials, caches and temporary debug instrumentation are excluded from this branch. The full local evidence remains under `research/vllm-accuracy`, `research/vllm-accuracy-latest-20261019` and `research/aime-regression-20261020` in the original workspace. The investigation document records exact retained artifact names and hashes for handoff.

The portable scripts adapt the qualified runner's paths and orchestration while retaining its evaluation logic. Offline checks reproduced all 24 current cells, including completion-length and length-stop statistics, directly from the retained closed results/raw responses. The evaluator's model-construction and sampling/grading calls and the AIME grader's function body match the original abstract syntax. Task staging and all eight native/four default launch commands were checked without loading models; unsupported default BF16 cells are rejected. All applicable repository pre-commit hooks pass. The portable scripts have not been used for another full GPU accuracy campaign. AI assistance was used to prepare the integration and this report.
