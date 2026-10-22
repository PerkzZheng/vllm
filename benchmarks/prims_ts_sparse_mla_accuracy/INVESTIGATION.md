# Remaining AIME investigation

## What needs explaining

The official V4 Flash native FP8 AIME 2026 score changed from 24/30 to 21/30. Zero-based questions **8, 10 and 16** changed from correct normal completions to 393,216-token length stops with empty final answers. Old completion lengths were 48,985 / 73,450 / 60,065 tokens. New repeated 32-gram fractions were 37.4% / 26.9% / 68.3%, versus 0.8% / 0.5% / 1.0% previously. Repetition and truncation describe the failed trajectories; they do not explain what initiated divergence. Historical default runs already looped/capped on questions 10 and 16.

Other AIME signals remain open: V4.1 native FP8 changed 30/30 → 28/30; GLM native BF16 changed 24/30 → 23/30 and native FP8 changed 23/30 → 22/30. GLM current FP8 is 22/30 versus its historical default's 25/30. The focused diagnostics below concern V4 Flash; their conclusions must not be extended to those other models without corresponding experiments. V4 Pro's question-level AIME correctness was unchanged.

GSM8K/GPQA changes, including V4.1 BF16 GPQA's 179/198 → 174/198, remain recorded in [COMPARISON.md](COMPARISON.md). No source-specific cause has been established for those changes either. The present investigation prioritizes the AIME losses; it does not certify the other cells as regression-free.

## Completed checks and their limits

| Check | Result | What it establishes / does not establish |
| --- | --- | --- |
| Official campaign integrity | 24 cells, 12,376 fresh responses, zero integrity failures; primary and strict grading reproduced | Input/grader/result provenance is sound. It does not attribute score changes to a kernel. |
| Offline trajectories | 270 default/old/current FP8 trajectories across Flash, V4.1 and GLM inspected | Original reasoning text exists, but historical token IDs/logprobs are null; re-tokenizing text cannot restore the exact original stream. |
| Relevant source and policy comparison | Sampler/runner/parser sources inspected are unchanged; 240 H64 policy points have no scheduling differences except the removed metadata selector | Rules out those particular code changes. Does not prove identical runtime scheduling or all generated machine code. |
| Same-input attention replay, job 1357419 | 60/60 old/current pairs bitwise identical: plain output, LSE, fused FP8 values and packed scales after a value-preserving layout bridge | Captured H64/D512, T=1/3/30, SWA-only/C4/C128, short/8k/32k prefixes. Both arms pass the unchanged FP64 bound. Eager component test; not full graph-coupled generation or every long-context input. |
| Production WO-A layout replay, job 1363832 | 60/60 old token-major/current group-major pairs bitwise identical; all 120 layout fixtures pass oracle/eager/graph checks | Real processed checkpoint weights and production DeepGEMM provider, five layers. Isolates the changed physical layout at these inputs; does not prove all consumer paths. |
| MoE GPU binary audit | 14 SM103a ELFs have identical code/constants/launch metadata; 2,013 sections match. Full ELF equality after normalizing compiler IDs in string tables. All 215 BMM cubins and 18 artifact headers match byte-for-byte | No GPU code difference found there. Host-wrapper, tactic choice and per-token launch identity are not globally proved identical. |
| Saved MoE profiles | 20 tuning entries/profile. Historical initial versus current differs in 13 choices; historical resume versus current differs in 12 | Concrete runtime confounder, not demonstrated causation. The historical 24/30 mixes 23 initial responses and 7 resumed responses; question 10 came from the resume, questions 8/16 did not. |

Both attention arms had zero violations of the unchanged independent FP64 bound; maximum relative RMSE was 0.0096701 for plain output, 0.029059 for fused output, and 8.73e-8 for LSE. The WO-A replay's maximum absolute error was 0.00780919 and relative RMSE 0.00169184 under its original rtol 0.008 / atol 0.001 gate. These bounds were not relaxed.

## Attention × MoE-profile diagnostic

Job 1364941 completed four independent full 30-question AIME cohorts with 120 fresh responses. Each cell used the common current auxiliary stack and either historical/current attention and historical-initial/current saved MoE tactics. The historical attention wrapper only bridges logical FP8 values to the current downstream layout; it performs no numerical re-quantization. All input, origin, tactic-load, graph-readiness, response and grader gates passed.

| Attention | Saved MoE profile | Correct | Length stops |
| --- | --- | ---: | ---: |
| Historical | Historical initial/global | 22/30 | 8 |
| Historical | Current | 23/30 | 7 |
| Current | Historical initial/global | 22/30 | 8 |
| Current | Current | 21/30 | 8 |

With historical-initial tactics, old/current attention have identical question-level correctness. With current tactics, current attention gains one and loses three questions: −6.67 percentage points, empirical paired 95% interval [−20.00, +6.67]. Changing tactics has opposite net effects depending on the attention arm. Therefore no one factor consistently explains the original drop. These are one run per cell, not independent repeated estimates. Fresh AIME-only server histories differ from the official GSM8K → GPQA → AIME history; asynchronous arrival and dynamic batch order were not forced identical. The third historical resume profile is not represented. These diagnostic scores do not replace official scores.

## Why TopK is the next target

All 120 diagnostic responses retain exact token IDs. Across the four factor-comparison edges, there are **108 correlated first divergences across 27 questions**. They are not 108 independent samples. Every first divergence occurs at logical input context ≥2,052; 101/108 occur between 2,052 and 2,136, and the maximum is 2,444. Three shorter completions match exactly across all four arms and end below context 2,048.

Context here means prompt length plus the zero-based index of the first differing generated token, i.e. the input context before that emission. For C4 compression, the number of compressed candidates is `floor(context / 4)`. With top-k 512, the first 513-candidate context is **2,052**. Through context 2,051, the selector can return all candidates in ascending order. Above that boundary the source uses the cooperative histogram/atomic-scatter path, whose output is unsorted. Actual V4 Flash geometry is H64, eight output groups, SWA 128 and compressed top-k 512; it is not the H128 benchmark geometry.

Two distinct hypotheses must be tested:

1. **Correct set, different order.** The same FP32 logits may produce the same selected KV set in different orders. Unsorted output is not inherently a TopK correctness bug. Different orders change KV tile grouping and FP8 online probability quantization, potentially changing attention output within its numerical bound. Repeated fixed-input selection and attention replay can measure this mechanism; neither alone proves an AIME cause.
2. **Incorrect membership near a coarse-bin boundary.** In pinned source `csrc/libtorch_stable/topk_histogram_4096.cuh`, consider N=513, K=512 and distinct scores `1 + i * 2**-20` for i=0…512. All 513 share one coarse FP16 histogram bin. The source retains only the first K boundary-bin arrivals before exact FP32 refinement. A legal atomic order can discard a higher-scoring candidate, which refinement cannot recover. This is a source-level counterexample involving distinct FP32 scores, not permissible equal-score tie selection. The compiled nightly provider has not yet been shown to use that exact algorithm or to exhibit this failure.

The actual `_C_stable_libtorch.abi3.so` hash is `d34b72362b9f1395bf7d1db637f861f64847ba465d7ed359a47110379e021f8e`. Its SM100 text contains cooperative K512/1024/2048 kernels, but symbol presence does not prove source correspondence or which runtime branch executes. Source-to-binary equivalence remains unproved. No live model logits/ordered indices/selected sets have yet been captured at the boundary.

## Queued actual-provider experiment

Job **1377619** is pending scheduler priority on batch/normal QoS, four GB300s (QoS minimum), 30 minutes. It uses the exact pinned nightly and frozen sources. No GPU result exists as of this publication. Do not create a duplicate job or call a source counterexample a confirmed nightly failure.

The original suite uses public `SparseIndexerTopk("auto")` and raw `torch.ops._C.cooperative_topk` in fresh workers, each with unique wide-spaced scores or controlled true cutoff ties. Geometry: rows 1/3/30; raw contexts 2,048/2,051/2,052/2,054/2,060; compressed valid counts 512/512/513/513/515; K512; FP32 physical capacity/row pitch 262,144; INT32 compressed lengths shaped [B,1]; output [B,512]; host max-sequence argument remains the **raw** context. Each case repeats 128 times eagerly and 128 times with CUDA Graphs on unchanged inputs. Ordered arrays and sorted selected sets have separate hashes/oracles; tied cases do not require stable ordering or a particular equally valid selected set.

If actual unique-score results show two orders of the same selected set, a conditional attention replay varies only compressed index order on retained T1/T3 C4 layer-2 snapshots. It preserves Q, both KV sources, 512 distinct physical compressed IDs, SWA order, lengths, scales, sinks and RoPE inputs. Fixed-order repeats control repeatability. This is an order-sensitivity experiment, not an end-to-end causal test.

A separate supplement runs afterward with the narrow distinct-score pattern, ascending/descending layouts, valid counts 512/513/515, both providers, rows 1/3/30, and 32 eager plus 32 graph repetitions. It records out-of-bounds, duplicate and valid-but-wrong-set rows separately, including missing/unexpected golden IDs. It continues after numerical failures to save all cases, then intentionally exits 2 if any set oracle failed. Such an exit must not erase or misclassify independently completed ordering results.

## Next actions, in order

1. Complete job 1377619. Check actual auto backend, operator schema/dispatch table, loaded library path/hash, geometry and frozen-file hashes before interpreting outputs. Record ordering variants separately from selected-set correctness. Preserve partial and failing outputs.
2. If membership fails, isolate the actual provider's failure and propose a generic boundary-bin refinement fix that considers every relevant candidate. Increasing a fixed buffer for just N=513 is insufficient. Validate distinct scores, true ties, valid bounds and eager/graph execution before a production change. Do not modify the frozen campaign checkouts or weaken the oracle.
3. If order changes, finish the conditional real-KV attention replay. Capture actual model-produced logits/ordered indices/sets, Q/KV and scales around contexts 2,048 / 2,051 / 2,052 / 2,054 / 2,060 using retained exact token prefixes. Match request, layer, compressed lengths, physical IDs and batch composition. Avoid reconstructed historical text as a claim of exact replay.
4. At the first divergent model step, compare selected sets, index order, plain attention/LSE, fused FP8 values/scales, WO-A output, then downstream logits and sampling. Separate a membership defect, permissible order sensitivity, consumer behavior and MoE tactic effects. Keep any instrumentation in isolated diagnostic copies.
5. Run matched repeated full AIME evaluations, including same-version repeats, controlled tactic/cache profiles and recorded token IDs. Preserve high reasoning, task order, maximum context, caps, graders, per-request seeds and all length stops. Match batching where feasible and report remaining differences. Estimate mean pass@1 over seeds, not pass@N; do not present one favorable rerun as a fix.
6. Revisit V4.1 and GLM AIME losses with model-specific captures if the mechanism does not explain them. Longer contexts beyond the 32k replay coverage and full graph-coupled generation remain unqualified by the existing component tests.

Separately, the historical Pro integration had post-readiness CuTe JIT misses. Raising its worker RPC timeout from 300 to 1,800 seconds allowed evaluation to finish; it did not fix warmup coverage. A representative Prims-TS warmup audit remains worthwhile. Additional arbitrary-FP32-scale WO-A experiments were incompatible with the pinned DeepGEMM einsum consumer; production UE8M0 checks passed. Do not describe unsupported FP32 consumer combinations as validated.

## Retained local evidence

These paths are relative to the original workspace's `research/` directory, not files bundled in this Git branch. They intentionally exclude raw model reasoning and licensed dataset content from publication.

| Artifact | SHA256 / purpose |
| --- | --- |
| `vllm-accuracy-latest-20261019/baseline/manifest.json` | `41569a562fa2e2def54889f0e826a03eb31a22370dd43732e1e979cb9cb9d4ab`: historical raw/protocol snapshot |
| `aime-regression-20261020/replay-comparison.json` | `487fd460cb4a0e8e69eb734d0ba7e35575dc1837d6d5c8eb408cb248f3b4d5e9`: 60 attention pairs |
| `aime-regression-20261020/source-controls/qualification.json` | `e8d864936bc7d629b41fc74eac4f3c1b2e87fb8fd3c7e6b2e650814984d55fe0`: exact historical control qualification |
| `aime-regression-20261020/consumer-layout/comparison.json` | `19a3f0c52f3b65f1438842c60b0fbb3b5c519878903013a74430e9e038ca2756`: 60 WO-A layout pairs |
| `aime-regression-20261020/paired-repeats/reviewer-final-factorial-audit.json` | `c1d6e7b1f5fb4cbef8181b1b7fcb38976dc1bc7ee2a49f72f6379dfc649fe623`: independent 120-response audit |
| `aime-regression-20261020/indexer-order/manifest.json` | `64c3929e7f28817d25f5a54adeab51cf6b59ca3d3e7053b3c9f50e2af8f3c250`: reviewed 115-file candidate |
| `aime-regression-20261020/indexer-order-narrow/STATIC_REVIEW.md` | Source counterexample and independent review; runtime pending |

No production kernel, selector, tolerance, grader, official response or official score has been changed to resolve this investigation. The evidence currently supports targeted follow-up, not a confirmed causal diagnosis.
