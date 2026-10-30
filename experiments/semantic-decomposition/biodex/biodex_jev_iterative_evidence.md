# Biodex: finer evidence selection and iterative Jev ranking

On the same five-paper pilot, RP@5 is **0.1600** with the previous probability ranking, **0.6400** with iterative selection using the full paper, and **0.6400** with iterative selection using Jev-selected evidence.

## Method

1. Reuse the saved exhaustive Jev scan. Every one of the 24,313 labels was considered in that scan. Both new variants receive the identical supported-label pool for each paper, in official vocabulary order (190–641 labels). There is no embedding retrieval or new candidate generation in this pilot.
2. Split every paper into numbered sentences, short headings, or delimited table rows using local deterministic rules. Classify every unit as a patient finding, necessary context, background, or uncertain. Each call shows a block of up to 16 target units, two neighboring units on each side, and associated headings.
3. Code copies findings, necessary context, and uncertain units verbatim from saved source offsets. It also retains associated headings and detected table headers. The prompt explicitly preserves negation, drug relationships, qualifiers, treatment failure, and drug-use issues. These prompts encourage semantic preservation; they do not guarantee perfect evidence recall.
4. Ask Jev to choose the best remaining supported label, remove that choice, and repeat up to ten times, stopping at NONE. For more than 253 candidates, select up to ten per group, combine survivors in original vocabulary order, and select the final ten. Independent group questions share a request; successive ranks depend on prior answers.
5. The selected-evidence variant can choose MORE_CONTEXT. Code then supplies the full paper and keeps it for that group’s remaining rounds. 253 labels plus NONE and MORE_CONTEXT fit the 255-option limit. Code assembles the final output by copying official label names.

**Comparison:** the ranking algorithm, initial candidate pool, and candidate order are held fixed. The evidence variant adds filtering and an explicit full-context fallback; this evaluates that policy, not a strictly forced-short-context ablation. The full-paper variant has no MORE_CONTEXT option.

## Accuracy

| Method | RP@5 | RP@10 |
|---|---:|---:|
| Previous full scan → probability sort | 0.1600 | 0.2500 |
| Same supported labels → iterative selection, full paper | 0.6400 | 0.6200 |
| Same supported labels → iterative selection, selected evidence | 0.6400 | 0.6200 |

The unchanged MOAR evaluator computes RP@k as correct labels among the first k predictions divided by min(k, number of ground-truth labels), averaged over papers. It is not whole-document accuracy.

| Paper | Supported labels | Previous RP@5 | Full-paper RP@5 | Selected-evidence RP@5 | Source characters retained | Full-context requests |
|---|---:|---:|---:|---:|---:|---:|
| 33811622 | 190 | 0.6000 | 0.4000 | 0.4000 | 69.0% | 0 |
| 34786403 | 384 | 0.0000 | 0.5000 | 0.5000 | 71.2% | 0 |
| 34703590 | 245 | 0.0000 | 1.0000 | 1.0000 | 48.6% | 0 |
| 33619451 | 299 | 0.0000 | 0.5000 | 0.5000 | 63.7% | 0 |
| 33552603 | 641 | 0.2000 | 0.8000 | 0.8000 | 65.2% | 0 |

Evidence selection classified **964 units**, retained **65.4% of source characters** (pooled across papers), and made 0 full-context requests during ranking. Decision counts: background=306, context=208, finding=449, uncertain=1.

| Paper | Full-paper survivors / GT recall | Selected-evidence survivors / GT recall |
|---|---:|---:|
| 33811622 | 190 / 100.0% | 190 / 100.0% |
| 34786403 | 20 / 50.0% | 20 / 50.0% |
| 34703590 | 245 / 100.0% | 245 / 100.0% |
| 33619451 | 20 / 50.0% | 20 / 50.0% |
| 33552603 | 26 / 21.4% | 26 / 21.4% |

Survivors are the labels entering the final ranking. For pools of at most 253, all initial supported labels enter directly. Ground truth is used only after model decisions.

## Cost and latency

| New stage, all five papers | API cost (USD) | Stage wall time (s) | API calls |
|---|---:|---:|---:|
| Sentence/row evidence classification + exact copying | $0.011580 | 15.83 | 63 |
| Iterative ranking with full paper | $0.035018 | 20.98 | 77 |
| Iterative ranking with selected evidence, including fallback | $0.029826 | 19.61 | 77 |

| Standalone variant after the saved scan | Incremental API cost | Incremental stage time |
|---|---:|---:|
| Full-paper iterative ranking | $0.035018 | 20.98s |
| Evidence selection + iterative ranking | $0.041406 | 35.44s |

Actual new API spend across both variants: **$0.076424**. Process wall time: **56.60s**. Retries: **0**; unknown-cost calls: **0**.

The prior full-vocabulary scan cost **$0.819949** and **250.53s** of composed stage time. It was reused, not rerun or charged again. Both methods still depend on that scan. Adding its historical cost and time gives the following composed estimates; these are not fresh end-to-end measurements:

| Variant including historical scan | Composed API cost | Composed stage time |
|---|---:|---:|
| Full-paper iterative ranking | $0.854967 | 271.50s |
| Evidence selection + iterative ranking | $0.861355 | 285.97s |

Jev cost uses recorded input usage × $0.042 per million tokens for jev-1.13.0; these are usage-based estimates, not invoices. Local computation is counted as $0. Stage times include request waiting, local processing, and checkpoint I/O. API requests run sequentially, while independent questions may share a request. No Gemini or embedding API calls were made.

## Interpretation and limits

- Iterative full-paper selection changes RP@5 by +0.4800 versus the saved probability sort.
- Adding selected evidence and fallback changes RP@5 by +0.0000 versus full-paper iterative selection. The filtering stage must be included when assessing cost or latency benefits.
- In this run, evidence selection preserves RP@5 but does not pay for itself in cost or latency. Iterative ranking provides the observed gain.
- This is a five-paper development pilot, with one run per variant. It does not establish statistical significance or generalization.
- The earlier binary scan already rejected three ground-truth labels on one paper. Ranking cannot recover labels outside its frozen pool.
- Group selection is a heuristic: it can discard correct labels before final ranking. The survivor recall table measures this loss.
- Sentence boundaries and table detection are heuristic. Flattened source tables cannot reliably recover their original layout. Every non-whitespace source character is assigned to a unit, and copied spans are verified against the source.
- We have not rescanned all 24,313 labels using the shorter evidence. This experiment therefore does not measure a cheaper exhaustive scan or independently measure evidence-selection recall against annotated source spans.

Validation: all 29 offline Biodex tests passed, including six new tests for source copying, negation preservation, grouping, iterative removal, and context fallback. A post-run audit verified all 217 successful calls, all 964 evidence decisions, exact source spans, unique official output IDs, and ledger totals. The full-context fallback was tested offline but was not requested in the live pilot.

## Saved artifacts

- [Implementation](iterative_evidence.py) and [tests](test_iterative_evidence.py).
- [Frozen protocol](runs/iterative-evidence-train5-v1/protocol.json) and [summary](runs/iterative-evidence-train5-v1/summary.json).
- [Full-paper predictions](runs/iterative-evidence-train5-v1/full_context_results.json).
- [Selected-evidence predictions](runs/iterative-evidence-train5-v1/selected_evidence_results.json).
- Per-paper folders contain exact copied spans, every evidence decision, ranking rounds, context requests, raw API request/response ledgers, usage, costs, and timing.

Reproduce (completed requests are checkpointed):

```bash
PYTHONDONTWRITEBYTECODE=1 TIKTOKEN_CACHE_DIR=/tmp/jev-tiktoken \
  /private/tmp/jev-comparison-venv/bin/python \
  experiments/semantic-decomposition/biodex/iterative_evidence.py \
  --output experiments/semantic-decomposition/biodex/runs/iterative-evidence-train5-v1
```
