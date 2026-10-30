# Biodex: select evidence before scanning labels

On the same five-paper pilot, selecting evidence before the exhaustive label scan gives **RP@5 = 0.5400** and **RP@10 = 0.5200**. The composed standalone API cost is **$0.826891**, 3.3% lower than scanning and ranking with full papers.

## Method

1. Jev classifies every numbered sentence, heading, or delimited table row as a patient finding, necessary context, background, or uncertain. Code copies findings, context, and uncertain units with associated headers from exact source offsets.
2. Use only those copied excerpts as the state for all 24,313 independent supported/not-supported label questions. The scan prompts, vocabulary order, 256-question batching, and model are unchanged. No embeddings or label retrieval are used.
3. Keep the supported labels and use the same iterative Jev ranking: choose the best remaining label, remove it, and repeat up to ten times. For pools above 253 labels, retain up to ten per group and rank the combined survivors. The ranking stage can request the original paper via MORE_CONTEXT; the scan cannot. Code copies official label names.

This run reuses exactly the evidence from the earlier pilot. It therefore changes where that evidence enters the pipeline, without changing the evidence-selection decisions. All evidence source hashes and copied spans were checked. Ground truth is used only after model decisions.

## Comparison on the same five papers

| Method | RP@5 | RP@10 | Standalone API cost | Composed stage time |
|---|---:|---:|---:|---:|
| Full-paper scan → full-paper iterative ranking | 0.6400 | 0.6200 | $0.854967 | 271.50s |
| Full-paper scan → evidence selection → iterative ranking | 0.6400 | 0.6200 | $0.861355 | 285.97s |
| **Evidence selection → short-text scan → iterative ranking** | **0.5400** | **0.5200** | **$0.826891** | **305.68s** |

Costs include every required stage once. Timings combine separately measured stages, including historical reused work; they are not fresh end-to-end measurements. API service latency varies between runs. RP@k uses the unchanged MOAR evaluator: correct labels among the first k divided by min(k, number of ground-truth labels), averaged across documents.

## Stage costs and latency

| Stage | API cost | Stage wall time | API calls |
|---|---:|---:|---:|
| Evidence selection + copying (reused) | $0.011580 | 15.83s | 63 |
| Scan all labels using short text | $0.784869 | 269.68s | 475 |
| Iterative ranking, including any context fallback | $0.030442 | 20.16s | 79 |

**New API spend this run: $0.815311.** New process wall time: **290.14s**. Evidence was reused and was not charged again, but its cost and original duration are included in the standalone comparison.

Unknown-cost new calls: 0. New retries: 0. Ranking full-context requests: 0. Costs use recorded Jev input tokens × $0.042 per million; these are usage estimates, not invoices. Local compute costs $0 by convention.

## Does filtering remove useful label evidence?

The copied evidence retains **65.4%** of source characters. Macro ground-truth recall among scan-supported labels changes from **0.9786** to **0.9786**. The supported-label oracle RP@5 is **1.0000**.

| Paper | Old / new supported-label count | Old / new GT recall | Previous / new RP@5 | Newly rejected GT labels | Newly supported GT labels |
|---|---:|---:|---:|---|---|
| 33811622 | 190 / 173 | 100.0% / 100.0% | 0.4000 / 0.4000 | — | — |
| 34786403 | 384 / 353 | 100.0% / 100.0% | 0.5000 / 0.5000 | — | — |
| 34703590 | 245 / 222 | 100.0% / 100.0% | 1.0000 / 1.0000 | — | — |
| 33619451 | 299 / 293 | 100.0% / 100.0% | 0.5000 / 0.0000 | — | — |
| 33552603 | 641 / 632 | 89.3% / 89.3% | 0.8000 / 0.8000 | — | — |

“Newly rejected” means supported in the earlier full-paper scan but rejected in this short-text scan. These are changed model decisions, not proof that the corresponding source sentence was removed. Context and model variability can also change decisions. Labels rejected during scanning cannot be recovered by a ranking-only context fallback.

For paper **33619451**, both ground-truth labels still passed the short-text scan, but neither survived the group shortlist. The previous ranking kept **Cardiac failure**; the new output includes **Cardiac failure congestive** instead. These are different official labels, so the unchanged exact-label metric gives no credit for the latter against the former. This paper accounts for the entire 0.10 drop in aggregate RP@5 and RP@10. The other four papers retain their previous scores. The changed candidate pool and a new ranking run can affect choices; this does not prove a loss of relevant source evidence.

## Interpretation

- This pilot saves only a small amount of API cost and does not show a latency improvement. Short-text scanning preserves the scan-supported ground-truth set, but downstream ranking is less accurate in this run.
- This is a single run on five development papers. It does not establish accuracy or latency on the 100-document test set.
- Shorter text reduces the document portion of repeated requests, but every request still contains label questions and instructions. A 34.6% reduction in source characters does not imply the same percentage reduction in API cost.
- Changed supported-label pools also change ranking groups. Accuracy differences reflect the entire new workflow, not evidence selection in isolation.
- All 33 offline tests passed before this run, including checks that the short-text scan uses the same label questions, covers all labels, sends only copied evidence, and rejects inconsistent cached source spans.

Post-run audit verified all 121,565 scan decisions across 475 scan calls and 79 ranking calls. All 554 new calls succeeded with no retries. Scan states contain only copied evidence, every label is covered once, output labels are valid and unique, and the cost ledger matches the summary. [Audit](runs/evidence-first-train5-v1/audit.json).

## Files

- [Implementation](evidence_first.py) · [Tests](test_evidence_first.py).
- [Protocol](runs/evidence-first-train5-v1/protocol.json) · [Summary](runs/evidence-first-train5-v1/summary.json).
- [Predictions](runs/evidence-first-train5-v1/results.json) · [Changed ground-truth decisions](runs/evidence-first-train5-v1/comparison.json).
- Per-paper folders retain copied evidence, all scan answers, ranking traces, exact request/response ledgers, costs, and timings.

```bash
python experiments/semantic-decomposition/biodex/evidence_first.py \
  --evidence-run experiments/semantic-decomposition/biodex/runs/iterative-evidence-train5-v1 \
  --output experiments/semantic-decomposition/biodex/runs/evidence-first-train5-v1
python experiments/semantic-decomposition/biodex/report_evidence_first.py
```
