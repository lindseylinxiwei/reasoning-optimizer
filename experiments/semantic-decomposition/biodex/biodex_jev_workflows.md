# Biodex: three Jev workflows

**None of the three new workflows improves training RP@5 over the prior retrieval + Jev baseline (0.2379).** Among the new workflows on the 40-document training split, **Evidence → dense retrieval → Jev** has the highest RP@5 (0.2367) and is selected by the prewritten rule: highest training RP@5, then lower API cost, then lower composed latency. This is a selection result, not an unbiased test estimate. On the 100-document test follow-up it reaches **0.1628**, versus **0.1950** for prior retrieval + Jev.

[Frozen workflow selection](runs/workflows-selection-v1.json)

## Methods

All three methods use Jev 1.13.0, return up to 10 official labels, and use the unchanged MOAR evaluator. No document answer is generated as free text; code copies source evidence and official label names.

1. **Evidence → dense retrieval → Jev:** Jev marks patient-relevant or uncertain passages; code copies them. The existing cached passage embeddings retrieve 64 labels. Jev verifies and ranks these labels.
2. **Full paper → hierarchy → Jev:** Jev explores relevant label groups using the complete paper, then verifies labels in the reached leaves.
3. **Evidence → hierarchy → Jev:** use the same copied evidence as method 1, then the same hierarchical search and verification as method 2.

Evidence questions retain symptoms, diagnoses, medical history, laboratory/genetic findings, treatment failure and drug-use issues. Overlapping source spans are merged; an empty selection gives an empty prediction. No dense label filter is applied after hierarchical search.

The tree clusters all 24,313 official labels using cached MOAR `text-embedding-3-small` vectors: **611 nodes, 402 leaves, at most 96 labels per leaf**. It uses up to 12 children, 16 representative labels per internal node, and a global beam of 6. This is an embedding-cluster search tree, not a medical ontology.

Jev independently marks groups supported, uncertain or none. Non-none paths compete by the geometric mean of `P(supported) + 0.5 × P(uncertain)`; finished leaves also occupy beam slots. Final label questions are binary, and supported labels are ranked by P(supported). Both ranking scores are uncalibrated heuristics. Branch pruning can still miss labels, and the workflows use different candidate budgets.

## Training comparison (40 documents)

| Workflow | RP@5 | RP@10 | Candidate recall | Oracle RP@5 | Mean candidates | API USD | Composed s | Errors / unknown-cost calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Evidence → dense retrieval → Jev | 0.2367 | 0.2183 | 0.2339 | 0.2850 | 62.4 | $0.059581 | 71.38 | 1 / 1 |
| Full paper → hierarchy → Jev | 0.0267 | 0.0258 | 0.0258 | 0.0267 | 238.2 | $0.179120 | 118.92 | 0 / 0 |
| Evidence → hierarchy → Jev | 0.0267 | 0.0258 | 0.0342 | 0.0350 | 233.6 | $0.186649 | 170.16 | 1 / 1 |

Costs and times above are the measured cached-stage execution; historical embeddings and one-time setup are excluded.

Prior full-paper dense retrieval + Jev at K=64 on this same split: **RP@5=0.2379**, RP@10=0.2221; candidate recall=0.2402, oracle RP@5=0.2913.

| Workflow vs prior Jev baseline | Paired ΔRP@5 | 95% bootstrap interval |
|---|---:|---:|
| Evidence → dense retrieval → Jev | -0.0012 | [-0.0187, +0.0150] |
| Full paper → hierarchy → Jev | -0.2112 | [-0.3183, -0.1154] |
| Evidence → hierarchy → Jev | -0.2112 | [-0.3183, -0.1154] |

Stage contributions:

| Workflow | Evidence USD / s | Search USD / s | Verification USD / s |
|---|---:|---:|---:|
| Evidence → dense retrieval → Jev | $0.031622 / 52.10 | $0.000000 / 1.28 | $0.027959 / 17.99 |
| Full paper → hierarchy → Jev | — | $0.055588 / 33.28 | $0.123533 / 85.64 |
| Evidence → hierarchy → Jev | $0.031622 / 52.10 | $0.046918 / 39.98 | $0.108110 / 78.08 |

The prior baseline's Jev ranking alone cost $0.033157 / 21.53s. Its original shared preparation cost $0.007812 / 387.95s. These are separate historical measurements, not a matched fresh-run timing comparison.

Actual new API spend for this split: **$0.393728 known**, 1 unknown-cost calls, 1010 total calls. Shared evidence is counted once here.

[Summary](runs/workflows-train-v1/summary.json) · [Protocol](runs/workflows-train-v1/protocol.json) · [Setup](runs/workflows-train-v1/setup.json)

- [Evidence → dense retrieval → Jev outputs](runs/workflows-train-v1/evidence_retrieval/results.json)
- [Full paper → hierarchy → Jev outputs](runs/workflows-train-v1/full_hierarchy/results.json)
- [Evidence → hierarchy → Jev outputs](runs/workflows-train-v1/evidence_hierarchy/results.json)

## Why the hierarchy failed in this run

The post-run trace audit locates where correct labels disappeared. Counts below are unique **ground-truth label/document pairs** (micro counts), not documents or the macro candidate recall above. Each pair has one pre-verification outcome.

| Workflow | Total pairs | Reached candidates | Rejected as none | Pruned by beam | API failure | Absent from vocabulary |
|---|---:|---:|---:|---:|---:|---:|
| Full paper → hierarchy → Jev | 199 | 4 | 190 | 4 | 0 | 1 |
| Evidence → hierarchy → Jev | 199 | 5 | 189 | 3 | 1 | 1 |

For full-paper routing, 179 pairs were already rejected at the first level. For both hierarchy variants, **none of the rejected groups explicitly displayed the missed ground-truth label** among its examples. The full-paper candidate oracle equals its final RP@5, so final verification cannot repair this loss.

For example, paper 33236195's label **death** belonged to a 3,044-label group represented by examples such as “Device psychogenic complication,” “Ainhum,” and “Military service.” The router chose none before ever seeing “death” as a candidate.

**This cluster-description design failed:** a few representative labels did not communicate the contents of large, mixed groups. The traces support routing failure; they do not prove Jev cannot classify biomedical labels or that all label hierarchies fail.

Evidence selection retained **74.0% of source characters** across 39 successful documents (pooled character ratio). It therefore removed relatively little text.

Shared evidence extraction failed for 33926386 (HTTP 520). This one request affects both evidence workflows, counts as a miss in each, and has one unknown charge in actual spend. No retry was made. Even a perfect result on that paper could not close the large gap between the selected retrieval workflow and either hierarchy workflow.

[Post-run routing audit](runs/workflows-train-v1/hierarchy_diagnostics.json)

## Test follow-up (100 documents)

| Workflow | RP@5 | RP@10 | Candidate recall | Oracle RP@5 | Mean candidates | API USD | Composed s | Errors / unknown-cost calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Evidence → dense retrieval → Jev | 0.1628 | 0.1817 | 0.1886 | 0.2115 | 64.0 | $0.148968 | 93.49 | 0 / 0 |

Costs and times above are the measured cached-stage execution; historical embeddings and one-time setup are excluded.

Prior full-paper dense retrieval + Jev at K=64 on this same split: **RP@5=0.1950**, RP@10=0.1973; candidate recall=0.2096, oracle RP@5=0.2330.

| Workflow vs prior Jev baseline | Paired ΔRP@5 | 95% bootstrap interval |
|---|---:|---:|
| Evidence → dense retrieval → Jev | -0.0322 | [-0.0608, -0.0110] |

Stage contributions:

| Workflow | Evidence USD / s | Search USD / s | Verification USD / s |
|---|---:|---:|---:|
| Evidence → dense retrieval → Jev | $0.077502 / 37.25 | $0.000000 / 3.34 | $0.071467 / 52.91 |

The prior baseline's Jev ranking alone cost $0.081616 / 52.29s. Its original shared preparation cost $0.012549 / 75.28s. These are separate historical measurements, not a matched fresh-run timing comparison.

Actual new API spend for this split: **$0.148968 known**, 0 unknown-cost calls, 320 total calls. Shared evidence is counted once here.

[Summary](runs/workflows-test-v1/summary.json) · [Protocol](runs/workflows-test-v1/protocol.json) · [Setup](runs/workflows-test-v1/setup.json)

- [Evidence → dense retrieval → Jev outputs](runs/workflows-test-v1/evidence_retrieval/results.json)

## Interpretation and accounting

RP@k is the number of correct labels in the first k outputs divided by min(k, number of ground-truth labels), averaged across labeled papers. Candidate recall measures ground-truth label coverage before verification; oracle RP@5 is the ceiling from that candidate set. Errors produce empty predictions and count as misses.

Paired percentile-bootstrap intervals use the same document IDs and ground truth, 10,000 resamples, seed 20260929. They describe variation across these documents, not repeated model runs; training comparisons are subject to selection and the intervals are not adjusted for multiple comparisons. They do not establish equivalence. Earlier test results informed this follow-up, so even the separate test follow-up is exploratory.

Shared evidence extraction ran once per split and was reused by both evidence workflows. Each standalone workflow total includes its evidence contribution; actual session spend counts those calls only once. Local computation is $0 by convention. Jev cost is recorded input usage × $0.042 per million tokens; these are estimates, not invoices. Unknown charges are not treated as free.

These are warm-cache workflow measurements: historical label and passage embedding creation charges are excluded. Evidence + dense retrieval depends on cached document-passage embeddings; a fresh deployment must also create them. The hierarchy methods depend on cached label embeddings but need no document embedding API calls. The earlier training preparation was cold-cache, so its total cannot be used to claim a workflow speedup.

Training setup was recorded separately: **2.81s**, $0.000000 API cost. It covers cache/tree loading or construction performed by the runner; any earlier tree prebuild is not included unless separately recorded. The same tree is reused for test. Setup is excluded from method tables.

Stage times are sequential wall-clock contributions summed into composed totals. Shared-stage reuse means these are not three fresh end-to-end runs, and loading/reporting/setup overhead is excluded. Interruptions, if present, may leave a marked API-time lower bound; completed artifacts are resumed without automatic API retries.

Total new API spend in the completed experiments above: **$0.542697 known**, with **1 unknown-cost calls**.

[Implementation](workflows.py) · [Tree builder](hierarchy.py) · [Previous experiments](biodex_jev.md) · [Paired statistics](biodex_jev_workflows_stats.json)
