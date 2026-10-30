# Biodex: Jev over the full label vocabulary

On the first **5 training papers**, removing retrieval **did not improve RP@5**: **0.1600**, versus **0.5000** for the previous K=64 pipeline on these same papers. All **121,565 label/document decisions** completed. This is a small exploratory pilot, not a 100-document result.

## Method

For each paper, Jev independently evaluates all **24,313 official labels** against the **complete document**. There is no embedding retrieval, evidence filter, hierarchy, or label preselection. Each question asks whether one label is supported; many labels may be supported in the same request.

The support question and model (`jev-1.13.0`) match the earlier Jev ranker. Requests contain up to 256 binary questions sharing one paper state, within conservative 24K state-plus-question and 48K whole-request token estimates. These are separate two-option questions, not a single winner among 256 labels. Each paper required 95 batches. Transient failures allow up to three total attempts, with every attempt logged.

Code keeps supported labels, orders them by P(supported), and copies up to 10 official names. Equal probabilities are resolved by official-vocabulary order. No probabilities are multiplied or treated as a joint distribution. Ground truth is used only after inference.

The first five papers in dataset order were fixed before calls. The protocol and every batch checkpoint are saved. The implementation reuses the existing question builder, transport, and MOAR evaluator.

## Quality on matched papers

| Method | MOAR RP@5 | MOAR RP@10 |
|---|---:|---:|
| Prior retrieval + Jev, K=64 (historical) | 0.5000 | 0.4800 |
| Full vocabulary → Jev (new run) | 0.1600 | 0.2500 |
| New answers restricted to prior K=64 (offline ablation) | 0.5400 | 0.4800 |

The offline ablation uses exactly the new full-scan answers, restricted to the original 64 labels. It uses vocabulary-order ties like the full scan and incurs no additional API cost. The historical baseline used smaller batches and retrieval-order ties; it is not a contemporaneous rerun.

| Diagnostic (macro average across papers) | Value |
|---|---:|
| Original K=64 candidate recall | 0.3943 |
| Complete-vocabulary coverage of ground truth | 1.0000 |
| Ground-truth recall after Jev support decisions | 0.9786 |
| Original candidate oracle RP@5 | 0.5800 |
| Complete-vocabulary oracle RP@5 | 1.0000 |
| Oracle RP@5 among Jev-supported labels | 1.0000 |

Jev marked **39 ground-truth label/document pairs** supported that were absent from the old shortlist. This measures recovered support decisions, not necessarily recovered top-five outputs.

| Paper | Prior RP@5 | Full-scan RP@5 | Supported labels | GT support recall | Supported oracle RP@5 | Labels tied at fifth score |
|---|---:|---:|---:|---:|---:|---:|
| 33811622 | 0.6000 | 0.6000 | 190 | 1.0000 | 1.0000 | 12 |
| 34786403 | 0.5000 | 0.0000 | 384 | 1.0000 | 1.0000 | 50 |
| 34703590 | 1.0000 | 0.0000 | 245 | 1.0000 | 1.0000 | 23 |
| 33619451 | 0.0000 | 0.0000 | 299 | 1.0000 | 1.0000 | 26 |
| 33552603 | 0.4000 | 0.2000 | 641 | 0.8929 | 1.0000 | 70 |

RP@k is the number of matching ground-truth labels in the first k predictions divided by min(k, number of ground-truth labels), then averaged across papers. Support recall shows whether Jev recognizes the annotated labels; the supported-label oracle separates support decisions from final ranking. A large tie at the fifth score means deterministic tie-breaking can affect RP@5.

## Cost and latency

New known API cost: **$0.819949**; **0 unknown-cost calls**. Requests: **475**, including **0 retries**. Recorded process wall time: **253.80s**. No embedding or Gemini API calls were made.

| Stage | API cost USD | Wall time s |
|---|---:|---:|
| Local request packing | $0 | 2.43 |
| Jev classification, including retry waits | $0.819949 | 247.20 |
| Local sorting, copying, and diagnostics | $0 | 0.89 |

Stage times exclude some startup, preflight estimation, and final reporting I/O; process wall time includes that work. Costs use provider-recorded input tokens at $0.042/M; local computation is counted as $0. Unknown failed-request charges are not treated as free. [Official Jev pricing and context limits](https://docs.typesafe.ai/models).

## Per-label audit

| Paper | Ground-truth label | P(supported) | Decision | Rank among supported labels |
|---|---|---:|---|---:|
| 33811622 | ankle fracture | 0.99 | supported | 26 |
| 33811622 | anti-transglutaminase antibody increased | 1.0 | supported | 11 |
| 33811622 | blood alkaline phosphatase increased | 0.98 | supported | 29 |
| 33811622 | blood calcium increased | 0.94 | supported | 53 |
| 33811622 | blood parathyroid hormone increased | 0.96 | supported | 43 |
| 33811622 | cholangitis sclerosing | 0.94 | supported | 52 |
| 33811622 | coeliac disease | 1.0 | supported | 3 |
| 33811622 | death | 0.99 | supported | 13 |
| 33811622 | fall | 0.73 | supported | 126 |
| 33811622 | femur fracture | 0.99 | supported | 14 |
| 33811622 | forearm fracture | 0.98 | supported | 32 |
| 33811622 | fracture | 0.99 | supported | 17 |
| 33811622 | hip fracture | 1.0 | supported | 5 |
| 33811622 | humerus fracture | 1.0 | supported | 10 |
| 33811622 | parathyroid tumour benign | 0.91 | supported | 63 |
| 33811622 | pelvic fracture | 0.99 | supported | 25 |
| 33811622 | rib fracture | 1.0 | supported | 8 |
| 33811622 | spinal fracture | 0.99 | supported | 15 |
| 33811622 | tibia fracture | 1.0 | supported | 4 |
| 33811622 | wrist fracture | 1.0 | supported | 7 |
| 34786403 | adenocarcinoma | 1.0 | supported | 22 |
| 34786403 | adenocarcinoma of colon | 1.0 | supported | 21 |
| 34786403 | metastases to liver | 1.0 | supported | 9 |
| 34786403 | neoplasm progression | 0.98 | supported | 79 |
| 34703590 | subdural haematoma | 1.0 | supported | 14 |
| 33619451 | cardiac failure | 0.99 | supported | 27 |
| 33619451 | respiratory disorder | 0.99 | supported | 29 |
| 33552603 | acute respiratory distress syndrome | 1.0 | supported | 18 |
| 33552603 | appendicitis | 0.99 | supported | 90 |
| 33552603 | bone marrow failure | 0.68 | supported | 473 |
| 33552603 | coagulopathy | 1.0 | supported | 27 |
| 33552603 | delirium | 1.0 | supported | 26 |
| 33552603 | drug ineffective | 0.09 | not_supported | — |
| 33552603 | drug ineffective for unapproved indication | 0.0 | not_supported | — |
| 33552603 | febrile neutropenia | 0.81 | supported | 341 |
| 33552603 | gastrointestinal haemorrhage | 1.0 | supported | 22 |
| 33552603 | haemodynamic instability | 0.97 | supported | 137 |
| 33552603 | haemoglobin decreased | 0.98 | supported | 113 |
| 33552603 | haemophagocytic lymphohistiocytosis | 1.0 | supported | 32 |
| 33552603 | hepatic failure | 0.99 | supported | 76 |
| 33552603 | histoplasmosis disseminated | 1.0 | supported | 42 |
| 33552603 | hyponatraemia | 1.0 | supported | 17 |
| 33552603 | hypotension | 1.0 | supported | 7 |
| 33552603 | lung infiltration | 0.99 | supported | 84 |
| 33552603 | malnutrition | 1.0 | supported | 43 |
| 33552603 | melaena | 0.99 | supported | 82 |
| 33552603 | multiple organ dysfunction syndrome | 0.93 | supported | 209 |
| 33552603 | off label use | 0.01 | not_supported | — |
| 33552603 | pancytopenia | 1.0 | supported | 9 |
| 33552603 | pleural effusion | 1.0 | supported | 11 |
| 33552603 | pulmonary granuloma | 0.77 | supported | 377 |
| 33552603 | rash | 1.0 | supported | 13 |
| 33552603 | renal failure | 1.0 | supported | 16 |
| 33552603 | respiratory failure | 1.0 | supported | 4 |
| 33552603 | sepsis | 1.0 | supported | 8 |

## Interpretation

Ground-truth support recall is **97.9%**, compared with **39.4%** candidate recall for the original shortlist. The scan recovers 39 annotated label/document pairs that the shortlist excluded. This directly tests the benefit of letting Jev see every label.

However, Jev supports **190–641 labels per paper**. Perfect ranking within that supported set could reach RP@5=1.0000, while probability sorting reaches 0.1600. High support recall alone does not establish high precision. Many supported labels are outside the annotations; source review would be needed to distinguish incorrect claims from valid but unannotated findings. The current support probabilities also do not reliably prioritize the benchmark labels, especially when many scores tie.

This experiment removes retrieval exclusions for every official label. It can reveal three remaining limits: Jev rejecting an annotated label, Jev supporting many labels and ranking annotated ones too low, or an annotation not matching the supplied vocabulary. The resulting scores should not be generalized beyond these five papers, and exact annotation matching is not a source-reviewed assessment of every supported medical claim.

[Protocol](runs/full-vocabulary-train5-v1/protocol.json) · [Predictions](runs/full-vocabulary-train5-v1/results.json) · [Metrics](runs/full-vocabulary-train5-v1/summary.json) · [Comparison data](runs/full-vocabulary-train5-v1/comparison.json) · [Runner](full_vocabulary.py)

Each paper folder contains `all_answers.json` for all label probabilities, plus per-batch raw requests/responses, retry logs, cost, timing, and source hashes.
