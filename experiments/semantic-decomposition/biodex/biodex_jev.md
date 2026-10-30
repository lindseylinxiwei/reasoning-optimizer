# Biodex: embedding retrieval + Jev

At the training-selected K=64, test RP@5 is **0.1950 for Jev**, **0.1867 for Gemini**, and **0.0938 for embedding-only**. Including shared test retrieval, Jev costs **$0.094165** and takes **127.57s**, versus **$0.181860** and **197.27s** for Gemini.

Jev uses 48.2% less API cost and 35.3% less composed wall time in this run. The quality difference is small and is not evidence of statistical superiority. **Retrieval is the main quality bottleneck:** candidate recall is only 21.0%, and even perfect ranking of these candidates is capped at RP@5=0.233. This experiment does not establish high absolute extraction quality.

We compare embedding-only ranking, retrieval + Jev, and retrieval + Gemini on the same candidate labels and document evidence. Each paper produces up to 10 unique official reaction labels. The primary metric is the existing MOAR RP@5.

## Method

We embed the official vocabulary and overlapping paper passages using OpenAI `text-embedding-3-small`, the embedding model used in MOAR. Passages contain 256 cl100k tokens with 64-token overlap. A label is scored by its maximum cosine similarity to any passage; the top K labels become candidates. This is our label-retrieval method, not an existing MOAR retrieval plan. There is no upstream LLM extraction.

- **Embedding only:** return labels in retrieval order.
- **Jev:** ask a supported/not-supported question for each candidate, sharing the paper state across questions. Keep supported labels and sort by P(supported); copy exact official names.
- **Gemini 3.1 Flash-Lite:** select and rank supported candidate IDs, then copy exact official names.

All 40 training papers and 100 test papers fit the shared full-document evidence budget; no paper was truncated. The saved passage offsets were validated against source text. Both classifiers may abstain; embedding-only returns its top labels. Jev probabilities across questions are an uncalibrated ranking heuristic. Gemini returns a ranked list directly, so this compares the two selection/ranking stages rather than identical model output parameterizations.

## Training selection (40 documents)

The rule was frozen before results: choose the highest mean training RP@5 across Jev and Gemini, breaking ties toward smaller K. Training and test IDs are disjoint. All K values reuse embeddings; each ranker uses identical candidate prefixes and frozen evidence.

| K | Candidate recall | Candidate oracle RP@5 | Embedding RP@5 | Jev RP@5 | Gemini RP@5 | Jev ranking USD / s | Gemini ranking USD / s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.2120 | 0.2575 | 0.1683 | 0.2267 | 0.2425 | $0.016582 / 9.25 | $0.061609 / 45.08 |
| 64 | 0.2402 | 0.2913 | 0.1683 | 0.2379 | 0.2429 | $0.033157 / 21.53 | $0.069450 / 45.60 |
| 128 | 0.2730 | 0.3321 | 0.1683 | 0.2429 | 0.2129 | $0.066304 / 42.05 | $0.082126 / 48.33 |

**Frozen test setting: K=64.** No test-set tuning.

Gemini had one HTTP 503 failure at K=128 (PMID 33573668); it counts as a miss and its charge is unknown. No retry was made. Even assigning that document a perfect score would only tie the selected K=64 mean; the smaller-K tie rule gives the same decision. All other training calls completed successfully.

## Held-out test (100 documents)

| Method | MOAR RP@5 | MOAR RP@10 | Errors | Ranking API cost (USD) | Ranking wall time (s) | Shared retrieval + ranking USD | Shared retrieval + ranking s |
|---|---:|---:|---:|---:|---:|---:|---:|
| embedding | 0.0938 | 0.1288 | 0 | $0.000000 | 0.08 | $0.012549 | 75.36 |
| jev | 0.1950 | 0.1973 | 0 | $0.081616 | 52.29 | $0.094165 | 127.57 |
| gemini | 0.1867 | 0.1986 | 0 | $0.169311 | 121.99 | $0.181860 | 197.27 |

Shared test retrieval cost: **$0.012549**; wall time: **75.28s**. This warm-cache run reuses the official-vocabulary embeddings built during training.
Candidate recall: **0.2096**. Candidate oracle RP@5: **0.2330**. These expose how much quality is lost before classification.

## Evaluation and interpretation

RP@5 is the number of correct unique labels in the first five predictions divided by min(5, number of ground-truth labels), then averaged across labeled papers. It does not distinguish ordering within the top five. Candidate recall measures the fraction of GT labels retrieved per paper; oracle RP@5 assumes perfect selection from the candidate set. No source-review corrections or alias adjudication are applied.

Jev changes RP@5 by +0.1012 versus embedding-only, and +0.0083 versus Gemini. These are observed results from one run, not a statistical equivalence claim.

## Accounting and artifacts

Costs use recorded token usage and explicit experiment rates: embeddings $0.02/M input; Jev $0.042/M input; Gemini $0.25/M input, $0.025/M cached input, $1.50/M output including thinking. These are token-equivalent estimates, not verified invoices. Local compute is counted as $0. Shared + ranking times are composed from one shared preparation plus each sequential ranker, not fresh end-to-end reruns.

Total known API cost for training selection plus test: **$0.600516**, with **1 calls with unknown cost**. Shared retrieval is counted once in this session total.
Training vocabulary/preparation setup: $0.007812, 387.95s. This setup is excluded from warm-cache test totals.

[Selection protocol](runs/protocol.json) · [Frozen K](runs/selected_k.json) · [Implementation](README.md)
[Test summary](runs/test-k64/summary.json) · [Shared inputs](runs/test-prepared/prepared.json)

Per-document outputs, raw API responses, request usage and latency are saved under each run directory. Errors are retained as empty predictions and count as misses; unknown costs are not assumed free.

## Follow-up: Gemini 3.1 Pro with high thinking

We ran `gemini-3.1-pro-preview` with **high thinking**, temperature 0, and a 32,768-token generation cap on the same 100 held-out papers. K remains 64, selected before this follow-up. Candidate IDs, paper evidence, task prompt and final output limit (10 labels) are unchanged. The output cap is higher than Flash-Lite’s 2,048 to allow reasoning; this tests the Pro/high configuration, not model identity alone.

Gemini 3.1 Pro is the current Pro model verified in the account’s model catalog. Newer Flash-family versions also exist; “Pro” is the selected flagship reasoning baseline, not a claim that it is best on every workload. [Model documentation](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview).

| Method | RP@5 | RP@10 | Errors | Ranking cost USD | Ranking time s | Including shared retrieval USD | Including shared retrieval s |
|---|---:|---:|---:|---:|---:|---:|---:|
| embedding | 0.0938 | 0.1288 | 0 | $0.000000 | 0.08 | $0.012549 | 75.36 |
| Jev | 0.1950 | 0.1973 | 0 | $0.081616 | 52.29 | $0.094165 | 127.57 |
| Gemini 3.1 Flash-Lite / minimal | 0.1867 | 0.1986 | 0 | $0.169311 | 121.99 | $0.181860 | 197.27 |
| Gemini 3.1 Pro / high | 0.2000 | 0.1989 | 0 | $3.574138 | 1549.54 | $3.586687 | 1624.82 |

Pro/high RP@5 differs from Flash-Lite by **+0.0133**, and from Jev by **+0.0050**. The shared candidate oracle remains **0.2330**; a stronger model cannot recover missing labels. This is one run per configuration, not a significance test.

New follow-up API cost: **$3.574138**, plus **0 unknown-cost calls**. Retrieval was reused, so its $0.012549 / 75.28s is included only in hypothetical standalone totals, not charged again to this run. Reported API latency is sequential wall time. Errors remain empty predictions and count as misses; no automatic retries or model fallback.
Reported token totals: input **624,563**, visible output **9,120**, thinking **184,631**. Missing usage is not assumed zero for cost accounting.
Pro prices used: $2/M input, $0.20/M cached input, $12/M output including thinking for prompts ≤200K tokens. All requests are below that threshold. [Official pricing](https://ai.google.dev/gemini-api/docs/pricing).

[Follow-up summary](runs/test-k64-pro-high/summary.json) · [Predictions](runs/test-k64-pro-high/gemini/results.json) · [API ledger](runs/test-k64-pro-high/gemini/calls.jsonl) · [Configuration](runs/test-k64-pro-high/config.json)
