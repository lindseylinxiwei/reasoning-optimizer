# Company extraction — 11-document experiment

Pipeline: full-document window scanning → organization candidates → Jev subject-company selection → exact source copying. Both runs use MPS, 320 word/punctuation-token windows, 64-token overlap (these are chunking units, not model subword tokens), extraction threshold 0.5, and the same 11 documents. Model checkpoints are pinned. Models are loaded once per run; documents and windows run sequentially. This is one run per method, not a repeated latency benchmark.

## Stage costs and latency

Local model loading, GLiNER extraction and deterministic processing cost $0 by the user's accounting convention. Jev cost is returned input tokens × $0.042/M, output free. No credits/free allowances are subtracted.

| Method | Stage | Monetary cost (USD) | Latency, all 11 documents (s) |
|---|---|---:|---:|
| GLiNER Small + Jev | Model loading (separate) | $0 | 4.727496 |
| GLiNER Small + Jev | Candidate extraction + merging | $0.000000000 | 44.211186 |
| GLiNER Small + Jev | Jev selection | $0.008866410 | 4.639395 |
| GLiNER Small + Jev | Copying + decision validation | $0.000000000 | 0.000239 |
| GLiNER Small + Jev | Pipeline total, excluding load | $0.008866410 | 48.967892 |
| GLiNER2.5 Base + Jev | Model loading (separate) | $0 | 3.617671 |
| GLiNER2.5 Base + Jev | Candidate extraction + merging | $0.000000000 | 29.691711 |
| GLiNER2.5 Base + Jev | Jev selection | $0.008343930 | 3.604643 |
| GLiNER2.5 Base + Jev | Copying + decision validation | $0.000000000 | 0.000227 |
| GLiNER2.5 Base + Jev | Pipeline total, excluding load | $0.008343930 | 33.446506 |

Extraction includes all windows, offset validation, and deduplication; selection includes payload construction, token-budget validation, HTTP and response processing. Copying includes deterministic ID lookup, span validation and slicing. Pipeline wall time also includes saving artifacts and bookkeeping, so it exceeds the sum of stages. First-inference warm-up is included; MPS is synchronized. Errors before a completed stage may have no stage timing; any such missing values are not zero-duration successful work.

## Completion and provisional name matching

| Method | Selected | Abstained | Errors | Candidate count range | Jev calls | Jev input tokens | Name-match proxy |
|---|---:|---:|---:|---|---:|---:|---:|
| GLiNER Small + Jev | 11 | 0 | 0 | 29–168 | 11 | 211,105 | 8/11 |
| GLiNER2.5 Base + Jev | 10 | 0 | 1 | 65–402 | 10 | 198,665 | 7/11 |

The name-match proxy compares each prediction against its own document’s existing GT company name using the MOAR name-similarity function at threshold ≥0.5. This is ID-aligned and differs from the original final-summary evaluator’s search over all GT names. Accepted source-span/alias annotations have not been reviewed, so neither this score nor the candidate-coverage proxy is a validated exact-span extraction accuracy. No model receives GT/company metadata.

## GLiNER Small + Jev

| ID | GT company | Copied name | Candidates | Windows | Extract (s) | Select (s) | Copy (ms) | Jev cost (USD) | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 4920 | Volvo Car Group | Volvo Car Group | 43 | 39 | 3.290498 | 0.415456 | 0.018166 | 0.000544 | selected |
| 8870 | LGIM | LGIM | 64 | 46 | 3.582177 | 0.787825 | 0.023292 | 0.000725 | selected |
| 4779 | Amega | Ammega Group | 29 | 49 | 3.747901 | 0.261622 | 0.017375 | 0.000598 | selected |
| 5962 | Deloite | Deloitte Audit & Assurance | 34 | 51 | 4.840733 | 0.297317 | 0.025000 | 0.000961 | selected |
| 8667 | News Corp | News Corporation | 124 | 59 | 4.581043 | 0.478095 | 0.023667 | 0.000983 | selected |
| 3678 | Nuance | Nuance | 58 | 56 | 4.292491 | 0.308936 | 0.020668 | 0.000855 | selected |
| 5466 | HEINEKEN UK | HEINEKEN UK | 59 | 38 | 2.960489 | 0.257370 | 0.018833 | 0.000606 | selected |
| 4546 | Nuday AS | Nuuday A/S | 52 | 44 | 3.375837 | 0.461482 | 0.018958 | 0.000671 | selected |
| 7523 | Enpro | Enpro Industries, Inc. | 64 | 49 | 3.800968 | 0.309120 | 0.020542 | 0.000792 | selected |
| 8170 | DigitalBridge | DigitalBridge | 71 | 69 | 5.239564 | 0.617900 | 0.019333 | 0.001054 | selected |
| 4922 | Volvo Group | Volvo Group | 168 | 59 | 4.499485 | 0.444273 | 0.032959 | 0.001078 | selected |

Invalid spans excluded after exact source-offset validation: 0. Invalid spans remain in the raw results for auditing; offsets are never repaired by guessing.


[Raw results](runs/sustainability_11_gliner_compact/results.json) · [Call ledger](runs/sustainability_11_gliner_compact/calls.jsonl) · [Run configuration](runs/sustainability_11_gliner_compact/config.json)

## GLiNER2.5 Base + Jev

| ID | GT company | Copied name | Candidates | Windows | Extract (s) | Select (s) | Copy (ms) | Jev cost (USD) | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 4920 | Volvo Car Group | Volvo Car Group | 150 | 39 | 2.112805 | 0.353122 | 0.018750 | 0.000669 | selected |
| 8870 | LGIM | LGIM | 93 | 46 | 2.324397 | 0.261400 | 0.024167 | 0.000756 | selected |
| 4779 | Amega | Ammega Group | 65 | 49 | 2.329466 | 0.272302 | 0.017292 | 0.000638 | selected |
| 5962 | Deloite | Deloitte Audit & Assurance | 90 | 51 | 4.435833 | 0.297210 | 0.017959 | 0.001024 | selected |
| 8667 | News Corp | News Corporation | 199 | 59 | 2.916686 | 0.488368 | 0.028208 | 0.001068 | selected |
| 3678 | Nuance | Nuance | 104 | 56 | 2.766476 | 0.761585 | 0.023625 | 0.000908 | selected |
| 5466 | HEINEKEN UK | HEINEKEN UK | 107 | 38 | 1.890391 | 0.267343 | 0.025000 | 0.000659 | selected |
| 4546 | Nuday AS | Nuuday A/S | 78 | 44 | 2.271294 | 0.271186 | 0.030833 | 0.000700 | selected |
| 7523 | Enpro | Enpro Industries | 85 | 49 | 2.365047 | 0.292070 | 0.021583 | 0.000814 | selected |
| 8170 | DigitalBridge | DigitalBridge | 120 | 69 | 3.296838 | 0.340055 | 0.019624 | 0.001109 | selected |
| 4922 | Volvo Group | — | 402 | 59 | 2.982478 | 0.000004 | — | 0.000000 | error |

Document 4922 error: Jev allows 255 choices including NONE; candidates were not pruned

Invalid spans excluded after exact source-offset validation: 1. Invalid spans remain in the raw results for auditing; offsets are never repaired by guessing.


[Raw results](runs/sustainability_11_gliner25_validated/results.json) · [Call ledger](runs/sustainability_11_gliner25_validated/calls.jsonl) · [Run configuration](runs/sustainability_11_gliner25_validated/config.json)

## Scope

This experiment measures the decomposed extraction pipeline only. It does not yet include a direct Gemini company-name extraction baseline, so no cost/latency savings claim against generation follows from these runs. Earlier Gemini reduce costs belong to a different task. Local compute is treated as $0 by convention, not estimated hardware or electricity cost.

## Preliminary runs and total spend

An initial GLiNER Small run included every candidate occurrence offset in the Jev state and estimated token length from serialized HTTP JSON. Five documents exceeded the conservative 24K estimate before any API request; six were selected. This was a request-format/preflight issue, not a 255-choice overflow or a provider-reported 32K overflow.

The final runs above keep all source offsets locally, send the full unmodified document plus every candidate ID/name, and estimate actual state/question text rather than escaped transport bytes. Both final runs use this same format, with the 24K guard retained and no content/candidate pruning.

Preliminary run: Jev cost $0.005078010; pipeline wall time 49.104048s; extraction 46.066071s, selection/preflight 2.959989s, copying 0.000106s; loading 5.444471s separately. Local monetary costs are $0.

**Total Jev spend for this experiment session, including both preliminary runs: $0.029608404.** Earlier synthetic smoke tests are excluded. The original preliminary artifacts are preserved under `runs/sustainability_11_gliner/`.

The preliminary GLiNER2.5 run cost $0.007320054 across nine Jev calls, with pipeline wall time 33.325821s and loading 3.732798s. It stopped one document at an invalid extracted span and another at the candidate limit. Its extraction stage total omitted the failed extraction's time, so use its pipeline wall time for accounting. The final GLiNER2.5 run records and excludes invalid spans, keeping all valid candidates. The local diagnostic between runs cost $0; its latency was not instrumented and is outside the benchmark. Preliminary artifacts: `runs/sustainability_11_gliner25_compact/`.

## Direct Gemini baseline follow-up

See [Gemini extraction comparison](gemini_extraction_comparison.md) for the same-document direct extraction baseline, costs, latency, provisional name matching and raw artifacts.
