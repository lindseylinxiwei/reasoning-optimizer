# Biodex shared-retrieval experiments

Three variants share the same cached candidates and evidence:
1. Embedding-only: rank labels by retrieval similarity.
2. Jev: classify each label as supported/not supported; rank supported labels by P(supported).
3. Gemini 3.1 Flash-Lite: select and rank supported candidate IDs.

## Retrieval

Embed the official vocabulary once with `text-embedding-3-small`, the embedding
model used in MOAR. Split papers into 256 cl100k-token passages with 64-token overlap.
A label's score is its maximum cosine similarity to any passage. Save the top 128;
each K experiment uses the same prefix. Ties follow the original vocabulary order.
No upstream LLM extraction. This is a new label retriever using MOAR's model, not a
reproduction of an existing MOAR label-retrieval plan.

Use full paper text below 22K estimated tokens. Otherwise select candidate-associated
passages first, then other passages within the budget, assembling in source order.
Evidence is frozen at preparation and identical across methods and K settings.

Jev uses shared-state binary questions, up to 32 per request plus existing token
guards. Multiple labels can be supported; no winner tournament. Rank by P(supported),
not winning-choice confidence. Scores across questions are an uncalibrated heuristic.
Gemini returns ranked IDs in one structured call. Both classifiers can abstain;
embedding-only always returns its top labels. Code copies exact vocabulary names.
Default output limit: 10. This compares an added selection/ranking stage, not identical
model output parameterizations. The existing Biodex evaluator is unchanged.

## Run from the repository root

Dependencies: numpy, requests, python-dotenv, tiktoken (existing experiment environment).
Repository `.env`: OPENAI_API_KEY for uncached embeddings, TYPESAFE_API_KEY for Jev,
GEMINI_API_KEY for Gemini. Keys are never written to output artifacts.

```bash
python experiments/semantic-decomposition/biodex/run.py prepare \
  --input experiments/reasoning/data/train/biodex.json \
  --output experiments/semantic-decomposition/biodex/runs/train-prepared

python experiments/semantic-decomposition/biodex/run.py run \
  --prepared experiments/semantic-decomposition/biodex/runs/train-prepared \
  --output experiments/semantic-decomposition/biodex/runs/train-k64 \
  --k 64 --method all
```

Repeat run with K=32 and 128 and new output directories. `--method embedding`, `jev`,
or `gemini` runs one variant. Select K on training data using recall, RP@5 and cost;
freeze before test. K selection is not automatic. Cache is reusable across splits:
it stores only embeddings keyed by model and exact text, never ground truth.

Output directories must be new. Failed preparation preserves partial.json and cached
vectors; rerun into a new directory with the same cache. No automatic retries or
silent classifier fallbacks. Existing experiments and engine code are unchanged.

## Artifacts

- prepared.json: source hash, candidates/scores, original passage offsets, shared
  evidence and hashes. No ground truth.
- embedding_calls.jsonl and retrieval_metrics.json: shared API usage, cost, time, cache hits.
- Each method: results.json, metrics.json, calls.jsonl for API calls.
- summary.json and report.md: candidate label recall, candidate oracle RP@5, MOAR
  RP@5/RP@10/term recall, errors, monetary cost, and latency.

Ground truth is attached only to results for evaluation, never retrieval/model inputs.
Invalid, duplicate, or out-of-vocabulary outputs are errors, not silently repaired.
Error predictions are empty and count as misses on labeled documents.

Shared preparation and incremental ranking costs/times are reported separately.
Standalone totals add shared work once per method; combined times are composed, not
fresh end-to-end measurements. Warm-cache preparation excludes historical embedding
creation charges. Unknown API costs remain unknown. API ledgers record HTTP calls;
semantic validation failures are recorded in document results.

Explicit pricing assumptions, USD per million tokens: embeddings .02 input; Jev .042
input (existing client); Gemini .25 input, .025 cached input, 1.50 output including
thinking. These are not verified invoices or a current-price guarantee. Local compute
is $0 by convention. The report records the rates used.

## Code and checks

core.py contains reusable retrieval, cache, and ranker adapters. run.py handles
prepare/run orchestration and reports. It reuses company-extraction/pipeline.py's
Jev transport and experiments/reasoning/evaluation/biodex.py without modifying them.

```bash
python -m unittest discover -s experiments/semantic-decomposition/biodex -v
```

Tests are offline and include all three methods with mocked model calls. Completed experiment artifacts are described in the reports below.

## Stronger Gemini follow-up

The same runner also supports `--gemini-model gemini-3.1-pro-preview --thinking high
--max-output-tokens 32768`, with Pro pricing ($2/M input, $0.20/M cached input,
$12/M output including thinking). Use `--method gemini --k 64` and a fresh output
directory. Existing Flash-Lite defaults remain unchanged.

The completed 100-document Pro/high follow-up is in `runs/test-k64-pro-high/`.
Run `python report_biodex.py` to regenerate the full `biodex_jev.md`, including the
Pro comparison when its artifacts exist. All ranking results reuse the same cached
candidates and evidence.


## Evidence selection and hierarchical search

`workflows.py` compares three Jev workflows with the same final label verifier:

- `evidence_retrieval`: classify every cached source passage, retain reported or
  uncertain patient findings, retrieve 64 labels from those passages, then verify.
- `full_hierarchy`: navigate the label search tree using the complete paper, then
  verify every label in reached leaves.
- `evidence_hierarchy`: navigate the same tree using selected verbatim evidence,
  then verify every label in reached leaves.

`hierarchy.py` builds a deterministic tree from the cached official-label vectors.
It is an embedding-cluster search tree, not a medical ontology. Each leaf contains
at most 96 labels, branching is at most 12, and internal nodes use 16 diverse
representative labels. Independent supported/uncertain/none decisions permit
multiple paths. A global beam of six paths includes finished leaves; path scores
are geometric means of P(supported) + 0.5 P(uncertain), a ranking heuristic.
There is no dense top-K filter after traversal. Code copies exact official labels.

```bash
python experiments/semantic-decomposition/biodex/workflows.py \
  --prepared experiments/semantic-decomposition/biodex/runs/train-prepared \
  --output experiments/semantic-decomposition/biodex/runs/workflows-train-v1
```

Use `--methods evidence_retrieval`, `full_hierarchy`, or `evidence_hierarchy` to run
one method. The default runs all three. `--beam` and `--limit` are explicit controls.
The embedding cache must already contain label and passage vectors; this runner
never makes new embedding calls. Complete documents are required in the prepared
artifact. Ground truth enters only evaluation. Evidence offsets are validated.

Outputs include a frozen protocol, per-document evidence/search/verification
artifacts and API ledgers, stage costs and times, outputs, and aggregate metrics.
Completed artifacts resume without new calls. Interrupted incomplete stages are
recorded as failures rather than silently retried; errors count as empty outputs.
Resume refuses a changed protocol, source, implementation, or hierarchy.

All monetary costs use recorded API tokens at $0.042/M Jev input; local work is $0.
Standalone method totals include shared evidence extraction once each; total session
spend counts that shared execution once. Cached embeddings exclude historical API
creation costs. Stage wall times are composed, not fresh end-to-end reruns. Consult
`biodex_jev_workflows.md` for measured results and accounting limitations.


## Full-vocabulary Jev pilot

`full_vocabulary.py` evaluates every official label against the complete document,
without retrieval, evidence filtering, or hierarchy routing. It shares the exact
binary support question builder with the earlier Jev ranker. Independent questions
are packed into requests of at most 256 questions within the existing token guards;
multiple labels can be supported. Code returns up to 10 supported labels sorted by
P(supported), breaking ties by official vocabulary order.

```bash
python experiments/semantic-decomposition/biodex/full_vocabulary.py \
  --output experiments/semantic-decomposition/biodex/runs/full-vocabulary-train5-v1 \
  --limit 5
python experiments/semantic-decomposition/biodex/report_full_vocabulary.py
```

`--plan-only` validates complete label coverage and estimates input cost without
calling the API or writing run artifacts. The default selects the first five train
papers in source order. `--batch-size` and `--max-attempts` are explicit controls.
The run freezes source/question/implementation hashes before calls. Each successful
batch is checkpointed; resume reuses saved answers. Transient HTTP/network failures
allow up to three total attempts by default, with all failed charges retained as
known or unknown. Non-transient failures are not retried.

Saved artifacts include all 24,313 label decisions per paper, exact requests and
responses, per-attempt cost and latency, final predictions, GT label probabilities
and ranks, support recall, and the supported-label RP@5 ceiling. Evaluation uses the
unchanged MOAR metric. Ground truth is never passed to Jev. The report compares the
same document IDs with the earlier K=64 baseline and an offline restriction of the
new answers to those 64 labels. No new embedding or Gemini calls are used.

Results: `biodex_jev_full_vocabulary.md`. This small pilot diagnoses access,
classification, and ranking separately; it is not a held-out 100-document result.


## Finer evidence and iterative ranking pilot

`iterative_evidence.py` reuses the completed five-paper full-vocabulary scan and
compares iterative label selection with full papers versus selected source evidence.
It uses the existing `nltk` dependency (Punkt tokenizer with local abbreviation
rules; no model/data download). Only `TYPESAFE_API_KEY` is needed; no embedding or Gemini calls.

Every source sentence/heading or delimited table row receives a Jev relevance
decision using surrounding units. Code copies selected source spans and associated
headers exactly. Ranking repeatedly selects the best remaining label. Groups hold
253 labels, leaving slots for NONE and MORE_CONTEXT. Larger pools first retain up
to ten labels per group, then rank the combined survivors. Independent questions
share requests, and MORE_CONTEXT switches that group to the original paper.

```bash
python experiments/semantic-decomposition/biodex/iterative_evidence.py \
  --output experiments/semantic-decomposition/biodex/runs/iterative-evidence-train5-v1
python experiments/semantic-decomposition/biodex/report_iterative_evidence.py
```

Successful calls and stages resume from checkpoints; changed protocols are rejected.
Request retries reuse the exhaustive-scan transport and retain all charges.
Nonrecoverable errors stop the pilot rather than producing a partial aggregate.
Saved files contain every evidence decision, exact offsets, group/final ranking
rounds, context fallbacks, raw responses, stage timing and cost, and MOAR evaluation.
Ground truth enters only evaluation. See `biodex_jev_iterative_evidence.md`.


## Select evidence before the exhaustive scan

`evidence_first.py` moves the same evidence-selection stage before all label
decisions. It imports the existing evidence selector, source copying, scan questions,
Jev transport, checkpointing, iterative ranker, and MOAR evaluator. The scan sees
only copied excerpts. Ranking can request the full paper, but cannot recover
labels rejected by the scan. No embedding or Gemini calls are made.

```bash
python experiments/semantic-decomposition/biodex/evidence_first.py \
  --evidence-run experiments/semantic-decomposition/biodex/runs/iterative-evidence-train5-v1 \
  --output experiments/semantic-decomposition/biodex/runs/evidence-first-train5-v1
python experiments/semantic-decomposition/biodex/report_evidence_first.py
```

`--input` and `--limit` select the dataset and count (default: first five training
papers). Omit `--evidence-run` to perform new evidence selection. With reuse, source
hashes and exact copied spans must match; the historical evidence cost is included
in standalone totals but excluded from new spend. New runs use separate output
folders. Successful requests resume from checkpoints; all retries are logged.

The protocol freezes input, dependency, prompt and reused-evidence hashes. Results
include scan-supported GT recall, its RP@5 ceiling, final predictions, costs and
latencies by stage, and context-fallback counts. `biodex_jev_evidence_first.md`
compares the five-paper run with the saved full-paper-scan variants.


## Full-paper iterative selection on the complete test set

`fullpaper.py` runs only the full-paper variant: every label receives an independent
binary support question, followed by the unchanged iterative ranking over supported
labels. It reuses `evidence_first.scan_labels` with the complete paper, the existing
Jev request checkpoints, and `iterative_evidence.rank_labels` with no context fallback.
No evidence selection, retrieval, or Gemini is involved. Every input document is used.

```bash
python experiments/semantic-decomposition/biodex/fullpaper.py \
  --input experiments/reasoning/data/test/biodex.json \
  --output experiments/semantic-decomposition/biodex/runs/fullpaper-test100-v1 --workers 8
python experiments/semantic-decomposition/biodex/report_fullpaper.py
```

`--plan-only` validates the full-paper context guards and estimates scan calls/cost
without API calls or output writes. `--workers` controls concurrency across documents
(default 8); each document owns its own client and request ledgers. The scan phase
finishes before ranking starts, giving separately measured batch wall times. Prompts
and per-document decision dependencies match the pilot. Ground truth enters only
post-run evaluation. The protocol freezes source and implementation hashes.

The report distinguishes actual process/phase wall time, per-document service time,
and the sum of overlapping document times. API costs include all scan and ranking
calls and retries; unknown charges remain explicit. Unresolved phase errors halt
the experiment instead of silently excluding documents. Successful stages and
requests resume without repeat calls; use a fresh output folder to measure a new
end-to-end run. Results: `biodex_jev_fullpaper_test100.md`.
