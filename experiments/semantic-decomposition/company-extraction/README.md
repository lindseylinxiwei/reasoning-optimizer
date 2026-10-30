# Company-name extraction pipeline

The first implementation is **candidate extraction → Jev selection → exact source copying**.
It is an experiment-level runner, not yet a new DocETL YAML operator. It does not
perform summarization or change the earlier filter/map experiments.

## Code

- `pipeline.py`: overlapping windows, offset validation/deduplication, two local
  extractor adapters, Jev selector, and deterministic copying. `extract_candidates`
  accepts any callable `extractor(text) -> spans`; `select_and_copy` accepts any
  callable `selector(text, candidates) -> decision`, so either backend can be replaced.
- `run_extraction.py`: JSON input/output, configuration, timings, call ledger, and errors.
- `test_pipeline.py`: offline boundary, overlap, Unicode, copying, and abstention tests.

## Behavior

1. Scan the entire document using 320 word/punctuation tokens per window with
   64-token overlap (configurable; runner caps windows at 384 for both models).
   Windows retain original characters; no normalization or truncation of the document.
2. Extract `organization` spans with GLiNER Small or GLiNER2.5, threshold 0.5.
   Validate offsets, deduplicate overlapping occurrences, and group identical surface
   strings. Keep every distinct spelling; do not merge aliases or consult GT labels.
3. Give Jev the complete original document and candidate IDs/names; offsets stay local.
   Up to 254 candidates use the original single question. Larger sets are partitioned
   into batches of 254 + `NONE`, asked as independent questions sharing one state.
   Discard `NONE` answers and repeat selection among winners until at most one remains.
   No survivors means abstention; one survivor is returned directly. Stable IDs and
   input order are retained. First-round mistakes can eliminate the correct candidate.
   Questions are packed into requests using conservative cl100k_base estimates:
   state + longest question <=24K, state + all questions <=48K. The provider's limits
   are 32K / 64K; this estimate is not Jev's tokenizer. Split requests when necessary;
   an oversized individual question raises an error. No text or candidates are pruned.
4. Copy the first occurrence of the chosen candidate using `text[start:end]`.
   Empty candidates or `NONE` produce an abstention with a null company name.
   HTTP failures and invalid answers are errors, not abstentions.

This selects the report's subject company, not an arbitrary organization mention.
NER scores describe candidate detection, not subject selection. Jev's raw answer
(including any returned probability/confidence fields) is retained separately.

## Run

The existing isolated runtime is `/private/tmp/gliner-smoke-venv`; the model cache
is `/private/tmp/gliner-smoke-hf`. Root `.env` supplies `TYPESAFE_API_KEY`; this runner
uses the official endpoint directly. Temporary runtimes may need recreating.
`requirements-pipeline.lock` records installed dependencies; downloaded checkpoint
revisions are pinned in `downloaded_revisions.json`.

Input is a JSON list with unique `id` and `tot_text_raw` fields (override the text
field with `--text-field`). Other columns, including company names and GT labels,
are never passed to either model. Output directories must be new.

```sh
# Synthetic end-to-end example (one paid Jev request):
TIKTOKEN_CACHE_DIR=/tmp/jev-tiktoken /private/tmp/gliner-smoke-venv/bin/python \
  experiments/semantic-decomposition/company-extraction/run_extraction.py \
  --input experiments/semantic-decomposition/company-extraction/smoke_input.json \
  --output-dir experiments/semantic-decomposition/company-extraction/runs/example \
  --extractor gliner25 --device mps
```

Use `--extractor gliner` for GLiNER Small. Add `--extract-only` to save candidates
without any Jev calls or API key. Use `--device cpu` if desired. Neither formal
11-document evaluation nor source-span annotations are automatically run.

## Saved artifacts and measurement

- `input.json`: text and IDs only; frozen snapshot.
- `config.json`: model/revision, settings, runtime versions, load time, pricing basis.
- `candidates.jsonl`: source spans, NER scores, candidate IDs, window counts and extraction timings;
  preserved even if selection subsequently fails.
- `calls.jsonl`: exact Jev requests, round/request numbers, raw responses, usage, API costs, errors and call timings. Selection cost and latency accumulate across all rounds/requests; partial failures retain billed calls in the ledger.
- `results.json`: copied name or explicit abstention/error, selected source span, decisions and timings.
- `metrics.json`: model load time separated from pipeline wall time, API usage cost and unknown-cost count.

Models load once per run. Timed extraction includes window scanning and merging;
MPS is synchronized. First inference warm-up remains in pipeline time. Jev calls
have no cache or automatic retries. At the recorded rate, input costs $0.042/M;
output is free. For the 11-document experiment, local extraction/loading/copying monetary cost is **$0 by user convention**. Their latency is measured; the convention is not a hardware/electricity cost estimate.

## Verification

```sh
/private/tmp/gliner-smoke-venv/bin/python -m unittest discover \
  -s experiments/semantic-decomposition/company-extraction -p 'test*.py' -v
```

Synthetic live checks are under `runs/`. They verify wiring and source copying,
not extraction accuracy on the sustainability dataset. The annotation sheet and
formal experiment remain separate work.

---

# Company extraction: local smoke tests

Both models ran successfully on CPU and Apple MPS (M5 Pro, 48 GB). This was a synthetic-text compatibility test only. No sustainability documents, subject-company selection, or paid model APIs were used.

| Model | Device | First inference (s) | Subsequent inferences (s) | Source offsets |
|---|---|---:|---|---|
| urchade/gliner_small-v2.1 | cpu | 0.0477 | 0.0287, 0.0273 | Verified |
| urchade/gliner_small-v2.1 | mps | 1.4766 | 0.0209, 0.0153 | Verified |
| fastino/gliner2.5-base-v1 | cpu | 0.0514 | 0.0316, 0.0308 | Verified |
| fastino/gliner2.5-base-v1 | mps | 1.4123 | 0.0343, 0.0114 | Verified |

Both detected Apple (two mentions) and Microsoft, with exact half-open character offsets. These confidence scores concern organization detection, not identification of the report’s subject.

## Reproduction

Environment: `/private/tmp/gliner-smoke-venv`; model cache: `/private/tmp/gliner-smoke-hf`. These temporary paths may be cleared by the OS; installed versions are recorded in `requirements-smoke.lock` and checkpoint revisions in `downloaded_revisions.json`.

From the repository root:

```sh
HF_HOME=/private/tmp/gliner-smoke-hf TOKENIZERS_PARALLELISM=false \
  /private/tmp/gliner-smoke-venv/bin/python \
  experiments/semantic-decomposition/company-extraction/smoke_models.py \
  --model gliner \
  --output experiments/semantic-decomposition/company-extraction/smoke_gliner.json
```

Use `--model gliner25` and `smoke_gliner25.json` for GLiNER2.5.

## Notes

- Both CPU and MPS use float32; CPU threads set to 4. MPS timings explicitly synchronize the device. Three inferences per device are only a functional check, not a latency benchmark.
- Loading/download time is recorded separately in JSON. GPU first-use initialization makes the first MPS inference slower.
- Added `protobuf` to satisfy tokenizer dependencies. GLiNER2.5 applies its built-in compatibility normalization to legacy tokenizer metadata and falls back to eager attention when SDPA is unsupported. No model weights or library source were patched.
- The loaded GLiNER Small checkpoint has 152,648,704 parameters by `sum(p.numel())`, although its model card lists 166M. GLiNER2.5 has 193,581,591. The measured counts are recorded for transparency.
- Full-document extraction, windowing and selector decisions are implemented below. Source-span annotation review and repeated quality/latency experiments remain pending.

## 11-document run

See `extraction_experiment.md` for stage costs/timings and provisional ID-aligned name matching. Rebuild it without inference using `report_extraction.py` in the earlier `/private/tmp/jev-comparison-venv` environment (which has the existing MOAR evaluator dependencies). Both preliminary runs are retained and their API charges are included separately. Invalid extracted spans are recorded and excluded after exact source-offset validation; valid candidates are preserved. The historical runs predate hierarchical selection and retain their original overflow errors. New runs use the batching method above.

## Direct Gemini extraction baseline

`run_gemini_extraction.py` runs one full-document request per record using Gemini 3.1 Flash-Lite with minimal thinking. It accepts `--input` (the frozen id/text snapshot) and a new `--output-dir`. `report_gemini_extraction.py` rebuilds the saved comparison without inference, using the earlier evaluation environment. See [Gemini comparison](gemini_extraction_comparison.md). The cheaper published 2.5 Flash-Lite model returned 404 for this key; those failed attempts are preserved separately.

## Window batching

For GLiNER Small, use `--extractor gliner --device mps --batch-size 16` to batch windows within each document. Default batch size remains 1 for reproducibility; GLiNER2.5 batching is not implemented. Full text, offsets, threshold and overlap are unchanged. `benchmark_windows.py` runs two counter-ordered full-dataset trials each at batch sizes 1, 8, 16 and 32 without API calls. `report_batch_benchmark.py` appends the benchmark and fresh end-to-end measurements to `gemini_extraction_comparison.md`. Rebuilding the base Gemini report should be followed by rebuilding this batching section.

## Overlapping extraction and selection

Add `--overlap-api` to overlap extraction of the next document with one Jev selection worker. GPU extraction stays on the caller thread; only one selection runs at a time, and outstanding work is bounded. Results remain in input order. Per-document latency and selection queue time are saved; stage sums are not additive under overlap. `run_overlap_trials.py` runs the four controlled trials in new directories; `report_api_overlap.py` appends their results without inference. Rebuild reports in order: Gemini, batching, overlap.

## Long-document evidence experiment

`evidence_selector.py` implements bounded verbatim evidence retrieval, token-budgeted candidate batches, winner selection, and one NONE-triggered expansion pass. `run_evidence_experiment.py` compares the same cached candidates on the 11-document pilot and then scans all 100 held-out test reports with a frozen policy. No document-length filtering or generated summaries are used. `report_long_evidence.py` evaluates saved outputs and writes the separate `long_document_evidence_report.md`; its accuracy figures remain approximate name-match proxies. Raw state/questions and evidence offsets are saved in the API ledger.

For the held-out direct baseline, run `run_gemini_extraction.py --input runs/long_document_evidence_v1/test_input.json --output-dir runs/long_document_evidence_v1/test_gemini` from this directory after the evidence run completes, using the existing Python environment. Then run `report_long_evidence.py` in the evaluation environment with `TIKTOKEN_CACHE_DIR=/tmp/jev-tiktoken`. The pilot full-document transport failure was retried separately; both attempts remain recorded.
