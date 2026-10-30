# Direct Gemini vs. candidate extraction + Jev

All methods receive the same 11 full documents. Gemini receives no extracted candidates or ground-truth metadata and directly returns a company name (or null) in JSON. Its prompt requests the complete subject-company name explicitly present in the source. Gemini uses temperature 0, minimal thinking, at most 256 output tokens, sequential standard requests, no retries or explicit caching. GLiNER costs $0 by convention. This is one run per method; latency is descriptive, not a statistically established speed difference.

## Overall comparison

| Method | Name-match proxy (all 11) | Successful outputs | API cost (USD) | Pipeline wall time, excluding model load (s) |
|---|---:|---:|---:|---:|
| GLiNER Small + Jev | 8/11 (72.7%) | 11/11 | $0.008866410 | 48.968 |
| GLiNER2.5 Base + Jev | 7/11 (63.6%) | 10/11 | $0.008343930 | 33.447 |
| Gemini 3.1 Flash-Lite, direct | 7/11 (63.6%) | 11/11 | $0.044546500 | 12.744 |

The GLiNER2.5 row is the saved pre-batching run: Volvo Group failed at 402 candidates and is counted as incorrect. The new hierarchical selector has not been rerun here. Its lower total cost therefore does not represent equal completion. Small + Jev completed all 11 and is the primary complete-run comparison. GLiNER model loading was 4.727s (Small) and 3.618s (Base), separately.

Accuracy uses the existing MOAR name-similarity function, threshold >=0.5, aligned by document ID. This is a provisional name-match proxy, not reviewed semantic correctness: GT aliases and suspected spelling errors remain unresolved (e.g., Amega, Deloite). Neither model sees GT. Exact source-substring checks are reported separately and do not establish that the correct company was selected.

## Gemini per-document results

| ID | GT | Gemini output | Match proxy | Exact substring | Cost (USD) | Latency (s) |
|---|---|---|---|---|---:|---:|
| 4920 | Volvo Car Group | Volvo Car AB (publ.) | False | True | $0.002813500 | 0.881 |
| 8870 | LGIM | Legal & General Investment Management | False | True | $0.003695000 | 0.852 |
| 4779 | Amega | Ammega | True | True | $0.003139250 | 1.319 |
| 5962 | Deloite | Deloitte Audit & Assurance | False | True | $0.004684250 | 0.966 |
| 8667 | News Corp | News Corp | True | True | $0.004757500 | 0.939 |
| 3678 | Nuance | Nuance | True | True | $0.004548500 | 0.954 |
| 5466 | HEINEKEN UK | HEINEKEN UK | True | True | $0.003054000 | 1.631 |
| 4546 | Nuday AS | Nuuday A/S | True | True | $0.003519750 | 1.198 |
| 7523 | Enpro | Enpro | True | True | $0.003954750 | 1.159 |
| 8170 | DigitalBridge | DigitalBridge | True | True | $0.005343750 | 1.564 |
| 4922 | Volvo Group | AB Volvo | False | True | $0.005036250 | 1.260 |

## Cost and provenance

Gemini totals: prompt tokens 177,154; cached prompt tokens 0; response tokens 172; thinking tokens 0. API/response-processing latency 12.725s. Actual returned model versions: ['gemini-3.1-flash-lite'].

Paid-tier equivalent cost uses $0.25/M uncached input, $0.025/M cached input, and $1.50/M output including thinking. This is usage-based list-price accounting, not a verified invoice; free tiers or credits are not subtracted. [Official pricing](https://ai.google.dev/gemini-api/docs/pricing).

Gemini 2.5 Flash-Lite was attempted first because its published text price is lower. All 11 requests returned HTTP 404, explicitly saying the model was no longer available to new users. There were no successful generations or returned usage for those calls; billed cost is unreported, not asserted to be zero. Failed-attempt wall time was 3.460s and is excluded from the successful-run latency above. Those artifacts remain in `runs/sustainability_11_gemini25_flash_lite/`. The runner now stops on permanent HTTP request/model errors.

[Gemini results](runs/sustainability_11_gemini31_flash_lite/results.json) · [Evaluation](runs/sustainability_11_gemini31_flash_lite/evaluation.json) · [Raw requests/responses and usage](runs/sustainability_11_gemini31_flash_lite/calls.jsonl) · [Configuration](runs/sustainability_11_gemini31_flash_lite/config.json) · [Earlier stage-level report](extraction_experiment.md)

## Window-batching latency experiment

Same Mac and MPS GPU, pinned GLiNER Small checkpoint, float precision unchanged, 11 documents / 559 windows, 320 word/punctuation units per window, overlap 64, threshold 0.5, CPU threads 4. Batches contain windows from one document at a time. No document truncation or candidate pruning. Two full-dataset trials per batch size, ordered 1, 8, 16, 32, 32, 16, 8, 1, with the model resident and one synthetic-window warm-up before all trials. First encounters of larger batch shapes may still include warm-up overhead. Timing includes candidate merging and source validation and synchronizes MPS before completion. Two trials are exploratory, not confidence intervals.

| Windows per batch | Trial 1 (s) | Trial 2 (s) | Mean extraction (s) | Speedup vs batch 1 | Added / removed source spans across both trials |
|---|---:|---:|---:|---:|---:|
| 1 | 44.622 | 45.001 | 44.812 | 1.00× | 0 / 0 |
| 8 | 22.422 | 21.436 | 21.929 | 2.04× | 0 / 0 |
| 16 | 21.369 | 20.656 | 21.013 | 2.13× | 0 / 0 |
| 32 | 24.445 | 23.864 | 24.154 | 1.86× | 0 / 0 |

Fastest tested mean: batch size **16**, **2.13×** extraction speedup. This is the fastest among the tested sizes, not a global optimum. Local benchmark API spend: **$0**. Candidate equality refers to names and exact source offsets; floating-point confidence scores need not be bit-identical.

### Fresh end-to-end run at the fastest tested batch size

| Stage | Wall time (s) | Monetary cost (USD) |
|---|---:|---:|
| Model loading, separate | 4.091 | $0 |
| extraction | 21.260301 | $0.000000000 |
| selection | 4.635506 | $0.008866410 |
| copying | 0.000153 | $0.000000000 |
| Pipeline total, excluding load | 26.017 | $0.008866410 |

Fresh-run total speedup against the original 48.968s run: **1.88×**. This end-to-end comparison is one run each; the repeated extraction benchmark above is the stronger evidence for the batching improvement. Successful selections: 11/11. Name-match proxy: **8/11** using the same existing GT and threshold; alias limitations still apply. Changed final names: [].

All original experiment outputs are preserved. The fresh run makes new Jev requests; it does not reuse historical decisions. Per-call usage and per-document stage latency are saved. The local benchmark contributes $0; the only additional API charge is the fresh run’s selection cost shown above.

[Batch benchmark records](runs/window_batch_benchmark/metrics.json) · [Fresh results](runs/sustainability_11_gliner_batch_16/results.json) · [Fresh call ledger](runs/sustainability_11_gliner_batch_16/calls.jsonl)

### Potential ways to further improve latency

These are untested follow-up experiments; no additional speedup is assumed.

- **Overlap extraction and API calls:** start GLiNER on the next document while Jev processes the previous one, with bounded request concurrency. This primarily improves total dataset completion time; it does not necessarily reduce one document's latency.
- **Tune batching further:** test sizes near 16 and group similarly sized windows, potentially across documents, to reduce padding and partially filled batches. Preserve document IDs and source offsets and verify candidate equivalence.
- **Reduce window overlap or adjust window length:** process fewer windows while staying within the extractor's input limit. Measure candidate recall, especially for names at window boundaries, before adopting the change.
- **Use selective extraction with a full-document fallback:** scan the opening sections first and expand when evidence is insufficient. This may save substantial work but can miss the subject or its complete name, so evaluate the stopping rule separately.
- **Benchmark alternative inference settings:** test supported lower-precision execution, optimized runtimes, or a different GPU. Measure actual speed and output changes; include any cloud GPU charges in the cost comparison.
- **Keep the model loaded between jobs:** avoid repeated startup costs in a persistent process. This improves cold-start latency, but does not reduce the reported 26.02-second pipeline time, which already excludes loading.

For any concurrency change, compare against a similarly configured Gemini baseline and report both per-document latency and total dataset completion time. Recheck extraction quality and repeat timings before claiming a reliable improvement.

## Overlapping extraction and Jev requests

Controlled follow-up at GLiNER Small batch size 16 on MPS. Order: sequential, overlapped, overlapped, sequential. Each trial processes the same 11 documents with a freshly loaded model; loading is timed separately. No candidate/decision caching or retries. In overlap mode, one selection worker processes a document while the main thread extracts the next. At most one selection is active; extraction remains sequential. All trials use the same runner, prompt, model versions, windows and threshold.

| Trial | Execution | Dataset wall time (s) | Extraction sum (s) | Selection sum (s) | Mean document latency (s) | API cost (USD) | Name-match proxy |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | serial | 26.556 | 21.420 | 5.017 | 2.404 | $0.008866410 | 8/11 |
| 2 | overlap | 21.585 | 21.102 | 9.133 | 2.753 | $0.008866410 | 8/11 |
| 3 | overlap | 21.417 | 20.988 | 9.770 | 2.800 | $0.008866410 | 7/11 |
| 4 | serial | 26.160 | 21.187 | 4.857 | 2.368 | $0.008866410 | 8/11 |

Mean dataset completion time: **26.358s sequential → 21.501s overlapped**, a **1.23× speedup (18.4% reduction)**. Two trials per mode provide exploratory evidence, not statistical confidence. These are dataset completion times, excluding model load; per-document latency runs from extraction start to selection/copy completion and includes any selection queue wait. It excludes waiting before a document's extraction starts. Stage-duration sums overlap in time and must not be added to estimate dataset wall time.

Changed final names versus the previous batch-16 run: 1 document-trials. Changed candidate-span sets: 0 document-trials. All 44 document executions completed successfully. Name-match scores retain the existing GT/alias limitations. Total additional API cost across these four trials: **$0.035465640**. Local computation is $0 by convention.

Use `--extractor gliner --device mps --batch-size 16 --overlap-api` to enable this scheduling. The default remains sequential for reproducibility. This experiment does not rebenchmark Gemini with concurrency, so it establishes the scheduling improvement within our method rather than a fair concurrent-throughput comparison against Gemini.

Raw artifacts include per-document start/completion timestamps, queue delay, stage timings, source spans, and complete API usage:

- [Trial 1 (serial)](runs/api_overlap_matched_1_serial/metrics.json): [outputs](runs/api_overlap_matched_1_serial/results.json), [calls](runs/api_overlap_matched_1_serial/calls.jsonl), [configuration](runs/api_overlap_matched_1_serial/config.json).
- [Trial 2 (overlap)](runs/api_overlap_matched_2_overlap/metrics.json): [outputs](runs/api_overlap_matched_2_overlap/results.json), [calls](runs/api_overlap_matched_2_overlap/calls.jsonl), [configuration](runs/api_overlap_matched_2_overlap/config.json).
- [Trial 3 (overlap)](runs/api_overlap_matched_3_overlap/metrics.json): [outputs](runs/api_overlap_matched_3_overlap/results.json), [calls](runs/api_overlap_matched_3_overlap/calls.jsonl), [configuration](runs/api_overlap_matched_3_overlap/config.json).
- [Trial 4 (serial)](runs/api_overlap_matched_4_serial/metrics.json): [outputs](runs/api_overlap_matched_4_serial/results.json), [calls](runs/api_overlap_matched_4_serial/calls.jsonl), [configuration](runs/api_overlap_matched_4_serial/config.json).

The only final-name change was document 4922: trial 3 returned `AB Volvo` instead of `Volvo Group`, reducing the automatic match score to 7/11. Its request payload was identical to the sequential trial (indeed all request payloads in these two trials matched). This is response variation under identical inputs, not evidence that batching changed candidates; the experiment cannot establish that scheduling caused the variation. The alias/subject-name distinction still requires review.

Mean document latency increased from approximately 2.39s sequentially to 2.78s under overlap, despite faster dataset completion. Selection-stage elapsed time also increased while extraction ran concurrently; these measurements alone do not isolate local contention from remote/network timing. Overlap improves throughput here, not individual-document response latency.


### Preliminary transport failure and session accounting

An earlier sequential trial had one TLS transport error on document 8667 and 10 successful selections. Its pipeline wall time was 25.137s and its known API charges were $0.007883862; one request has unknown billed usage. It is preserved at `runs/api_overlap_trial_1_serial/` and excluded from the matched successful-run timing table. Total known charges for this overlap experiment, including that preliminary run, are **$0.043349502, plus any unreported charge for the failed request**. No automatic retries were used; the matched trials are fresh full runs.

