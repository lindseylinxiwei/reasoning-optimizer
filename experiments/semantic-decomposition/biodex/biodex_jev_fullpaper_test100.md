# Biodex: iterative Jev selection with full-paper context — 100 test documents

All **100 test documents** completed. **RP@5 = 0.3380**, **RP@10 = 0.3875**. Total known Jev API cost: **$16.457382** ($0.164574 per document). Measured end-to-end process wall time: **692.90s (11.55 min)** with up to **8 documents in flight**.

## Method and evaluation

For each complete paper, Jev independently evaluates every official label as supported or not supported. Up to 256 binary questions share one request and the same full-paper state: 24,313 labels and 95 scan calls per paper. All supported labels enter iterative comparative selection. Jev selects the best remaining label, code removes it, and selection repeats up to ten times or until NONE. Pools larger than 253 labels are split into groups; up to ten labels per group advance to the final selection. Code copies official label names into the ranked output.

Both stages use the complete paper. There is no evidence filter, embedding retrieval, Gemini call, or free-text answer generation. The model (jev-1.13.0), prompts, scan batch size, vocabulary order, group size, and output limit match the five-paper full-context pilot. The execution change is parallel processing across eight documents; decisions within each document retain their dependencies.

The source is the complete `experiments/reasoning/data/test/biodex.json` file, with IDs and hashes frozen before API calls. Ground truth enters only evaluation. The unchanged MOAR evaluator defines RP@k as correct labels among the first k divided by min(k, number of ground-truth labels), averaged across documents. It is not whole-document accuracy.

| Evaluation on these same test documents | RP@5 | RP@10 |
|---|---:|---:|
| Same exhaustive scan, probability sorting (offline baseline) | 0.2558 | 0.3303 |
| **Exhaustive scan + iterative full-paper selection** | **0.3380** | **0.3875** |

The probability-sort baseline is computed from the same saved scan answers and makes no additional API calls. The earlier five-paper development result was RP@5=0.64 and RP@10=0.62; that pilot is not a matched accuracy comparison with these 100 papers.

## Monetary cost and latency

| Stage | Known API cost | Measured batch wall time | Mean service time/document | API calls including retries |
|---|---:|---:|---:|---:|
| Exhaustive label scan | $15.910089 | 647.36s | 49.90s | 9500 |
| Iterative label selection | $0.547293 | 42.41s | 3.25s | 1436 |
| **Total stages** | **$16.457382** | **689.77s** | **53.15s** | **10936** |

End-to-end process wall time is **692.90s**; the difference from summed phase wall time (3.13s) covers preflight, aggregation, evaluation, and output processing.

Per-document scan + ranking service time: mean **53.15s**, median **52.67s**, p95 **57.87s**. This adds each document’s two measured stages and excludes queueing between them. It is not batch time divided by 100, and it is not submission-to-result latency: the runner completes the scan phase for the batch before starting ranking.

The sum of all document stage times is **5315.01s**; it exceeds wall time because documents overlap. This sum is not a measured single-worker run. Latency depends on concurrency and API conditions, so it should not be compared directly with the earlier single-worker pilot or MOAR runtimes measured differently.

The scan accounts for **96.7%** of total cost. Cost uses recorded provider input tokens × $0.042 per million; it is a usage-based estimate rather than an invoice. Local compute is counted as $0. No prior pilot costs, optimizer search costs, or report-generation costs are included.

Unknown-cost calls: **0**. Retry calls: **0**. Unresolved document errors: **0**. All retry charges with reported usage are counted; calls lacking usage remain unknown rather than being assumed free.

## Label coverage

- Exhaustive label decisions: **2,431,300**.
- Macro ground-truth recall after binary support decisions: **0.8142**.
- Oracle RP@5 from scan-supported labels: **0.8510**.
- Macro ground-truth recall after group selection: **0.6100**.
- Oracle RP@5 among group survivors: **0.6678**.

For candidate pools of at most 253, all supported labels enter final ranking directly. Larger pools can lose correct labels during group selection. Coverage and oracle metrics are evaluation-only diagnostics, not inputs to any Jev decision.

## Per-document results

| Document | RP@5 | RP@10 | Supported labels | Scan GT recall | API cost | Scan + ranking service time |
|---|---:|---:|---:|---:|---:|---:|
| 34456717 | 0.0000 | 0.0000 | 305 | 0.5000 | $0.160523 | 51.15s |
| 34433719 | 1.0000 | 1.0000 | 250 | 1.0000 | $0.163000 | 50.38s |
| 34790421 | 0.2000 | 0.3333 | 246 | 1.0000 | $0.156656 | 48.35s |
| 34188921 | 1.0000 | 1.0000 | 110 | 1.0000 | $0.151580 | 47.36s |
| 34073672 | 0.4000 | 0.6000 | 259 | 1.0000 | $0.158730 | 51.06s |
| 33054123 | 0.2000 | 0.4000 | 295 | 1.0000 | $0.169095 | 52.02s |
| 34055543 | 0.3333 | 0.3333 | 344 | 1.0000 | $0.163289 | 50.69s |
| 34107924 | 0.0000 | 0.0000 | 507 | 0.3333 | $0.173178 | 52.69s |
| 34201591 | 0.0000 | 0.0000 | 306 | 1.0000 | $0.183917 | 58.43s |
| 33732653 | 1.0000 | 0.5000 | 252 | 0.8333 | $0.164386 | 52.21s |
| 33883542 | 0.6000 | 0.6000 | 209 | 0.6000 | $0.158740 | 50.76s |
| 34541338 | 0.6000 | 0.5714 | 239 | 1.0000 | $0.166191 | 53.61s |
| 34485017 | 0.2500 | 0.2500 | 255 | 0.7500 | $0.155187 | 50.71s |
| 34635150 | 0.0000 | 0.0000 | 165 | 1.0000 | $0.156103 | 49.65s |
| 32773727 | 0.0000 | 0.0000 | 273 | 0.0000 | $0.159752 | 52.01s |
| 33557784 | 0.6000 | 0.4286 | 264 | 0.8571 | $0.158556 | 50.44s |
| 33482449 | 0.8000 | 0.5000 | 310 | 0.9167 | $0.158202 | 51.05s |
| 33888146 | 0.0000 | 0.0000 | 265 | 0.5000 | $0.163733 | 52.29s |
| 32526071 | 0.0000 | 0.0000 | 207 | 0.8889 | $0.162099 | 51.35s |
| 34633074 | 0.5000 | 1.0000 | 74 | 1.0000 | $0.192389 | 57.12s |
| 33995989 | 0.0000 | 0.0000 | 411 | 0.8000 | $0.162963 | 52.11s |
| 34707681 | 0.0000 | 0.0000 | 391 | 1.0000 | $0.168066 | 55.13s |
| 34664862 | 0.0000 | 0.0000 | 257 | 1.0000 | $0.180864 | 56.24s |
| 33889835 | 0.0000 | 0.0000 | 157 | 1.0000 | $0.155499 | 47.76s |
| 34809625 | 0.2000 | 0.2000 | 338 | 0.8000 | $0.173834 | 55.95s |
| 34249743 | 0.0000 | 0.0000 | 277 | 1.0000 | $0.164401 | 53.11s |
| 34055491 | 1.0000 | 1.0000 | 95 | 1.0000 | $0.173025 | 52.24s |
| 34877251 | 0.3333 | 0.3333 | 154 | 0.3333 | $0.161300 | 49.39s |
| 34431691 | 0.6000 | 0.3000 | 172 | 0.4167 | $0.166367 | 50.78s |
| 33485376 | 0.0000 | 0.3333 | 332 | 1.0000 | $0.159137 | 50.41s |
| 33763319 | 1.0000 | 1.0000 | 333 | 1.0000 | $0.156892 | 54.31s |
| 33953631 | 0.2000 | 0.2000 | 221 | 0.8000 | $0.164091 | 54.73s |
| 33614197 | 0.5000 | 0.5000 | 364 | 0.5000 | $0.158462 | 57.86s |
| 34593488 | 0.0000 | 0.0000 | 101 | 0.0000 | $0.213469 | 64.34s |
| 34871273 | 0.0000 | 0.0000 | 150 | 1.0000 | $0.156846 | 52.52s |
| 33941104 | 0.2500 | 0.2500 | 46 | 0.7500 | $0.170761 | 56.06s |
| 33138347 | 0.2500 | 0.5000 | 181 | 0.5000 | $0.182397 | 58.12s |
| 34039375 | 0.0000 | 0.5000 | 167 | 1.0000 | $0.172072 | 55.61s |
| 34646694 | 0.0000 | 0.5000 | 159 | 1.0000 | $0.152067 | 50.45s |
| 34567798 | 0.0000 | 0.0000 | 207 | 0.5000 | $0.159191 | 50.43s |
| 33996609 | 0.0000 | 0.0000 | 309 | 1.0000 | $0.159981 | 52.78s |
| 33860147 | 0.5000 | 0.5000 | 156 | 1.0000 | $0.155405 | 47.84s |
| 34820246 | 1.0000 | 1.0000 | 327 | 1.0000 | $0.155897 | 50.97s |
| 34479480 | 0.2000 | 0.4000 | 318 | 0.8000 | $0.169402 | 53.81s |
| 34887925 | 1.0000 | 1.0000 | 338 | 1.0000 | $0.162440 | 51.22s |
| 34306782 | 0.0000 | 0.0000 | 353 | 1.0000 | $0.157733 | 50.82s |
| 34345762 | 0.4000 | 0.5000 | 140 | 0.8333 | $0.152513 | 48.31s |
| 33281154 | 1.0000 | 1.0000 | 268 | 1.0000 | $0.160162 | 52.60s |
| 34408682 | 0.0000 | 0.0000 | 124 | 1.0000 | $0.156139 | 52.25s |
| 34234701 | 0.2000 | 0.1667 | 333 | 0.8333 | $0.195541 | 61.47s |
| 34436497 | 0.6000 | 0.3000 | 82 | 0.5455 | $0.175151 | 55.25s |
| 34622867 | 0.6000 | 0.4286 | 314 | 0.8571 | $0.171284 | 56.80s |
| 34083820 | 0.2000 | 0.2500 | 149 | 1.0000 | $0.165943 | 56.45s |
| 34447684 | 0.5000 | 0.5000 | 100 | 1.0000 | $0.164528 | 54.53s |
| 34093674 | 0.3333 | 0.3333 | 327 | 0.6667 | $0.156575 | 52.66s |
| 33528846 | 0.2500 | 0.2500 | 188 | 0.7500 | $0.165650 | 54.78s |
| 33552335 | 1.0000 | 1.0000 | 183 | 1.0000 | $0.163457 | 55.18s |
| 33653851 | 0.7500 | 0.7500 | 227 | 0.7500 | $0.156560 | 54.16s |
| 34721009 | 0.4000 | 0.3333 | 304 | 0.8333 | $0.159177 | 56.65s |
| 34821704 | 0.0000 | 1.0000 | 243 | 1.0000 | $0.160534 | 55.32s |
| 34479520 | 0.4000 | 0.5000 | 469 | 1.0000 | $0.163868 | 57.32s |
| 34178120 | 0.0000 | 0.0000 | 195 | 1.0000 | $0.156977 | 54.56s |
| 34804436 | 0.5000 | 0.5000 | 288 | 0.5000 | $0.160825 | 55.95s |
| 33912470 | 0.0000 | 0.0000 | 234 | 1.0000 | $0.161003 | 52.76s |
| 34250484 | 0.5000 | 0.5000 | 140 | 1.0000 | $0.159630 | 51.92s |
| 34595213 | 0.0000 | 0.0000 | 286 | 0.6667 | $0.158101 | 50.61s |
| 34007372 | 0.8000 | 0.7500 | 367 | 1.0000 | $0.160780 | 50.37s |
| 34400601 | 0.0000 | 0.6667 | 247 | 1.0000 | $0.160704 | 49.04s |
| 34377045 | 0.2500 | 0.2500 | 324 | 0.5000 | $0.166480 | 51.47s |
| 33113098 | 0.4000 | 0.2500 | 179 | 0.7500 | $0.152760 | 47.82s |
| 34633155 | 0.2000 | 0.1250 | 183 | 0.5000 | $0.157611 | 47.68s |
| 33936120 | 0.0000 | 1.0000 | 475 | 1.0000 | $0.177043 | 53.74s |
| 34568441 | 0.5000 | 0.5000 | 319 | 1.0000 | $0.166471 | 51.25s |
| 33237657 | 0.8000 | 0.5556 | 166 | 0.7778 | $0.172698 | 54.08s |
| 33542643 | 0.0000 | 0.5000 | 240 | 1.0000 | $0.171321 | 54.90s |
| 33681590 | 0.0000 | 0.0000 | 9 | 0.0000 | $0.172647 | 52.67s |
| 33796234 | 0.0000 | 0.0000 | 323 | 1.0000 | $0.157776 | 50.17s |
| 34140335 | 0.0000 | 0.0000 | 248 | 1.0000 | $0.174803 | 55.40s |
| 34759667 | 0.0000 | 0.0000 | 90 | 0.5000 | $0.163358 | 55.36s |
| 34267798 | 0.8000 | 0.9000 | 400 | 1.0000 | $0.185713 | 58.66s |
| 33747646 | 0.5000 | 0.5000 | 243 | 1.0000 | $0.153775 | 50.22s |
| 34485080 | 0.6667 | 0.6667 | 337 | 1.0000 | $0.161341 | 55.09s |
| 34667538 | 0.2000 | 0.5000 | 239 | 1.0000 | $0.153869 | 50.80s |
| 34114997 | 0.4000 | 0.5000 | 315 | 1.0000 | $0.157844 | 52.44s |
| 33997204 | 0.0000 | 0.0000 | 65 | 1.0000 | $0.170364 | 50.51s |
| 34126959 | 0.6667 | 0.6667 | 441 | 1.0000 | $0.176103 | 56.62s |
| 33580193 | 0.6000 | 0.6667 | 192 | 0.8889 | $0.169317 | 52.26s |
| 33851013 | 1.0000 | 1.0000 | 138 | 1.0000 | $0.154422 | 49.87s |
| 34222286 | 0.0000 | 0.0000 | 224 | 1.0000 | $0.160273 | 51.43s |
| 33622999 | 0.8000 | 0.5000 | 365 | 0.7857 | $0.166718 | 55.01s |
| 33997647 | 0.0000 | 0.0000 | 208 | 0.0000 | $0.152681 | 50.66s |
| 33585092 | 0.6000 | 0.8000 | 329 | 1.0000 | $0.158611 | 53.91s |
| 34035966 | 0.4000 | 0.8000 | 381 | 1.0000 | $0.155711 | 54.29s |
| 34248548 | 0.5000 | 0.5000 | 266 | 0.5000 | $0.160257 | 56.49s |
| 33055478 | 0.3333 | 0.3333 | 304 | 0.3333 | $0.159029 | 56.04s |
| 33791385 | 0.3333 | 0.3333 | 198 | 0.6667 | $0.162909 | 54.93s |
| 33708966 | 0.0000 | 0.0000 | 300 | 0.0000 | $0.183758 | 56.70s |
| 33613847 | 0.5000 | 0.5000 | 150 | 1.0000 | $0.169004 | 55.33s |
| 34743690 | 0.2000 | 0.5000 | 568 | 0.7500 | $0.168147 | 56.02s |
| 33987101 | 0.2000 | 0.1429 | 444 | 0.8571 | $0.171600 | 55.94s |

## Scope and reproducibility

- This is one execution per test paper, not a repeated-run variance estimate. Earlier experiments on this test set informed method development, so this is a full test-set evaluation rather than a claim of a completely untouched holdout.
- Comparison with a paper’s MOAR frontier requires matching the document set, metric, cost accounting, and concurrency. This report does not infer a MOAR comparison from an unmatched plot.
- All 35 offline tests passed before the run, including checks for per-document client isolation, checkpoint reuse, source copying, label coverage, and ranking behavior. Unresolved phase errors stop the runner rather than silently dropping papers.

The post-run audit verified all 100 documents, all 2,431,300 scan decisions, all 10,936 successful API calls, complete original-paper contexts, valid unique output labels, recomputed evaluation metrics, and costs against provider usage. No retry or failed call occurred. The scan-question hash also matches the original pilot exactly.

Artifacts:

- [Protocol](runs/fullpaper-test100-v1/protocol.json) · [Summary](runs/fullpaper-test100-v1/summary.json) · [Audit](runs/fullpaper-test100-v1/audit.json).
- [Predictions and ground-truth diagnostics](runs/fullpaper-test100-v1/results.json).
- [Probability-sort predictions](runs/fullpaper-test100-v1/probability_results.json).
- [Implementation](fullpaper.py) · [Tests](test_fullpaper.py).
- Each document folder contains every scan answer, ranking trace, request/response ledger, API usage, cost, and timing.

```bash
python experiments/semantic-decomposition/biodex/fullpaper.py \
  --input experiments/reasoning/data/test/biodex.json \
  --output experiments/semantic-decomposition/biodex/runs/fullpaper-test100-v1 --workers 8
python experiments/semantic-decomposition/biodex/report_fullpaper.py
```

Successful requests are checkpointed. Reusing an output folder resumes saved work; its new process wall time then measures a resumed execution, not a fresh run. Use a new output folder for a fresh end-to-end timing measurement.

## Comparison with MOAR on the 100-document test set

Source: [MOAR frontier JSON](https://drive.google.com/file/d/1HHJleZNRb7x3P9MdMnY2_HSDFkyzg_ow/view). Use **test_accuracy**, **test_cost**, and **test_latency**. The user confirms these test fields evaluate the same 100-document set. The adjacent `total_documents: 40`, `avg_rp_at_5`, and `cost` fields describe training and are not used below.

| Plan | Test RP@5 | Test API cost | Recorded test latency | Cost–accuracy test frontier? |
|---|---:|---:|---:|---|
| gpt_nano_config_18.json | 0.0000 | $0.1254 | 68.39s | Yes |
| gpt_nano_config_11-2-acc.json | 0.2587 | $0.8937 | 117.00s | Yes |
| gpt_nano_config_17.json | 0.2420 | $0.8960 | 94.12s | No |
| gpt_nano_config_5.json | 0.2945 | $0.9008 | 60.99s | Yes |
| gpt_mini_config_22.json | 0.3353 | $4.3948 | 303.89s | Yes |
| gpt_config_20.json | 0.3702 | $22.1868 | 335.44s | Yes |
| **Iterative Jev selection, full paper** | **0.3380** | **$16.4574** | **692.90s** | **Nondominated among these measured points** |

- Against `gpt_mini_config_22`: Jev adds **0.0027 RP@5 (0.27 percentage points)** at **3.74× the cost**. Its recorded wall time is **2.28×** as long. The small accuracy difference is not established as statistically significant.
- Against the highest-accuracy plan, `gpt_config_20`: Jev costs **25.8% less**, but RP@5 is **0.0322 lower (3.22 percentage points)**. Its recorded wall time is **2.07×** as long.
- Jev is technically nondominated in cost–accuracy among these discrete measurements: the cheaper mini plan has slightly lower accuracy, and the more accurate full model costs more. This is a weak tradeoff improvement, not evidence of a compelling efficiency advantage.
- The training-selected frontier changes on test: `gpt_nano_config_17` is dominated in cost–accuracy by `gpt_nano_config_11-2-acc`. The zero-accuracy config 18 remains a mathematical frontier endpoint solely because it is cheapest.
- Latencies are recorded whole-workload times from different executions. Jev used eight concurrent documents; the Drive summary does not specify MOAR hardware/concurrency. These are historical runtime comparisons, not a controlled speed benchmark.

[Machine-readable comparison](runs/fullpaper-test100-v1/moar_comparison.json) · [Saved source JSON](runs/fullpaper-test100-v1/moar_test_frontier_source.json)
