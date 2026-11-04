# CUAD evaluation: what was fixed and why

`cuad.py` now has two evaluators:

| Function | Status |
|---|---|
| `evaluate_results` | **Corrected version** (`evaluator_version = "cuad-official-match-v1"`). Used by MOAR search and the Pareto frontier through `utils.get_evaluate_func`. |
| `evaluate_results_original` | The original evaluator, kept unchanged for reference and for reproducing earlier numbers. |

Both take `(method_name, results_file, ground_truth_file, original_json_file)` and return the keys
the callers read: `avg_f1` (MOAR's search reward, `docetl/mcts/mcts.py`), `avg_precision`,
`avg_recall`, `nan_fraction`, `avg_clause_length` and `per_metric`. In the corrected version
`avg_f1` is the pooled (micro) F1 described below. The corrected version also returns the
original's numbers as `original_avg_f1`, `original_avg_precision` and `original_avg_recall`, so
every run can be compared both ways. It also takes an optional `categories=[...]` to score a
subset of the 41 categories, for example the 8 types of the earlier Jev retrieval experiments.

## How the original scores a run

For each of 41 categories:

1. Pick one predicted entry per contract. Its `clause_type` is the one closest by edit distance
   to the category name, if the two share a word.
2. Join the gold spans of that category into one string, and compare it with the predicted
   string by word-set Jaccard. Above 0.15 is a true positive, otherwise a false positive. A
   prediction with no gold is a false positive, and gold with no prediction is a false negative.
3. Compute each category's precision and recall, average the F1 of the categories where both are
   defined, and report that as `avg_f1`.

## Problems with it

The checks below used controls on `train_50` (50 contracts, 41 categories = 2,050 tasks, 1,437
gold answers). The scripts are listed at the end of this file.

### 1. 14 of the 41 categories are never scored

Most metric names match the ground-truth CSV headers only after lowercasing and replacing
spaces. Headers with hyphens or slashes ("Anti-Assignment", "Non-Compete", "Rofr/Rofo/Rofn",
"Unlimited/All-You-Can-Eat-License", "Affiliate License-Licensor", ...) never match. Neither do
headers whose wording differs ("Notice Period To Terminate Renewal", "Price Restrictions"). Those
14 categories get NaN and are silently left out of the average.

### 2. A wrong span never counts as a miss, and an all-wrong category disappears

A wrong span counts as a false positive only, so it never lowers recall. If every prediction in
a category is wrong, recall is 0/0 = NaN, so F1 is NaN, and the category is dropped from the
average instead of scoring 0. Returning the first 3,000 characters of the contract for every
category gets recall 1.0 in every category the original scores.

### 3. A correctly omitted clause is scored with a similar clause's span

Types are matched by edit distance. If a pipeline correctly leaves out "Uncapped Liability" but
returns "Cap on Liability", the cap span is scored as the uncapped-liability answer and counts as
a false positive. A perfect prediction (the gold itself) scores **0.90 to 0.98** in the original,
depending only on whether absent categories are listed with empty strings or omitted.

### 4. Matching rules differ from CUAD's

- The Jaccard threshold is 0.15 instead of the official 0.5.
- Gold spans are joined into one string, and so is a multi-span prediction.
- The CSV joins the parts of a multi-part answer with `<omitted>`, while the official CUAD QA data
  lists each part as its own answer.

### 5. Contracts are matched by position

Each output row is matched to the CSV filename with the most equal characters at the same
positions, and the two tables are then joined by row order after sorting. A missing or extra row
can shift the join. `rstrip(".TXT")` also strips any trailing `T`, `X` or `.` characters, not the
suffix.

## The corrected rule

It follows the official CUAD matching, as frozen in the Jev retrieval experiments
(`MOAR-Jev/research/jev_retrieval/jev-preserve-fill-f1-20261030-001/reference/upstream-evaluate.py`).

- **Task:** one (input contract, category). The contracts are all rows of `original_json_file`,
  including any that the pipeline dropped.
- **Gold:** the CSV span list for that contract and category, with each span split on
  `<omitted>`. For the 320 tasks of the handover cohort, this is identical to the official CUAD QA
  answers (320/320 tasks, 278 answers).
- **Matching:** `official_jaccard` is an exact copy of the official `get_jaccard`:
  - remove `. , ; :`;
  - lowercase;
  - replace `/` with a space;
  - split on the space character;
  - take word sets and compute Jaccard.

  A match needs Jaccard ≥ 0.5. For Parties, a prediction that contains the gold string also
  matches, as in the official evaluator. Checked identical to the official function on 22,000
  string pairs.
- **Counting** (many-to-many, per task):
  - TP: each gold answer matched by at least one prediction.
  - FN: each gold answer matched by none.
  - FP: each prediction matched by no gold answer, which includes every prediction on a task
    without gold.
- **Aggregation:**
  - Counts are pooled over all tasks: precision = TP/(TP+FP), recall = TP/(TP+FN), F1 =
    2TP/(2TP+FP+FN). This pooled F1 is the headline (`avg_f1` = `f1_micro`).
  - `f1_macro` is the mean of the per-category pooled F1. Categories with no gold and no
    predictions in the cohort have an undefined F1 and are left out (`nan_fraction`).
  - Precision is 0 when there are no predictions, as in the handover's summaries.
- **Negative tasks:** a task without gold where the pipeline returns nothing is a correct
  abstention. It adds no counts and is reported in `negative_task_accuracy`.
- **Missing or failed output:** a contract with no output row, or with no row whose `clauses` is a
  list, gets no predictions. Its gold answers become FNs, and its negative tasks do not count as
  correct abstentions. These are counted in `missing_contracts`, `failed_contracts` and
  `failed_tasks`. An input contract with no CSV row raises an error.

### Reading the pipeline output

The pipeline returns `clauses: list[{clause_type, text_spans}]` (current `pipelines/cuad.yaml`) or the
older `clauses: list[{clause_type, text_span}]`; both are read.

- **Type:** `clause_type` is normalized: lowercased, a leading list number such as `24) ` or `1. `
  dropped, and non-alphanumerics joined with `_`. It must then equal a metric key, the CSV header
  or an explicit alias (e.g. `right_of_first_refusal_offer_or_negotiation`,
  `non_solicit_of_employees`), also after making every word singular ("Third Party Beneficiaries"
  maps to `third_party_beneficiary`; a name that would become ambiguous raises an error). A known name followed by more words also maps to that name, the
  longest that fits. This covers models that copy the whole prompt line, e.g. "Governing Law:
  Which state/country's law governs ...". There is no nearest-name guessing.
  - Types that still don't map are ignored and reported in `unknown_clause_types`, e.g.
    "Confidentiality".
  - Prefix matches are reported in `prefix_matched_clause_types`.
- **Quotes:** `text_spans` (or `text_span`) may be a string or a list of strings.
  - Repeated entries of a type add more quotes.
  - Exactly empty strings are dropped, and exact duplicates are removed, keeping order.
  - Strings are not split on commas or repaired, and need not occur literally in the contract.
    Placeholders such as "not specified" or "Yes" count as predictions, because the prompt asks
    for an empty string.
- **Contracts:** output rows are matched to input contracts by `id`, then by normalized filename
  (no directory, no `.pdf`/`.txt`, uppercase letters and digits only). This normalization is
  unique across the 510 CSV contracts. Several rows for one contract are pooled. Rows for
  contracts outside the input list are ignored and counted in `extra_output_rows`.

## Control results (`train_50`, all 41 categories)

| Control | Corrected F1 (micro / macro) | P / R | Original `avg_f1` |
|---|---|---|---|
| Gold, one entry per answer part | **1.000 / 1.000** | 1.00 / 1.00 | 0.900 |
| Gold, list-valued `text_span` | 1.000 / 1.000 | 1.00 / 1.00 | 0.921 |
| Gold, absent categories listed with "" | 1.000 / 1.000 | 1.00 / 1.00 | 0.976 |
| Gold, multiple answers joined with ", " | 0.789 / 0.738 | 0.96 / 0.67 | 0.924 |
| Gold on positives, one sentence on every negative | 0.681 / 0.554 | 0.52 / 1.00 | 0.517 |
| Empty output | 0.000 / 0.000 | 0 / 0 | 0.000 |
| First 3,000 characters for every category | 0.114 / 0.021 | 0.09 / 0.14 | 0.087 |

- In the last row, all 207 TPs are Parties, through the official containment rule.
- Dropping 5 contracts from the gold output turns their 193 gold answers into FNs, F1 = 0.928.
- A row with `clauses: null` counts as a failed contract, not as an abstention.

## Reproducing the earlier 8-type results

The Jev retrieval experiment's saved answers
(`jev-preserve-fill-f1-20261030-001/analysis/per_case.json`) were rescored: 40 contracts × 8 types
= 320 tasks, 3 arms × 5 caps. Its 40 contracts are exactly `data/train/cuad.json`.

- `official_task_counts` gives the same TP/FP/FN on all 4,800 rows.
- Writing each arm and cap as a pipeline output and scoring it with `evaluate_results(...,
  categories=[the 8 types])` reproduces `analysis/metrics.json` exactly: TP/FP/FN, precision,
  recall, F1, macro F1, and negative tasks with false extractions. The maximum difference is 0.0.

## Earlier MOAR outputs, rescored

38 saved MCTS outputs (`experiments/reasoning/outputs/cuad_mcts*/cuad_mcts/*.json`, 50 contracts
of `train_50`; the baseline uses the 40 of `train`):

- Original `avg_f1`: 0.43 to 0.75. Corrected F1: 0.27 to 0.59. Spearman correlation between the
  two: 0.74.
- Typical corrected precision is 0.55–0.68, recall 0.30–0.53, and negative-task accuracy
  0.71–0.95.
- Some plans move a lot:
  - `gpt_nano_config_10`: 0.733 → 0.490.
  - `gpt_4.1_mini_config_21` rises from 0.554 to 0.559, into the top four, because the original
    could not read its "1. Document Name: …" types.
- The recomputed original matches the `evaluation_metrics.json` that MOAR wrote, where both exist.

## Notes for the pipeline

- **One answer per entry.** The prompt in `pipelines/cuad.yaml` asks for multiple spans "as a
  comma-separated list". Under the official rule, one joined string usually matches none of its
  parts. Even with perfect spans, joining costs 21 F1 points (1.000 → 0.789). It is better to ask
  for one `{clause_type, text_span}` entry per separate span, or to make `text_span` a
  `list[string]`.
- **Omit absent categories.** Ask the pipeline to omit categories that are absent, or return "".
  The prompt's yes/no questions ("Is there …?") make some models answer "Yes", which counts as a
  false positive. Phrasing each category as "the clause that …" avoids that. These changes are
  recommended, not made; changing the prompt changes the baseline.
- **Split mismatch.** `utils.get_evaluate_func("cuad")` and `evaluate_cuad_results` pass
  `data/train/cuad.json` (40 contracts) as the input list. A run on `train_50` is scored on 40 of
  its 50 contracts; the other 10 appear in `extra_output_rows`. The same happened with the
  original.

## Reproduce

These are offline, need no API calls, and are run from the repository root. The scripts are not
committed; they live in the session scratchpad
(`/private/tmp/claude-501/-Users-lindseywei/e91e8e1c-908a-4651-9df4-aedebcb1912a/scratchpad/cuad_eval_tests/`):

- `controls.py`: the control table and assertions, plus a check that `official_jaccard` equals
  the official function.
- `reproduce_handover.py`: the 8-type reproduction (reads the handover folder; read-only).
- `rescore_saved.py`: original vs corrected on saved MOAR outputs.
