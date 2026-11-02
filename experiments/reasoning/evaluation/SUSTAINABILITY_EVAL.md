# Sustainability evaluation: what was fixed and why

`sustainability.py` now has two evaluators:

| Function | Status |
|---|---|
| `evaluate_results` | **Corrected version.** Used by MOAR search and the Pareto frontier through `utils.get_evaluate_func`. |
| `evaluate_results_original` | The original evaluator, kept unchanged for reference and for reproducing earlier numbers. |

Both have the same signature and return the same keys. `economic_activity_accuracy` is MOAR's
search reward for this workload (`docetl/mcts/mcts.py`); `combined_score` is used for the reported
Pareto frontier. The corrected version also returns the original's three scores as
`original_economic_activity_accuracy`, `original_company_name_accuracy` and
`original_combined_score`, so every run can be compared both ways.

## How the original scores a run

For every company the pipeline outputs:

1. Compare its name with all 286 GT company names using `calculate_name_similarity`, and take the
   best GT record if the similarity is at least 0.5.
2. If found, count a correct name, and a correct activity if the output's activity equals that GT
   record's `GT economic_activity`.
3. Divide both counts by the number of input reports.

## Problems with it

### 1. Full legal names fail the name match

When one name contains the other, `calculate_name_similarity` returns the length ratio of the two
strings. A model that writes the full legal name scores below 0.5:

| Output name | GT name | Original similarity |
|---|---|---|
| Enpro Industries, Inc. | Enpro | 0.23 |
| Deloitte Italy S.p.A. | Deloite | 0.25 |
| NIPPON EXPRESS HOLDINGS, INC. (NIPPON EXPRESS GROUP) | NIPON EXPRES GROUP | 0.33 |
| Unibail-Rodamco-Westfield SE | UNIBAIL ROD | 0 |
| Legal & General Investment Management Limited | LGIM | 0 |

A failed name match also makes the activity count as wrong, so this leaks into the search metric.
The effect depends on how verbose the model is, not on how good the pipeline is. With
gemini-3.5-flash-lite, the full-text pipeline found the company in 45 of 50 reports but the
original scored its names at 0.56 and its activity at 0.44 (corrected: 0.93 and 0.69). Shorter
input text made the model write shorter names, so the original score went *up* as text was cut
(0.50 for full text, 0.66 for the first 1k tokens).

### 2. Outputs are matched against all 286 GT records

An output can match a GT record that was not in the input, for example a different report from a
company with a similar name. That match counts as correct.

### 3. Duplicates and hedging can add credit

Every output entry counts. If the same company appears twice, both entries count. If a company is
listed under two activities, one of them is counted as correct.

### 4. A correct filter is penalized

The pipeline starts with a filter that keeps sustainability reports, but the original divides by
all input reports. `train_50` has 42 sustainability reports and 8 annual or financial reports.
A filter that correctly drops the 8 loses 8 reports' worth of name and activity credit, while a
lenient filter, or no filter at all, scores higher. With full text the filter let more annual
reports through, so part of the full-text pipeline's higher score came from that.

## What the corrected version does

1. **Name matching** (`name_match_score`, a match is score >= 0.5). Both names are lowercased,
   apostrophes are dropped, doubled letters are collapsed (the GT typos are mostly dropped double
   letters: "Acenture", "Swis Re", "Ferero"), corporate-form words are removed ("Inc.", "S.p.A.",
   "Group", "Holdings"...), and parenthetical alternatives are tried separately. Then:

   | Score | Rule | Example |
   |---|---|---|
   | 1.0 | same core name | "Enpro Industries, Inc." vs "Enpro Industries" |
   | 0.9 | one core name is a word sequence inside the other (a shorter output must keep the GT name's first word) | "Enpro Industries, Inc." vs "Enpro" |
   | 0.85 | same letters once spaces are removed | "Allcargo Logistics" vs "ALCAR GO LOGISTIC S", "Unibail-Rodamco-Westfield" vs "UNIBAIL ROD" |
   | 0.8 | acronym | "Legal & General Investment Management" vs "LGIM" |
   | 0.7 | the GT name's first core word (3+ letters) appears in the output | "Sun Life Assurance" vs "Sun Life Financial" |
   | <= 0.65 | the original word-level typo matching | |

   Rejected on purpose: "Energy" vs "XYZ Energy Holdings", "ABC Energy" vs "XYZ Energy",
   "Bank" vs "Center invest Bank", and "Eni" for the Versalis report (the model named the parent
   company). The first core word is used because it is usually the distinctive part of a name.

2. **Only input reports are candidates** (from `original_json_file`). The output goes to the
   closest name, so "Volvo Group" and "Volvo Car Group" stay separate. When names tie exactly
   (a company's annual report and its sustainability report, e.g. "Nuday" and "Nuday AS"), it
   goes to a report not yet credited, then to the sustainability report.

3. **Each input report counts once.** Duplicate entries add nothing. A report is correct on
   activity only if every entry credited to it has the GT activity.

4. **Only what the filter is meant to keep is scored.** Targets are input reports whose
   `GT document_category` is "Sustainability Report". Name and activity accuracy are computed
   over them. Each non-target report (annual, financial, ...) that reaches the output adds one
   to the denominator with no credit, so letting it through lowers the score and filtering it
   out costs nothing:

   `accuracy = correct target reports / (target reports + non-target reports in the output)`

   Records without a `GT document_category` are treated as targets. Without
   `original_json_file`, every GT record is a candidate and unmatched output entries are added
   to the denominator.

Extra diagnostics in the result: `target_reports`, `non_target_reports`,
`non_target_reports_in_output`, `filter_recall` (targets found / targets), `filter_precision`
(targets found / all scored reports in the output), `unmatched_output_entries`,
`reports_with_conflicting_activities` and `economic_activity_accuracy_excl_gt_other`. The last
one leaves out target reports whose GT label is `other`, which often conflicts with an obvious
specific sector (Meiji, Versalis, Nippon Express) and caps every method's accuracy.

Not changed: the GT labels themselves, the activity comparison (exact string match), and
`calculate_name_similarity`, which other scripts import.

## Effect on results

50 reports from `experiments/reasoning/data/train_50/sustainability.json` (42 sustainability
reports, 8 annual or financial). Pipelines on gemini-3.5-flash-lite; the two Jev runs use
jev-1.13.0 for the filter and/or classification step. The trimming variants come from the
Sustainability data-decomposition study (branch `sustainability-decomposition`).

| Run | Cost | Activity | Names | Combined | Sustainability reports found | Non-sustainability let through | Original activity | Original combined |
|---|---|---|---|---|---|---|---|---|
| first 1k tokens, Jev filter + classify (one call) | $0.026 | 0.67 | 0.95 | 0.81 | 41/42 | 1/8 | 0.50 | 0.58 |
| search plan 15 (first 120 + last 60 words) | $0.051 | 0.67 | 0.91 | 0.79 | 39/42 | 1/8 | 0.50 | 0.59 |
| first 1k tokens | $0.066 | 0.68 | 0.93 | 0.81 | 41/42 | 2/8 | 0.56 | 0.66 |
| first 4k tokens | $0.195 | 0.70 | 0.95 | 0.83 | 42/42 | 2/8 | 0.44 | 0.51 |
| search plan 17 | $0.729 | 0.67 | 0.91 | 0.79 | 41/42 | 3/8 | 0.54 | 0.62 |
| full text (root) | $2.402 | 0.69 | 0.93 | 0.81 | 42/42 | 3/8 | 0.44 | 0.50 |

Search plan 13 is left out: its `hierarchical_reduce` step groups by the input's
`company_name` field, which is the GT label, and copies it into the output.

With the original evaluator, the full-text pipeline looked worst (0.50 combined). Corrected, it
is in the middle, and the first 4k tokens score as well or better at 8% of the cost. The
original search outputs (gemini-2.5-flash-lite) change rank too.

## Using it

```python
from experiments.reasoning.evaluation.sustainability import evaluate_results, evaluate_results_original

m = evaluate_results("run", "output.json", "experiments/reasoning/data/company_reports_gt.json",
                     "experiments/reasoning/data/train_50/sustainability.json")
m["economic_activity_accuracy"], m["original_economic_activity_accuracy"]
```

Numbers recorded before this change (search logs, `evaluation_metrics.json`, earlier findings
docs) were computed with the original. To reproduce them, call `evaluate_results_original`.
`experiments/semantic-decomposition/evaluate_comparison.py` calls `evaluate_results`, so it reports
corrected numbers if rerun.
