# CUAD: GLiNER2.5 + Jev

An off-the-shelf, source-preserving extraction experiment. No training or generated
contract text: GLiNER proposes `(category, start, end)` candidates; Jev independently
keeps or rejects them; Python copies and groups the accepted spans.

The schema is read from `experiments/reasoning/pipelines/cuad.yaml`. The existing
company-extraction loader pins GLiNER2.5's revision. All 41 types are extracted
together with overlap allowed. Source windows default to 1,000 tokenizer tokens
with 250-token overlap and prefer paragraph boundaries. The real GLiNER processor
checks the complete encoded schema + text; windows shrink instead of truncating.
Candidates must exactly match their source offsets. Exact duplicates are collapsed
by `(category, start, end)`; distinct occurrences, overlapping spans and labels survive.
GLiNER's native sparse proposal limits still apply: this is not exhaustive span enumeration.

Jev sees each candidate's category definition, source text, and containing window.
Questions sharing a window are packed under the existing 24K/48K token guards.
`NEEDS_CONTEXT` expands to neighboring windows once; unresolved candidates remain
auditable and are omitted from the output. This version only verifies: it does not
repair boundaries, relabel candidates, or recover GLiNER omissions. Every category
is serialized once; absent categories get an empty string, multiple spans a comma-separated string.

Run from the repository root, in the existing company-extraction Python environment.
`requirements.txt` reuses its pinned dependencies and adds the MOAR evaluator's
`pandas` and `Levenshtein` dependencies:

```sh
python -m pip install -r experiments/semantic-decomposition/cuad/requirements.txt
python experiments/semantic-decomposition/cuad/run.py --device mps \
  --limit 10 --output experiments/semantic-decomposition/cuad/runs/train10 --extract-only
# Reuse those exact candidates and add Jev verification (uses TYPESAFE_API_KEY from .env):
python experiments/semantic-decomposition/cuad/run.py --device mps \
  --limit 10 --output experiments/semantic-decomposition/cuad/runs/train10
python experiments/semantic-decomposition/cuad/evaluate.py \
  --run experiments/semantic-decomposition/cuad/runs/train10
python -m unittest discover -s experiments/semantic-decomposition/cuad -p 'test_*.py'
```

Use `HF_HOME` to select an existing model cache. Run directories checkpoint extraction
and successful Jev requests. Source/config/code changes require a new directory.
Transport errors propagate, never masquerade as absent clauses. API requests are
logged individually; failed calls with unknown billing are reported explicitly.
There are no automatic network retries. Re-running reuses successful checkpoints.

Outputs include `gliner.json`, `gliner_jev.json`, per-document candidates/raw predictions,
verification traces, `model.json`, `protocol.json`, and `metrics.json`. Timings separate
model loading, extraction and verification; resumed execution time is labeled separately.
Local compute is reported in seconds, API spend uses the existing client's price assumption.

Evaluation is offline and never sends gold labels to either model. It reports the
unchanged MOAR evaluator plus exact-filename candidate coverage, precision and Jev
false rejections using diagnostic word-set Jaccard >=0.5. The diagnostic is not the
official CUAD metric (including its special Parties rule); MOAR itself uses >0.15
and fuzzy category matching. Do not interpret a smoke test as a benchmark result.

Known limits: spans longer than a window, cross-references outside the expanded
context, incorrect categories/boundaries, and model proposal limits can all reduce
recall. GLiNER scores are used as thresholds, not assumed calibrated probabilities.
