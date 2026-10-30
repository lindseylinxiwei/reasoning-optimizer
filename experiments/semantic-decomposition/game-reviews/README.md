# GameReviews: Jev decisions + deterministic processing

The input already supplies review IDs and timestamps. A strict parser recovers
every review; Jev classifies each as positive, negative, or neutral. Code selects
up to ten reviews per polarity, minimizing total timestamp distance to ten
evenly spaced targets across the full game's time range. A second Jev choice
selects one source sentence/line as an extractive summary for each selected review.
IDs, timestamps, and summaries are copied from the source. Identical duplicate
records are classified once; conflicting records with the same ID are rejected.

No GLiNER, fine-tuning, or generative LLM is needed in this decomposition. VADER
is used only by offline evaluation. Neutral reviews never fill a sentiment quota;
shortages are recorded instead of inventing reviews or changing their polarity.

Run from the repository root with the existing Jev comparison environment:

```sh
python experiments/semantic-decomposition/game-reviews/run.py \
  --output experiments/semantic-decomposition/game-reviews/runs/train40 --dry-run
python experiments/semantic-decomposition/game-reviews/run.py \
  --output experiments/semantic-decomposition/game-reviews/runs/train40
python experiments/semantic-decomposition/game-reviews/evaluate.py \
  --run experiments/semantic-decomposition/game-reviews/runs/train40
python -m unittest discover -s experiments/semantic-decomposition/game-reviews -p 'test_*.py'
```

The runner uses `TYPESAFE_API_KEY` from the repository `.env` and the existing
company-extraction Jev client. Requests contain at most 64 independent questions
and respect that client's 24K per-question / 48K per-request token guards. No
review text is truncated. Successful requests are checkpointed by model and
payload; changed code or inputs require a new run directory. Errors propagate,
and there are no automatic network retries. API costs and failed calls with
unknown billing remain in the request ledger.

The evaluator compares all input games with the saved baseline using exact IDs
and identical source text. The baseline is not rerun. It reports the unchanged
local MOAR score plus ID validity, duplicate selections, exact timestamps,
per-list ordering, requested list sizes, and normalized distance to time targets.
The cached VADER lexicon is required; only the legacy module's redundant download
attempt is suppressed, and parsed reference labels are memoized across methods.

Important evaluation limits:

- Local MOAR sentiment labels come from VADER, not human or Steam ratings.
- Its temporal score concatenates positive and negative lists, although the task
  requests two separately sorted lists. Per-list ordering is audited separately.
- Its score does not measure temporal evenness, summaries, or completion of 10+10
  slots. Agreement per requested slot exposes the effect of returning fewer reviews.
- Extractive summaries preserve wording but can still omit nuance or context.
- These are training-set diagnostics, not held-out benchmark results.

Outputs include source snapshots, parsing counts, per-review sentiment decisions,
summary source offsets, selection pools, the final output, API logs, and evaluation.
