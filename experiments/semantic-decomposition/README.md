# Jev semantic-operator experiment

Set `model: jev` on a map/filter with boolean or `enum[...]` output fields.
The rendered prompt is sent unchanged as state. Boolean fields use Noul >= 0.5;
enum fields use Choice with exactly the schema labels. Other output types fail
explicitly. One document per call; the existing reduce uses Gemini 3.8 Flash with its original prompt.
No automatic prompt rewrite, truncation, or fallback. Validation rules run once;
Jev does not generate correction attempts. Existing temperature kwargs are unused.

## Prepare and run (from repository root, in the DocETL Python environment)

```sh
python experiments/semantic-decomposition/prepare_sustainability.py --split train
docetl run experiments/semantic-decomposition/sustainability_train_classifiers.yaml
# Full pipeline, including the original reduce prompt executed by Gemini:
docetl run experiments/semantic-decomposition/sustainability_train_pipeline.yaml
```

Preparation writes a shared retained input subset and an exclusion manifest.
It limits the serialized request to 24,000 estimated cl100k_base tokens (including
prompt and questions), leaving headroom below Jev's 32K limit. This is an estimate
using a different tokenizer, not a guarantee of fit. Retained
text is never shortened. Both generated YAMLs preserve original prompt strings.
An API overflow still raises an error; authentication failures are never treated
as long documents. Generated data/results are ignored by git.

## Netlify AI Gateway (no separate Jev key)

This directory includes a small protected Netlify Function bridge. Link it to a
Netlify project with AI Gateway enabled on a credit-based plan. Netlify requires
an initial production deploy before gateway activation. Set `JEV_PROXY_TOKEN`
to a private shared token in Netlify; never commit it. Deploy only the `public`
directory and the Function, not the local dataset. The gateway supplies
`TYPESAFE_API_KEY` and `TYPESAFE_BASE_URL` to the Function.

For a deployed bridge, set locally:

```sh
export JEV_PROXY_URL='https://YOUR-SITE.netlify.app/.netlify/functions/jev'
export JEV_PROXY_TOKEN='YOUR-PRIVATE-SHARED-TOKEN'
```

Alternatively run `netlify dev` in this directory after linking/activation and
use `http://localhost:8888/.netlify/functions/jev` as the proxy URL. If running
Python in an environment already supplied with Netlify's TypeSafe credentials,
the adapter also reads `TYPESAFE_API_KEY` and `TYPESAFE_BASE_URL` directly.
A direct TypeSafe key works with the default provider endpoint.

`jev` pins `jev-1.13.0`. Cache keys include endpoint, model, prompt, and questions.
The supplied YAML bypasses the response cache, as the original pipeline does.
Reported cost is a provider-token estimate at $0.042/M input tokens, not Netlify
credit billing; override `JEV_INPUT_PRICE_PER_MILLION` if needed.

References: https://docs.typesafe.ai/models and
https://docs.netlify.com/build/ai-gateway/overview/

## Private-project local smoke test

Run `bash experiments/semantic-decomposition/run_local_gateway.sh` from the repository root in a regular terminal. This machine-specific helper uses the installed temporary CLI/Python and bundled Node, obtains gateway credentials through `netlify dev:exec`, and makes one tiny live call. It does not change visibility or print credentials. Temporary runtimes may need reinstalling after a reboot.

The local launcher sets Node DNS preference and the Python Jev transport to IPv4 to avoid Netlify source-IP mismatch across IPv4/IPv6 routes. TLS verification remains enabled. Run `bash experiments/semantic-decomposition/run_local_gateway.sh pipeline` for the 11-document, two-operator smoke pipeline. The project remains private.

## Gemini for non-Jev operators

The experiment YAMLs use `default_model: gemini/gemini-3.8-flash`. Set `GEMINI_API_KEY` in the repository-root `.env`. Filter/map explicitly use `model: jev`; the full pipeline reduce inherits Gemini. The preparation script accepts `--default-model gemini/gemini-3.5-flash-lite` (or another Gemini model) for alternate experiments. Historical MOAR baseline YAMLs outside this project are unchanged.

## Evaluate a saved comparison

```sh
python experiments/semantic-decomposition/evaluate_comparison.py \
  --run-dir experiments/semantic-decomposition/runs/20260928T183815Z
```

This makes no model calls. It imports the existing MOAR sustainability evaluator
unchanged, selects original ground-truth records by IDs in the run's `input.json`,
and uses that same input snapshot to set the scoring denominator (11 for this run).
It fails on missing/duplicate IDs or missing labels. No labels are regenerated.
Outputs are under `<run-dir>/evaluation/`: `ground_truth.json`, per-variant metrics,
`evaluation.json` with provenance and scoring limitations, and `summary.md`.

MOAR scores company-name matching and economic-sector correctness in the final
reduce output, not the factual quality of sustainability findings. Matching remains
fuzzy and is not one-to-one. Its `missing_companies` field counts unmatched generated
names, not omitted input companies. Its `total_companies_processed` is the input
count. The subset restricts name-matching candidates to the evaluated documents;
the original generic MOAR wrapper instead searches the full ground-truth dataset.

## Additional Gemini model comparisons

`run_comparison.py` accepts `--gemini-model` (3.8 Flash, 3.1 Flash-Lite,
or 3.5 Flash-Lite), `--input-file`, and `--pipeline-file`. It records the
selected model's standard input, cached-input, and output rates. Use
`--variant gemini_only` for an all-Gemini run and a fresh `--output-dir`.
Existing experiment directories are never overwritten.

The two Flash-Lite experiments are saved under the original run directory as
`flash_lite_3_1/` and `flash_lite_3_5/`. Rebuild their combined report without
model calls using:

```sh
python experiments/semantic-decomposition/report_flash_lite.py \
  --run-dir experiments/semantic-decomposition/runs/20260928T183815Z
```

This appends/replaces the Flash-Lite section of the existing `summary.md`,
retaining earlier sections, and saves machine-readable evaluation results.
