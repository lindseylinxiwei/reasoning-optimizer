# Jev directive selection for MOAR

This checkout is based on `lindseylinxiwei/reasoning-optimizer`, branch
`experiment-code`, commit `a6aa69c2d55a7bc78f325c6f763c66e66e17384e`.

Jev replaces the selection call in `docetl/mcts/mcts.py:MCTS.expand`.
`--model` still controls directive instantiation. The default selector is `llm`,
so existing commands retain their original behavior. The directive registry,
MCTS node selection, evaluation, and multi-instantiation mechanism are unchanged.

## Local run

Use the repository's normal Python environment (`pip install -e .` if setting
up a new one). The Jev adapter itself uses only Python's standard library.
Set `TYPESAFE_API_KEY` in the environment or the repository's ignored `.env`;
keep your existing instantiation/execution provider credentials configured too.
Do not put API keys in command-line arguments.

From the repository root, add `--selector_backend jev` to your normal run:

```bash
python -m experiments.reasoning.run_mcts \
  --yaml_path /absolute/path/to/your/pipeline.yaml \
  --dataset_path /absolute/path/to/your/dataset.json \
  --dataset cuad \
  --experiment_name cuad_jev \
  --max_iterations 5 \
  --model gpt-4.1 \
  --selector_backend jev
```

Use your real pipeline, dataset, provider-qualified model, and dataset name.
Run a baseline with `--selector_backend llm` and a different experiment name.
No dataset or paid model execution is performed by the offline tests below.

## Modal and batch experiments

The same settings are forwarded through `run_mcts_remote` and the Modal local
entrypoint. Modal exposes the arguments as `--selector-backend`, `--jev-model`,
and `--jev-input-price`. The existing `modal.Secret.from_dotenv()` must include
`TYPESAFE_API_KEY` in the local `.env` for remote execution.

For `run_all.py`, add these fields inside each experiment's existing `mcts` block:

```json
{
  "mcts": {
    "max_iterations": 30,
    "model": "gpt-4.1",
    "selector_backend": "jev",
    "jev_model": "jev-1.13.0",
    "jev_input_price": 0.042
  }
}
```

## Selection behavior

1. Reuse MOAR's selection messages (pipeline, samples, goals, search history) and
   its allowed `(operator, directive)` pairs.
2. Jev `Choice` selects a directive, using the registry descriptions.
3. A second Choice selects its target operator or ordered operator pair.
   `chunk_header_summary` needs split/gather, `map_reduce_fusion` needs map/reduce,
   and `operator_fusion` needs two operators. Impossible pairs are omitted.
4. Directives accepting multiple targets (summarization, compression, model
   changes) can add targets through further Choices with a stop option.
   `arbitrary_rewrite` targets the whole pipeline and returns an empty target list.
5. Convert the decision to the original `{directive, operators}` JSON and append
   it to message history. The original LLM then instantiates the rewrite.

Forced choices with only one candidate are resolved locally. This is a sequential
selection policy, not an identical joint decision to the original generative
selector. It performs no learned pruning, confidence gating, or silent fallback
to the original LLM. When adding custom directives, extend the target-cardinality
rules in `jev_selector.py` if they do not take exactly one operator.

## Accounting and limits

- `jev_selection_calls.jsonl` logs each successful HTTP response with node/goal,
  actual model version, answer, selected option, usage, latency, and estimated cost.
  It does not log credentials or the full input dataset. Target descriptions can
  include operator configs/prompts, so treat logs as experiment data.
- `experiment_summary.json` records the selector configuration, Jev call count,
  Jev selection cost, and total search cost. Jev estimates are included in the
  existing total as calls finish, including calls with invalid choice values.
- Default model is pinned to `jev-1.13.0`. Its documented price on 2026-09-22 is
  $0.042 per million input tokens, with free output. Override `--jev_input_price`
  if your price changes; these are estimates, not billing reconciliation.
- Calls time out after 60 seconds. HTTP 429/5xx are retried up to three attempts;
  authentication errors fail immediately. Provider failures abort the experiment
  instead of silently treating a different selector as Jev.
- A Choice allows at most 255 options. Oversized candidate sets fail explicitly;
  they are never truncated. Jev 1.13 has a 64k total request limit and a 32k limit
  for state plus the longest question. This adapter preserves MOAR's messages;
  unusually long histories may therefore require a separate context-compaction
  experiment. API context errors are reported rather than silently dropping data.
- No live Jev call or full workload benchmark has been run for this change.

Official references: [HTTP API](https://docs.typesafe.ai/api),
[Choice](https://docs.typesafe.ai/primitives/choice),
[models and pricing](https://docs.typesafe.ai/models).

## Offline validation

```bash
LITELLM_LOCAL_MODEL_COST_MAP=True python -m unittest discover -s tests/mcts -v
```

Tests mock HTTP and paid LLM execution, covering selection constraints, multi-target
rewrites, API errors/retries, accounting, and the real `MCTS.expand` handoff for
both Jev and the original LLM selector.

## Working-folder migration and MEDEC volume check (2026-09-22)

The integration is now installed in `/Users/lindseywei/Documents/reasoning-optimizer`.
The working folder's 19-directive registry, local action-space-exhaustion handling,
and other uncommitted experiment changes have been preserved. Jev API errors also
propagate through the extra expansion error handler in this working copy.

Read-only checks of Modal volume `docetl-ro-experiments`, directory `outputs/medec`,
confirmed all seven expected first-layer configurations and referenced output JSONs:
`gpt_4.1_nano`, `gpt_4o_mini`, `gpt_5_nano`, `gpt_4.1`, `gpt_5_mini`,
`gemini_2.5_pro`, and `gpt_5`.

Each cached output contains 50 records. The local train set contains 40 records;
all 40 IDs occur in each cached output, and none of the local 100 test IDs do.
All seven cached pipeline prompts match the local MEDEC pipeline prompt.
A representative GPT-5 cache comparison confirms the shared texts and error flags
match exactly; missing sentence labels are represented as NaN, which caused raw
Python equality checks to report differences even though the values correspond.

The YAML input path is relative (`experiments/reasoning/data/train/medec.json`).
It exists locally but not inside the volume; it must be available in the Modal
image/runtime. No container was launched to verify that runtime path.

Do not interpret the cached 50-row accuracy/cost statistics as results on the
current 40-row train set. For a consistent comparison, regenerate first-layer
results on the current split, or deliberately restore the original 50-row setup.
No remote files were modified and no paid experiment was launched.
