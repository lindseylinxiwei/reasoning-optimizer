# Offline search on CUAD and Sustainability

This file covers the code for the offline search of rewrite plans on the CUAD and Sustainability
workloads. The search is MOAR's MCTS run unattended on Modal under a dollar budget. Sweeps run
beforehand, outside the search, choose its first layer, its model statistics and its prior. Everything
listed here serves that search. It is not needed to run a DocETL pipeline.

"Offline" means two things here:

- The search and the sweeps run in batch jobs before any plan is used, and can take hours.
- They use train-split labels only to score plans. Ground truth never reaches the optimizer agent or
  the container that executes candidate pipelines (see "Label isolation" below).

## Search engine (`docetl/mcts/`)

| File | Role |
|---|---|
| `mcts.py` | The search. Budget and cost guard, failure accounting, exploration turns, which rewrites may target which operators, Jev rules, and the parameter-grid fan-out. |
| `mcts_utils.py` | Expansion prompts. Rewrites are listed least-tried first, with per-rewrite stats from this search; rewrites that do not fit the plan's shape are dropped; models outside the allowed list are rejected. |
| `param_grid.py` | Adaptive grid. A rewrite with a main parameter (head words, top-k size, chunk size, model) runs one child per selected value. Values are probed, refined and pruned using one shared results table. |
| `search_state.py` | State that survives a crash: `checkpoint.json`, `nodes.jsonl`, `costs.jsonl` (the budget ledger) and `param_table.jsonl` in the run's output directory. |
| `tree_log.py` | Readable tree, one line per plan (`mcts_tree_log.txt`). Run `python -m docetl.mcts.tree_log <run_dir>` to print it. |
| `jev_fit.py` | Static check that an operation's input fits Jev's 32K-token context. |
| `Node.py` | Never selects a node that has no rewrite left. |

Supporting changes in `docetl/reasoning_optimizer/`:

- **Spend tracking:** `directives/agent_utils.py` counts every agent call toward the budget, including
  calls made by instantiations that fail. A model the pricing table does not cover is priced from its usage.
- **Chunk sizes:** `effective_length.py` holds per-model effective input lengths, measured on CUAD. The
  chunking rewrites (`doc_chunking.py`, `doc_chunking_topk.py`) suggest chunk sizes from it.
- **Data views:** `data_views.py` accepts a minimum kept share and level guidance per pipeline.
- **Rewrite fixes:**
  - `take_head_tail` writes its cut text to a field only the target operation reads.
  - `clarify_instructions` checks that reduce prompts keep their `inputs` loop.
  - Gleaning is capped at one round.
  - `arbitrary_rewrite` asks for a task or data decomposition and only allows models from `AVAILABLE_MODELS`.
- **Model statistics:** `directives/change_model_cost.py` holds accuracy and cost per Gemini model,
  taken from the first-layer sweeps below.

## Label isolation (`experiments/reasoning/`)

Candidate pipelines can contain agent-written code, so the container that runs them must not hold any
ground-truth file.

- `label_isolation.py` decides, file by file, what goes into the label-free search image. Anything it
  cannot classify is left out.
- `utils.py` builds three images: `image` (everything), `eval_image` (code plus `reasoning/data`) and
  `search_image` (label-free; sets `DOCETL_LABEL_FREE=1`).
- `remote_eval.py` scores plans on `eval_image`. Only pipeline output goes in and only metrics come
  back. `evaluation/utils.py` routes to it when it runs inside the label-free container.

## Running the search

```bash
modal run --detach experiments/reasoning/run_mcts.py \
    --yaml-path <pipeline.yaml> --dataset-path <train.json> --dataset <cuad|sustainability> \
    --experiment-name <name> --max-budget-usd 50 [--prior-path prior.json]
```

- The search is spawned, so it keeps running if the client disconnects. Follow it with
  `modal app logs <app id>`; results are on the Modal volume.
- Running the same command again resumes from `checkpoint.json`. The budget already spent is read from
  `costs.jsonl`.
- **`--exclude-directives` / `--only-directives`:** choose the rewrites. By default `hierarchical_reduce`,
  `chunk_header_summary` and the generic `change model cost` are left out.
- **`--cost-guard-fraction`:** skips a child whose estimated cost is above this share of the budget left.

## CUAD sweeps (`experiments/cuad-decomposition/`)

All sweeps run on the 40 train contracts unless noted. Each script's docstring lists its phases
(propose, check, run, score).

| Script | What it does |
|---|---|
| `run_first_layer_sweep.py` | Original pipeline once per Gemini model: the search's first layer and the CUAD rows of `change_model_cost.py`. |
| `measure_effective_length.py` | Per-model effective length from the first-layer sweep; the source of `effective_length.py`. |
| `sweep.py` | Level 1: each rewrite on its own (data views, isolating subtasks, presence gate, chunking). |
| `sweep_level2.py` | Level 2: combinations of the Level 1 rewrites that helped. |
| `sweep_level3.py` | Level 3: the Level 1+2 frontier plans with every LLM step switched to one model. |
| `jev_retrieval_sweep.py` | Jev-only retrieval sweep for per-group retrieval (no LLM calls). |
| `moar_rerun.py` | Reruns the original MOAR CUAD frontier plans with the list output format. |
| `run_test_frontier.py` | Runs chosen plans once on the 100-contract test split. |
| `plot_level1.py`, `plot_level2.py`, `plot_level3.py`, `report_level1.py` | Cost-vs-F1 plots and the Level 1 report. |

Outputs go to `experiments/cuad-decomposition/runs/`. That folder is not committed (it is over 300 MB);
regenerate it with the scripts.

## Sustainability sweeps (`experiments/sustainability-decomposition/`)

The data-view and model sweeps (`view_sweep.py`, `plot_model_sweep.py` and others) were committed
earlier. This change adds:

| Script | What it does |
|---|---|
| `run_first_layer_sweep.py` | Original pipeline once per Gemini model: the search's first layer and the Sustainability rows of `change_model_cost.py`. |
| `build_search_prior.py` | Turns the earlier data-view sweep into the search prior (`--prior-path`). The prior holds aggregate scores and costs only, no labels. |
| `run_moar500_test.py` | Runs plan YAMLs on label-free train or test reports (by default the `sustainability_moar_500` frontier plans). |
| `plot_moar500_test.py`, `plot_all_nodes.py` | Test-set frontier of the new search's plans against the original MOAR frontier. |

## Tests

`tests/mcts/` covers these areas:

- **Parameter grid:** `test_param_grid`, `test_grid_siblings`
- **Logs:** `test_tree_log`
- **Dry runs:** `test_dry_run`, using `dryrun_harness.py`
- **Failures and skipped visits:** `test_search_failures`, `test_skipped_visits`
- **Exploration and action stats:**
  - `test_exploration`
  - `test_action_stats`
  - `test_tried_elsewhere`
  - `test_target_choice`
  - `test_data_view_prior`
- **Spend tracking:** `test_agent_cost_tracking`
- **Effective length:** `test_effective_length`
- **Label isolation:** `test_label_isolation`
- **Jev rewrites in the search:** `test_jev_classify`, `test_jev_fusion`, `test_jev_selector`

Run them with:

```bash
python -m pytest tests/mcts
```
