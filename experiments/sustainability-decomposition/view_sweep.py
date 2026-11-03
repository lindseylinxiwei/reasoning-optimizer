"""Data decomposition sweep: shared data views on the Sustainability pipeline.

Every plan = one data view (head_tail | topk | jev | summarization, with agent-chosen
parameters) inserted before the first operator and written to tot_text_raw, then the original
filter -> classify -> summary reduce, all on gemini-3.5-flash-lite (Jev steps on Jev).

Phases (run separately, from the repository root):
  python experiments/sustainability-decomposition/view_sweep.py propose   # agent sweeps (~$0.05-0.2)
  python experiments/sustainability-decomposition/view_sweep.py check     # offline, free
  python experiments/sustainability-decomposition/view_sweep.py run --local [plan ...]
  MODAL_PROFILE=personal modal run experiments/sustainability-decomposition/view_sweep.py   # same, on Modal
  python experiments/sustainability-decomposition/view_sweep.py score     # frontier, free

Inputs come from the label-free copy of train_50; labels are only read when scoring.
"""
import json
import sys
import time
from pathlib import Path

import modal
import yaml

HERE = Path(__file__).resolve().parent
# Inside a Modal container the script sits near the filesystem root; repo paths are only used locally.
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
sys.path[:0] = [str(REPO), str(REPO / "experiments/reasoning")]

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

ROOT_YAML = REPO / "experiments/reasoning/pipelines/sustainability.yaml"
DATA = REPO / "experiments/reasoning/data/train_50/sustainability.json"
GT = REPO / "experiments/reasoning/data/company_reports_gt.json"
MODEL = "gemini/gemini-3.5-flash-lite"
AGENT = "gemini/gemini-3.8-flash"
TEXT_KEY = "tot_text_raw"
OUT = HERE / "runs" / "view_sweep"
CONFIGS = OUT / "configs.json"
TECHNIQUES = ["head_tail", "topk", "jev", "summarization"]


# ---- inputs ---------------------------------------------------------------------------

def label_free_docs(path=DATA):
    from docetl.mcts.label_guard import strip_labels
    return strip_labels(json.load(open(path)), "sustainability")


# ---- phase 1: agent proposals ---------------------------------------------------------

def propose():
    from docetl.reasoning_optimizer.data_views import SWEEP_TYPES, check_view, proposal_prompt
    from docetl.reasoning_optimizer.directives import JevRetrievalDirective
    from docetl.reasoning_optimizer.directives.agent_utils import agent_completion
    from docetl.reasoning_optimizer.directives.jev_retrieval import _jev_explanation, _validation_sample
    from docetl.reasoning_optimizer.instantiate_schemas import JevIndexDesignSchema
    from docetl.retrieval.index_spec import IndexSpec, corpus_profile

    docs = label_free_docs()
    texts = [d[TEXT_KEY] for d in docs]
    lengths = [len(t) for t in texts]
    profile = corpus_profile(texts)
    ops = yaml.safe_load(open(ROOT_YAML))["operations"]
    result, total_cost = {"agent": AGENT, "profile": {k: v for k, v in profile.items() if k != "excerpts"}}, 0.0

    def ask(prompt, schema, validate, history=None):
        nonlocal total_cost
        history = history if history is not None else []
        history.append({"role": "user", "content": prompt})
        for _ in range(3):
            resp, cost = agent_completion(model=AGENT, messages=history, response_format=schema)
            total_cost += cost
            content = resp.choices[0].message.content
            history.append({"role": "assistant", "content": content})
            try:
                return validate(schema(**json.loads(content))), history
            except Exception as err:
                history.append({"role": "user", "content": f"Validation error: {err}\nPlease try again."})
        raise RuntimeError(f"agent failed to produce valid {schema.__name__}")

    for technique in TECHNIQUES:
        extra, index_spec = "", None
        if technique == "jev":
            # Call 1: index design (shared by the 4 Jev configurations).
            directive = JevRetrievalDirective()
            samples = _validation_sample(texts)

            def design_ok(design):
                spec = IndexSpec(levels=[l.model_dump(exclude_none=True) for l in design.levels],
                                 leaf_chars=design.leaf_chars, min_chars=design.min_chars, fanout=design.fanout)
                return spec, directive.check_index(spec, samples), design.rationale

            (spec, stats, rationale), _ = ask(directive.index_design_prompt(ops[1], profile),
                                              JevIndexDesignSchema, design_ok)
            index_spec = spec.model_dump(exclude_none=True)
            result["jev_index"] = {"spec": index_spec, "stats": stats, "rationale": rationale}
            extra = (f"{_jev_explanation()}\nIndex already designed: {json.dumps(index_spec)}\n"
                     f"Index built on sample documents: {json.dumps({k: v for k, v in stats.items() if k != 'levels'})}\n"
                     f"Fields: targets (1-8 evidence targets with name and description, phrased as the evidence "
                     f"needed, never the answer), regions_per_request (1-8; regions x targets per request <= 32 is "
                     f"handled by automatic bundling), cap_levels (evidence_chars per target) and decision_levels "
                     f"(max_decisions, 16-256, enough to reach the evidence given the index shape). Policy is "
                     f"preserve_fill for the levels; an adaptive comparison is added automatically.")

        def sweep_ok(sweep, technique=technique):
            for name, config in sweep.configs().items():
                try:
                    check_view(technique, config, lengths)
                except ValueError as err:
                    raise ValueError(f"{name}: {err}")
            return sweep

        sweep, _ = ask(proposal_prompt(technique, ops, profile, extra), SWEEP_TYPES[technique], sweep_ok)
        result[technique] = {"sweep": sweep.model_dump(),
                             "plans": {n: c.model_dump() for n, c in sweep.configs().items()}}
        print(f"{technique}: {len(result[technique]['plans'])} plans")
    result["agent_cost"] = total_cost
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(CONFIGS, "w"), indent=1)
    print(f"agent cost ${total_cost:.4f}; wrote {CONFIGS}")


# ---- plans ------------------------------------------------------------------------------

def plans():
    """{plan name: (technique, config dict, index spec)} from configs.json."""
    from docetl.reasoning_optimizer.data_views import VIEW_TYPES
    configs = json.load(open(CONFIGS))
    out = {}
    for technique in TECHNIQUES:
        for name, raw in configs.get(technique, {}).get("plans", {}).items():
            index = configs.get("jev_index", {}).get("spec") if technique == "jev" else None
            out[name] = (technique, VIEW_TYPES[technique](**raw), index)
    return out


def pipeline_config(technique, config, index_spec, data_path, output_path):
    from docetl.reasoning_optimizer.data_views import view_ops
    base = yaml.safe_load(open(ROOT_YAML))
    view = view_ops(technique, config, TEXT_KEY, MODEL, index_spec=index_spec)
    ops = view + base["operations"]
    for op in ops:  # every LLM operator on the fixed model; Jev and code operators keep theirs
        if op["type"] in ("map", "filter", "reduce"):
            op["model"] = MODEL
    step = base["pipeline"]["steps"][0]
    step["operations"] = [op["name"] for op in view] + step["operations"]
    base.update(default_model=MODEL, operations=ops, bypass_cache=True)
    base["datasets"]["company_reports"]["path"] = str(data_path)
    base["pipeline"]["output"]["path"] = str(output_path)
    return base


# ---- phase 2: offline check -------------------------------------------------------------

def check():
    from docetl.reasoning_optimizer.data_views import check_view
    from docetl.runner import DSLRunner
    lengths = [len(d[TEXT_KEY]) for d in label_free_docs()]
    print(f"{'plan':16s} {'kept (median doc)':>18s}  config")
    for name, (technique, config, index) in plans().items():
        stats = check_view(technique, config, lengths)
        cfg = pipeline_config(technique, config, index, DATA, OUT / name / "output.json")
        DSLRunner(cfg, max_threads=1)  # DocETL validates the full plan config
        shown = {k: v for k, v in config.model_dump().items() if k != "rationale"}
        if technique == "jev":
            shown["targets"] = [t["name"] for t in shown["targets"]]
        print(f"{name:16s} {stats['kept_share_median']:9.1%} ({stats['kept_chars_median']:>6d} ch)  {json.dumps(shown)[:140]}")


# ---- phase 3: run -----------------------------------------------------------------------

def run_plan(config: dict, docs: list) -> dict:
    """Execute one plan config on `docs`; returns cost, seconds and output records."""
    import tempfile
    from docetl.runner import DSLRunner
    with tempfile.TemporaryDirectory() as tmp:
        data, out = Path(tmp) / "in.json", Path(tmp) / "out.json"
        data.write_text(json.dumps(docs))
        config["datasets"]["company_reports"]["path"] = str(data)
        config["pipeline"]["output"]["path"] = str(out)
        start = time.time()
        cost = DSLRunner(config, max_threads=16).load_run_save()
        return {"cost": cost, "seconds": round(time.time() - start, 1), "output": json.loads(out.read_text())}


def save_result(name, technique, config, result):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    json.dump(result["output"], open(d / "output.json", "w"))
    json.dump({"plan": name, "technique": technique, "config": config.model_dump(), "cost": result["cost"],
               "seconds": result["seconds"], "model": MODEL}, open(d / "run_meta.json", "w"), indent=1)
    print(f"{name}: ${result['cost']:.4f} in {result['seconds']}s")


def selected_plans(names):
    all_plans = plans()
    unknown = set(names) - set(all_plans)
    if unknown:
        raise SystemExit(f"unknown plans: {sorted(unknown)}")
    return {n: all_plans[n] for n in (names or all_plans)}


def run_local(names):
    docs = label_free_docs()
    for name, (technique, config, index) in selected_plans(names).items():
        cfg = pipeline_config(technique, config, index, DATA, OUT / name / "output.json")
        save_result(name, technique, config, run_plan(cfg, docs))


# Modal: one container per plan. Only API calls are paid there; inputs are passed in.
app = modal.App("sustainability-view-sweep")


def _skip(path):
    # Inputs are passed in at call time, so data and result folders stay out of the image.
    parts = set(Path(path).parts)
    return (bool(parts & {".venv", "__pycache__", "outputs", "runs", "out", "data", "othersystems"})
            or Path(path).name.startswith(".") or ".chroma-" in str(path))


_image = (
    modal.Image.debian_slim(python_version="3.10")
    .uv_sync()
    .add_local_python_source("experiments", ignore=_skip)
    .add_local_python_source("docetl", ignore=_skip)
)


@app.function(image=_image, secrets=[modal.Secret.from_dotenv(REPO)], timeout=60 * 60, max_containers=16)
def run_plan_remote(config: dict, docs: list, name: str) -> dict:
    # Errors are returned as text: provider exception classes do not always deserialize locally.
    try:
        return {"plan": name, **run_plan(config, docs)}
    except Exception as exc:
        import traceback
        return {"plan": name, "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}"}


@app.local_entrypoint()
def modal_main(plans_csv: str = ""):
    names = [n for n in plans_csv.split(",") if n]
    docs = label_free_docs()
    todo = selected_plans(names)
    names = list(todo)
    calls = [(pipeline_config(t, c, i, DATA, OUT / n / "output.json"), docs, n) for n, (t, c, i) in todo.items()]
    failed = {}
    for result in run_plan_remote.starmap(calls, return_exceptions=True, order_outputs=False):
        if isinstance(result, BaseException):
            print(f"FAILED (plan unknown at this point): {str(result)[:300]}")
            continue
        name = result.pop("plan")
        if "error" in result:
            failed[name] = result["error"]
            print(f"FAILED {name}: {result['error'][:300]}")
        else:
            technique, config, _ = todo[name]
            save_result(name, technique, config, result)
    done = [n for n in names if (OUT / n / "run_meta.json").exists()]
    print(f"saved {len(done)}/{len(names)}; failed: {sorted(failed) or 'none'}")


# ---- model sweep on the frontier plan (head_tail_1) -------------------------------------
# Same view (head 500 tokens), every LLM step on one model; "h1_jev" replaces the filter and
# the activity map with one Jev call answering both, the reduce stays on MODEL.

MODEL_OUT = HERE / "runs" / "model_sweep"
# Split -> (input data, output folder). The test split is the 100-report held-out set.
SPLITS = {"train_50": (DATA, MODEL_OUT),
          "test": (REPO / "experiments/reasoning/data/test/sustainability.json", HERE / "runs" / "model_sweep_test")}
SWEEP_MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.7-flash",
                "gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.1-pro-preview", "gemini-4-argon"]
MODEL_PLANS = {f"h1_{m.removeprefix('gemini-')}": f"gemini/{m}" for m in SWEEP_MODELS} | {"h1_jev": "jev"}

COMBINED_PROMPT = """Answer two questions about the following company document.

Document text:
{{ input.tot_text_raw }}

Output field is_sustainability_report: determine if this document is a sustainability report or contains significant sustainability-related content.
A sustainability report typically includes information about:
- Environmental performance and initiatives
- Social responsibility and community impact
- Corporate governance and sustainability practices
- ESG (Environmental, Social, Governance) metrics
- Climate change initiatives
- Sustainable business practices
- Corporate social responsibility activities
True if this document is primarily a sustainability report or contains substantial sustainability-related content, false otherwise.

Output field economic_activity: determine the primary economic sector/activity of the company. Choose from these exact categories:
- manufacturing
- other
- information_communication
- finance_insurance
- agriculture
- energy_supply
- health_social
- trade_repair
- real_estate
- professional_scientific
- transportation
- construction
- accommodation_food
"""


def model_pipeline_config(name, output_path):
    technique, config, index = plans()["head_tail_1"]
    cfg = pipeline_config(technique, config, index, DATA, output_path)
    model = MODEL_PLANS[name]
    if model != "jev":
        for op in cfg["operations"]:
            if op["type"] in ("map", "filter", "reduce"):
                op["model"] = model
        cfg["default_model"] = model
        return cfg
    ops = {op["name"]: op for op in cfg["operations"]}
    enum = ops["determine_economic_activity"]["output"]["schema"]["economic_activity"]
    combined = {"name": "classify_report_jev", "type": "map", "model": "jev", "prompt": COMBINED_PROMPT,
                "output": {"schema": {"is_sustainability_report": "boolean", "economic_activity": enum}}}
    keep = {"name": "keep_sustainability_reports", "type": "code_filter",
            "code": "def transform(input_doc):\n    return bool(input_doc.get('is_sustainability_report'))\n"}
    cfg["operations"] = [op for op in cfg["operations"]
                         if op["name"] not in ("filter_sustainability_reports", "determine_economic_activity")]
    cfg["operations"] += [combined, keep]
    steps = cfg["pipeline"]["steps"][0]["operations"]
    i = steps.index("filter_sustainability_reports")
    steps[i:i + 2] = ["classify_report_jev", "keep_sustainability_reports"]
    return cfg


@app.local_entrypoint()
def model_main(plans_csv: str = "", split: str = "train_50"):
    names = [n for n in plans_csv.split(",") if n] or list(MODEL_PLANS)
    unknown = set(names) - set(MODEL_PLANS)
    if unknown:
        raise SystemExit(f"unknown plans: {sorted(unknown)}")
    data, MODEL_OUT = SPLITS[split]
    docs = label_free_docs(data)
    calls = []
    for n in names:
        cfg = model_pipeline_config(n, MODEL_OUT / n / "output.json")
        (MODEL_OUT / n).mkdir(parents=True, exist_ok=True)
        yaml.safe_dump(cfg, open(MODEL_OUT / n / "pipeline.yaml", "w"), sort_keys=False)
        calls.append((cfg, docs, n))
    failed = []
    for result in run_plan_remote.starmap(calls, return_exceptions=True, order_outputs=False):
        if isinstance(result, BaseException):
            print(f"FAILED (plan unknown at this point): {str(result)[:300]}")
            continue
        name = result.pop("plan")
        if "error" in result:
            failed.append(name)
            print(f"FAILED {name}: {result['error'][:300]}")
            continue
        json.dump(result["output"], open(MODEL_OUT / name / "output.json", "w"))
        json.dump({"plan": name, "view": "head_tail_1", "model": MODEL_PLANS[name], "cost": result["cost"],
                   "seconds": result["seconds"]}, open(MODEL_OUT / name / "run_meta.json", "w"), indent=1)
        print(f"{name}: ${result['cost']:.4f} in {result['seconds']}s")
    print(f"failed: {sorted(failed) or 'none'}")


def score_models(split="train_50"):
    from evaluation.sustainability import evaluate_results
    DATA, MODEL_OUT = SPLITS[split]
    rows = []
    for meta_path in sorted(MODEL_OUT.glob("*/run_meta.json")):
        meta = json.load(open(meta_path))
        m = evaluate_results(meta["plan"], str(meta_path.parent / "output.json"), str(GT), str(DATA))
        rows.append({"plan": meta["plan"], "model": meta["model"], "cost": meta["cost"],
                     "activity": m["economic_activity_accuracy"], "names": m["company_name_accuracy"],
                     "combined": m["combined_score"], "filter_recall": m["filter_recall"],
                     "filter_precision": m["filter_precision"],
                     "activity_excl_gt_other": m["economic_activity_accuracy_excl_gt_other"]})
    json.dump(rows, open(MODEL_OUT / "scores.json", "w"), indent=1)
    for r in sorted(rows, key=lambda r: r["cost"]):
        print(f"{r['plan']:26s} ${r['cost']:7.4f} | activity {r['activity']:.3f} names {r['names']:.3f} "
              f"combined {r['combined']:.3f} | recall {r['filter_recall']:.3f} precision "
              f"{r['filter_precision']:.3f} activity excl other {r['activity_excl_gt_other']:.3f}")


# ---- phase 4: score ---------------------------------------------------------------------

def score():
    from evaluation.sustainability import evaluate_results
    rows = []
    for meta_path in sorted(OUT.glob("*/run_meta.json")):
        meta = json.load(open(meta_path))
        m = evaluate_results(meta["plan"], str(meta_path.parent / "output.json"), str(GT), str(DATA))
        rows.append({"plan": meta["plan"], "technique": meta["technique"], "cost": meta["cost"],
                     "activity": m["economic_activity_accuracy"], "names": m["company_name_accuracy"],
                     "combined": m["combined_score"]})
    json.dump(rows, open(OUT / "scores.json", "w"), indent=1)
    for r in sorted(rows, key=lambda r: r["cost"]):
        print(f"{r['plan']:16s} ${r['cost']:7.4f} | activity {r['activity']:.2f} names {r['names']:.2f} "
              f"combined {r['combined']:.2f}")


if __name__ == "__main__":
    command, rest = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    if command == "propose":
        propose()
    elif command == "check":
        check()
    elif command == "run" and "--local" in rest:
        run_local([a for a in rest if a != "--local"])
    elif command == "score":
        score()
    elif command == "score-models":
        score_models(*rest[:1])
    else:
        raise SystemExit(__doc__)
