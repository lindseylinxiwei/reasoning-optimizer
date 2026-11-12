"""CUAD Level 3: model sweep over the Level 1+2 Pareto-frontier plans, on the same 40 contracts.

Each cell = one frozen frontier plan (exact config from runs/level1 or runs/level2) with EVERY LLM operator
(map, parallel_map, gated_map, their per-prompt models, and split token-count models) switched to one model.
Jev operators (jev_retrieve, presence_check) and code operators are unchanged.

Plans:  Base (baseline), C1 (Jev gate at 0.2), I3a (isolating, 3 groups), A1 (Jev gate -> isolating),
        J5 (Jev view, ~70%), H5 (head/tail view, ~69%)
Models: gemini-3.1-flash-lite, gemini-3.8-flash, gemini-3.5-flash, gemini-3.1-pro-preview, gemini-4-argon,
        gpt-4.1-mini, gpt-4.1, gpt-5-nano (OpenAI via OPENAI_API_KEY). gemini-3.5-flash-lite is Levels 1-2.

Phases (from the repository root):
  python experiments/cuad-decomposition/sweep_level3.py avail          # tiny call per model (fractions of a cent)
  python experiments/cuad-decomposition/sweep_level3.py smoke          # Base and C1 per model on 2 contracts
  python experiments/cuad-decomposition/sweep_level3.py check          # free: validate + estimate -> check.json
  MODAL_PROFILE=personal modal run experiments/cuad-decomposition/sweep_level3.py --max-usd X --cells-csv a,b
  python experiments/cuad-decomposition/sweep_level3.py score
OpenAI cells run one at a time (one container), with max_threads 4 and a token-rate limit, because the
organization hit gpt-4.1's rate limit before; Gemini cells run in parallel.
"""
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
sys.path[:0] = [str(REPO), str(REPO / "experiments/reasoning"), str(HERE)]

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

L1, L2 = HERE / "runs" / "level1", HERE / "runs" / "level2"
OUT = HERE / "runs" / "level3"
DATA = REPO / "experiments/reasoning/data/train/cuad.json"
GT = REPO / "experiments/reasoning/data/CUAD-master_clauses.csv"
BASE_MODEL = "gemini/gemini-3.5-flash-lite"
LLM_TYPES = {"map", "parallel_map", "filter", "reduce", "resolve", "extract", "gated_map", "rank", "cluster",
             "equijoin"}

# code -> (level, plan name in that level's configs.json)
PLANS = {
    "Base": (1, "baseline"),
    "C1": (2, "C1_gate_jev_threshold_0.2"),
    "I3a": (1, "isolating_subtasks_1"),
    "A1": (2, "A1_gate_jev_isolating"),
    "J5": (1, "jev_5"),
    "H5": (1, "head_tail_5"),
}
CORE = ["Base", "C1", "I3a", "A1"]
# Jev (non-LLM) part of each plan's flash-lite cost; the rest scales with the model.
JEV_USD = {"Base": 0.0, "H5": 0.0, "I3a": 0.0, "C1": 0.329 - 0.283, "A1": 0.329 - 0.283, "J5": 0.036}

MODELS = {
    "g31fl": "gemini/gemini-3.1-flash-lite",
    "g38f": "gemini/gemini-3.8-flash",
    "g35f": "gemini/gemini-3.5-flash",
    "g31pro": "gemini/gemini-3.1-pro-preview",
    "g4argon": "gemini/gemini-4-argon",
    "gpt41mini": "gpt-4.1-mini",
    "gpt41": "gpt-4.1",
    "gpt5nano": "gpt-5-nano",
}
OPENAI = {k for k, m in MODELS.items() if not m.startswith("gemini/")}
EXPENSIVE = {"g35f", "g31pro", "g4argon", "gpt41"}
OPENAI_RATE_LIMITS = {"llm_tokens": [{"count": 150000, "per": 1, "unit": "minute"}]}


def label_free_docs():
    import sweep
    return sweep.label_free_docs()


def base_plan(code):
    level, name = PLANS[code]
    return json.load(open((L1 if level == 1 else L2) / "configs.json"))["plans"][name]


def base_cost(code):
    level, name = PLANS[code]
    rows = json.load(open((L1 if level == 1 else L2) / "scores.json"))
    return next(r for r in rows if r["plan"] == name)["cost"]


def swap_models(ops, model):
    ops = deepcopy(ops)
    for op in ops:
        if op.get("type") in LLM_TYPES and not str(op.get("model", "")).startswith("jev"):
            op["model"] = model
            for sub in op.get("prompts") or []:
                if isinstance(sub, dict) and "model" in sub:
                    sub["model"] = model
        if op.get("type") == "split" and isinstance(op.get("method_kwargs"), dict) and "model" in op["method_kwargs"]:
            op["method_kwargs"]["model"] = model
    return ops


def cells():
    out = {}
    for code in PLANS:
        plan = base_plan(code)
        for tag, model in MODELS.items():
            out[f"{code}__{tag}"] = {"plan": code, "model_tag": tag, "model": model,
                                     "family": plan["family"], "operations": swap_models(plan["operations"], model)}
    return out


def cell_config(name, cell, data_path, output_path):
    import sweep
    cfg = sweep.pipeline_config(name, {"operations": cell["operations"]}, data_path, output_path)
    cfg["default_model"] = cell["model"]
    if cell["model_tag"] in OPENAI:
        cfg["rate_limits"] = OPENAI_RATE_LIMITS
    return cfg


def max_threads(cell):
    return 4 if cell["model_tag"] in OPENAI else 16


# ---- availability and smoke ----------------------------------------------------------------

def avail():
    from litellm import completion
    res = {}
    for tag, model in MODELS.items():
        try:
            r = completion(model=model, messages=[{"role": "user", "content": "Reply with the word ok."}],
                           max_tokens=200 if "gpt-5" in model else 20)
            res[tag] = {"ok": True, "reply": (r.choices[0].message.content or "")[:20]}
        except Exception as err:
            res[tag] = {"ok": False, "error": f"{type(err).__name__}: {str(err)[:300]}"}
        print(tag, res[tag])
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "avail.json", "w"), indent=1)


def smoke_docs():
    """Two contracts near the 25th and 50th percentile of length (a representative input/output mix)."""
    docs = sorted(label_free_docs(), key=lambda d: len(d["document"]))
    return [docs[len(docs) // 4], docs[len(docs) // 2]]


def run_plan(config: dict, docs: list, threads: int = 16) -> dict:
    import tempfile
    from docetl.runner import DSLRunner
    with tempfile.TemporaryDirectory() as tmp:
        data, out = Path(tmp) / "in.json", Path(tmp) / "out.json"
        data.write_text(json.dumps(docs))
        config["datasets"]["contracts_data"]["path"] = str(data)
        config["pipeline"]["output"]["path"] = str(out)
        start = time.time()
        cost = DSLRunner(config, max_threads=threads).load_run_save()
        return {"cost": cost, "seconds": round(time.time() - start, 1), "output": json.loads(out.read_text())}


def smoke(tags=None):
    """Base and C1 on 2 contracts for each model and for flash-lite (the calibration reference)."""
    docs = smoke_docs()
    path = OUT / "smoke.json"
    results = json.load(open(path)) if path.exists() else {}
    avail_ok = json.load(open(OUT / "avail.json")) if (OUT / "avail.json").exists() else {}
    todo = ["ref"] + [t for t in MODELS if (tags is None or t in tags)]
    for tag in todo:
        if tag != "ref" and not avail_ok.get(tag, {}).get("ok"):
            continue
        for code in ("Base", "C1"):
            key = f"{code}__{tag}"
            if key in results and results[key].get("ok"):
                continue
            model = BASE_MODEL if tag == "ref" else MODELS[tag]
            cell = {"plan": code, "model_tag": tag, "model": model,
                    "operations": swap_models(base_plan(code)["operations"], model)}
            try:
                cfg = cell_config(key, cell, DATA, OUT / "smoke_tmp.json")
                r = run_plan(cfg, docs, threads=4)
                rows = r["output"]
                ok_rows = [x for x in rows if isinstance(x.get("clauses"), list)
                           and all(isinstance(c.get("text_spans"), list) for c in x["clauses"])]
                results[key] = {"ok": len(ok_rows) == len(docs), "cost": r["cost"], "seconds": r["seconds"],
                                "entries": [len(x.get("clauses") or []) for x in rows]}
            except Exception as err:
                results[key] = {"ok": False, "error": f"{type(err).__name__}: {str(err)[:600]}"}
            print(key, results[key])
            json.dump(results, open(path, "w"), indent=1)


# ---- check / estimate ------------------------------------------------------------------------

def price_ratio(model):
    """Cost ratio vs flash-lite from list prices, at the 70/30 input/output cost split measured on the baseline."""
    from litellm import model_cost
    ref, m = model_cost[BASE_MODEL], model_cost[model]
    return 0.7 * m["input_cost_per_token"] / ref["input_cost_per_token"] + \
        0.3 * m["output_cost_per_token"] / ref["output_cost_per_token"]


def smoke_ratio(tag):
    """Measured cost ratio vs flash-lite on the same 2 contracts (includes reasoning tokens), LLM part only."""
    path = OUT / "smoke.json"
    if not path.exists():
        return None
    s = json.load(open(path))
    ratios = []
    for code in ("Base", "C1"):
        a, b = s.get(f"{code}__{tag}"), s.get(f"{code}__ref")
        if a and b and a.get("ok") and b.get("ok"):
            jev = 0.0 if code == "Base" else JEV_USD["C1"] * len(smoke_docs()) / 40  # rough Jev share
            ra = max(1e-9, b["cost"] - jev)
            ratios.append(max(0.0, a["cost"] - jev) / ra)
    return sum(ratios) / len(ratios) if ratios else None


def estimate(code, tag):
    llm = max(0.0, base_cost(code) - JEV_USD[code])
    r_price = price_ratio(MODELS[tag])
    r_smoke = smoke_ratio(tag)
    ratio = max(r_price, r_smoke) if r_smoke is not None else r_price * 3  # unmeasured: assume heavy reasoning
    return {"usd": round(llm * ratio + JEV_USD[code], 4), "price_ratio": round(r_price, 2),
            "measured_ratio": None if r_smoke is None else round(r_smoke, 2)}


def check():
    from docetl.runner import DSLRunner
    avail_ok = json.load(open(OUT / "avail.json")) if (OUT / "avail.json").exists() else {}
    rows = {}
    for name, cell in cells().items():
        row = {"plan": cell["plan"], "model": cell["model"], "available": avail_ok.get(cell["model_tag"], {}).get("ok")}
        try:
            DSLRunner(cell_config(name, cell, DATA, OUT / name / "output.json"), max_threads=1)
            row["valid"], row["error"] = True, None
        except Exception as err:
            row["valid"], row["error"] = False, f"{type(err).__name__}: {err}"[:600]
        row.update(estimate(cell["plan"], cell["model_tag"]))
        rows[name] = row
    json.dump({"checked_at": time.strftime("%Y-%m-%d %H:%M:%S"), "cells": rows}, open(OUT / "check.json", "w"), indent=1)
    tags = list(MODELS)
    print(f"{'plan':5s} " + " ".join(f"{t:>10s}" for t in tags))
    for code in PLANS:
        print(f"{code:5s} " + " ".join(
            (f"{rows[f'{code}__{t}']['usd']:>10.2f}" if rows[f"{code}__{t}"]["valid"] else f"{'INVALID':>10s}")
            for t in tags))
    print("ratio  " + " ".join(f"{rows[f'Base__{t}']['measured_ratio'] or rows[f'Base__{t}']['price_ratio']:>10}" for t in tags))
    bad = [n for n, r in rows.items() if not r["valid"]]
    print("invalid:", bad or "none")


# ---- run on Modal ----------------------------------------------------------------------------

def todo_cells(names, max_usd):
    all_cells = cells()
    unknown = set(names) - set(all_cells)
    if unknown:
        raise SystemExit(f"unknown cells: {sorted(unknown)}")
    if not names:
        raise SystemExit("pass --cells-csv explicitly (a budget tier)")
    checked = json.load(open(OUT / "check.json"))["cells"]
    todo = {n: all_cells[n] for n in names if not (OUT / n / "run_meta.json").exists()}
    bad = [n for n in todo if not checked[n]["valid"] or not checked[n]["available"]]
    if bad:
        raise SystemExit(f"invalid or unavailable cells: {bad}")
    est = {n: checked[n]["usd"] for n in todo}
    spent = sum(json.load(open(m))["cost"] for m in OUT.glob("*/run_meta.json"))
    smoke_cost = sum(r.get("cost", 0) for r in json.load(open(OUT / "smoke.json")).values()) \
        if (OUT / "smoke.json").exists() else 0.0
    print(f"already spent ${spent:.2f} + ${smoke_cost:.2f} smoke; {len(todo)} to run, estimated "
          f"${sum(est.values()):.2f}; cap ${max_usd:.2f}")
    if spent + smoke_cost + sum(est.values()) > max_usd:
        raise SystemExit("estimated total exceeds the cap")
    return todo, est


def save_result(name, cell, config, result, est):
    import yaml
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    json.dump(result["output"], open(d / "output.json", "w"))
    yaml.safe_dump(config, open(d / "pipeline.yaml", "w"), sort_keys=False)
    json.dump({"cell": name, "plan": cell["plan"], "model": cell["model"], "family": cell["family"],
               "cost": result["cost"], "estimated_usd": est, "seconds": result["seconds"]},
              open(d / "run_meta.json", "w"), indent=1)
    print(f"{name}: ${result['cost']:.4f} in {result['seconds']}s")


app = modal.App("cuad-model-sweep-level3")


def _skip(path):
    parts = set(Path(path).parts)
    return (bool(parts & {".venv", "__pycache__", "outputs", "runs", "out", "data", "othersystems"})
            or Path(path).name.startswith(".") or ".chroma-" in str(path))


_image = (
    modal.Image.debian_slim(python_version="3.10")
    .uv_sync()
    .add_local_python_source("experiments", ignore=_skip)
    .add_local_python_source("docetl", ignore=_skip)
)


@app.function(image=_image, secrets=[modal.Secret.from_dotenv(REPO)], timeout=2 * 60 * 60, max_containers=16)
def run_cell_remote(config: dict, docs: list, name: str, threads: int) -> dict:
    try:
        return {"cell": name, **run_plan(config, docs, threads)}
    except Exception as exc:
        import traceback
        return {"cell": name, "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}"}


def _handle(result, todo, configs, est, failed):
    name = result.pop("cell")
    if "error" in result:
        failed[name] = result["error"]
        (OUT / name).mkdir(parents=True, exist_ok=True)
        (OUT / name / "error.txt").write_text(result["error"])
        print(f"FAILED {name}: {result['error'][:300]}")
    else:
        save_result(name, todo[name], configs[name], result, est[name])


@app.local_entrypoint()
def modal_main(cells_csv: str = "", max_usd: float = 0.0):
    names = [n for n in cells_csv.split(",") if n]
    docs = label_free_docs()
    todo, est = todo_cells(names, max_usd)
    configs = {n: cell_config(n, c, DATA, OUT / n / "output.json") for n, c in todo.items()}
    failed = {}
    gem = [n for n in todo if todo[n]["model_tag"] not in OPENAI]
    oai = sorted((n for n in todo if todo[n]["model_tag"] in OPENAI), key=lambda n: est[n])
    calls = [(deepcopy(configs[n]), docs, n, max_threads(todo[n])) for n in gem]
    for result in run_cell_remote.starmap(calls, return_exceptions=True, order_outputs=False):
        if isinstance(result, BaseException):
            print(f"FAILED (cell unknown): {str(result)[:300]}")
            continue
        _handle(result, todo, configs, est, failed)
    for n in oai:  # one OpenAI cell at a time: the organization's rate limit is shared
        _handle(run_cell_remote.remote(deepcopy(configs[n]), docs, n, max_threads(todo[n])), todo, configs, est, failed)
    done = [n for n in todo if (OUT / n / "run_meta.json").exists()]
    print(f"saved {len(done)}/{len(todo)}; failed: {sorted(failed) or 'none'}")


# ---- score ------------------------------------------------------------------------------------

def score():
    from evaluation.cuad import evaluate_results
    rows = []
    for meta_path in sorted(OUT.glob("*/run_meta.json")):
        meta = json.load(open(meta_path))
        m = evaluate_results(meta["cell"], str(meta_path.parent / "output.json"), str(GT), str(DATA))
        rows.append({"cell": meta["cell"], "plan": meta["plan"], "model": meta["model"], "cost": meta["cost"],
                     "estimated_usd": meta.get("estimated_usd"), "f1": m["avg_f1"], "f1_macro": m["f1_macro"],
                     "precision": m["precision"], "recall": m["recall"],
                     "negative_task_accuracy": m["negative_task_accuracy"],
                     "failed_contracts": m["failed_contracts"], "missing_contracts": m["missing_contracts"]})
    json.dump(rows, open(OUT / "scores.json", "w"), indent=1)
    for r in sorted(rows, key=lambda r: r["cost"]):
        print(f"{r['cell']:22s} ${r['cost']:7.3f}  F1 {r['f1']:.3f}  P {r['precision']:.2f}  R {r['recall']:.2f}")


if __name__ == "__main__":
    command, rest = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    if command == "avail":
        avail()
    elif command == "smoke":
        smoke(rest or None)
    elif command == "check":
        check()
    elif command == "score":
        score()
    else:
        raise SystemExit(__doc__)
