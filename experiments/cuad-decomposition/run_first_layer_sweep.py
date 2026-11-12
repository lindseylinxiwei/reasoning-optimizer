"""Run the original CUAD pipeline once per Gemini model to seed MOAR's first layer.

Same procedure as experiments/sustainability-decomposition/run_first_layer_sweep.py: each run is
the root pipeline (one extraction map, train split of 40 contracts) with every operator on one
model. The pipeline is fixed (no agent-written code). Results go to runs/first_layer_sweep/<stem>/;
`score` writes runs/first_layer_sweep/results.json with the search's own evaluation function
(the official-match CUAD evaluator, avg_f1 is the search reward).

Usage (from the repository root):
  python experiments/cuad-decomposition/run_first_layer_sweep.py run [--cap 30] [model ...]
  python experiments/cuad-decomposition/run_first_layer_sweep.py score
  python experiments/cuad-decomposition/run_first_layer_sweep.py volume-yamls
"""
import json
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

ROOT_YAML = REPO / "experiments/reasoning/pipelines/cuad.yaml"
DATA = "experiments/reasoning/data/train/cuad.json"
OUT = HERE / "runs/first_layer_sweep"
VOLUME_DIR = "/mnt/docetl-ro-experiments/outputs/cuad"
# Cheapest first (LiteLLM prices), so the budget check sees real costs before the expensive runs.
MODELS = [
    "gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3-flash-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash", "gemini-3.5-flash", "gemini-4-argon", "gemini-3.1-pro-preview",
]


def stem(model):
    """gemini-3.5-flash-lite -> gemini_3.5_flash_lite_config (the first-layer naming convention)."""
    return model.replace("-", "_") + "_config"


def build_config(model, output_path):
    config = yaml.safe_load(open(ROOT_YAML))
    config["default_model"] = f"gemini/{model}"
    config["datasets"]["contracts_data"]["path"] = DATA
    config["pipeline"]["output"]["path"] = str(output_path)
    return config


def price_estimate(model, out_tokens_per_doc=4000):
    """Rough cost from input tokens and an assumed output length (used only for the budget check)."""
    import tiktoken
    from litellm import model_cost
    enc = tiktoken.get_encoding("cl100k_base")
    docs = json.load(open(REPO / DATA))
    prompt = yaml.safe_load(open(ROOT_YAML))["operations"][0]["prompt"]
    tin = sum(len(enc.encode(d["document"], disallowed_special=())) for d in docs) + len(enc.encode(prompt)) * len(docs)
    info = model_cost[f"gemini/{model}"]
    return tin * info["input_cost_per_token"] + len(docs) * out_tokens_per_doc * info["output_cost_per_token"]


def run_one(model):
    from docetl.runner import DSLRunner
    out_dir = OUT / stem(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{stem(model)}.json"
    config = build_config(model, output)
    yaml.safe_dump(config, open(out_dir / f"{stem(model)}.yaml", "w"), sort_keys=False)
    start = time.time()
    meta = {"model": model}
    try:
        meta["cost"] = DSLRunner(config, max_threads=16).load_run_save()
    except Exception as e:  # recorded, not retried here
        meta["error"] = f"{type(e).__name__}: {e}"[:2000]
    meta["seconds"] = round(time.time() - start, 1)
    json.dump(meta, open(out_dir / "run_meta.json", "w"), indent=1)
    print(json.dumps(meta), flush=True)
    return meta


def run(models, cap):
    """Runs models in order; stops before a model if spend so far plus the projected cost of the
    remaining models (estimates scaled by the observed cost/estimate ratio) would exceed `cap`."""
    models = models or MODELS
    est = {m: price_estimate(m) for m in models}
    spent, ratio = 0.0, 1.0
    for k, m in enumerate(models):
        meta_path = OUT / stem(m) / "run_meta.json"
        if meta_path.exists() and "error" not in json.load(open(meta_path)):
            done = json.load(open(meta_path))
            spent += done["cost"]
            ratio = max(ratio, done["cost"] / est[m])
            print(f"skip {m}: already run (${done['cost']:.2f})", flush=True)
            continue
        projected = spent + ratio * sum(est[x] for x in models[k:])
        if projected > cap:
            print(f"STOP before {m}: spent ${spent:.2f}, projected total ${projected:.2f} > cap ${cap:.2f}", flush=True)
            return
        meta = run_one(m)
        if "cost" in meta:
            spent += meta["cost"]
            ratio = max(ratio, meta["cost"] / est[m])
        print(f"spent so far ${spent:.2f} (cost/estimate ratio {ratio:.2f})", flush=True)


def score():
    from experiments.reasoning.evaluation.utils import get_evaluate_func
    evaluate = get_evaluate_func("cuad", mode="train")
    rows = []
    for model in MODELS:
        out_dir = OUT / stem(model)
        meta_path = out_dir / "run_meta.json"
        if not meta_path.exists():
            continue
        meta = json.load(open(meta_path))
        row = {"model": model, "config": f"{stem(model)}.yaml", **meta}
        output = out_dir / f"{stem(model)}.json"
        if "error" not in meta and output.exists():
            metrics = evaluate(stem(model), str(output))
            row["acc"] = metrics["avg_f1"]
            row["metrics"] = {k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}
        rows.append(row)
    ok = [r for r in rows if "acc" in r]
    for r in ok:
        r["on_frontier"] = not any(
            o["cost"] <= r["cost"] and o["acc"] >= r["acc"] and (o["cost"] < r["cost"] or o["acc"] > r["acc"])
            for o in ok)
    json.dump(rows, open(OUT / "results.json", "w"), indent=1)
    for r in sorted(rows, key=lambda r: r.get("cost") or 0):
        if "acc" in r:
            m = r["metrics"]
            print(f"{r['model']:24s} ${r['cost']:7.3f}  F1 {r['acc']:.3f}  P {m['avg_precision']:.3f}  "
                  f"R {m['avg_recall']:.3f}  {'FRONTIER' if r['on_frontier'] else ''}")
        else:
            print(f"{r['model']:24s} FAILED: {r.get('error', 'no output')[:150]}")


def volume_yamls():
    """Copies of each config whose output path points at the Modal volume, for upload."""
    dest = OUT / "volume"
    dest.mkdir(exist_ok=True)
    for model in MODELS:
        out_dir = OUT / stem(model)
        output = out_dir / f"{stem(model)}.json"
        if not output.exists():
            continue
        config = build_config(model, f"{VOLUME_DIR}/{stem(model)}.json")
        yaml.safe_dump(config, open(dest / f"{stem(model)}.yaml", "w"), sort_keys=False)
        (dest / output.name).write_bytes(output.read_bytes())


if __name__ == "__main__":
    load_dotenv(REPO / ".env")
    cmd, *rest = sys.argv[1:]
    if cmd == "run":
        cap = 30.0
        if "--cap" in rest:
            i = rest.index("--cap")
            cap = float(rest[i + 1])
            rest = rest[:i] + rest[i + 2:]
        run(rest, cap)
    elif cmd == "score":
        score()
    elif cmd == "volume-yamls":
        volume_yamls()
