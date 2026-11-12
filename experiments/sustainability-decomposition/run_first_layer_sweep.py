"""Run the original Sustainability pipeline once per Gemini model to seed MOAR's first layer.

Each run is the root pipeline (filter -> map -> reduce on full text, train split) with every
operator on one model. Results go to runs/first_layer_sweep/<config stem>/; scores are written
to runs/first_layer_sweep/results.json with the search's own evaluation function.

Usage (from the repository root):
  python experiments/sustainability-decomposition/run_first_layer_sweep.py run gemini-3.5-flash-lite
  python experiments/sustainability-decomposition/run_first_layer_sweep.py score
  python experiments/sustainability-decomposition/run_first_layer_sweep.py volume-yamls
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

ROOT_YAML = REPO / "experiments/reasoning/pipelines/sustainability.yaml"
DATA = "experiments/reasoning/data/train/sustainability.json"
OUT = HERE / "runs/first_layer_sweep"
VOLUME_DIR = "/mnt/docetl-ro-experiments/outputs/sustainability"
MODELS = [
    "gemini-3.1-flash-lite", "gemini-3-flash-preview", "gemini-3.5-flash-lite", "gemini-3.5-flash",
    "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash", "gemini-3.1-pro-preview", "gemini-4-argon",
]


def stem(model):
    """gemini-3.5-flash-lite -> gemini_3.5_flash_lite_config (the first-layer naming convention)."""
    return model.replace("-", "_") + "_config"


def build_config(model, output_path):
    config = yaml.safe_load(open(ROOT_YAML))
    config["default_model"] = f"gemini/{model}"
    config["datasets"]["company_reports"]["path"] = DATA
    config["pipeline"]["output"]["path"] = str(output_path)
    return config


def run(model):
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
    except Exception as e:  # recorded, not retried here; the driver decides
        meta["error"] = f"{type(e).__name__}: {e}"[:2000]
    meta["seconds"] = round(time.time() - start, 1)
    json.dump(meta, open(out_dir / "run_meta.json", "w"), indent=1)
    print(json.dumps(meta), flush=True)


def score():
    sys.path.insert(0, str(REPO / "experiments/reasoning"))
    from experiments.reasoning.evaluation.utils import get_evaluate_func
    evaluate = get_evaluate_func("sustainability", mode="train")
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
            row["acc"] = metrics["economic_activity_accuracy"]
            row["combined_score"] = metrics["combined_score"]
            row["metrics"] = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
        rows.append(row)
    ok = [r for r in rows if "acc" in r]
    for r in ok:
        r["on_frontier"] = not any(
            o["cost"] <= r["cost"] and o["acc"] >= r["acc"] and (o["cost"] < r["cost"] or o["acc"] > r["acc"])
            for o in ok)
    json.dump(rows, open(OUT / "results.json", "w"), indent=1)
    for r in sorted(rows, key=lambda r: r.get("cost") or 0):
        if "acc" in r:
            print(f"{r['model']:24s} ${r['cost']:8.3f}  acc {r['acc']:.3f}  combined {r['combined_score']:.3f}"
                  f"  {'FRONTIER' if r['on_frontier'] else ''}")
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
        for m in rest:
            run(m)
    elif cmd == "score":
        score()
    elif cmd == "volume-yamls":
        volume_yamls()
