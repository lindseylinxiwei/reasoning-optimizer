"""Run plan YAMLs on the label-free train or test reports (default: the sustainability_moar_500
frontier plans #64 and #39 on test).

Reuses view_sweep's Modal runner: label-free documents are passed in at call time.
Usage (from the repo root):
  modal run experiments/sustainability-decomposition/run_moar500_test.py --yaml-dir DIR --out-dir DIR
  modal run ... --plans name=file.yaml,name2=file2.yaml --split train
"""
import json
from pathlib import Path

import yaml

from view_sweep import REPO, SPLITS, app, label_free_docs, run_plan_remote

# "train" is the 40-report set the MCTS search ran on (view_sweep's train_50 is a different set).
DATA = {"train": REPO / "experiments/reasoning/data/train/sustainability.json", "test": SPLITS["test"][0]}
PLANS = {"moar500_64": "gemini_flash_preview_config_64.yaml", "moar500_39": "gemini_argon_config_39.yaml"}


@app.local_entrypoint()
def main(yaml_dir: str, out_dir: str, plans: str = "", split: str = "test"):
    chosen = dict(p.split("=", 1) for p in plans.split(",") if p) or PLANS
    docs = label_free_docs(DATA[split])
    calls = [(yaml.safe_load(open(Path(yaml_dir) / f)), docs, name) for name, f in chosen.items()]
    for result in run_plan_remote.starmap(calls, order_outputs=False):
        name = result.pop("plan")
        if "error" in result:
            print(f"FAILED {name}: {result['error'][:1500]}")
            continue
        d = Path(out_dir) / name
        d.mkdir(parents=True, exist_ok=True)
        json.dump(result["output"], open(d / "output.json", "w"))
        json.dump({"plan": name, "split": split, "cost": result["cost"], "seconds": result["seconds"]},
                  open(d / "run_meta.json", "w"))
        print(f"{name} ({split}): ${result['cost']:.4f} in {result['seconds']}s")
