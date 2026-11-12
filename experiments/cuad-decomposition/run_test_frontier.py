"""Run CUAD plans (search frontier + first-layer baselines) once on the 100-contract test split.

Each plan YAML is rewritten to read experiments/reasoning/data/test/cuad.json and write its
output to the volume under outputs/<experiment>/. Runs in parallel on Modal, scores each output
with the search's own CUAD evaluator (official matching, avg_f1 = pooled F1) in test mode, and
writes outputs/<experiment>/results.json. The orchestrator is spawned, so the run continues if
this client disconnects.

Usage (from the repository root):
  modal run --detach experiments/cuad-decomposition/run_test_frontier.py --plans-dir DIR [--experiment NAME]
DIR holds the plan YAMLs; files named baseline_*.yaml are labelled as baselines.
"""
import json
import time
import traceback
from pathlib import Path

import modal
import yaml

from experiments.reasoning.utils import app, eval_image, volume, VOLUME_MOUNT_PATH

TEST_DATA = "experiments/reasoning/data/test/cuad.json"


def _to_test(config: dict, out_path: str) -> dict:
    for ds in config["datasets"].values():
        if ds.get("type") == "file" and "cuad.json" in ds.get("path", ""):
            ds["path"] = TEST_DATA
    config["pipeline"]["output"]["path"] = out_path
    config.get("pipeline", {}).get("output", {}).pop("intermediate_dir", None)
    config["bypass_cache"] = True
    return config


@app.function(image=eval_image, secrets=[modal.Secret.from_dotenv()],
              volumes={VOLUME_MOUNT_PATH: volume}, timeout=4 * 3600)
def run_plan(name: str, yaml_text: str, experiment: str) -> dict:
    from docetl.runner import DSLRunner
    from experiments.reasoning.evaluation.utils import get_evaluate_func

    out_dir = Path(VOLUME_MOUNT_PATH) / "outputs" / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"
    config = _to_test(yaml.safe_load(yaml_text), str(out_path))
    (out_dir / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    row = {"name": name}
    start = time.time()
    try:
        row["cost"] = DSLRunner(config, max_threads=16).load_run_save()
        metrics = get_evaluate_func("cuad", mode="test")(name, str(out_path))
        row["f1"] = metrics["avg_f1"]
        row["metrics"] = {k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"[:2000]
        row["traceback"] = traceback.format_exc()[-4000:]
    row["seconds"] = round(time.time() - start, 1)
    (out_dir / f"{name}.result.json").write_text(json.dumps(row, indent=1, default=str))
    volume.commit()
    print(json.dumps({k: row.get(k) for k in ("name", "cost", "f1", "error", "seconds")}), flush=True)
    return row


@app.function(image=eval_image, volumes={VOLUME_MOUNT_PATH: volume}, timeout=5 * 3600)
def run_all(plans: dict, experiment: str) -> list:
    rows = list(run_plan.starmap([(n, t, experiment) for n, t in plans.items()], return_exceptions=True))
    rows = [r if isinstance(r, dict) else {"error": repr(r)} for r in rows]
    out = Path(VOLUME_MOUNT_PATH) / "outputs" / experiment / "results.json"
    out.write_text(json.dumps(rows, indent=1, default=str))
    volume.commit()
    total = sum(r.get("cost") or 0 for r in rows)
    for r in sorted(rows, key=lambda r: r.get("cost") or 0):
        print(f"{r.get('name')!s:45s} ${r.get('cost') or 0:6.2f}  F1 {r.get('f1') or 0:.3f}  {r.get('error', '')[:120]}")
    print(f"TOTAL ${total:.2f}", flush=True)
    return rows


@app.local_entrypoint()
def main(plans_dir: str, experiment: str = "cuad_moar_300_test"):
    plans = {p.stem: p.read_text() for p in sorted(Path(plans_dir).glob("*.yaml"))}
    print(f"{len(plans)} plans: {', '.join(plans)}")
    call = run_all.spawn(plans, experiment)
    print(f"Spawned on Modal (function call {call.object_id}); results go to outputs/{experiment}/")
