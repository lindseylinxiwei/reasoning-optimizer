"""End-to-end dry runs of the search with a stub agent and stub pipelines (no LLM calls).

Each run is a subprocess, because the harness patches classes globally."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    not (REPO / "experiments/sustainability-decomposition/runs/first_layer_sweep/volume").exists(),
    reason="needs the cached first-layer runs",
)


def run(out: Path, iterations: int, budget: float, crash_at=None, resume=False, guard=None):
    import os
    args = [sys.executable, "-m", "tests.mcts.dryrun_harness", str(out), str(iterations), str(budget),
            str(crash_at) if crash_at else "-"] + (["--resume"] if resume else [])
    env = dict(os.environ, **({"DRYRUN_GUARD": str(guard)} if guard is not None else {}))
    p = subprocess.run(args, cwd=REPO, capture_output=True, text=True, timeout=900, env=env)
    return p.returncode, p.stdout + p.stderr


def rows(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def check_invariants(out: Path):
    ckpt = json.loads((out / "checkpoint.json").read_text())
    ids = [n["id"] for n in ckpt["nodes"]]
    assert len(ids) == len(set(ids)), "duplicate node ids in the tree"
    logged = [r["id"] for r in rows(out / "nodes.jsonl")]
    assert len(logged) == len(set(logged)), "a node id was reused"
    # every evaluated grid child is in the parameter table
    grid_ids = {str(r["id"]) for r in rows(out / "nodes.jsonl") if r["grid"] and r["status"] != "skipped"}
    table_ids = {r["node_id"] for r in rows(out / "param_table.jsonl")}
    assert grid_ids <= table_ids
    # no operator ever gets a second gleaning
    for n in ckpt["nodes"]:
        y = (out / Path(n["yaml"]).name)
        if y.exists() and n["latest_action"] == "gleaning":
            assert n["memo"].count(["gleaning", n["memo"][-1][1]]) == 1
    return ckpt


def test_dry_run_completes(tmp_path):
    out = tmp_path / "out"
    code, log = run(out, 20, 500)
    assert code == 0, log[-3000:]
    ckpt = check_invariants(out)
    assert ckpt["iteration_count"] == 20
    assert "not offered" in log                     # invalid agent choices were caught and retried
    spent = sum(r["usd"] for r in rows(out / "costs.jsonl"))
    assert f"spent={spent:.2f}" in log


def test_cost_guard_stops_cleanly_when_budget_is_tight(tmp_path):
    out = tmp_path / "out"
    code, log = run(out, 100, 60)
    assert code == 0, log[-3000:]
    assert "only produced children the cost guard skipped" in log or "Stopping: spent" in log
    assert sum(r["usd"] for r in rows(out / "costs.jsonl")) <= 60


def test_budget_stop(tmp_path):
    out = tmp_path / "out"
    code, log = run(out, 100, 60, guard=1.0)      # guard only refuses children above the remainder
    assert code == 0, log[-3000:]
    assert "Stopping: spent" in log or "only produced children the cost guard skipped" in log
    spent = sum(r["usd"] for r in rows(out / "costs.jsonl"))
    assert 45 <= spent <= 60 + 3 * 2          # close to the cap; only agent calls in flight can pass it
    ckpt = json.loads((out / "checkpoint.json").read_text())
    assert ckpt["iteration_count"] < 100


def test_crash_then_resume(tmp_path):
    out = tmp_path / "out"
    code, log = run(out, 20, 500, crash_at=30)
    assert code == 17, log[-3000:]
    before = json.loads((out / "checkpoint.json").read_text())
    assert 0 < before["iteration_count"] < 20
    spent_at_crash = sum(r["usd"] for r in rows(out / "costs.jsonl"))

    code, log = run(out, 20, 500, resume=True)
    assert code == 0, log[-3000:]
    assert "Resumed from" in log
    after = check_invariants(out)
    assert after["iteration_count"] == 20
    old_ids = {n["id"] for n in before["nodes"]}
    assert old_ids <= {n["id"] for n in after["nodes"]}         # nothing from before the crash is lost
    assert sum(r["usd"] for r in rows(out / "costs.jsonl")) >= spent_at_crash


def test_fresh_run_refuses_a_used_directory(tmp_path):
    out = tmp_path / "out"
    assert run(out, 3, 500)[0] == 0
    code, log = run(out, 3, 500)
    assert code != 0 and "already holds a search" in log


def test_crash_in_the_first_iteration_can_resume(tmp_path):
    out = tmp_path / "out"
    code, log = run(out, 5, 500, crash_at=1)
    assert code == 17, log[-3000:]
    code, log = run(out, 5, 500, resume=True)
    assert code == 0, log[-3000:]
    assert check_invariants(out)["iteration_count"] == 5


def test_jev_is_only_applied_on_inputs_that_fit(tmp_path):
    """jev_classify on a long input creates a shortening parent and the Jev child in one iteration."""
    import yaml
    from docetl.mcts.jev_fit import SHORTENING_FOR_JEV, fits_jev
    out = tmp_path / "out"
    code, log = run(out, 40, 500)
    assert code == 0, log[-3000:]
    ckpt = check_invariants(out)
    by_id = {n["id"]: n for n in ckpt["nodes"]}
    jev_nodes = [n for n in ckpt["nodes"] if n["latest_action"] == "jev_classify"]
    assert "Jev coupling on" in log
    assert jev_nodes, "no Jev child was created in 40 iterations"
    for n in jev_nodes:
        cfg = yaml.safe_load((out / Path(n["yaml"]).name).read_text())
        jev_ops = [o["name"] for o in cfg["operations"] if o.get("model") == "jev"]
        assert jev_ops and all(fits_jev(cfg["operations"], o)[0] for o in jev_ops), n["yaml"]
    coupled = [n for n in jev_nodes if by_id.get(n["parent"], {}).get("latest_action") in SHORTENING_FOR_JEV]
    assert coupled, "no coupled shortening -> Jev pair"
