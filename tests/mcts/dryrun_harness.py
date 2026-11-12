"""Dry run of the MCTS search with a stub agent and stub pipelines: no LLM calls, no cost.

The search code is real: selection, expansion bookkeeping, the parameter grid, the cost guard,
the budget stop, the frontier, the logs, the checkpoint and resume. Stubbed: the agent's choice
(picks a random offered action), directive instantiation (deterministic edits), pipeline
execution (cost from the plan) and scoring (score from the plan).

Run as a script to drive a search in a subprocess (used by the crash/resume test):
    python -m tests.mcts.dryrun_harness OUT_DIR ITERATIONS BUDGET [CRASH_AT_SIMULATION] [--resume]
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import sys
import threading
from copy import deepcopy
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
SWEEP = REPO / "experiments/sustainability-decomposition/runs/first_layer_sweep/volume"
ACTIONS = ("take_head_tail", "gleaning", "clarify_instructions", "chaining", "jev_retrieval", "jev_classify")
COUPLINGS = []

_state = threading.local()
_RNG = random.Random(0)
CALLS = {"options": [], "simulations": 0}
_lock = threading.Lock()


def _h(text: str) -> float:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def prepare_volume(tmp: Path) -> Path:
    """Local copy of the first-layer runs with output paths pointing at the copy."""
    vol = tmp / "volume"
    dst = vol / "outputs" / "sustainability"
    dst.mkdir(parents=True, exist_ok=True)
    for f in SWEEP.iterdir():
        if f.suffix == ".json":
            shutil.copy(f, dst / f.name)
    for f in SWEEP.glob("*.yaml"):
        cfg = yaml.safe_load(f.read_text())
        cfg["pipeline"]["output"]["path"] = str(dst / (f.stem + ".json"))
        (dst / f.name).write_text(yaml.safe_dump(cfg, sort_keys=False))
    return vol


def install_stubs(crash_at_simulation: int | None = None):
    import docetl.mcts.mcts as M
    from docetl.mcts.Node import Node
    from docetl.reasoning_optimizer.directives import ALL_DIRECTIVES, Directive
    from docetl.reasoning_optimizer.directives.change_model_cost import ChangeModelCostDirective, MODEL_COSTS
    from docetl.reasoning_optimizer.instantiate_schemas import (
        JevRetrievalInstantiateSchema, JevRetrievalTargetSchema, TakeHeadTailInstantiateSchema)

    costs = MODEL_COSTS["sustainability"]

    def op_model(cfg, op):
        return str(op.get("model") or cfg.get("default_model")).replace("gemini/", "")

    def execute_plan(self, max_threads=None):
        with _lock:
            CALLS["simulations"] += 1
            n = CALLS["simulations"]
        if crash_at_simulation is not None and n == crash_at_simulation:
            print(f"DRYRUN: simulated crash at simulation {n}", flush=True)
            os._exit(17)
        cfg = self.parsed_yaml
        text = json.dumps(cfg["operations"], sort_keys=True)
        if "FAIL_ME" in text:
            self.cost = -1
            raise RuntimeError("stub pipeline failure")
        cost = 0.0
        for op in cfg["operations"]:
            share = 0.45 if op["type"] in ("filter", "map") else 0.1
            frac = 1.0
            for o in cfg["operations"]:
                if o.get("type") == "code_map" and "head_words" in json.dumps(o):
                    frac = min(1.0, 1.39 * int(o.get("_stub_head", 16000)) / 33093)
            per = 0.05 if op_model(cfg, op) == "jev" else costs.get(op_model(cfg, op), 2.0)
            cost += per * share * frac
        self.cost = round(cost * (1.2 if "gleaning" in text else 1.0), 4)
        return self.cost, []

    def evaluate_node(self, node):
        if node.cost == -1:
            return float("-inf")
        text = json.dumps(node.parsed_yaml["operations"], sort_keys=True)
        model = str(node.parsed_yaml.get("default_model")).replace("gemini/", "")
        base = M.MODEL_STATS["sustainability"].get(model, {"acc": 0.7})["acc"]
        return round(min(1.0, base + 0.06 * (_h(text) - 0.5)), 4)

    def prompt(self, node, action_options, input_query):
        _state.options = list(action_options)
        with _lock:
            CALLS["options"].append((node.get_id(), json.dumps(node.parsed_yaml["operations"]), list(action_options)))
        return "stub prompt", "stub prompt"

    class _Msg:
        def __init__(self, content):
            self.content = content

    class _Resp:
        def __init__(self, content):
            self.choices = [type("C", (), {"message": _Msg(content)})()]

    def agent_completion(model, messages, response_format=None, **kw):
        if getattr(response_format, "__name__", "") == "ShorteningChoice":
            names = [n for n in ("take_head_tail", "jev_retrieval") if n in messages[-1]["content"]]
            with _lock:
                COUPLINGS.append(names)
                pick = _RNG.choice(names) if names else "nothing"
            return _Resp(json.dumps({"directive": pick, "reason": "stub"})), 0.002
        options = _state.options
        with _lock:
            invalid = _RNG.random() < 0.15      # an invalid choice now and then, to exercise validation
            op, name = _RNG.choice(options)
        if invalid:
            return _Resp(json.dumps({"directive": name, "operators": ["no_such_operator"]})), 0.002
        return _Resp(json.dumps({"directive": name, "operators": [op]})), 0.002

    def generic_instantiate(self, operators, target_ops, agent_llm, message_history=[], **kw):
        ops = deepcopy(operators)
        for op in ops:
            if op["name"] in target_ops:
                if isinstance(self, ChangeModelCostDirective):
                    op["model"] = f"gemini/{self.target_model}"
                elif self.name == "gleaning":
                    op["gleaning"] = {"validation_prompt": "Check.", "num_rounds": 1}
                else:
                    op["prompt"] = op.get("prompt", "") + f"\n# {self.name} {len(message_history)}"
        return ops, message_history, 0.01

    def head_tail_instantiate(self, operators, target_ops, agent_llm, message_history=[], **kw):
        rw = TakeHeadTailInstantiateSchema(name=f"head_{target_ops[0]}", document_key="tot_text_raw",
                                           head_words=random.Random(len(message_history)).choice([700, 2000, 9000]),
                                           tail_words=0)
        ops = self.apply(operators, target_ops[0], rw)
        for op in ops:
            if op.get("type") == "code_map":
                op["_stub_head"] = rw.head_words
        return ops, message_history, 0.01

    def jev_instantiate(self, operators, target_ops, agent_llm, message_history=[], **kw):
        rw = JevRetrievalInstantiateSchema(
            name="jev_business", document_key="tot_text_raw", evidence_chars=10000,
            targets=[JevRetrievalTargetSchema(name="business", description="What the company does.")])
        return self.apply(operators, target_ops[0], rw, None), message_history, 0.01

    def jev_classify_instantiate(self, operators, target_ops, agent_llm, message_history=[], **kw):
        from docetl.reasoning_optimizer.instantiate_schemas import JevClassifyInstantiateSchema
        op = [o for o in operators if o["name"] == target_ops[0]][0]
        lines = []
        for f, kind in op["output"]["schema"].items():
            labels = kind[5:-1] if str(kind).startswith("enum[") else "true or false"
            lines.append(f"Output field {f}: {labels}")
        rw = JevClassifyInstantiateSchema(prompt=op["prompt"] + "\n" + "\n".join(lines), rationale="stub")
        return self.apply(operators, target_ops[0], rw), message_history, 0.01

    M.Node.execute_plan = execute_plan
    M.MCTS.evaluate_node = evaluate_node
    M.MCTS.expansion_prompt_acc = prompt
    M.MCTS.expansion_prompt_cost = prompt
    M.agent_completion = agent_completion
    for d in ALL_DIRECTIVES:
        cls = type(d)
        if d.name == "take_head_tail":
            cls.instantiate = head_tail_instantiate
        elif d.name == "jev_retrieval":
            cls.instantiate = jev_instantiate
        elif d.name == "jev_classify":
            cls.instantiate = jev_classify_instantiate
        else:
            cls.instantiate = generic_instantiate
    ChangeModelCostDirective.instantiate = generic_instantiate
    return M


def build(out_dir: Path, volume: Path, iterations: int, budget: float, resume: bool = False,
          concurrent: int = 3, crash_at_simulation: int | None = None):
    M = install_stubs(crash_at_simulation)
    from docetl.reasoning_optimizer.directives import ALL_DIRECTIVES
    M.VOLUME_MOUNT_PATH = str(volume)
    random.seed(0)
    actions = {d for d in ALL_DIRECTIVES if d.name in ACTIONS}
    root = REPO / "experiments/reasoning/pipelines/sustainability.yaml"
    mcts = M.MCTS(
        root_yaml_path=str(root), available_actions=actions, sample_input=[{"id": 1, "tot_text_raw": "x"}],
        dataset_stats="", dataset_name="sustainability", max_iterations=iterations,
        output_dir=str(out_dir), build_first_layer=True, max_budget_usd=budget, resume=resume,
        cost_guard_fraction=float(os.environ.get("DRYRUN_GUARD", "0.10")),
    )
    mcts.max_concurrent_agents = concurrent
    return mcts


if __name__ == "__main__":
    out, iters, budget = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    crash = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "-" else None
    resume = "--resume" in sys.argv
    os.chdir(REPO)
    vol = Path(out).parent / "volume"
    if not vol.exists():
        prepare_volume(Path(out).parent)
    m = build(out, vol, iters, budget, resume=resume, crash_at_simulation=crash)
    m.search()
    print(f"DRYRUN DONE iterations={m.iteration_count} spent={m.total_search_cost:.2f}")
