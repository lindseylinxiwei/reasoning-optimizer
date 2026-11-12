"""The agent's (directive, operators) choice is checked before any instantiation is paid for."""
from pathlib import Path
from types import SimpleNamespace

import yaml

from docetl.mcts.mcts import MCTS

REPO = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((REPO / "experiments/reasoning/pipelines/sustainability.yaml").read_text())
F, M, R = (o["name"] for o in CFG["operations"])
NODE = SimpleNamespace(parsed_yaml=CFG, op_dict={o["name"]: o for o in CFG["operations"]})
FAKE = SimpleNamespace(PAIR_TARGETS=MCTS.PAIR_TARGETS, MULTI_TARGETS=MCTS.MULTI_TARGETS)
OFFERED = {(R, "map_reduce_fusion"), (F, "operator_fusion"), (M, "operator_fusion"), (F, "take_head_tail"),
           (M, "take_head_tail"), (F, "doc_summarization"), (M, "doc_summarization")}


def err(name, ops):
    return MCTS._target_error(FAKE, NODE, name, ops, OFFERED)


def test_pairs():
    assert err("map_reduce_fusion", [M, R]) is None            # the choice the old check refused
    assert "exactly two" in err("map_reduce_fusion", [R])
    assert "map followed by" in err("map_reduce_fusion", [R, M]) or "consecutive" in err("map_reduce_fusion", [R, M])
    assert err("operator_fusion", [F, M]) is None
    assert "filter/map" in err("operator_fusion", [M, R])
    assert "consecutive" in err("operator_fusion", [F, R])


def test_single_and_multi():
    assert err("take_head_tail", [F]) is None
    assert "exactly one" in err("take_head_tail", [F, M])
    assert err("doc_summarization", [F, M]) is None
    assert "not offered" in err("doc_summarization", [F, M, R])
    assert "no operator" in err("take_head_tail", ["nope"])
    assert "twice" in err("take_head_tail", [F, F])
