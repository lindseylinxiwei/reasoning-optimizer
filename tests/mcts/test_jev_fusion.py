"""jev_fusion: two consecutive Jev classifications become one Jev call."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from docetl.mcts.mcts import MCTS
from docetl.reasoning_optimizer.directives import ALL_DIRECTIVES, DIRECTIVE_REGISTRY
from docetl.reasoning_optimizer.directives.jev_fusion import JevFusionDirective

D = JevFusionDirective()
F = {"name": "f", "type": "filter", "model": "jev", "output": {"schema": {"is_report": "boolean"}},
     "prompt": "Is it a report?\n\nOutput field is_report: true if a report.\n\nDocument text:\n{{ input.text }}"}
M = {"name": "m", "type": "map", "model": "jev", "output": {"schema": {"sector": "enum[a, b]"}},
     "prompt": "Pick the sector.\n\nOutput field sector:\n- a: when a\n- b: when b\n\nDocument text:\n{{ input.text }}"}
R = {"name": "r", "type": "reduce", "reduce_key": "sector", "prompt": "{{ inputs }}",
     "output": {"schema": {"s": "string"}}}


def test_filter_then_map_becomes_one_jev_map_and_a_code_filter():
    ops = D.apply(deepcopy([F, M, R]), ["f", "m"])
    assert [o["type"] for o in ops] == ["map", "code_filter", "reduce"]
    fused = ops[0]
    assert fused["model"] == "jev" and fused["output"]["schema"] == {"is_report": "boolean", "sector": "enum[a, b]"}
    p = fused["prompt"]
    assert p.count("{{ input.text }}") == 1 and p.count("Document text:") == 1
    assert "Output field is_report:" in p and "Output field sector:" in p and "- b: when b" in p
    assert "is_report" in ops[1]["code"]


def test_map_map_needs_no_code_filter():
    m2 = {**M, "name": "m2", "output": {"schema": {"size": "enum[s, l]"}},
          "prompt": "Size?\nOutput field size: s or l\n{{ input.text }}"}
    ops = D.apply(deepcopy([M, m2]), ["m", "m2"])
    assert [o["type"] for o in ops] == ["map"]


@pytest.mark.parametrize("ops,targets,msg", [
    ([F, R], ["f", "r"], "fuses filters and maps"),
    ([{**F, "model": "gemini/gemini-3-flash-preview"}, M], ["f", "m"], "does not run on Jev"),
    ([F, {**M, "prompt": M["prompt"].replace("input.text", "input.other")}], ["f", "m"], "different inputs"),
    ([F, R, M], ["f", "m"], "consecutive"),
    ([F, {**M, "output": {"schema": {"sector": "string"}}}], ["f", "m"], "not boolean or enum"),
])
def test_invalid_pairs_are_rejected(ops, targets, msg):
    assert msg in (D.check(ops, targets) or "")


def test_registered_and_offered_only_on_jev_ops():
    assert "jev_fusion" in DIRECTIVE_REGISTRY and "jev_fusion" in MCTS.JEV_OP_ALLOWED
    assert "jev_fusion" in MCTS.PAIR_TARGETS
    acts = set(ALL_DIRECTIVES)

    def excluded(op):
        fake = SimpleNamespace(available_actions=acts, TARGET_TYPES=MCTS.TARGET_TYPES, JEV_OP_ALLOWED=MCTS.JEV_OP_ALLOWED)
        return {d.name for d in MCTS._gleaning_exclusions(fake, SimpleNamespace(op_dict={op["name"]: op}), op["name"])}

    assert "jev_fusion" not in excluded(F) and "jev_fusion" not in excluded(M)
    assert "jev_fusion" in excluded({**M, "model": "gemini/gemini-3-flash-preview"})


def test_search_target_check_uses_the_fusion_rules():
    node = SimpleNamespace(parsed_yaml={"operations": [F, M, R]}, op_dict={o["name"]: o for o in (F, M, R)})
    fake = SimpleNamespace(PAIR_TARGETS=MCTS.PAIR_TARGETS, MULTI_TARGETS=MCTS.MULTI_TARGETS)
    offered = {("f", "jev_fusion"), ("m", "jev_fusion")}
    assert MCTS._target_error(fake, node, "jev_fusion", ["f", "m"], offered) is None
    assert "two" in MCTS._target_error(fake, node, "jev_fusion", ["f"], offered)
