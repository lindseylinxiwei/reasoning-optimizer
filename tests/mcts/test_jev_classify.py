"""jev_classify: validation of the agent's rewrite, the rewritten operation, and the search hooks."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from docetl.mcts.param_grid import GRIDS
from docetl.reasoning_optimizer.directives import ALL_DIRECTIVES, DIRECTIVE_REGISTRY
from docetl.reasoning_optimizer.directives.jev_classify import (
    SHORTENING_REWRITES, JevClassifyDirective, jev_compatible)
from docetl.reasoning_optimizer.instantiate_schemas import JevClassifyInstantiateSchema

REPO = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((REPO / "experiments/reasoning/pipelines/sustainability.yaml").read_text())
OPS = CFG["operations"]
FILTER, MAP, REDUCE = OPS[0], OPS[1], OPS[2]
LABELS = MAP["output"]["schema"]["economic_activity"][5:-1].split(", ")
D = JevClassifyDirective()


def filter_rw(**kw):
    base = dict(rationale="Type is a classification.",
                prompt="Is this a sustainability report?\nOutput field is_sustainability_report: true if it is "
                       "a sustainability or ESG report, false otherwise.\n{{ input.tot_text_raw }}")
    base.update(kw)
    return JevClassifyInstantiateSchema(**base)


def map_rw():
    lines = "\n".join(f"- {l}: when the company's main activity is {l}" for l in LABELS)
    return JevClassifyInstantiateSchema(
        rationale="Sector is a classification.",
        prompt=f"Classify the company's sector.\nOutput field economic_activity:\n{lines}\n{{{{ input.tot_text_raw }}}}")


def test_registered_with_no_parameter():
    assert "jev_classify" in DIRECTIVE_REGISTRY and any(d.name == "jev_classify" for d in ALL_DIRECTIVES)
    assert "jev_classify" not in GRIDS
    assert set(JevClassifyInstantiateSchema.model_fields) == {"prompt", "rationale"}
    assert jev_compatible(FILTER) and jev_compatible(MAP) and not jev_compatible(REDUCE)


def test_agent_is_told_the_limit_and_how_to_shorten():
    text = D.to_string_for_instantiate(FILTER) + D.to_string_for_plan()
    assert "32K tokens" in text and "does not truncate" in text
    for name in ("take_head_tail", "doc_chunking_topk", "jev_retrieval", "doc_summarization",
                 "deterministic_doc_compression", "doc_chunking"):
        assert name in text
    assert len(SHORTENING_REWRITES) == 4


@pytest.mark.parametrize("kw,msg", [
    ({"prompt": "Decide.\n{{ input.tot_text_raw }}"}, "Output field is_sustainability_report"),
    ({"prompt": "Output field is_sustainability_report: x {{ input.tot_text_raw }} {{ input.id }}"}, "does not"),
    ({"prompt": "Output field is_sustainability_report: true if so."}, "must include the input"),
    ({"prompt": "Output field is_sustainability_report: " + "word " * 3000 + "{{ input.tot_text_raw }}"}, "under 2000"),
])
def test_invalid_rewrites_are_rejected(kw, msg):
    with pytest.raises(ValueError, match=msg):
        filter_rw(**kw).check_against(FILTER)


def test_enum_labels_must_all_be_described():
    rw = map_rw()
    rw.check_against(MAP)
    bad = rw.model_copy(update={"prompt": rw.prompt.replace("accommodation_food", "hotels")})
    with pytest.raises(ValueError, match="accommodation_food"):
        bad.check_against(MAP)


def test_string_outputs_are_rejected():
    with pytest.raises(ValueError, match="boolean and enum"):
        filter_rw().check_against(REDUCE)


def test_apply_only_switches_the_operation():
    new = D.apply(deepcopy(OPS), FILTER["name"], filter_rw())
    assert [o["name"] for o in new] == [o["name"] for o in OPS]        # nothing inserted
    op = new[0]
    assert op["model"] == "jev" and op["type"] == "filter" and "litellm_completion_kwargs" not in op
    assert op["prompt"] == filter_rw().prompt
    assert op["output"]["schema"] == FILTER["output"]["schema"]
    assert new[1:] == OPS[1:]


def test_rewritten_operations_pass_docetl_validation():
    from docetl.operations.filter import FilterOperation
    from docetl.operations.map import MapOperation
    new = D.apply(deepcopy(OPS), MAP["name"], map_rw())
    new = D.apply(new, FILTER["name"], filter_rw())
    runner = SimpleNamespace(config={}, default_model="gemini/gemini-3.5-flash-lite", console=None,
                             max_threads=1, is_cancelled=False, status=None)
    FilterOperation(runner, new[0], default_model="gemini/gemini-3.5-flash-lite", max_threads=1)
    MapOperation(runner, new[1], default_model="gemini/gemini-3.5-flash-lite", max_threads=1)


def test_search_offers_jev_only_where_it_fits():
    from docetl.mcts.mcts import MCTS
    acts = set(ALL_DIRECTIVES)

    def excluded(op):
        node = SimpleNamespace(op_dict={op["name"]: op})
        fake = SimpleNamespace(available_actions=acts, TARGET_TYPES=MCTS.TARGET_TYPES,
                               JEV_OP_ALLOWED=MCTS.JEV_OP_ALLOWED)
        return {d.name for d in MCTS._gleaning_exclusions(fake, node, op["name"])}

    assert "jev_classify" not in excluded(FILTER) and "jev_classify" not in excluded(MAP)
    assert "jev_classify" in excluded(REDUCE)
    jev_op = D.apply(deepcopy(OPS), FILTER["name"], filter_rw())[0]
    allowed = {d.name for d in acts} - excluded(jev_op)
    assert allowed == MCTS.JEV_OP_ALLOWED & {d.name for d in acts}
    assert "gleaning" not in allowed and "jev_classify" not in allowed and "doc_chunking" not in allowed
