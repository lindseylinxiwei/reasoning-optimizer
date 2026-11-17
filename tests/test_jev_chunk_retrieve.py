"""Offline tests for the jev_chunk_retrieve operator and the jev_retrieval rewrite (Jev on chunks; no network)."""
import json
import re

import pytest
import yaml

import docetl.retrieval.jev_backend as jev_backend
from docetl.operations.jev_chunk_retrieve import chunk_bounds, kept_chunks, payload, plan
from docetl.reasoning_optimizer.directives import DIRECTIVE_REGISTRY, JevRetrievalDirective
from docetl.reasoning_optimizer.instantiate_schemas import JevRetrievalInstantiateSchema
from docetl.runner import DSLRunner

CHUNK = re.compile(r"⟦CHUNK\|BEGIN (C\d+)⟧\n(.*?)\n⟦CHUNK\|END \1⟧", re.S)
SUSTAINABILITY = "experiments/reasoning/pipelines/sustainability.yaml"
TARGET = {"name": "business_description", "description": "Passages describing what the company does."}


def scripted_jev(keywords, calls):
    """Score chunk Cn 0.9 for target Tj if it contains keywords[j], else 0.1."""

    def request(payload, timeout):
        calls.append(payload)
        chunks = dict(CHUNK.findall(payload["state"]["source_text"]))
        answers = {}
        for key in payload["questions"]:
            chunk, target = key.rsplit("_T", 1)
            hit = keywords[int(target)] in chunks.get(chunk, "")
            answers[key] = {"type": "noul", "noul": 0.9 if hit else 0.1}
        return {"model": payload["model"], "answers": answers,
                "usage": {"input_tokens": len(json.dumps(payload)) // 4}}

    return request


def long_doc(needle_at=40):
    paragraphs = [f"Section {i}. Routine sustainability boilerplate paragraph number {i}. " * 6 for i in range(80)]
    paragraphs[needle_at] = "ACME Corp is a leading manufacturer of industrial pumps and valves in Ohio. " * 3
    return "\n\n".join(paragraphs)


def run_pipeline(tmp_path, docs, **overrides):
    data = tmp_path / "in.json"
    data.write_text(json.dumps(docs))
    op = {"name": "retrieve_business", "type": "jev_chunk_retrieve", "text_key": "text",
          "targets": [TARGET], "share": 0.05, "chunk_tokens": 100, **overrides}
    config = {"default_model": "gpt-4o-mini", "bypass_cache": True,
              "datasets": {"docs": {"type": "file", "path": str(data)}}, "operations": [op],
              "pipeline": {"steps": [{"name": "s", "input": "docs", "operations": [op["name"]]}],
                           "output": {"type": "file", "path": str(tmp_path / "out.json")}}}
    cost = DSLRunner(config, max_threads=4).load_run_save()
    return json.loads((tmp_path / "out.json").read_text()), cost


@pytest.fixture
def jev_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(jev_backend, "request_jev", scripted_jev(["manufacturer", "Ohio"], calls))
    return calls


# ---- operator ----------------------------------------------------------------------------

def test_keeps_the_best_chunks_up_to_the_share(tmp_path, jev_calls):
    text = long_doc()
    out, cost = run_pipeline(tmp_path, [{"id": 1, "text": text}])
    view, diag = out[0]["text_jev"], out[0]["_retrieve_business_retrieval"]
    assert "manufacturer of industrial pumps" in view
    assert diag["kept_chars"] <= 0.05 * len(text) * 1.6 and diag["kept_chunks"] >= 1
    assert cost > 0 and jev_calls
    # Every chunk is asked about exactly once; context chunks get no question.
    asked = [k for p in jev_calls for k in p["questions"]]
    assert len(asked) == len(set(asked)) == diag["chunks"]


def test_each_request_fits_the_limits(jev_calls):
    text = long_doc()
    bounds = chunk_bounds(text, 450)
    for first, last in plan(text, bounds, [TARGET], "jev-1.13.0", 28000):
        p = payload(text, bounds, first, last, [TARGET], "jev-1.13.0")
        assert len(p["questions"]) <= 32 and len(json.dumps(p, ensure_ascii=False).encode()) <= 28000


def test_two_targets_share_requests_and_take_the_best_score(tmp_path, jev_calls):
    second = {"name": "location", "description": "Where the company is based."}
    out, _ = run_pipeline(tmp_path, [{"id": 1, "text": long_doc()}], targets=[TARGET, second])
    assert all(len(p["questions"]) <= 32 and len(p["state"]["task"]["targets"]) == 2 for p in jev_calls)
    assert "Ohio" in out[0]["text_jev"]


def test_short_document_passes_through_without_calls(tmp_path, jev_calls):
    out, cost = run_pipeline(tmp_path, [{"id": 1, "text": "A short report."}], share=0.2)
    assert out[0]["text_jev"] == "A short report." and not jev_calls and cost == 0


def test_ties_go_to_the_stronger_neighbour_then_document_order():
    assert kept_chunks([0.5, 0.9, 0.5, 0.1, 0.5], 2) == [1, 0]      # 0 and 2 tie with neighbour 0.9; 0 first
    assert kept_chunks([0.2, 0.5, 0.2, 0.5, 0.8], 1) == [4]
    assert kept_chunks([0.5, 0.2, 0.5, 0.8], 2) == [3, 2]          # chunk 2 has the 0.8 neighbour


def test_schema_rejects_bad_configs():
    from docetl.operations.jev_chunk_retrieve import JevChunkRetrieveOperation
    base = {"name": "r", "type": "jev_chunk_retrieve", "text_key": "text", "targets": [TARGET]}
    with pytest.raises(Exception):
        JevChunkRetrieveOperation.schema(**base, share=0)
    with pytest.raises(Exception):
        JevChunkRetrieveOperation.schema(**{**base, "targets": [TARGET, TARGET]}, share=0.1)


# ---- jev_retrieval rewrite ---------------------------------------------------------------

def rewrite(**kw):
    base = {"name": "retrieve_business", "document_key": "tot_text_raw", "targets": [TARGET], "share": 0.05}
    return JevRetrievalInstantiateSchema(**{**base, **kw})


def test_registered_rewrite_is_the_chunk_method():
    assert isinstance(DIRECTIVE_REGISTRY["jev_retrieval"], JevRetrievalDirective)
    ops = yaml.safe_load(open(SUSTAINABILITY))["operations"]
    new = JevRetrievalDirective().apply(ops, "determine_economic_activity", rewrite())
    by_name = {o["name"]: o for o in new}
    names = [o["name"] for o in new]
    assert names.index("retrieve_business") == names.index("determine_economic_activity") - 1
    assert by_name["retrieve_business"]["type"] == "jev_chunk_retrieve"
    assert by_name["retrieve_business"]["share"] == 0.05
    assert "{{ input.tot_text_raw_jev_determine_economic_activity }}" in by_name["determine_economic_activity"]["prompt"]
    assert "{{ input.tot_text_raw }}" in by_name["filter_sustainability_reports"]["prompt"]


def test_reapplying_replaces_the_view_instead_of_stacking():
    ops = yaml.safe_load(open(SUSTAINABILITY))["operations"]
    once = JevRetrievalDirective().apply(ops, "determine_economic_activity", rewrite())
    twice = JevRetrievalDirective().apply(
        once, "determine_economic_activity",
        rewrite(share=0.2, document_key="tot_text_raw_jev_determine_economic_activity"))
    views = [o for o in twice if o["type"] == "jev_chunk_retrieve"]
    assert len(views) == 1 and views[0]["share"] == 0.2 and views[0]["text_key"] == "tot_text_raw"


def test_rewrite_rejects_bad_values():
    with pytest.raises(Exception):
        rewrite(share=0.5)
    with pytest.raises(Exception):
        rewrite(targets=[TARGET] * 2)
    ops = yaml.safe_load(open(SUSTAINABILITY))["operations"]
    with pytest.raises(ValueError, match="does not reference"):
        JevRetrievalDirective().apply(ops, "determine_economic_activity", rewrite(document_key="missing"))


def test_instantiate_is_one_agent_call_with_document_lengths(tmp_path, monkeypatch):
    import docetl.reasoning_optimizer.directives.jev_retrieval as module

    data = tmp_path / "in.json"
    data.write_text(json.dumps([{"id": i, "tot_text_raw": long_doc() * (1 + i)} for i in range(3)]))
    prompts = []

    class Resp:
        def __init__(self, content):
            self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]

    def fake_agent(model, messages, response_format):
        prompts.append(messages[-1]["content"])
        return Resp(rewrite(share=0.1).model_dump_json()), 0.002

    monkeypatch.setattr(module, "agent_completion", fake_agent)
    config = yaml.safe_load(open(SUSTAINABILITY))
    new_ops, _, cost = JevRetrievalDirective().instantiate(
        operators=config["operations"], target_ops=["determine_economic_activity"], agent_llm="m",
        message_history=[], input_file_path=str(data))
    assert len(prompts) == 1 and cost == pytest.approx(0.002)
    assert "3 documents; characters: median" in prompts[0]
    assert [o for o in new_ops if o["type"] == "jev_chunk_retrieve"][0]["share"] == 0.1
    # The rewritten pipeline passes DocETL's own config validation.
    config["operations"] = new_ops
    step = config["pipeline"]["steps"][0]
    step["operations"].insert(step["operations"].index("determine_economic_activity"), "retrieve_business")
    config["datasets"]["company_reports"]["path"] = str(data)
    DSLRunner(config, max_threads=1)


def test_grid_ladder_is_share_of_report():
    from docetl.mcts.param_grid import GRIDS
    spec = GRIDS["jev_retrieval"]
    assert spec.param == "share" and spec.ladder == (0.025, 0.05, 0.1, 0.2)
    assert spec.get(spec.set(rewrite(), 0.1)) == 0.1


def test_jev_fit_treats_an_uncapped_share_as_unbounded():
    from docetl.mcts.jev_fit import input_token_bound
    ops = yaml.safe_load(open(SUSTAINABILITY))["operations"]
    new = JevRetrievalDirective().apply(ops, "determine_economic_activity", rewrite())
    bound, why = input_token_bound(new, "determine_economic_activity")
    assert bound is None and "no size limit" in why


def test_shortening_for_jev_caps_the_chunk_retrieval_to_fit():
    """Before a Jev switch, jev_retrieval may shorten the input: max_chars is set from the room left."""
    from docetl.mcts.jev_fit import (CHARS_PER_TOKEN_SAFE, JEV_FIT_TOKENS, SHORTENING_FOR_JEV,
                                     cap_chunk_retrieval, fits_jev)
    assert "jev_retrieval" in SHORTENING_FOR_JEV
    ops = yaml.safe_load(open(SUSTAINABILITY))["operations"]
    new = JevRetrievalDirective().apply(ops, "determine_economic_activity", rewrite())
    capped = cap_chunk_retrieval(new, "determine_economic_activity")
    view = [o for o in capped if o["type"] == "jev_chunk_retrieve"][0]
    assert 0 < view["max_chars"] <= JEV_FIT_TOKENS * CHARS_PER_TOKEN_SAFE
    fits, bound, _ = fits_jev(capped, "determine_economic_activity")
    assert fits and JEV_FIT_TOKENS - 1_100 < bound <= JEV_FIT_TOKENS - 1_000   # keeps 1k headroom for the Jev prompt
    assert "max_chars" not in [o for o in new if o["type"] == "jev_chunk_retrieve"][0]   # input not mutated
    assert cap_chunk_retrieval(capped, "determine_economic_activity") == capped          # already capped


def test_max_chars_bounds_chunks_and_separators(tmp_path, jev_calls):
    text = long_doc()
    out, _ = run_pipeline(tmp_path, [{"id": 1, "text": text}], share=0.3, max_chars=1500)
    assert 0 < len(out[0]["text_jev"]) <= 1500
