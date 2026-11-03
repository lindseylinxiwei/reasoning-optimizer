"""Offline tests for the jev_retrieve operator and the jev_retrieval directive (no network)."""
import json
import re

import pytest

import docetl.retrieval.jev_backend as jev_backend
from docetl.runner import DSLRunner

REGION = re.compile(r"⟦REGION\|BEGIN (C\d+)⟧(.*?)⟦REGION\|END \1⟧", re.S)


def scripted_jev(keywords, calls):
    """Score a region 0.9 for target Ti if it contains keywords[i], else 0.1."""

    def request(payload, timeout):
        calls.append(payload)
        regions = dict(REGION.findall(payload["state"]["source_text"]))
        answers = {}
        for key in payload["questions"]:
            region, target = key.rsplit("_T", 1)
            hit = keywords[int(target)] in regions.get(region, "")
            answers[key] = {"type": "noul", "noul": 0.9 if hit else 0.1}
        return {"model": payload["model"], "answers": answers,
                "usage": {"input_tokens": len(json.dumps(payload)) // 4}}

    return request


def long_doc(needle_at=30):
    paragraphs = [f"Section {i}. Routine sustainability boilerplate paragraph number {i}. " * 6
                  for i in range(60)]
    paragraphs[needle_at] = ("Section about us. ACME Corp is a leading manufacturer of industrial "
                             "pumps and valves, headquartered in Ohio. ") * 3
    return "\n\n".join(paragraphs)


def run_pipeline(tmp_path, docs, op_overrides=None):
    data = tmp_path / "in.json"
    data.write_text(json.dumps(docs))
    op = {"name": "retrieve_business", "type": "jev_retrieve", "text_key": "text",
          "targets": [{"name": "business_description",
                       "description": "Passages describing what the company does."}],
          "evidence_chars": 1500, "max_decisions": 32, **(op_overrides or {})}
    config = {"default_model": "gpt-4o-mini", "bypass_cache": True,
              "datasets": {"docs": {"type": "file", "path": str(data)}},
              "operations": [op],
              "pipeline": {"steps": [{"name": "s", "input": "docs", "operations": [op["name"]]}],
                           "output": {"type": "file", "path": str(tmp_path / "out.json")}}}
    runner = DSLRunner(config, max_threads=4)
    cost = runner.load_run_save()
    return json.loads((tmp_path / "out.json").read_text()), cost


@pytest.fixture
def jev_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(jev_backend, "request_jev", scripted_jev(["manufacturer"], calls))
    return calls


def test_retrieves_the_relevant_passage(tmp_path, jev_calls):
    doc = long_doc()
    out, cost = run_pipeline(tmp_path, [{"id": 1, "text": doc}])
    evidence = out[0]["text_jev"]
    assert "leading manufacturer of industrial pumps" in evidence
    assert len(evidence) < len(doc) / 4
    # Every rendered passage is a literal slice of the source.
    assert all(part in doc for part in evidence.split("\n[...]\n"))
    diagnostics = out[0]["_retrieve_business_retrieval"]
    assert diagnostics["passthrough"] is False and diagnostics["evidence_chars"] <= 1500
    assert jev_calls and cost > 0


def test_short_documents_pass_through_without_calls(tmp_path, jev_calls):
    out, cost = run_pipeline(tmp_path, [{"id": 1, "text": "ACME is a manufacturer."}])
    assert out[0]["text_jev"] == "ACME is a manufacturer."
    assert out[0]["_retrieve_business_retrieval"]["passthrough"] is True
    assert jev_calls == [] and cost == 0


def test_adaptive_policy_without_fill(tmp_path, jev_calls):
    out, _ = run_pipeline(tmp_path, [{"id": 1, "text": long_doc()}], {"policy": "adaptive"})
    assert "leading manufacturer" in out[0]["text_jev"]


def test_scoring_failure_fails_the_operation(tmp_path, monkeypatch):
    def broken(payload, timeout):
        raise RuntimeError("Jev HTTP 500")

    monkeypatch.setattr(jev_backend, "request_jev", broken)
    with pytest.raises(Exception, match="Jev"):
        run_pipeline(tmp_path, [{"id": 1, "text": long_doc()}])


def test_target_validation():
    from docetl.operations.jev_retrieve import JevRetrieveOperation
    base = {"name": "r", "type": "jev_retrieve", "text_key": "text"}
    with pytest.raises(Exception):
        JevRetrieveOperation.schema(**base, targets=[])
    with pytest.raises(Exception):
        JevRetrieveOperation.schema(**base, targets=[{"name": "a", "description": "x"}] * 2)


# ---- jev_retrieval directive --------------------------------------------------------------

import yaml  # noqa: E402

from docetl.reasoning_optimizer.directives import JevRetrievalDirective  # noqa: E402
from docetl.reasoning_optimizer.instantiate_schemas import JevRetrievalInstantiateSchema  # noqa: E402

SUSTAINABILITY = "experiments/reasoning/pipelines/sustainability.yaml"


def rewrite(**kw):
    base = {"name": "retrieve_business", "document_key": "tot_text_raw",
            "targets": [{"name": "business", "description": "Passages describing what the company does."}],
            "evidence_chars": 4000}
    return JevRetrievalInstantiateSchema(**{**base, **kw})


def test_directive_rewires_only_the_target_operation():
    ops = yaml.safe_load(open(SUSTAINABILITY))["operations"]
    new = JevRetrievalDirective().apply(ops, "determine_economic_activity", rewrite())
    names = [o["name"] for o in new]
    assert names.index("retrieve_business") == names.index("determine_economic_activity") - 1
    retrieve = new[names.index("retrieve_business")]
    assert retrieve["type"] == "jev_retrieve" and retrieve["index"]["levels"] == [{"kind": "blank_line"}]
    target = new[names.index("determine_economic_activity")]
    assert "{{ input.tot_text_raw_jev_determine_economic_activity }}" in target["prompt"]
    # The filter and the reduce still read the full text.
    by_name = {o["name"]: o for o in new}
    assert "{{ input.tot_text_raw }}" in by_name["filter_sustainability_reports"]["prompt"]
    assert "item.tot_text_raw" in by_name["sustainability_summary"]["prompt"]


def test_directive_handles_loop_variables_in_reduce_prompts():
    ops = yaml.safe_load(open(SUSTAINABILITY))["operations"]
    new = JevRetrievalDirective().apply(ops, "sustainability_summary", rewrite())
    prompt = {o["name"]: o for o in new}["sustainability_summary"]["prompt"]
    assert "item.tot_text_raw_jev_sustainability_summary" in prompt and "item.tot_text_raw }}" not in prompt


def test_directive_rejects_a_field_the_prompt_does_not_read():
    ops = yaml.safe_load(open(SUSTAINABILITY))["operations"]
    with pytest.raises(ValueError, match="does not reference"):
        JevRetrievalDirective().apply(ops, "determine_economic_activity", rewrite(document_key="missing"))


def test_directive_instantiate_two_calls(tmp_path, monkeypatch):
    """Call 1 designs the index from the corpus profile; call 2 configures retrieval."""
    import docetl.reasoning_optimizer.directives.jev_retrieval as module
    from docetl.reasoning_optimizer.instantiate_schemas import JevIndexDesignSchema

    body = "\n".join("Body text sentence number one continues here." for _ in range(30))
    report = "\n".join(f"{i}. Section heading {i}\n{body}\n" for i in range(1, 30))
    data = tmp_path / "in.json"
    data.write_text(json.dumps([{"id": i, "tot_text_raw": report * (1 + i % 3)} for i in range(6)]))
    prompts = []
    design = {"rationale": "numbered sections, then paragraphs",
              "levels": [{"kind": "heading", "style": "numbered", "pattern": None, "chars": None},
                         {"kind": "blank_line", "style": None, "pattern": None, "chars": None}],
              "leaf_chars": 2000, "min_chars": 200, "fanout": 6}

    class Resp:
        def __init__(self, content):
            self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]

    def fake_agent(model, messages, response_format):
        prompts.append(messages[-1]["content"])
        if response_format is JevIndexDesignSchema:
            return Resp(json.dumps(design)), 0.001
        return Resp(rewrite(regions_per_request=6).model_dump_json()), 0.002

    monkeypatch.setattr(module, "agent_completion", fake_agent)
    config = yaml.safe_load(open(SUSTAINABILITY))
    new_ops, history, cost = JevRetrievalDirective().instantiate(
        operators=config["operations"], target_ops=["determine_economic_activity"],
        agent_llm="gpt-4.1", message_history=[], input_file_path=str(data))
    assert len(prompts) == 2 and cost == pytest.approx(0.003)
    assert "\"styles\"" in prompts[0] and "Sample excerpts" in prompts[0]        # profile shown in call 1
    assert "Index built on sample documents" in prompts[1]                         # index stats shown in call 2
    retrieve = [o for o in new_ops if o["type"] == "jev_retrieve"][0]
    assert retrieve["index"]["levels"][0] == {"kind": "heading", "style": "numbered"}
    assert retrieve["regions_per_request"] == 6
    # The rewritten pipeline passes DocETL's own config validation.
    config["operations"] = new_ops
    step = config["pipeline"]["steps"][0]
    step["operations"].insert(step["operations"].index("determine_economic_activity"), "retrieve_business")
    config["datasets"]["company_reports"]["path"] = str(data)
    DSLRunner(config, max_threads=1)


def test_index_design_is_checked_and_retried(tmp_path, monkeypatch):
    """A heading pattern that matches nothing is sent back to the agent."""
    import docetl.reasoning_optimizer.directives.jev_retrieval as module
    from docetl.reasoning_optimizer.instantiate_schemas import JevIndexDesignSchema

    data = tmp_path / "in.json"
    data.write_text(json.dumps([{"id": 1, "tot_text_raw": long_doc()}]))
    designs = iter([{"rationale": "x", "levels": [{"kind": "heading", "pattern": "ZZZ NEVER"}]},
                    {"rationale": "y", "levels": [{"kind": "blank_line"}]}])
    feedback = []

    class Resp:
        def __init__(self, content):
            self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]

    def fake_agent(model, messages, response_format):
        feedback.append(messages[-1]["content"])
        if response_format is JevIndexDesignSchema:
            return Resp(json.dumps(next(designs))), 0.0
        return Resp(rewrite().model_dump_json()), 0.0

    monkeypatch.setattr(module, "agent_completion", fake_agent)
    ops = yaml.safe_load(open(SUSTAINABILITY))["operations"]
    new_ops, _, _ = JevRetrievalDirective().instantiate(
        operators=ops, target_ops=["determine_economic_activity"], agent_llm="m",
        message_history=[], input_file_path=str(data))
    assert any("matches no line" in f for f in feedback)
    assert [o for o in new_ops if o["type"] == "jev_retrieve"][0]["index"]["levels"] == [{"kind": "blank_line"}]


# ---- index specs --------------------------------------------------------------------------

from docetl.retrieval.index_spec import IndexSpec, build_spec_index, corpus_profile  # noqa: E402


def leaves(index):
    return [(n.start, n.end) for n in index.leaves()]


def test_heading_levels_follow_document_structure():
    text = "Intro paragraph that is long enough. " * 10 + "\n" + "".join(
        f"{i}. Heading {i}\n" + "Body sentence here. " * 30 + "\n" for i in range(1, 6))
    spec = IndexSpec(levels=[{"kind": "heading", "pattern": r"\d+\.\s"}], leaf_chars=5000, min_chars=50, fanout=8)
    index = build_spec_index(text, spec)
    root_children = [index.nodes[c] for c in index.nodes[index.root_id].children]
    assert len(root_children) == 6                       # preface + 5 sections
    assert all(text[c.start:].startswith(f"{i}. Heading") for i, c in enumerate(root_children[1:], 1))


def test_spec_index_covers_text_and_respects_sizes():
    text = long_doc()
    for spec in [IndexSpec(), IndexSpec(levels=[{"kind": "blank_line"}], leaf_chars=800, fanout=3),
                 IndexSpec(levels=[{"kind": "fixed", "chars": 3000}], leaf_chars=1000)]:
        index = build_spec_index(text, spec)       # validate(): full coverage, exact partitions
        assert max(e - s for s, e in leaves(index)) <= spec.leaf_chars
        assert all(len(n.children) <= spec.fanout for n in index.nodes.values())
        assert leaves(index) == leaves(build_spec_index(text, spec))   # deterministic


def test_min_chars_merges_tiny_segments():
    text = "\n\n".join(["a"] * 50 + ["word " * 100])
    tiny = build_spec_index(text, IndexSpec(levels=[{"kind": "blank_line"}], min_chars=0, leaf_chars=5000))
    merged = build_spec_index(text, IndexSpec(levels=[{"kind": "blank_line"}], min_chars=200, leaf_chars=5000))
    assert len(leaves(merged)) < len(leaves(tiny))


def test_invalid_specs_are_rejected():
    for bad in [{"levels": [{"kind": "heading"}]}, {"levels": [{"kind": "heading", "pattern": "("}]},
                {"levels": [{"kind": "fixed", "chars": 10}]}, {"leaf_chars": 50}, {"fanout": 1}]:
        with pytest.raises(Exception):
            IndexSpec(**bad)


def test_corpus_profile_is_label_free_stats():
    profile = corpus_profile([long_doc(), long_doc(10)])
    assert profile["docs"] == 2 and "numbered" in profile["styles"] and len(profile["excerpts"]) >= 1


def test_operator_with_designed_index_and_regions_per_request(tmp_path, jev_calls):
    spec = {"levels": [{"kind": "blank_line"}], "leaf_chars": 600, "min_chars": 100, "fanout": 8}
    out, _ = run_pipeline(tmp_path, [{"id": 1, "text": long_doc()}], {"index": spec, "regions_per_request": 2})
    assert "leading manufacturer" in out[0]["text_jev"]
    assert max(len(p["questions"]) for p in jev_calls) <= 2


def test_index_check_rejects_a_pattern_matching_ordinary_text():
    spec = IndexSpec(levels=[{"kind": "heading", "pattern": r"\w"}])
    with pytest.raises(ValueError, match="limit 5%"):
        JevRetrievalDirective.check_index(spec, [long_doc()])


def test_targets_are_bundled_to_fit_the_question_limit(tmp_path, monkeypatch):
    """10 targets with 4 regions per request -> bundles of 8 and 2; every request stays <= 32 questions."""
    calls = []

    def request(payload, timeout):
        calls.append(payload)
        regions = dict(REGION.findall(payload["state"]["source_text"]))
        targets = payload["state"]["task"]["targets"]
        answers = {}
        for key in payload["questions"]:
            region, label = key.rsplit("_", 1)
            word = targets[label]["description"].split()[-1]
            answers[key] = {"type": "noul", "noul": 0.9 if word in regions.get(region, "") else 0.1}
        return {"model": payload["model"], "answers": answers, "usage": {"input_tokens": 100}}

    monkeypatch.setattr(jev_backend, "request_jev", request)
    targets = [{"name": f"t{i}", "description": f"Passages mentioning word{i}"} for i in range(10)]
    doc = long_doc().replace("pumps", "word9 pumps")
    out, _ = run_pipeline(tmp_path, [{"id": 1, "text": doc}], {"targets": targets})
    assert max(len(p["questions"]) for p in calls) <= 32
    assert {len(p["state"]["task"]["targets"]) for p in calls} == {8, 2}
    assert set(out[0]["_retrieve_business_retrieval"]["targets"]) == {t["name"] for t in targets}
    assert "word9 pumps" in out[0]["text_jev"]


def test_named_styles_and_custom_patterns():
    from docetl.retrieval.index_spec import HEADING_STYLES, Splitter
    assert Splitter(kind="heading", style="numbered").regex == HEADING_STYLES["numbered"][0]
    assert Splitter(kind="heading", pattern=r"Item \d").regex == r"Item \d"
    for bad in [{"kind": "heading", "style": "nope"}, {"kind": "heading", "style": "numbered", "pattern": "x"},
                {"kind": "page_break", "style": "numbered"}]:
        with pytest.raises(Exception):
            Splitter(**bad)


def test_rejection_shows_the_lines_a_pattern_matched():
    from docetl.retrieval.index_spec import check_spec
    text = "\n".join(f"{i} apples" for i in range(400))
    with pytest.raises(ValueError, match="Lines it matched: '0 apples'"):
        check_spec(IndexSpec(levels=[{"kind": "heading", "pattern": r"\d+ "}]), [text])
