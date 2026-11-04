"""Offline tests for the presence_check operator and the presence_gate directive (no network)."""
import json

import pytest
import yaml
from jinja2 import Template

import docetl.retrieval.jev_backend as jev_backend
from docetl.operations.presence_check import chunk_text
from docetl.reasoning_optimizer.directives import PresenceGateDirective
from docetl.reasoning_optimizer.instantiate_schemas import PresenceGateInstantiateSchema
from docetl.runner import DSLRunner

CUAD = "experiments/reasoning/pipelines/cuad.yaml"


def scripted_jev(keywords, calls):
    """Answer 0.9 for check `key` if keywords[key] occurs in the request's text, else 0.1."""

    def request(payload, timeout):
        calls.append(payload)
        answers = {k: {"type": "noul", "noul": 0.9 if keywords.get(k, "\0") in payload["state"] else 0.1}
                   for k in payload["questions"]}
        return {"model": payload["model"], "answers": answers, "usage": {"input_tokens": len(payload["state"]) // 4}}

    return request


def config_for(tmp_path, docs, ops):
    data = tmp_path / "in.json"
    data.write_text(json.dumps(docs))
    return {"default_model": "gpt-4o-mini", "bypass_cache": True,
            "datasets": {"docs": {"type": "file", "path": str(data)}},
            "operations": ops,
            "pipeline": {"steps": [{"name": "s", "input": "docs", "operations": [o["name"] for o in ops]}],
                         "output": {"type": "file", "path": str(tmp_path / "out.json")}}}


def run(tmp_path, docs, ops):
    runner = DSLRunner(config_for(tmp_path, docs, ops), max_threads=4)
    cost = runner.load_run_save()
    return json.loads((tmp_path / "out.json").read_text()), cost


def contract(n_paragraphs=80, inserts=None):
    paragraphs = [f"{i}. The parties agree to routine obligation number {i} under this agreement." * 4
                  for i in range(n_paragraphs)]
    for at, text in (inserts or {}).items():
        paragraphs[at] = text
    return "\n\n".join(paragraphs)


CHECKS = [{"key": "governing_law", "question": "Does the text state which law governs the agreement?"},
          {"key": "audit_rights", "question": "Does the text give a party the right to audit the other's records?"}]
KEYWORDS = {"governing_law": "governed by the laws", "audit_rights": "right to audit"}


def presence_op(**overrides):
    return {"name": "clauses", "type": "presence_check", "text_key": "text", "checks": CHECKS,
            "chunk_chars": 4000, **overrides}


def test_chunks_cover_the_text_with_overlap():
    text = contract()
    spans = chunk_text(text, 4000, 400)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(e - s <= 4000 for s, e in spans)
    assert all(s2 < e1 for (_, e1), (s2, _) in zip(spans, spans[1:]))  # consecutive chunks overlap
    assert chunk_text("short", 4000, 400) == [(0, 5)]


def test_presence_flags_any_and_early_stop(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(jev_backend, "request_jev", scripted_jev(KEYWORDS, calls))
    text = contract(inserts={2: "This Agreement is governed by the laws of Delaware."})
    out, cost = run(tmp_path, [{"id": 1, "text": text}, {"id": 2, "text": contract()}], [presence_op()])
    first, second = out
    assert first["clauses_governing_law"] is True and first["clauses_audit_rights"] is False
    assert first["clauses_any"] is True and second["clauses_any"] is False
    diag = first["_clauses_presence"]
    assert diag["max_probability"]["governing_law"] == 0.9 and diag["chunks"] > 1
    # governing_law was found in the first chunk of doc 1 and is not asked again in its later chunks
    doc1_later = [p for p in calls if "(part 1 of" not in p["state"]
                  and p["state"].count("routine obligation") and "Delaware" not in p["state"]
                  and set(p["questions"]) != {"governing_law", "audit_rights"}]
    assert doc1_later and all(set(p["questions"]) == {"audit_rights"} for p in doc1_later)
    assert len([p for p in calls if "(part 1 of" in p["state"]]) == 2
    assert cost > 0


def test_threshold_is_the_recall_knob(tmp_path, monkeypatch):
    def lukewarm(payload, timeout):
        return {"answers": {k: {"type": "noul", "noul": 0.25} for k in payload["questions"]},
                "usage": {"input_tokens": 10}}

    monkeypatch.setattr(jev_backend, "request_jev", lukewarm)
    out, _ = run(tmp_path, [{"id": 1, "text": "short contract"}], [presence_op(threshold=0.2)])
    assert out[0]["clauses_any"] is True
    out, _ = run(tmp_path, [{"id": 1, "text": "short contract"}], [presence_op(threshold=0.3, bypass_cache=True)])
    assert out[0]["clauses_any"] is False


def test_more_than_32_checks_are_bundled(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(jev_backend, "request_jev", scripted_jev({}, calls))
    checks = [{"key": f"c{i}", "question": f"Is item {i} present?"} for i in range(41)]
    run(tmp_path, [{"id": 1, "text": "tiny"}], [presence_op(checks=checks)])
    assert sorted(len(p["questions"]) for p in calls) == [9, 32]


def test_invalid_jev_answer_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(jev_backend, "request_jev",
                        lambda p, t: {"answers": {}, "usage": {"input_tokens": 1}})
    with pytest.raises(Exception, match="invalid Jev answer"):
        run(tmp_path, [{"id": 1, "text": "tiny"}], [presence_op()])


# ---- directive -------------------------------------------------------------------------

def cuad_ops():
    return yaml.safe_load(open(CUAD))["operations"]


def cuad_rewrite(**overrides):
    prompt = cuad_ops()[0]["prompt"]
    lines = {line.strip().split(":")[0]: line.strip() for line in prompt.splitlines() if line.strip()[:2].rstrip(".").isdigit()}
    fields = dict(name="clause_presence", document_key="document", granularity="category", checker="jev",
                  context="Which clauses a commercial contract contains.",
                  checks=[{"key": "governing_law", "question": "Does the text state the governing law?",
                           "prompt_fragment": lines["8. Governing Law"]},
                          {"key": "audit_rights", "question": "Does a party have audit rights?",
                           "prompt_fragment": lines["34. Audit Rights"]}])
    fields.update(overrides)
    return PresenceGateInstantiateSchema(**fields)


def test_category_gate_wraps_only_the_gated_items():
    ops = PresenceGateDirective().apply(cuad_ops(), "extract_contract_info", cuad_rewrite())
    assert [o["type"] for o in ops] == ["presence_check", "gated_map", "code_map"]
    check, gated, fill = ops
    assert gated["gate_key"] == "clause_presence_any"
    prompt = gated["prompt"]
    render = lambda **flags: Template(prompt).render(input={"document": "DOC", **flags})
    on = render(clause_presence_governing_law=True, clause_presence_audit_rights=False)
    assert "8. Governing Law" in on and "34. Audit Rights" not in on
    assert "1. Document Name" in on                     # ungated categories stay
    assert "do not output them" in on
    assert check["checks"][0] == {"key": "governing_law", "question": "Does the text state the governing law?"}


def test_skipped_documents_get_empty_outputs_not_none(tmp_path, monkeypatch):
    """No check present -> the map is never called (no LLM needed) and clauses becomes []."""
    monkeypatch.setattr(jev_backend, "request_jev", scripted_jev({}, []))
    ops = PresenceGateDirective().apply(cuad_ops(), "extract_contract_info", cuad_rewrite())
    ops[0]["chunk_chars"] = 4000
    out, _ = run(tmp_path, [{"id": 1, "name": "a.txt", "document": contract()}], ops)
    assert out[0]["clauses"] == [] and out[0]["clause_presence_any"] is False


def test_fragments_must_be_verbatim_and_disjoint():
    directive = PresenceGateDirective()
    bad = cuad_rewrite(checks=[{"key": "x", "question": "q", "prompt_fragment": "8. Governing law (paraphrased)"}])
    with pytest.raises(ValueError, match="verbatim"):
        directive.apply(cuad_ops(), "extract_contract_info", bad)
    overlap = cuad_rewrite(checks=[{"key": "a", "question": "q", "prompt_fragment": "8. Governing Law"},
                                   {"key": "b", "question": "q", "prompt_fragment": "Governing Law: Which"}])
    with pytest.raises(ValueError, match="overlap"):
        directive.apply(cuad_ops(), "extract_contract_info", overlap)


def test_schema_validation():
    with pytest.raises(ValueError):
        cuad_rewrite(granularity="document")                       # document takes one check
    with pytest.raises(ValueError):
        cuad_rewrite(checker="llm")                                # needs checker_model
    with pytest.raises(ValueError):
        cuad_rewrite(threshold=0.99)
    with pytest.raises(ValueError):
        PresenceGateDirective().apply(cuad_ops(), "extract_contract_info", cuad_rewrite(document_key="body"))


def test_document_granularity_gates_the_whole_map():
    rewrite = cuad_rewrite(granularity="document",
                           checks=[{"key": "is_license", "question": "Is this a license agreement?"}])
    ops = PresenceGateDirective().apply(cuad_ops(), "extract_contract_info", rewrite)
    assert ops[1]["gate_key"] == "clause_presence_is_license"
    assert ops[1]["prompt"] == cuad_ops()[0]["prompt"]


def test_llm_checker_builds_a_boolean_map_and_any_flag():
    ops = PresenceGateDirective().apply(cuad_ops(), "extract_contract_info",
                                        cuad_rewrite(checker="llm", checker_model="gemini-3.1-flash-lite"))
    assert [o["type"] for o in ops] == ["map", "code_map", "gated_map", "code_map"]
    assert ops[0]["model"] == "gemini/gemini-3.1-flash-lite"
    assert ops[0]["output"]["schema"] == {"clause_presence_governing_law": "boolean",
                                          "clause_presence_audit_rights": "boolean"}
    ns = {}
    exec(ops[1]["code"], ns)
    assert ns["transform"]({"clause_presence_audit_rights": True}) == {"clause_presence_any": True}
    with pytest.raises(ValueError, match="checker_model"):
        PresenceGateDirective().apply(cuad_ops(), "extract_contract_info",
                                      cuad_rewrite(checker="llm", checker_model="gpt-5"))


def test_instantiate_retries_and_produces_a_valid_config(tmp_path, monkeypatch):
    import docetl.reasoning_optimizer.directives.presence_gate as module

    data = tmp_path / "in.json"
    data.write_text(json.dumps([{"id": i, "name": f"c{i}.txt", "document": contract(20 + i)} for i in range(4)]))
    good = cuad_rewrite().model_dump_json()
    bad = cuad_rewrite().model_dump()
    bad["checks"][0]["prompt_fragment"] = "Governing law, paraphrased"
    replies, prompts = [json.dumps(bad), good], []

    class Resp:
        def __init__(self, content):
            self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]

    def fake_agent(model, messages, response_format):
        prompts.append(messages[-1]["content"])
        return Resp(replies.pop(0)), 0.001

    monkeypatch.setattr(module, "agent_completion", fake_agent)
    config = yaml.safe_load(open(CUAD))
    new_ops, history, cost = PresenceGateDirective().instantiate(
        operators=config["operations"], target_ops=["extract_contract_info"], agent_llm="gemini/x",
        message_history=[], input_file_path=str(data))
    assert len(prompts) == 2 and "verbatim" in prompts[1] and cost == pytest.approx(0.002)
    assert "Sample documents" in prompts[0] and "Document lengths" in prompts[0]
    config["operations"] = new_ops
    config["pipeline"]["steps"][0]["operations"] = [o["name"] for o in new_ops]
    config["datasets"]["contracts_data"]["path"] = str(data)
    DSLRunner(config, max_threads=1)  # passes DocETL's config validation
