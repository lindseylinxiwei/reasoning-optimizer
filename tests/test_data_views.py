"""Offline tests for shared data views (no paid calls)."""
import json

import pytest

import docetl.retrieval.jev_backend as jev_backend
from docetl.reasoning_optimizer.data_views import (
    HeadTailView, JevView, SummaryView, TopKView, check_view, view_ops,
)
from docetl.runner import DSLRunner
from tests.test_jev_retrieve import long_doc, scripted_jev

MODEL = "gemini/gemini-3.5-flash-lite"


def run_view(tmp_path, ops, docs):
    data = tmp_path / "in.json"
    data.write_text(json.dumps(docs))
    config = {"default_model": MODEL, "bypass_cache": True,
              "datasets": {"d": {"type": "file", "path": str(data)}}, "operations": ops,
              "pipeline": {"steps": [{"name": "s", "input": "d", "operations": [o["name"] for o in ops]}],
                           "output": {"type": "file", "path": str(tmp_path / "out.json")}}}
    cost = DSLRunner(config, max_threads=4).load_run_save()
    return json.loads((tmp_path / "out.json").read_text()), cost


def test_head_tail_view_replaces_the_text_field(tmp_path):
    doc = long_doc()
    ops = view_ops("head_tail", HeadTailView(rationale="r", head_tokens=200, tail_tokens=50), "text", MODEL)
    out, cost = run_view(tmp_path, ops, [{"id": 1, "text": doc}])
    head, tail = out[0]["text"].split("\n[...]\n")
    assert doc.startswith(head) and doc.endswith(tail) and len(out[0]["text"]) < len(doc) / 5
    assert cost == 0


def test_topk_view_joins_retrieved_chunks_in_document_order(tmp_path):
    docs = [{"id": i, "company": f"c{i}", "text": long_doc(needle_at=10 + 10 * i)} for i in range(3)]
    config = TopKView(rationale="r", chunk_tokens=120, k=3, method="fts",
                      query="leading manufacturer industrial pumps valves headquartered")
    out, cost = run_view(tmp_path, view_ops("topk", config, "text", MODEL), docs)
    assert len(out) == 3 and cost == 0
    for doc, row in zip(sorted(docs, key=lambda d: d["id"]), sorted(out, key=lambda r: r["id"])):
        parts = row["text"].split("\n[...]\n")
        assert 1 <= len(parts) <= 3
        positions = [doc["text"].find(p) for p in parts]
        assert all(p >= 0 for p in positions) and positions == sorted(positions)   # literal, in order
        assert "manufacturer" in row["text"]
        assert row["company"] == doc["company"]                                     # other fields kept


def test_jev_view_writes_back_to_the_text_field(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(jev_backend, "request_jev", scripted_jev(["manufacturer"], calls))
    config = JevView(rationale="r", targets=[{"name": "business", "description": "What the company does."}],
                     evidence_chars=1500, max_decisions=32)
    out, _ = run_view(tmp_path, view_ops("jev", config, "text", MODEL), [{"id": 1, "text": long_doc()}])
    assert "leading manufacturer" in out[0]["text"] and len(out[0]["text"]) <= 1500 + 50


def test_summarization_view_is_a_valid_map():
    ops = view_ops("summarization", SummaryView(rationale="r", summary_words=300, focus="Keep the company name."),
                   "text", MODEL)
    assert ops[0]["type"] == "map" and ops[0]["output"]["schema"] == {"text": "string"}
    assert "{{ input.text }}" in ops[0]["prompt"] and ops[0]["model"] == MODEL


def test_check_view_rejects_views_that_keep_too_much():
    lengths = [100_000] * 5
    check_view("topk", TopKView(rationale="r", chunk_tokens=500, k=4, method="fts", query="q"), lengths)
    with pytest.raises(ValueError, match="limit 30%"):
        check_view("topk", TopKView(rationale="r", chunk_tokens=12000, k=8, method="fts", query="q"), lengths)


def test_sweeps_expand_to_levels_plus_one_change_comparisons():
    from docetl.reasoning_optimizer.data_views import JevSweep, TopKSweep
    topk = TopKSweep(chunk_tokens=400, chunk_rationale="r", method="embedding", method_rationale="r", query="q",
                     query_rationale="r", k_levels=[1, 2, 4, 8, 16], levels_rationale="r").configs()
    assert len(topk) == 7
    mid, fts, half = topk["topk_3"], topk["topk_3_fts"], topk["topk_3_half_chunks"]
    assert (fts.method, fts.k, fts.chunk_tokens) == ("fts", mid.k, mid.chunk_tokens)          # one change: method
    assert half.chunk_tokens * half.k == mid.chunk_tokens * mid.k and half.method == mid.method  # same budget
    jev = JevSweep(targets=[{"name": "a", "description": "b"}], targets_rationale="r", regions_per_request=4,
                   regions_rationale="r", cap_levels=[1000, 2000, 4000, 8000, 16000],
                   decision_levels=[16, 24, 32, 48, 64], levels_rationale="r").configs()
    assert jev["jev_3_adaptive"].policy == "adaptive" and jev["jev_3"].policy == "preserve_fill"
    assert jev["jev_3_adaptive"].evidence_chars == jev["jev_3"].evidence_chars
    with pytest.raises(Exception):
        TopKSweep(chunk_tokens=400, chunk_rationale="r", method="fts", method_rationale="r", query="q",
                  query_rationale="r", k_levels=[4, 2, 8, 16, 32], levels_rationale="r")


def test_topk_embedding_view_batches_and_embeds_whole_chunks(tmp_path, monkeypatch):
    """Gemini allows 100 texts per embedding request; chunks are embedded whole, not cut to 1000 chars."""
    from docetl.operations import clustering_utils
    from docetl.operations.utils.api import APIWrapper
    batches = []

    def fake_embedding(self, model, input):
        texts = json.loads(input) if isinstance(input, str) else input
        batches.append(texts)
        return {"data": [{"embedding": [float("manufacturer" in t), 1.0]} for t in texts]}

    monkeypatch.setattr(APIWrapper, "gen_embedding", fake_embedding)
    monkeypatch.setattr(clustering_utils, "completion_cost", lambda response: 0.0)
    doc = long_doc() * 6                      # ~145k chars -> ~250 chunks of 120 tokens
    config = TopKView(rationale="r", chunk_tokens=500, k=3, method="embedding", query="manufacturer")
    out, _ = run_view(tmp_path, view_ops("topk", config, "text", MODEL), [{"id": 1, "text": doc}])
    assert batches and max(len(b) for b in batches if len(b) > 1) <= 100
    assert max(len(t) for b in batches for t in b) > 1000            # whole chunks, not first 1000 chars
