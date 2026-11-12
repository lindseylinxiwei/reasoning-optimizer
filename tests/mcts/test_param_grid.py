from types import SimpleNamespace

from docetl.mcts.param_grid import GRIDS, ParamTable, Result, estimate_child_cost, prune

HEAD = GRIDS["take_head_tail"]
RANK = {"gemini-3.1-flash-lite": 1.62, "gemini-4-argon": 12.31}


def test_tie_goes_to_cheaper_value():
    kept, pruned = prune([Result(2000, 0.10, 0.74), Result(4000, 0.20, 0.76), Result(8000, 0.40, 0.70)], 0.05)
    assert kept == 2000
    assert pruned == {"4000", "8000"}


def test_clear_winner_is_kept_and_cheaper_worse_values_survive():
    kept, pruned = prune([Result(500, 0.02, 0.50), Result(2000, 0.10, 0.74), Result(8000, 0.40, 0.66)], 0.05)
    assert kept == 2000
    assert pruned == {"8000"}      # 500 is cheaper, so it stays as a cost option


def test_failed_values_are_pruned():
    kept, pruned = prune([Result(500, None, None), Result(1000, 0.05, 0.7)], 0.05)
    assert kept == 1000 and pruned == {"500"}


def test_probe_is_agent_value_and_neighbours():
    t = ParamTable()
    assert t.choose(HEAD, "map", "gemini-3.1-flash-lite", 2100) == [1000, 2000, 4000]
    assert t.choose(HEAD, "map", "gemini-3.1-flash-lite", 300) == [500, 1000, 2000]
    assert t.choose(HEAD, "map", "gemini-3.1-flash-lite", 50000) == [4000, 8000, 16000]


def test_refine_steps_past_best_edge_then_finishes():
    t = ParamTable()
    for v, c, s in [(1000, 0.05, 0.60), (2000, 0.10, 0.66), (4000, 0.20, 0.75)]:
        t.record("take_head_tail", "map", "m", v, c, s)
    assert t.choose(HEAD, "map", "m", 2000) == [8000]
    t.record("take_head_tail", "map", "m", 8000, 0.40, 0.76)     # a tie with 4000: 4000 kept
    assert t.choose(HEAD, "map", "m", 2000) == [4000]


def test_refine_fills_gap_between_two_best():
    t = ParamTable()
    for v, c, s in [(500, 0.03, 0.70), (4000, 0.20, 0.62), (16000, 0.8, 0.40)]:
        t.record("take_head_tail", "map", "m", v, c, s)
    t.record("take_head_tail", "map", "m", 1000, 0.05, 0.50)
    # kept 500; next best alive is 1000 (4000 and 16000 cost more and score lower: pruned)
    assert t.choose(HEAD, "map", "m", 500) == [500]


def test_probe_cheap_apply_expensive():
    t = ParamTable()
    for v, c, s in [(1000, 0.05, 0.60), (2000, 0.10, 0.70), (4000, 0.20, 0.72)]:
        t.record("take_head_tail", "map", "gemini-3.1-flash-lite", v, c, s)
    out = t.choose(HEAD, "map", "gemini-4-argon", 8000, model_cost_rank=RANK, expensive_parent=True)
    assert out == [2000]          # only the best value from the cheaper model


def test_globally_pruned_after_two_contexts():
    t = ParamTable()
    for ctx in ("a", "b"):
        t.record("take_head_tail", "map", ctx, 2000, 0.1, 0.70)
        t.record("take_head_tail", "map", ctx, 4000, 0.2, 0.71)
    assert "4000" in t.globally_pruned("take_head_tail", "map")
    assert 4000 not in t.choose(HEAD, "map", "c", 2000)


def test_table_persists_and_reloads(tmp_path):
    p = tmp_path / "param_table.jsonl"
    t = ParamTable.load(str(p))
    t.record("presence_gate", "filter", "m", ("jev", None), 0.01, 0.7)
    t2 = ParamTable.load(str(p))
    assert t2.rows == t.rows
    assert t2._results("presence_gate", "filter")[0].value == ("jev", None)


def test_setters_change_only_the_parameter():
    from docetl.reasoning_optimizer.instantiate_schemas import TakeHeadTailInstantiateSchema
    rw = TakeHeadTailInstantiateSchema(name="head_tail", document_key="tot_text_raw", head_words=1000, tail_words=0)
    rw2 = HEAD.set(rw, 4000)
    assert rw2.head_words == 4000 and rw.head_words == 1000 and rw2.document_key == "tot_text_raw"
    topk = GRIDS["doc_chunking_topk"]
    fake = SimpleNamespace(chunk_size=1000, topk_config=SimpleNamespace(k=2))
    assert topk.get(fake) == 2000


def test_cost_estimate_scales_with_kept_tokens():
    assert estimate_child_cost(HEAD, 16000, 2.0) > estimate_child_cost(HEAD, 1000, 2.0)
    assert estimate_child_cost(None, 0, 2.0) == 2.0
