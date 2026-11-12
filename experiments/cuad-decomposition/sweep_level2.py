"""CUAD rewrite sweep, Level 2: combinations of the Level 1 rewrites that helped, on the same 40 contracts.

Level 2 reuses Level 1's frozen agent proposals (runs/level1/configs.json), so plans differ from Level 1 only in
the combination:
  I3a       isolating_subtasks_1: 3 groups, each its own prompt over the contract, then an LLM combine step
  GJ2 / GL  presence_gate_2: 36 per-category checks (Jev checker, or the same questions on flash-lite)
  views     jev_5, topk_bm25_5, topk_emb_5, head_tail_5 (about 70% of the median contract)
  C16k      chunking_16000: 16k-token chunks (only the longest contracts are split), previous chunk as context

Plans:
  A1/A2  Jev / LLM gate -> I3a with one gated_map per group; a group whose categories are all gated and all
         absent is skipped (its output becomes ""), the others list only their present categories
  A3     C16k -> Jev gate per chunk (agent-instantiated presence_gate on the chunk map)
  A5     C16k -> isolating subtasks on the chunk map (agent-instantiated)
  R1-R6  I3a with per-group retrieval: each group reads its own retrieved text document_g{i}
         (BM25 / embedding top-k with one agent-proposed query per group; Jev with one target per category),
         at about 35% and 70% of the median contract per group
  R7-R9  global views J5 / B5 / E5 -> I3a unchanged
  B1-B3  views J5 / B5 / H5 -> GJ2
  C1/C2  GJ2 with threshold 0.2 / 0.5
  D      unchanged reruns of baseline, gate_jev_2, gate_llm_1, isolating_subtasks_1, isolating_subtasks_2

Phases (from the repository root):
  python experiments/cuad-decomposition/sweep_level2.py propose     # 2-3 agent calls (cents)
  python experiments/cuad-decomposition/sweep_level2.py unit        # offline sanity checks, free
  python experiments/cuad-decomposition/sweep_level2.py check       # offline, free -> runs/level2/check.json
  python experiments/cuad-decomposition/sweep_level2.py smoke [plan ...]   # 2 shortest contracts, cents
  MODAL_PROFILE=personal modal run experiments/cuad-decomposition/sweep_level2.py --max-usd 25 [--plans-csv a,b]
  python experiments/cuad-decomposition/sweep_level2.py score
"""
import json
import re
import sys
import time
from copy import deepcopy
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
sys.path[:0] = [str(REPO), str(REPO / "experiments/reasoning"), str(HERE)]

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

L1 = HERE / "runs" / "level1"
OUT = HERE / "runs" / "level2"
CONFIGS = OUT / "configs.json"
DATA = REPO / "experiments/reasoning/data/train/cuad.json"
GT = REPO / "experiments/reasoning/data/CUAD-master_clauses.csv"
MODEL = "gemini/gemini-3.5-flash-lite"
AGENT = "gemini/gemini-3.8-flash"
TARGET = "extract_contract_info"
CHUNK_MAP = "map_extract_contract_info_chunks"
GATE = "clause_presence"          # checker op name for A1/A2 (fields clause_presence_<key>)
MEDIAN_CHARS = 34461              # median contract (corpus profile)
CHARS_PER_TOKEN = 4.5
BUDGETS = {"35": 0.35, "70": 0.70}  # share of the median contract each group reads
TOPK_CHUNK_TOKENS = 400            # Level 1 topk chunk size
JEV_DECISIONS = 64


def level1():
    return json.load(open(L1 / "configs.json"))


def l1_scores():
    return {r["plan"]: r for r in json.load(open(L1 / "scores.json"))}


# ---- building blocks ----------------------------------------------------------------------

def i3a_ops(l1):
    """(parallel_map op, aggregate map op) of isolating_subtasks_1."""
    ops = deepcopy(l1["plans"]["isolating_subtasks_1"]["operations"])
    pm = next(op for op in ops if op["type"] == "parallel_map")
    agg = next(op for op in ops if op["type"] == "map")
    return pm, agg


_CAT = re.compile(r"(?<![\d.])(\d{1,2})\. ([^:]{2,80}): (.*?)(?=\s+\d{1,2}\. [A-Z]|\s+Output Schema:|\s*$)", re.S)


def group_categories(prompt):
    """[(number, name, definition)] of the numbered category items in a group prompt."""
    return [(int(n), name.strip(), " ".join(d.split())) for n, name, d in _CAT.findall(prompt)]


def locate(prompt, fragment):
    """The fragment as it appears in this prompt: verbatim, or with double quotes the agent turned single."""
    for cand in (fragment, fragment.replace('"', "'")):
        if prompt.count(cand) == 1:
            return cand
    return None


def gate_design(l1):
    from docetl.reasoning_optimizer.instantiate_schemas import PresenceGateInstantiateSchema
    return PresenceGateInstantiateSchema(**l1["proposals"]["presence_gate_2"]["design"])


def checker_ops(design, checker):
    """The GJ2/GL checker operators exactly as the presence_gate directive builds them (name GATE)."""
    from docetl.reasoning_optimizer.directives import PresenceGateDirective
    import sweep
    base = sweep.base_ops()  # the original map, which holds every check's prompt fragment
    update = {"name": GATE, "checker": checker, "skip_when_none": False,
              "checker_model": MODEL.split("/", 1)[1] if checker == "llm" else None}
    ops = PresenceGateDirective().apply(base, TARGET, design.model_copy(update=update))
    return ops[:[o["name"] for o in ops].index(TARGET)]


def gated_group_prompt(prompt, design, prefix):
    """Wrap this group's gated category lines in their presence condition (directive logic), plus the note.

    Returns (prompt, gated keys, ungated category count)."""
    from docetl.reasoning_optimizer.directives import PresenceGateDirective
    checks = []
    for ch in design.checks:
        frag = locate(prompt, ch.prompt_fragment)
        if frag is not None:
            checks.append(ch.model_copy(update={"prompt_fragment": frag}))
    ungated = len(group_categories(prompt)) - len(checks)
    if not checks:
        return prompt, [], ungated
    rewrite = design.model_copy(update={"checks": checks})
    return PresenceGateDirective._gate_prompt(prompt, rewrite, prefix), [c.key for c in checks], ungated


def gate_then_isolating(l1, checker):
    """A1/A2: checker once, then one gated_map per I3a group, then I3a's combine step."""
    design = gate_design(l1)
    pm, agg = i3a_ops(l1)
    prefix = f"{GATE}_"
    ops = checker_ops(design, checker)
    flags, groups, meta = {}, [], {}
    for i, sub in enumerate(pm["prompts"], 1):
        prompt, keys, ungated = gated_group_prompt(sub["prompt"], design, prefix)
        flags[f"g{i}_run"] = ([f"{prefix}{k}" for k in keys], ungated > 0)
        out_key = sub["output_keys"][0]
        groups.append({"name": f"extract_group_{i}", "type": "gated_map", "gate_key": f"g{i}_run",
                       "prompt": prompt, "model": MODEL, "litellm_completion_kwargs": {"temperature": 0.0},
                       "output": {"schema": {out_key: "string"}}})
        meta[f"group_{i}"] = {"name": sub["name"], "gated": len(keys), "ungated": ungated}
    code = "def transform(doc):\n    out = {}\n"
    for flag, (fields, always) in flags.items():
        code += f"    out[{flag!r}] = {always!r} or any(bool(doc.get(k)) for k in {fields!r})\n"
    code += "    return out\n"
    fill_keys = [g["output"]["schema"] and next(iter(g["output"]["schema"])) for g in groups]
    fill = ("def transform(doc):\n"
            f"    return {{k: '' for k in {fill_keys!r} if doc.get(k) is None}}\n")
    ops += [{"name": "group_run_flags", "type": "code_map", "code": code}] + groups + \
           [{"name": "skipped_groups_empty", "type": "code_map", "code": fill}, agg]
    return ops, meta


def per_group_topk(l1, query_list, method, share):
    from docetl.reasoning_optimizer.data_views import TopKView, view_ops
    pm, agg = i3a_ops(l1)
    k = max(1, round(share * MEDIAN_CHARS / CHARS_PER_TOKEN / TOPK_CHUNK_TOKENS))
    copy = {"name": "copy_document_per_group", "type": "code_map",
            "code": ("def transform(doc):\n"
                     f"    return {{f'document_g{{i}}': doc['document'] for i in range(1, {len(pm['prompts']) + 1})}}\n")}
    ops = [copy]
    for i, (sub, query) in enumerate(zip(pm["prompts"], query_list), 1):
        cfg = TopKView(rationale="per-group query", chunk_tokens=TOPK_CHUNK_TOKENS, k=k, method=method, query=query)
        ops += view_ops("topk", cfg, f"document_g{i}", MODEL, prefix=f"g{i}")
        sub["prompt"] = sub["prompt"].replace("{{ input.document }}", f"{{{{ input.document_g{i} }}}}")
        assert f"input.document_g{i}" in sub["prompt"]
    return ops + [pm, agg], {"k_per_group": k, "chunk_tokens": TOPK_CHUNK_TOKENS, "method": method}


def per_group_jev(l1, share):
    pm, agg = i3a_ops(l1)
    index = l1["proposals"]["jev"]["index"]
    ops, meta = [], {}
    for i, sub in enumerate(pm["prompts"], 1):
        cats = group_categories(sub["prompt"])
        cap = int(share * MEDIAN_CHARS / len(cats))
        targets = [{"name": re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_"),
                    "description": f"Passages of the contract relevant to '{name}': {definition}"}
                   for _, name, definition in cats]
        ops.append({"name": f"g{i}_jev", "type": "jev_retrieve", "text_key": "document",
                    "output_key": f"document_g{i}", "targets": targets, "evidence_chars": cap,
                    "policy": "preserve_fill", "max_decisions": JEV_DECISIONS, "regions_per_request": 4,
                    "index": index, "diagnostics_key": f"_g{i}_jev_retrieval"})
        sub["prompt"] = sub["prompt"].replace("{{ input.document }}", f"{{{{ input.document_g{i} }}}}")
        meta[f"group_{i}"] = {"targets": len(targets), "evidence_chars_per_target": cap}
    return ops + [pm, agg], meta


def per_group_jev_tuned(l1):
    """R10: I3a + per-group Jev retrieval with the offline sweep's best config (broad_cs_c3k_d64_long):
    4 broad targets per group, group budget 3k chars x categories, 64 decisions, contract_structure index,
    preserve_fill, 4 regions per request; contracts of at most LONG_N chars skip retrieval (full text)."""
    import jev_retrieval_sweep as js
    pm, agg = i3a_ops(l1)
    groups = js.groups()
    long_text = {"name": "long_contract_text", "type": "code_map",
                 "code": ("def transform(doc):\n"
                          f"    text = doc['document'] or ''\n"
                          f"    return {{'document_long': text if len(text) > {js.LONG_N} else ''}}\n")}
    ops, meta = [long_text], {"sweep_config": "broad_cs_c3k_d64_long", "long_n": js.LONG_N}
    for i, sub in enumerate(pm["prompts"], 1):
        cats = groups[i]
        cap = js.broad_cap(3000, cats)
        targets = js.broad_targets(cats)
        # Short contracts have an empty document_long, so jev_retrieve passes them through with no Jev calls.
        ops.append({"name": f"g{i}_jev", "type": "jev_retrieve", "text_key": "document_long",
                    "output_key": f"document_g{i}", "targets": targets, "evidence_chars": cap,
                    "policy": "preserve_fill", "max_decisions": 64, "regions_per_request": 4, "max_concurrency": 16,
                    "index": "contract_structure", "leaf_chars": 2400, "fanout": 8,
                    "diagnostics_key": f"_g{i}_jev_retrieval"})
        sub["prompt"] = sub["prompt"].replace("{{ input.document }}", f"{{{{ input.document_g{i} }}}}")
        assert f"input.document_g{i}" in sub["prompt"]
        meta[f"group_{i}"] = {"categories": len(cats), "targets": len(targets), "evidence_chars_per_target": cap}
    full = {"name": "short_contract_full_text", "type": "code_map",
            "code": ("def transform(doc):\n"
                     f"    if doc.get('document_long'):\n"
                     f"        return {{}}\n"
                     f"    return {{f'document_g{{i}}': doc['document'] for i in range(1, {len(pm['prompts']) + 1})}}\n")}
    return ops + [full, pm, agg], meta


def view_part(l1, plan):
    ops = l1["plans"][plan]["operations"]
    return deepcopy([op for op in ops if op["name"] != TARGET])


# ---- propose ------------------------------------------------------------------------------

def propose():
    import sweep
    from pydantic import BaseModel
    from docetl.reasoning_optimizer.directives import IsolatingSubtasksDirective, PresenceGateDirective
    from docetl.reasoning_optimizer.directives.agent_utils import agent_completion

    l1 = level1()
    OUT.mkdir(parents=True, exist_ok=True)
    state = json.load(open(CONFIGS)) if CONFIGS.exists() else {}
    proposals, agent_cost = state.get("proposals", {}), state.get("agent_cost", {})
    plans = {}

    def save():
        json.dump({"agent": AGENT, "model": MODEL, "plans": plans, "proposals": proposals, "agent_cost": agent_cost},
                  open(CONFIGS, "w"), indent=1)

    def add(name, family, ops, meta=None):
        plans[name] = {"family": family, "operations": sweep.fix_models(ops), "meta": meta or {}}

    # A1/A2
    for name, checker in (("A1_gate_jev_isolating", "jev"), ("A2_gate_llm_isolating", "llm")):
        ops, meta = gate_then_isolating(l1, checker)
        add(name, f"A:gate_{checker}->isolating", ops, meta)
    # A1b: A1 with the Jev gate threshold lowered to 0.2 (C1's threshold)
    ops, meta = gate_then_isolating(l1, "jev")
    next(op for op in ops if op["type"] == "presence_check")["threshold"] = 0.2
    add("A1b_gate_jev_t02_isolating", "A:gate_jev->isolating", ops, dict(meta, threshold=0.2))

    # A3: C16k -> Jev gate per chunk (agent instantiates the gate on the chunk map)
    c16 = deepcopy(l1["plans"]["chunking_16000"]["operations"])
    chunk_map = next(op for op in c16 if op["name"] == CHUNK_MAP)
    if "chunk_gate" not in proposals:
        directive = PresenceGateDirective()
        rewrite, history, cost = directive.llm_instantiate(chunk_map, AGENT, [], [])
        agent_cost["chunk_gate"] = cost
        proposals["chunk_gate"] = {"design": rewrite.model_dump(), "history": history}
        save()
    from docetl.reasoning_optimizer.instantiate_schemas import PresenceGateInstantiateSchema
    agent_design = PresenceGateInstantiateSchema(**proposals["chunk_gate"]["design"])
    # Jev checker as in GJ2. The directive's skip gates on the gated categories only, so skipping a chunk with
    # nothing present is safe only when every category is gated; otherwise ungated ones (document name,
    # parties, dates) would be dropped from that chunk.
    all_gated = len(agent_design.checks) == len(group_categories(chunk_map["prompt"].replace("\n", " ")))
    design = agent_design.model_copy(update={"checker": "jev", "checker_model": None,
                                             "skip_when_none": agent_design.skip_when_none and all_gated})
    ops = PresenceGateDirective().apply(c16, CHUNK_MAP, design)
    add("A3_chunk16k_gate_per_chunk", "A:chunk16k->gate_jev_per_chunk", ops,
        {"agent_choices": {k: agent_design.model_dump()[k] for k in ("granularity", "checker", "threshold",
                                                                      "chunk_chars", "skip_when_none")},
         "checks": len(design.checks), "document_key": design.document_key,
         "skip_when_none": design.skip_when_none})

    # A5: C16k -> isolating subtasks on the chunk map
    if "chunk_isolating" not in proposals:
        new_ops, history, cost = IsolatingSubtasksDirective().instantiate(
            operators=deepcopy(c16), target_ops=[CHUNK_MAP], agent_llm=AGENT, optimize_goal="acc",
            global_default_model=MODEL, message_history=[], input_file_path=str(DATA),
            pipeline_code={"operations": deepcopy(c16)}, dataset="cuad")
        agent_cost["chunk_isolating"] = cost
        proposals["chunk_isolating"] = {"operations": new_ops, "history": history}
        save()
    ops = proposals["chunk_isolating"]["operations"]
    groups = len(next(op for op in ops if op["type"] == "parallel_map")["prompts"])
    add("A5_chunk16k_isolating", "A:chunk16k->isolating", ops, {"groups": groups})

    # R1-R6: per-group retrieval
    pm, _ = i3a_ops(l1)
    if "group_queries" not in proposals:
        class GroupQueries(BaseModel):
            rationale: str
            queries: list[str]

        groups_text = "\n\n".join(
            f"Group {i} ({sub['name']}):\n" + "\n".join(f"- {n}: {d}" for _, n, d in group_categories(sub["prompt"]))
            for i, sub in enumerate(pm["prompts"], 1))
        prompt = (f"A contract-analysis pipeline extracts clause text spans in {len(pm['prompts'])} groups of "
                  f"categories; each group runs on its own retrieved excerpt of the contract. Each excerpt is "
                  f"built by splitting the contract into {TOPK_CHUNK_TOKENS}-token chunks and keeping the chunks "
                  f"that best match one query (BM25 keyword matching, or embedding similarity).\n\n{groups_text}\n\n"
                  f"Write one retrieval query per group, in group order: a description of the contract passages "
                  f"that contain that group's clauses, using the words such clauses typically contain. Phrase "
                  f"each query as the evidence to find, never as an answer. Return the queries in `queries` "
                  f"(exactly {len(pm['prompts'])}) and a short rationale.")
        history, cost = [{"role": "user", "content": prompt}], 0.0
        for _ in range(3):
            resp, c = agent_completion(model=AGENT, messages=history, response_format=GroupQueries)
            cost += c
            content = resp.choices[0].message.content
            history.append({"role": "assistant", "content": content})
            try:
                parsed = GroupQueries(**json.loads(content))
                if len(parsed.queries) != len(pm["prompts"]) or not all(q.strip() for q in parsed.queries):
                    raise ValueError(f"give exactly {len(pm['prompts'])} non-empty queries")
                proposals["group_queries"] = {**parsed.model_dump(), "history": history}
                break
            except Exception as err:
                history.append({"role": "user", "content": f"Validation error: {err}\nPlease try again."})
        agent_cost["group_queries"] = cost
        if "group_queries" not in proposals:
            raise RuntimeError("agent did not produce valid group queries")
        save()
    queries = proposals["group_queries"]["queries"]
    for b, share in BUDGETS.items():
        for method, tag, num in (("fts", "bm25", {"35": "R1", "70": "R2"}), ("embedding", "emb", {"35": "R3", "70": "R4"})):
            ops, meta = per_group_topk(l1, queries, method, share)
            add(f"{num[b]}_isolating_{tag}_per_group_{b}", f"R:isolating+{tag}_per_group", ops,
                {**meta, "share_per_group": share})
        ops, meta = per_group_jev(l1, share)
        add(f"{ {'35': 'R5', '70': 'R6'}[b] }_isolating_jev_per_group_{b}".replace(" ", ""),
            "R:isolating+jev_per_group", ops, {**meta, "share_per_group": share})

    # R10: per-group Jev retrieval tuned by the offline Jev-only sweep (runs/jev_sweep)
    ops, meta = per_group_jev_tuned(l1)
    add("R10_isolating_jev_broad_cs_c3k_long", "R:isolating+jev_tuned", ops, meta)

    # R7-R9: global view -> I3a
    pm, agg = i3a_ops(l1)
    for name, view in (("R7_view_jev5_isolating", "jev_5"), ("R8_view_bm25_5_isolating", "topk_bm25_5"),
                       ("R9_view_emb5_isolating", "topk_emb_5")):
        add(name, "R:global_view->isolating", view_part(l1, view) + [deepcopy(pm), deepcopy(agg)], {"view": view})

    # B1-B3: view -> GJ2
    gj2 = deepcopy(l1["plans"]["gate_jev_2"]["operations"])
    for name, view in (("B1_view_jev5_gate", "jev_5"), ("B2_view_bm25_5_gate", "topk_bm25_5"),
                       ("B3_view_head5_gate", "head_tail_5")):
        add(name, "B:view->gate_jev", view_part(l1, view) + deepcopy(gj2), {"view": view})

    # C1/C2: GJ2 thresholds
    for name, thr in (("C1_gate_jev_threshold_0.2", 0.2), ("C2_gate_jev_threshold_0.5", 0.5)):
        ops = deepcopy(gj2)
        next(op for op in ops if op["type"] == "presence_check")["threshold"] = thr
        add(name, "C:gate_jev_threshold", ops, {"threshold": thr})

    # D: unchanged reruns
    for plan in ("baseline", "gate_jev_2", "gate_llm_1", "isolating_subtasks_1", "isolating_subtasks_2"):
        add(f"D_{plan}_rerun", "D:rerun", deepcopy(l1["plans"][plan]["operations"]), {"rerun_of": plan})

    save()
    print(f"{len(plans)} plans; agent cost ${sum(agent_cost.values()):.4f}; wrote {CONFIGS}")


# ---- offline checks -----------------------------------------------------------------------

def plans():
    return json.load(open(CONFIGS))["plans"]


def unit():
    """Offline sanity checks, no LLM and no paid calls."""
    from jinja2 import Template
    l1 = level1()
    p = plans()
    # 1. Gated group prompts list only present categories; a group with everything absent is skipped.
    a1 = p["A1_gate_jev_isolating"]
    flags = next(op for op in a1["operations"] if op["name"] == "group_run_flags")
    ns = {}
    exec(flags["code"], ns)
    gate_fields = [k for k in next(op for op in a1["operations"] if op["type"] == "presence_check")["checks"]]
    none_present = {f"{GATE}_{c['key']}": False for c in gate_fields}
    run = ns["transform"](none_present)
    print("all absent ->", run)
    assert run["g1_run"] is True and run["g2_run"] is False and run["g3_run"] is False, run
    one = dict(none_present, **{f"{GATE}_audit_rights": True})
    print("only audit_rights present ->", ns["transform"](one))
    for op in a1["operations"]:
        if op["type"] != "gated_map":
            continue
        doc = {"document": "DOC", **one}
        text = Template(op["prompt"]).render(input=doc)
        listed = sorted(int(n) for n in re.findall(r"(?<![\d.])(\d{1,2})\. [A-Z]", text))
        print(op["name"], "categories shown with only audit_rights present:", listed)
        assert all(n in (1, 2, 3, 4, 8, 34) for n in listed), listed
    fill = next(op for op in a1["operations"] if op["name"] == "skipped_groups_empty")
    ns2 = {}
    exec(fill["code"], ns2)
    assert ns2["transform"]({"subtask_1_output": "x", "subtask_2_output": None}) == {"subtask_2_output": "", "subtask_3_output": ""}
    # 2. Per-group retrieval fields differ (BM25 is free: run the retrieval ops alone on 2 contracts).
    import sweep
    from docetl.runner import DSLRunner
    docs = sorted(sweep.label_free_docs(), key=lambda d: len(d["document"]))[10:12]
    r1 = p["R1_isolating_bm25_per_group_35"]
    ops = [op for op in r1["operations"] if op["type"] != "parallel_map" and op["name"] != "extract_contract_info_aggregate"]
    cfg = sweep.pipeline_config("u", {"operations": ops}, DATA, OUT / "unit_out.json")
    data = OUT / "unit_in.json"
    data.write_text(json.dumps(docs))
    cfg["datasets"]["contracts_data"]["path"] = str(data)
    cost = DSLRunner(cfg, max_threads=4).load_run_save()
    out = json.load(open(OUT / "unit_out.json"))
    for r in out:
        g = [r[f"document_g{i}"] for i in (1, 2, 3)]
        print(r["name"][:40], "chars", len(r["document"]), [len(x) for x in g], "distinct:", len(set(g)))
        assert len(set(g)) == 3 and all(len(x) < len(r["document"]) for x in g)
    assert cost == 0
    pm = next(op for op in r1["operations"] if op["type"] == "parallel_map")
    assert [f"input.document_g{i}" in s["prompt"] for i, s in enumerate(pm["prompts"], 1)] == [True] * 3
    print("unit checks passed")


def estimate_usd(name, plan, s1):
    """Calibrated on Level 1 actual costs (not the Level 1 upper estimates, which ran ~1.75x high)."""
    fam, meta = plan["family"], plan["meta"]
    base = s1["baseline"]["cost"]
    i3a = s1["isolating_subtasks_1"]["cost"]
    jev_gate = s1["gate_jev_2"]["cost"] - base            # Jev presence check over the contract
    llm_gate = s1["gate_llm_1"]["cost"] - base            # flash-lite presence check
    jev_view = s1["jev_5"]["cost"] - s1["head_tail_5"]["cost"]   # Jev retrieval at 8 targets, 128 decisions
    emb = s1["topk_emb_5"]["cost"] - s1["topk_bm25_5"]["cost"]   # one embedding pass over the contracts
    if fam.startswith("D:"):
        return s1[meta["rerun_of"]]["cost"]
    if fam.startswith("C:"):
        return s1["gate_jev_2"]["cost"]
    if fam == "A:gate_jev->isolating":
        return i3a + jev_gate                                # upper: no group skipped
    if fam == "A:gate_llm->isolating":
        return i3a + llm_gate
    if fam == "A:chunk16k->gate_jev_per_chunk":
        return s1["chunking_16000"]["cost"] + jev_gate
    if fam == "A:chunk16k->isolating":
        return i3a * s1["chunking_16000"]["cost"] / base
    if fam.startswith("R:isolating+"):
        share = meta.get("share_per_group", 0.39)
        cost = i3a * share
        if "emb" in fam:
            cost += 3 * emb
        if fam == "R:isolating+jev_tuned":  # offline sweep: groups read ~39% of I3a's text; Jev standalone $0.278
            return i3a * 0.39 + 0.278
        if "jev" in fam:  # per group: ~13 targets, 64 decisions, vs Level 1's 8 targets x 128 decisions
            targets = sum(v["targets"] for k, v in meta.items() if k.startswith("group_"))
            cost += jev_view * targets / 8 * JEV_DECISIONS / 128 * 2  # x2 margin: navigation is not shared
        return cost
    if fam == "R:global_view->isolating":
        view = meta["view"]
        view_cost = {"jev_5": jev_view, "topk_emb_5": emb}.get(view, 0.0)
        return view_cost + i3a * 0.70
    if fam == "B:view->gate_jev":
        return s1[meta["view"]]["cost"] + jev_gate * 0.7
    raise ValueError(fam)


def check():
    import sweep
    from docetl.runner import DSLRunner
    s1 = l1_scores()
    rows = {}
    print(f"{'plan':38s} {'est. $':>7s}  operators")
    for name, plan in plans().items():
        row = {"family": plan["family"], "operators": [op["type"] for op in plan["operations"]],
               "estimated_usd": round(estimate_usd(name, plan, s1), 4)}
        try:
            DSLRunner(sweep.pipeline_config(name, plan, DATA, OUT / name / "output.json"), max_threads=1)
            row["valid"], row["error"] = True, None
        except Exception as err:
            row["valid"], row["error"] = False, f"{type(err).__name__}: {err}"[:1000]
        rows[name] = row
        flag = "" if row["valid"] else "  INVALID: " + row["error"][:150]
        print(f"{name:38s} {row['estimated_usd']:7.3f}  {row['operators']}{flag}")
    total = sum(r["estimated_usd"] for r in rows.values())
    invalid = sorted(n for n, r in rows.items() if not r["valid"])
    json.dump({"checked_at": time.strftime("%Y-%m-%d %H:%M:%S"), "plans": rows, "estimated_total_usd": round(total, 4),
               "invalid": invalid, "calibration": "Level 1 actual costs of the reused parts (see estimate_usd)"},
              open(OUT / "check.json", "w"), indent=1)
    print(f"{len(rows)} plans, estimated total ${total:.2f} (calibrated on Level 1 actual costs); invalid: {invalid or 'none'}")


def smoke(names):
    """Run plans on the 2 shortest contracts and check the output; results -> runs/level2/smoke.json."""
    import sweep
    docs = sorted(sweep.label_free_docs(), key=lambda d: len(d["document"]))[:2]
    all_plans = plans()
    results = json.load(open(OUT / "smoke.json")) if (OUT / "smoke.json").exists() else {}
    for name in names:
        plan = all_plans[name]
        cfg = sweep.pipeline_config(name, plan, DATA, OUT / "smoke_out.json")
        try:
            r = run_plan(deepcopy(cfg), docs)
            rows = r["output"]
            ok = len(rows) == 2 and all(isinstance(x.get("clauses"), list) and
                                        all(isinstance(c, dict) and isinstance(c.get("text_spans"), list)
                                            for c in x["clauses"]) for x in rows)
            extra = {}
            for x in rows:
                for k in ("g1_run", "g2_run", "g3_run"):
                    if k in x:
                        extra.setdefault(k, []).append(x[k])
            results[name] = {"ok": ok, "cost": r["cost"], "entries": [len(x.get("clauses") or []) for x in rows],
                             **extra}
        except Exception as err:
            results[name] = {"ok": False, "error": f"{type(err).__name__}: {err}"[:800]}
        print(name, results[name])
        json.dump(results, open(OUT / "smoke.json", "w"), indent=1)


# ---- run ----------------------------------------------------------------------------------

def run_plan(config: dict, docs: list) -> dict:
    import tempfile
    from docetl.runner import DSLRunner
    with tempfile.TemporaryDirectory() as tmp:
        data, out = Path(tmp) / "in.json", Path(tmp) / "out.json"
        data.write_text(json.dumps(docs))
        config["datasets"]["contracts_data"]["path"] = str(data)
        config["pipeline"]["output"]["path"] = str(out)
        start = time.time()
        cost = DSLRunner(config, max_threads=16).load_run_save()
        return {"cost": cost, "seconds": round(time.time() - start, 1), "output": json.loads(out.read_text())}


def todo_plans(names, max_usd):
    all_plans = plans()
    unknown = set(names) - set(all_plans)
    if unknown:
        raise SystemExit(f"unknown plans: {sorted(unknown)}")
    todo = {n: all_plans[n] for n in (names or all_plans) if not (OUT / n / "run_meta.json").exists()}
    checked = json.load(open(OUT / "check.json")) if (OUT / "check.json").exists() else None
    if checked is None or set(todo) - set(checked["plans"]):
        raise SystemExit("run `check` first (every plan must be validated before it runs)")
    bad = sorted(n for n in todo if not checked["plans"][n]["valid"])
    if bad:
        raise SystemExit(f"invalid plans (see check.json): {bad}")
    est = {n: checked["plans"][n]["estimated_usd"] for n in todo}
    spent = sum(json.load(open(m))["cost"] for m in OUT.glob("*/run_meta.json"))
    smoke_cost = sum(r.get("cost", 0) for r in json.load(open(OUT / "smoke.json")).values()) \
        if (OUT / "smoke.json").exists() else 0.0
    print(f"already spent ${spent:.2f} on runs + ${smoke_cost:.2f} smoke; {len(todo)} to run, "
          f"estimated ${sum(est.values()):.2f}; cap ${max_usd:.2f}")
    if spent + smoke_cost + sum(est.values()) > max_usd:
        raise SystemExit("estimated total exceeds the cap; run fewer plans or raise --max-usd")
    return dict(sorted(todo.items(), key=lambda kv: est[kv[0]]))


def save_result(name, plan, config, result, estimate):
    import yaml
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    json.dump(result["output"], open(d / "output.json", "w"))
    yaml.safe_dump(config, open(d / "pipeline.yaml", "w"), sort_keys=False)
    json.dump({"plan": name, "family": plan["family"], "meta": plan["meta"], "cost": result["cost"],
               "estimated_usd": estimate, "seconds": result["seconds"], "model": MODEL},
              open(d / "run_meta.json", "w"), indent=1)
    print(f"{name}: ${result['cost']:.4f} in {result['seconds']}s")


app = modal.App("cuad-rewrite-sweep-level2")


def _skip(path):
    parts = set(Path(path).parts)
    return (bool(parts & {".venv", "__pycache__", "outputs", "runs", "out", "data", "othersystems"})
            or Path(path).name.startswith(".") or ".chroma-" in str(path))


_image = (
    modal.Image.debian_slim(python_version="3.10")
    .uv_sync()
    .add_local_python_source("experiments", ignore=_skip)
    .add_local_python_source("docetl", ignore=_skip)
)


@app.function(image=_image, secrets=[modal.Secret.from_dotenv(REPO)], timeout=60 * 60, max_containers=16)
def run_plan_remote(config: dict, docs: list, name: str) -> dict:
    try:
        return {"plan": name, **run_plan(config, docs)}
    except Exception as exc:
        import traceback
        return {"plan": name, "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}"}


@app.local_entrypoint()
def modal_main(plans_csv: str = "", max_usd: float = 25.0):
    import sweep
    names = [n for n in plans_csv.split(",") if n]
    docs = sweep.label_free_docs()
    todo = todo_plans(names, max_usd)
    est = json.load(open(OUT / "check.json"))["plans"]
    configs = {n: sweep.pipeline_config(n, p, DATA, OUT / n / "output.json") for n, p in todo.items()}
    calls = [(deepcopy(configs[n]), docs, n) for n in todo]
    failed = {}
    for result in run_plan_remote.starmap(calls, return_exceptions=True, order_outputs=False):
        if isinstance(result, BaseException):
            print(f"FAILED (plan unknown at this point): {str(result)[:300]}")
            continue
        name = result.pop("plan")
        if "error" in result:
            failed[name] = result["error"]
            (OUT / name).mkdir(parents=True, exist_ok=True)
            (OUT / name / "error.txt").write_text(result["error"])
            print(f"FAILED {name}: {result['error'][:300]}")
        else:
            save_result(name, todo[name], configs[name], result, est[name]["estimated_usd"])
    done = [n for n in todo if (OUT / n / "run_meta.json").exists()]
    print(f"saved {len(done)}/{len(todo)}; failed: {sorted(failed) or 'none'}")


def run_local_cached_jev(names, max_usd):
    """Run plans locally; Jev requests may come from the local cache (the offline sweep's requests), LLM calls never
    do. run_meta cost = standalone cost (LLM + every Jev request priced as new); actual spend is recorded too."""
    import sweep
    import jev_retrieval_sweep as js
    import docetl.retrieval.jev_backend as jb
    js.patch_backend()
    orig_init = jb.JevRetrievalBackend.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        self.bypass_cache = False
    jb.JevRetrievalBackend.__init__ = init
    docs = sweep.label_free_docs()
    todo = todo_plans(names, max_usd)
    est = json.load(open(OUT / "check.json"))["plans"]
    for name, plan in todo.items():
        js.COUNT.reset()
        config = sweep.pipeline_config(name, plan, DATA, OUT / name / "output.json")
        result = run_plan(deepcopy(config), docs)
        actual = result["cost"]
        jev_standalone, jev_new = js.COUNT.tokens * js.PRICE, js.COUNT.new_tokens * js.PRICE
        result["cost"] = actual - jev_new + jev_standalone
        save_result(name, plan, config, result, est[name]["estimated_usd"])
        meta_path = OUT / name / "run_meta.json"
        meta = json.load(open(meta_path))
        meta.update(actual_spend=actual, jev_standalone=jev_standalone, jev_new=jev_new,
                    jev_requests=js.COUNT.requests, jev_new_requests=js.COUNT.new_requests,
                    note="cost = standalone (every Jev request priced); actual_spend reused cached Jev responses")
        json.dump(meta, open(meta_path, "w"), indent=1)
        print(f"{name}: standalone ${result['cost']:.4f} (actual ${actual:.4f}; Jev {js.COUNT.requests} requests, "
              f"{js.COUNT.new_requests} new, standalone ${jev_standalone:.3f})")


# ---- score --------------------------------------------------------------------------------

def score():
    from evaluation.cuad import evaluate_results
    rows = []
    for meta_path in sorted(OUT.glob("*/run_meta.json")):
        meta = json.load(open(meta_path))
        m = evaluate_results(meta["plan"], str(meta_path.parent / "output.json"), str(GT), str(DATA))
        rows.append({"plan": meta["plan"], "family": meta["family"], "cost": meta["cost"],
                     "estimated_usd": meta.get("estimated_usd"), "f1": m["avg_f1"], "f1_macro": m["f1_macro"],
                     "precision": m["precision"], "recall": m["recall"],
                     "negative_task_accuracy": m["negative_task_accuracy"],
                     "failed_contracts": m["failed_contracts"], "missing_contracts": m["missing_contracts"],
                     "unknown_clause_types": m["unknown_clause_types"], "tp": m["tp"], "fp": m["fp"], "fn": m["fn"]})
    json.dump(rows, open(OUT / "scores.json", "w"), indent=1)
    print(f"{'plan':38s} {'cost':>8s}  {'F1':>5s} {'macro':>5s} {'P':>5s} {'R':>5s} {'neg':>5s}  fail")
    for r in sorted(rows, key=lambda r: r["cost"]):
        print(f"{r['plan']:38s} ${r['cost']:7.4f}  {r['f1']:.3f} {r['f1_macro']:.3f} {r['precision']:.3f} "
              f"{r['recall']:.3f} {r['negative_task_accuracy']:.3f}  {r['failed_contracts'] + r['missing_contracts']}")


if __name__ == "__main__":
    command, rest = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    if command == "propose":
        propose()
    elif command == "unit":
        unit()
    elif command == "check":
        check()
    elif command == "smoke":
        smoke(rest)
    elif command == "score":
        score()
    elif command == "run-local-cached-jev":
        cap = float(rest[rest.index("--max-usd") + 1]) if "--max-usd" in rest else 25.0
        run_local_cached_jev([a for i, a in enumerate(rest) if a != "--max-usd" and (i == 0 or rest[i - 1] != "--max-usd")], cap)
    else:
        raise SystemExit(__doc__)
