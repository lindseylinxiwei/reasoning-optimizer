"""CUAD rewrite sweep, Level 1: every selected rewrite on its own, on the 40 MOAR train contracts.

Data decomposition reuses the shared data views of the Sustainability sweep
(docetl/reasoning_optimizer/data_views.py): head_tail, topk (BM25 and embedding, full ladders
for both) and jev, with levels proposed by the agent from a label-free corpus profile.
Task decomposition uses the rewrite directives, instantiated by the agent:
isolating_subtasks, presence_gate (Jev checker and LLM checker, same questions),
doc_chunking and chunk_header_summary (both at a fixed chunk-size ladder, each chunk with the previous
chunk as context, plus doc_chunking exactly as the agent configured it), isolating_subtasks also at
6, 8 and 12 groups, and a structured variant of 3/6/8/12 groups (same groups and prompts; each group
returns the clauses list and the lists are concatenated in code instead of an LLM aggregation step). arbitrary_rewrite was
dropped: its one valid instance only changed the model, which the sweep fixes anyway.
The agent-heavy rewrites (isolating_subtasks, presence_gate) are instantiated twice.

Every LLM step runs on gemini-3.5-flash-lite; Jev steps on Jev. Inputs are label-free; ground
truth is read only by `score`.

Phases (run from the repository root):
  python experiments/cuad-decomposition/sweep.py propose      # agent calls (~$1-3), writes runs/level1/configs.json
  python experiments/cuad-decomposition/sweep.py check        # offline, free: validates configs, estimates cost
  MODAL_PROFILE=personal modal run experiments/cuad-decomposition/sweep.py --max-usd 25 [--plans-csv a,b]
  python experiments/cuad-decomposition/sweep.py run --local [plan ...]
  python experiments/cuad-decomposition/sweep.py score        # free
"""
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import modal
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
sys.path[:0] = [str(REPO), str(REPO / "experiments/reasoning")]

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

ROOT_YAML = REPO / "experiments/reasoning/pipelines/cuad.yaml"
DATA = REPO / "experiments/reasoning/data/train/cuad.json"      # 40 contracts, fields document/name/id only
GT = REPO / "experiments/reasoning/data/CUAD-master_clauses.csv"
MODEL = "gemini/gemini-3.5-flash-lite"
AGENT = "gemini/gemini-3.8-flash"
TEXT_KEY = "document"
TARGET = "extract_contract_info"
OUT = HERE / "runs" / "level1"
CONFIGS = OUT / "configs.json"
CHUNK_LADDER = [2000, 4000, 8000, 16000]                         # tokens per chunk
REPEATED = ["isolating_subtasks", "presence_gate"]  # instantiated twice (isolating at the directive's own 2-4 groups)
ISOLATING_GROUPS = [6, 8, 12]  # finer splits: the directive's prompt asks for 2-4 subtasks; these ask for exactly N
CHUNK_CONTEXT = {"previous": {"tail": {"count": 1}}}  # ladder: each chunk sees only the chunk before it
# CUAD's 41 clause types are spread over the whole contract (unlike Sustainability's head-only evidence),
# so data views keep 15-70% of the median contract instead of Sustainability's "small, at most 30%".
MIN_SHARE, MAX_SHARE = 0.15, 0.70
LEVEL_GUIDANCE = ("increasing, spread from about 15% to about 70% of the median document (e.g. roughly "
                  "15/25/35/50/70%). The pipeline needs evidence for 41 clause types that can sit anywhere in a "
                  "contract, so even the smallest level must keep a substantial part of it")
JEV_TARGETS = (6, 10)
TOPK_CHUNK_TOKENS = (300, 600)
LLM_TYPES = {"map", "parallel_map", "filter", "reduce", "resolve", "extract", "gated_map", "rank", "cluster",
             "equijoin"}
BASELINE_USD = 0.32  # measured: flash-lite full pass over these 40 contracts (scaled from the smoke test)


def label_free_docs():
    docs = json.load(open(DATA))
    extra = {k for d in docs for k in d} - {"document", "name", "id"}
    if extra:
        raise SystemExit(f"unexpected (possibly label) fields in {DATA}: {sorted(extra)}")
    return docs


def base_ops():
    return yaml.safe_load(open(ROOT_YAML))["operations"]


def fix_models(ops):
    """Every LLM operator (and token-count splitter) on MODEL; Jev and code operators keep theirs."""
    ops = deepcopy(ops)
    for op in ops:
        if op.get("type") in LLM_TYPES and not str(op.get("model", "")).startswith("jev"):
            op["model"] = MODEL
            for sub in op.get("prompts") or []:  # parallel_map: per-prompt model overrides
                if isinstance(sub, dict) and "model" in sub:
                    sub["model"] = MODEL
        if op.get("type") == "split" and isinstance(op.get("method_kwargs"), dict) and "model" in op["method_kwargs"]:
            op["method_kwargs"]["model"] = MODEL
    return ops


def structured_isolating(ops, output_schema):
    """Same groups and prompts as an isolating_subtasks plan, but each group returns the original structured
    list and the lists are concatenated in code (no LLM aggregation step that re-copies the quotes)."""
    ops = deepcopy(ops)
    (field, list_type), = output_schema.items()
    pm = next(op for op in ops if op["type"] == "parallel_map")
    pm["output"] = {"schema": {}}
    keys = []
    for i, sub in enumerate(pm["prompts"], 1):
        key = f"subtask_{i}_{field}"
        sub["output_keys"] = [key]
        pm["output"]["schema"][key] = list_type
        keys.append(key)
    aggregate = next(op for op in ops[ops.index(pm) + 1:] if op["type"] == "map")
    merge = {"name": aggregate["name"].replace("_aggregate", "") + "_merge", "type": "code_map",
             "code": ("def transform(doc):\n"
                      f"    merged = []\n"
                      f"    for key in {keys!r}:\n"
                      f"        merged.extend(doc.get(key) or [])\n"
                      f"    return {{{field!r}: merged}}\n")}
    ops[ops.index(aggregate)] = merge
    return ops


def set_gather(ops, peripheral):
    ops = deepcopy(ops)
    gathers = [op for op in ops if op.get("type") == "gather"]
    if len(gathers) != 1:
        raise ValueError(f"expected one gather operator, found {len(gathers)}")
    gathers[0]["peripheral_chunks"] = deepcopy(peripheral)
    return ops


def gather_factor(peripheral, n_chunks):
    """Chunk-texts sent per document by a gather config: each chunk plus its context chunks."""
    prev = (peripheral or {}).get("previous", {})
    nxt = (peripheral or {}).get("next", {})
    total = 0
    for i in range(n_chunks):
        before = i if "middle" in prev else min(i, prev.get("head", {}).get("count", 0) + prev.get("tail", {}).get("count", 0))
        after = (n_chunks - 1 - i) if "middle" in nxt else min(n_chunks - 1 - i, nxt.get("head", {}).get("count", 0) + nxt.get("tail", {}).get("count", 0))
        total += 1 + before + after
    return total


def set_chunk_tokens(ops, tokens):
    ops = deepcopy(ops)
    splits = [op for op in ops if op.get("type") == "split"]
    if len(splits) != 1:
        raise ValueError(f"expected one split operator, found {len(splits)}")
    splits[0]["method_kwargs"]["num_tokens"] = tokens
    return ops


# ---- phase 1: agent proposals ---------------------------------------------------------

def propose():
    from docetl.reasoning_optimizer.data_views import SWEEP_TYPES, check_view, proposal_prompt, view_ops
    from docetl.reasoning_optimizer.directives import (
        ChunkHeaderSummaryDirective, DocumentChunkingDirective,
        IsolatingSubtasksDirective, JevRetrievalDirective, PresenceGateDirective)
    from docetl.reasoning_optimizer.directives.agent_utils import agent_completion
    from docetl.reasoning_optimizer.directives.jev_retrieval import _jev_explanation, _validation_sample
    from docetl.reasoning_optimizer.instantiate_schemas import JevIndexDesignSchema
    from docetl.retrieval.index_spec import IndexSpec, corpus_profile

    docs = label_free_docs()
    texts = [d[TEXT_KEY] for d in docs]
    lengths = [len(t) for t in texts]
    profile = corpus_profile(texts)
    ops = base_ops()
    OUT.mkdir(parents=True, exist_ok=True)
    configs = json.load(open(CONFIGS)) if CONFIGS.exists() else {}
    # Plans are rebuilt from the (cached) proposals on every run; delete a proposal to re-propose it.
    result = {"agent": AGENT, "model": MODEL, "profile": {k: v for k, v in profile.items() if k != "excerpts"},
              "plans": {}, "proposals": configs.get("proposals", {}),
              "agent_cost": configs.get("agent_cost", {})}
    plans, proposals, agent_cost = result["plans"], result["proposals"], result["agent_cost"]

    def save():
        json.dump(result, open(CONFIGS, "w"), indent=1)

    def add(name, family, ops_list, meta=None):
        plans[name] = {"family": family, "operations": fix_models(ops_list), "meta": meta or {}}

    add("baseline", "baseline", ops)

    # -- data views (same code and prompts as the Sustainability view sweep) --
    def ask(prompt, schema, validate, key):
        history, cost = [{"role": "user", "content": prompt}], 0.0
        for _ in range(3):
            resp, c = agent_completion(model=AGENT, messages=history, response_format=schema)
            cost += c
            content = resp.choices[0].message.content
            history.append({"role": "assistant", "content": content})
            try:
                out = validate(schema(**json.loads(content)))
                agent_cost[key] = agent_cost.get(key, 0) + cost
                return out
            except Exception as err:
                history.append({"role": "user", "content": f"Validation error: {err}\nPlease try again."})
        agent_cost[key] = agent_cost.get(key, 0) + cost
        raise RuntimeError(f"agent failed to produce a valid {schema.__name__}")

    def sweep_ok(technique):
        def check(sweep):
            if technique == "jev" and not JEV_TARGETS[0] <= len(sweep.targets) <= JEV_TARGETS[1]:
                raise ValueError(f"give {JEV_TARGETS[0]}-{JEV_TARGETS[1]} targets, one per group of related clause "
                                 f"types; got {len(sweep.targets)}")
            if technique == "topk" and not TOPK_CHUNK_TOKENS[0] <= sweep.chunk_tokens <= TOPK_CHUNK_TOKENS[1]:
                raise ValueError(f"chunk_tokens must be {TOPK_CHUNK_TOKENS[0]}-{TOPK_CHUNK_TOKENS[1]}")
            for name, config in sweep.configs().items():
                if name.endswith("_half_chunks"):
                    continue  # derived control, same budget as level 3
                try:
                    check_view(technique, config, lengths, max_share=MAX_SHARE, min_share=MIN_SHARE)
                except ValueError as err:
                    raise ValueError(f"{name}: {err}")
            return sweep
        return check

    def prompt(technique, extra=""):
        return proposal_prompt(technique, ops, profile, extra, max_share=MAX_SHARE, level_guidance=LEVEL_GUIDANCE)

    if "head_tail" not in proposals:
        sweep = ask(prompt("head_tail"), SWEEP_TYPES["head_tail"], sweep_ok("head_tail"), "head_tail")
        proposals["head_tail"] = sweep.model_dump(); save()
    sweep = SWEEP_TYPES["head_tail"](**proposals["head_tail"])
    for name, config in sweep.configs().items():
        add(name, "view:head_tail", view_ops("head_tail", config, TEXT_KEY, MODEL) + ops, config.model_dump())

    if "topk" not in proposals:
        sweep = ask(prompt("topk", f"chunk_tokens must be {TOPK_CHUNK_TOKENS[0]}-{TOPK_CHUNK_TOKENS[1]} tokens (about one "
                                     f"clause or section with its context)."), SWEEP_TYPES["topk"], sweep_ok("topk"), "topk")
        proposals["topk"] = sweep.model_dump(); save()
    sweep = SWEEP_TYPES["topk"](**proposals["topk"])
    configs_ = sweep.configs()
    for i, k in enumerate(sweep.k_levels, 1):        # full ladders for both ranking methods
        for method, tag in (("fts", "bm25"), ("embedding", "emb")):
            config = configs_[f"topk_{i}"].model_copy(update={"method": method})
            add(f"topk_{tag}_{i}", f"view:topk_{tag}", view_ops("topk", config, TEXT_KEY, MODEL) + ops,
                config.model_dump())
    half = configs_["topk_3_half_chunks"]
    add(f"topk_{'bm25' if half.method == 'fts' else 'emb'}_3_half_chunks", "view:topk_control",
        view_ops("topk", half, TEXT_KEY, MODEL) + ops, half.model_dump())

    if "jev" not in proposals:
        directive, samples = JevRetrievalDirective(), _validation_sample(texts)

        def design_ok(design):
            spec = IndexSpec(levels=[l.model_dump(exclude_none=True) for l in design.levels],
                             leaf_chars=design.leaf_chars, min_chars=design.min_chars, fanout=design.fanout)
            return spec, directive.check_index(spec, samples), design.rationale

        spec, stats, rationale = ask(directive.index_design_prompt(ops[0], profile), JevIndexDesignSchema,
                                     design_ok, "jev")
        index = spec.model_dump(exclude_none=True)
        extra = (f"{_jev_explanation()}\nIndex already designed: {json.dumps(index)}\n"
                 f"Index built on sample documents: {json.dumps({k: v for k, v in stats.items() if k != 'levels'})}\n"
                 f"Fields: targets ({JEV_TARGETS[0]}-{JEV_TARGETS[1]} evidence targets, one per group of related "
                 f"clause types; each description names the specific clause types of its group, phrased as the "
                 f"evidence needed, never the answer), regions_per_request (1-8; regions x targets per request <= 32 is "
                 f"handled by automatic bundling), cap_levels (evidence_chars per target) and decision_levels "
                 f"(max_decisions, 16-256, enough to reach the evidence given the index shape). Policy is "
                 f"preserve_fill for the levels; an adaptive comparison is added automatically.")
        sweep = ask(prompt("jev", extra), SWEEP_TYPES["jev"], sweep_ok("jev"), "jev")
        proposals["jev"] = {"sweep": sweep.model_dump(), "index": index, "index_stats": stats,
                            "index_rationale": rationale}
        save()
    sweep = SWEEP_TYPES["jev"](**proposals["jev"]["sweep"])
    for name, config in sweep.configs().items():
        add(name, "view:jev", view_ops("jev", config, TEXT_KEY, MODEL, index_spec=proposals["jev"]["index"]) + ops,
            config.model_dump())
    save()

    # -- task decomposition (directives, instantiated like MCTS does) --
    def instantiate(directive, operators, target_ops, key):
        if key in proposals:
            return proposals[key]["operations"]
        new_ops, history, cost = directive.instantiate(
            operators=deepcopy(operators), target_ops=target_ops, agent_llm=AGENT, optimize_goal="acc",
            global_default_model=MODEL, message_history=[], input_file_path=str(DATA),
            pipeline_code={"operations": deepcopy(operators)}, dataset="cuad")
        agent_cost[key] = cost
        proposals[key] = {"operations": new_ops, "history": history}
        save()
        return new_ops

    for rep in (1, 2):
        new = instantiate(IsolatingSubtasksDirective(), ops, [TARGET], f"isolating_subtasks_{rep}")
        add(f"isolating_subtasks_{rep}", "task:isolating_subtasks", new)

        key = f"presence_gate_{rep}"
        if key not in proposals:
            directive = PresenceGateDirective()
            rewrite, history, cost = directive.llm_instantiate(ops[0], AGENT, texts, [])
            agent_cost[key] = cost
            proposals[key] = {"design": rewrite.model_dump(), "history": history}
            save()
        from docetl.reasoning_optimizer.instantiate_schemas import PresenceGateInstantiateSchema
        design = PresenceGateInstantiateSchema(**proposals[key]["design"])
        # Same checks and questions; only the checker differs.
        jev = design.model_copy(update={"checker": "jev", "checker_model": None})
        llm = design.model_copy(update={"checker": "llm", "checker_model": MODEL.split("/", 1)[1]})
        add(f"gate_jev_{rep}", "task:presence_gate_jev", PresenceGateDirective().apply(ops, TARGET, jev),
            {"agent_checker": design.checker, "threshold": design.threshold, "checks": len(design.checks)})
        add(f"gate_llm_{rep}", "task:presence_gate_llm", PresenceGateDirective().apply(ops, TARGET, llm),
            {"agent_checker": design.checker, "checks": len(design.checks)})

    class IsolatingN(IsolatingSubtasksDirective):
        """isolating_subtasks asking for exactly n subtasks instead of the prompt's 2-4."""
        n: int = 3

        def to_string_for_instantiate(self, original_op):
            text = super().to_string_for_instantiate(original_op)
            assert "into 2-4 logical subtasks" in text
            return text.replace("into 2-4 logical subtasks",
                                f"into exactly {self.n} logical subtasks (groups of related categories)")

    for n in ISOLATING_GROUPS:
        key = f"isolating_subtasks_g{n}"
        for _ in range(3):
            new = instantiate(IsolatingN(n=n), ops, [TARGET], key)
            got = len(next(op for op in new if op["type"] == "parallel_map")["prompts"])
            if got == n:
                break
            print(f"{key}: agent made {got} subtasks, retrying")
            proposals.pop(key)
        else:
            raise RuntimeError(f"{key}: agent did not produce {n} subtasks")
        add(key, "task:isolating_subtasks", new, {"groups": n})

    # Structured variants: same groups and prompts, groups return the clauses list, merged in code.
    schema = ops[0]["output"]["schema"]
    for key, name in [("isolating_subtasks_1", "isolating_structured_1"), ("isolating_subtasks_g6", "isolating_structured_g6"),
                      ("isolating_subtasks_g8", "isolating_structured_g8"), ("isolating_subtasks_g12", "isolating_structured_g12")]:
        source = proposals[key]["operations"]
        groups = len(next(op for op in source if op["type"] == "parallel_map")["prompts"])
        add(name, "task:isolating_structured", structured_isolating(source, schema),
            {"groups": groups, "same_groups_as": key})

    chunked = instantiate(DocumentChunkingDirective(), ops, [TARGET], "doc_chunking")
    split = next(op["name"] for op in chunked if op["type"] == "split")
    gather = next(op["name"] for op in chunked if op["type"] == "gather")
    headed = instantiate(ChunkHeaderSummaryDirective(), chunked, [split, gather], "chunk_header_summary")
    agent_size = next(op for op in chunked if op["type"] == "split")["method_kwargs"]["num_tokens"]
    for tokens in CHUNK_LADDER:  # chunk size is the only difference; context = the previous chunk
        add(f"chunking_{tokens}", "task:doc_chunking", set_gather(set_chunk_tokens(chunked, tokens), CHUNK_CONTEXT),
            {"chunk_tokens": tokens, "agent_chunk_tokens": agent_size, "context": CHUNK_CONTEXT})
        add(f"chunk_header_{tokens}", "task:chunk_header_summary",
            set_gather(set_chunk_tokens(headed, tokens), CHUNK_CONTEXT),
            {"chunk_tokens": tokens, "agent_chunk_tokens": agent_size, "context": CHUNK_CONTEXT})
    agent_context = next(op for op in chunked if op["type"] == "gather")["peripheral_chunks"]
    add(f"chunking_agent_{agent_size}", "task:doc_chunking", chunked,  # exactly as the agent configured it
        {"chunk_tokens": agent_size, "agent_chunk_tokens": agent_size, "context": agent_context})

    save()  # chunking plans
    save()
    print(f"{len(plans)} plans; agent cost ${sum(agent_cost.values()):.4f}; wrote {CONFIGS}")


# ---- plans ------------------------------------------------------------------------------

def plans():
    return json.load(open(CONFIGS))["plans"]


def pipeline_config(name, plan, data_path, output_path):
    base = yaml.safe_load(open(ROOT_YAML))
    base.update(default_model=MODEL, operations=deepcopy(plan["operations"]), bypass_cache=True)
    base["datasets"]["contracts_data"]["path"] = str(data_path)
    base["pipeline"]["steps"][0]["operations"] = [op["name"] for op in base["operations"]]
    base["pipeline"]["output"]["path"] = str(output_path)
    return base


def estimate_usd(plan, lengths):
    """Rough upper estimate before running: LLM text passes x baseline, plus Jev and embeddings."""
    ops, fam = plan["operations"], plan["family"]
    median_chars = sorted(lengths)[len(lengths) // 2]
    total_chars = sum(lengths)
    cost = 0.0
    view = next((op for op in ops if op["name"].startswith("view_")), None)
    llm_ops = [op for op in ops if op.get("type") in LLM_TYPES and not str(op.get("model", "")).startswith("jev")]
    if fam.startswith("view:"):
        kept = plan["meta"]
        if view["type"] == "code_map":
            share = min(1, (kept["head_tokens"] + kept["tail_tokens"]) * 4.5 * len(lengths) / total_chars)
        elif view["type"] == "split":
            share = min(1, kept["chunk_tokens"] * kept["k"] * 4.5 * len(lengths) / total_chars)
            if kept["method"] == "embedding":
                cost += total_chars / 4 * 0.15 / 1e6
        else:
            share = min(1, kept["evidence_chars"] * len(kept["targets"]) * len(lengths) / total_chars)
        jev = next((op for op in ops if op["type"] == "jev_retrieve"), None)
        if jev:  # upper bound: every target uses all its decisions, one request each (~$0.0001)
            cost += len(lengths) * len(jev["targets"]) * jev["max_decisions"] * 0.0001
        return cost + BASELINE_USD * max(share, 0.05)
    if fam == "task:doc_chunking" or fam == "task:chunk_header_summary":
        import math
        tokens = plan["meta"]["chunk_tokens"]
        gather = next(op for op in ops if op["type"] == "gather")["peripheral_chunks"]
        doc_tokens = [n / 4.5 for n in lengths]
        sent = 0.0
        for t in doc_tokens:
            n = max(1, math.ceil(t / tokens))
            sent += gather_factor(gather, n) * min(t, tokens) + 1500 * n  # chunk texts + the 41-category prompt
        if fam == "task:chunk_header_summary":
            sent += sum(doc_tokens)  # header/summary pass over every chunk
        return BASELINE_USD * sent / sum(doc_tokens) + 0.1
    if fam == "task:presence_gate_jev":
        return BASELINE_USD + 0.05
    if fam == "task:presence_gate_llm":
        return 2 * BASELINE_USD
    # Each prompt that reads the contract text is one full pass; prompts over earlier outputs are small.
    passes = 0.0
    for op in llm_ops:
        prompts = [p.get("prompt", "") for p in op.get("prompts") or []] or [op.get("prompt", "")]
        passes += sum(1.0 if f"input.{TEXT_KEY}" in p else 0.1 for p in prompts)
    return BASELINE_USD * max(1.0, passes)


def kept_share(plan, lengths):
    """Offline: median share of the contract a data view keeps (None for other plans)."""
    from docetl.reasoning_optimizer.data_views import VIEW_TYPES, kept_chars
    technique = plan["family"].removeprefix("view:")
    technique = {"topk_bm25": "topk", "topk_emb": "topk", "topk_control": "topk"}.get(technique, technique)
    if technique not in VIEW_TYPES:
        return None
    config = VIEW_TYPES[technique](**plan["meta"])
    shares = sorted(kept_chars(technique, config, n) / n for n in lengths)
    return round(shares[len(shares) // 2], 4)


def check():
    """Validate every plan with DocETL, estimate its cost and kept share; save to runs/level1/check.json."""
    from docetl.runner import DSLRunner
    lengths = [len(d[TEXT_KEY]) for d in label_free_docs()]
    rows = {}
    print(f"{'plan':28s} {'family':26s} {'est. $':>7s} {'kept':>6s}  ops")
    for name, plan in plans().items():
        row = {"family": plan["family"], "operators": [op["type"] for op in plan["operations"]],
               "estimated_usd": round(estimate_usd(plan, lengths), 4), "kept_share_median": kept_share(plan, lengths)}
        try:
            DSLRunner(pipeline_config(name, plan, DATA, OUT / name / "output.json"), max_threads=1)
            row["valid"], row["error"] = True, None
        except Exception as err:
            row["valid"], row["error"] = False, f"{type(err).__name__}: {err}"[:1000]
        rows[name] = row
        kept = f"{row['kept_share_median']:.1%}" if row["kept_share_median"] is not None else "-"
        flag = "" if row["valid"] else "  INVALID: " + row["error"][:120]
        print(f"{name:28s} {plan['family']:26s} {row['estimated_usd']:7.2f} {kept:>6s}  {row['operators']}{flag}")
    total = sum(r["estimated_usd"] for r in rows.values())
    invalid = sorted(n for n, r in rows.items() if not r["valid"])
    json.dump({"checked_at": time.strftime("%Y-%m-%d %H:%M:%S"), "plans": rows,
               "estimated_total_usd": round(total, 4), "invalid": invalid}, open(OUT / "check.json", "w"), indent=1)
    print(f"{len(rows)} plans, estimated total ${total:.2f} (rough upper estimate); invalid: {invalid or 'none'}")


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


def save_result(name, plan, config, result):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    json.dump(result["output"], open(d / "output.json", "w"))
    yaml.safe_dump(config, open(d / "pipeline.yaml", "w"), sort_keys=False)
    lengths = [len(doc[TEXT_KEY]) for doc in label_free_docs()]
    json.dump({"plan": name, "family": plan["family"], "meta": plan["meta"], "cost": result["cost"],
               "estimated_usd": round(estimate_usd(plan, lengths), 4), "seconds": result["seconds"],
               "model": MODEL}, open(d / "run_meta.json", "w"), indent=1)
    print(f"{name}: ${result['cost']:.4f} in {result['seconds']}s")


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
    lengths = [len(d[TEXT_KEY]) for d in label_free_docs()]
    est = {n: estimate_usd(p, lengths) for n, p in todo.items()}
    spent = sum(json.load(open(m))["cost"] for m in OUT.glob("*/run_meta.json"))
    print(f"already spent ${spent:.2f} on {len(list(OUT.glob('*/run_meta.json')))} plans; "
          f"{len(todo)} to run, estimated ${sum(est.values()):.2f}; cap ${max_usd:.2f}")
    if spent + sum(est.values()) > max_usd:
        raise SystemExit("estimated total exceeds the cap; run fewer plans or raise --max-usd")
    return dict(sorted(todo.items(), key=lambda kv: est[kv[0]]))


def run_local(names, max_usd=25.0):
    docs = label_free_docs()
    for name, plan in todo_plans(names, max_usd).items():
        config = pipeline_config(name, plan, DATA, OUT / name / "output.json")
        save_result(name, plan, config, run_plan(deepcopy(config), docs))


app = modal.App("cuad-rewrite-sweep")


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
    names = [n for n in plans_csv.split(",") if n]
    docs = label_free_docs()
    todo = todo_plans(names, max_usd)
    configs = {n: pipeline_config(n, p, DATA, OUT / n / "output.json") for n, p in todo.items()}
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
            save_result(name, todo[name], configs[name], result)
    done = [n for n in todo if (OUT / n / "run_meta.json").exists()]
    print(f"saved {len(done)}/{len(todo)}; failed: {sorted(failed) or 'none'}")


# ---- score ----------------------------------------------------------------------------------

def score():
    from evaluation.cuad import evaluate_results
    rows = []
    for meta_path in sorted(OUT.glob("*/run_meta.json")):
        meta = json.load(open(meta_path))
        m = evaluate_results(meta["plan"], str(meta_path.parent / "output.json"), str(GT), str(DATA))
        rows.append({"plan": meta["plan"], "family": meta["family"], "cost": meta["cost"],
                     "estimated_usd": meta.get("estimated_usd"),
                     "f1": m["avg_f1"], "f1_macro": m["f1_macro"], "precision": m["precision"],
                     "recall": m["recall"], "negative_task_accuracy": m["negative_task_accuracy"],
                     "failed_contracts": m["failed_contracts"], "missing_contracts": m["missing_contracts"],
                     "unknown_clause_types": m["unknown_clause_types"], "tp": m["tp"], "fp": m["fp"], "fn": m["fn"]})
    json.dump(rows, open(OUT / "scores.json", "w"), indent=1)
    print(f"{'plan':28s} {'cost':>8s}  {'F1':>5s} {'macro':>5s} {'P':>5s} {'R':>5s} {'neg':>5s}  fail")
    for r in sorted(rows, key=lambda r: r["cost"]):
        print(f"{r['plan']:28s} ${r['cost']:7.4f}  {r['f1']:.3f} {r['f1_macro']:.3f} {r['precision']:.3f} "
              f"{r['recall']:.3f} {r['negative_task_accuracy']:.3f}  {r['failed_contracts'] + r['missing_contracts']}")


if __name__ == "__main__":
    command, rest = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    if command == "propose":
        propose()
    elif command == "check":
        check()
    elif command == "run" and "--local" in rest:
        run_local([a for a in rest if a != "--local"])
    elif command == "score":
        score()
    else:
        raise SystemExit(__doc__)
