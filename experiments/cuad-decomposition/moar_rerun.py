"""Rerun the 8 MOAR CUAD search frontier plans (~/Downloads/cuad) on the 40 train contracts with the new
list output format, keeping each plan's original models except Gemini 2.5 -> 3.5 (2.5 is no longer available
to this key). OpenAI models run through OpenAI directly (azure/<m> -> <m>, OPENAI_API_KEY).

Only the output format changes in the prompts: the comma-joined / empty-string instructions and the schema
(see REPLACEMENTS, recorded per plan in runs/moar_rerun/<node>/changes.json). Code steps, split/topk/gather
settings and all other prompt text are MOAR's.

  python experiments/cuad-decomposition/moar_rerun.py build        # convert configs (free)
  python experiments/cuad-decomposition/moar_rerun.py check        # DSLRunner validation + cost estimate (free)
  python experiments/cuad-decomposition/moar_rerun.py smoke NODE [NODE ...]   # 2 shortest contracts, local
  python experiments/cuad-decomposition/moar_rerun.py keycheck     # one tiny OpenAI request
  MODAL_PROFILE=personal modal run experiments/cuad-decomposition/moar_rerun.py --max-usd 8
  python experiments/cuad-decomposition/moar_rerun.py score        # free
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

SRC = Path.home() / "Downloads" / "cuad"
DATA = REPO / "experiments/reasoning/data/train/cuad.json"
GT = REPO / "experiments/reasoning/data/CUAD-master_clauses.csv"
OUT = HERE / "runs" / "moar_rerun"
LLM_TYPES = {"map", "parallel_map", "filter", "reduce", "resolve", "extract", "gated_map"}
SCHEMA_OLD = "list[{clause_type: string, text_span: string}]"
SCHEMA_NEW = "list[{clause_type: string, text_spans: list[string]}]"

STANDARD_OLD = ("If a category is not present or cannot be determined, return an empty string. If there are multiple "
                "text spans for a category, return them as a comma-separated list of text spans.")
STANDARD_NEW = ("Return one entry per category found, with clause_type set to the category name as listed below and "
                "text_spans a list of the matching text spans, each copied verbatim from the contract as a separate "
                "list item (do not join several spans into one string). If a category is not present or cannot be "
                "determined, return no entry for it.")
CUSTOM_28_20 = [  # nodes 28 (map) and 20 (reduce) share this wording
    ("If there are multiple relevant text spans for a category, include each as a separate item in a comma-separated list.",
     "If there are multiple relevant text spans for a category, include each as a separate item in the text_spans list."),
    ("If a category is not present or cannot be determined from the non-redacted portion of the document, return an empty string for that category.",
     "If a category is not present or cannot be determined from the non-redacted portion of the document, return no entry for that category."),
    ("'text_span' (the extracted span)", "'text_spans' (the list of extracted spans)"),
    ("Return the results as a list of objects with fields 'clause_type' and 'text_span'.",
     "Return the results as a list of objects with fields 'clause_type' and 'text_spans'."),
]
# (operation name, old text, new text); each old text must occur exactly once in that operation's prompt.
REPLACEMENTS = {
    34: [("extract_contract_info", STANDARD_OLD, STANDARD_NEW)],
    31: [("reduce_extract_contract_info", STANDARD_OLD, STANDARD_NEW)],
    32: [("extract_contract_info", STANDARD_OLD, STANDARD_NEW)],
    14: [("reduce_extract_contract_info", STANDARD_OLD, STANDARD_NEW)],
    28: [("extract_contract_info", o, n) for o, n in CUSTOM_28_20],
    20: [("reduce_extract_contract_info", o, n) for o, n in CUSTOM_28_20],
    30: [("reduce_extract_contract_info", o, n) for o, n in [
        ("extract all relevant text spans as a comma-separated list of full sentences or paragraphs, not fragments.",
         "extract all relevant text spans as separate items in the text_spans list (full sentences or paragraphs, not fragments)."),
        ("If a category is not present or cannot be determined, return an empty string for that category.",
         "If a category is not present or cannot be determined, return no entry for that category."),
        ("{clause_type: <category>, text_span: <extracted verbatim text>}",
         "{clause_type: <category>, text_spans: [<extracted verbatim text>, ...]}"),
        ("Return your output as a JSON list, where each element contains the category (clause_type) and the verbatim "
         "extracted text (text_span) as specified above. If multiple non-overlapping sections apply to one category, "
         "include all as a comma-separated list of spans. If none found, use empty string.",
         "Return your output as a JSON list, where each element contains the category (clause_type) and the list of "
         "verbatim extracted texts (text_spans) as specified above. If multiple non-overlapping sections apply to one "
         "category, include each as a separate item in text_spans. If none found, return no entry for that category."),
    ]],
    9: [("map_extract_contract_info_chunks", STANDARD_OLD, STANDARD_NEW),
        ("reduce_extract_contract_info",
         "Only include each clause_type once, concatenating multiple spans with commas, and return matching the "
         "schema: clauses: list[{clause_type: string, text_span: string}]",
         "Only include each clause_type once, combining its text spans from all chunks into one text_spans list (one "
         "item per span), and return matching the schema: clauses: list[{clause_type: string, text_spans: list[string]}]")],
}


def map_model(model):
    if not model:
        return model
    m = str(model)
    if m.startswith("gemini/gemini-2.5-flash-lite"):
        return "gemini/gemini-3.5-flash-lite"
    if m.startswith("gemini/gemini-2.5-flash"):
        return "gemini/gemini-3.5-flash"
    if m.startswith("gemini/gemini-2.5-pro"):
        return "gemini/gemini-3.1-pro-preview"
    if m.startswith("azure/"):
        return m.removeprefix("azure/")
    return m


def frontier_points():
    return json.load(open(SRC / "pareto_frontier_cuad.json"))["frontier_points"]


def label_free_docs():
    docs = json.load(open(DATA))
    extra = {k for d in docs for k in d} - {"document", "name", "id"}
    if extra:
        raise SystemExit(f"unexpected (possibly label) fields: {sorted(extra)}")
    return docs


def convert(node, cfg):
    cfg = deepcopy(cfg)
    changes = []
    default = cfg.get("default_model")
    cfg["default_model"] = map_model(default)
    if cfg["default_model"] != default:
        changes.append({"what": "default_model", "old": default, "new": cfg["default_model"]})
    ops = {op["name"]: op for op in cfg["operations"]}
    for op in cfg["operations"]:
        if "model" in op and map_model(op["model"]) != op["model"]:
            changes.append({"what": f"{op['name']}.model", "old": op["model"], "new": map_model(op["model"])})
            op["model"] = map_model(op["model"])
        mk = op.get("method_kwargs")
        if isinstance(mk, dict) and "model" in mk and map_model(mk["model"]) != mk["model"]:
            changes.append({"what": f"{op['name']}.method_kwargs.model", "old": mk["model"], "new": map_model(mk["model"])})
            mk["model"] = map_model(mk["model"])
        if "embedding_model" in op and map_model(op["embedding_model"]) != op["embedding_model"]:
            changes.append({"what": f"{op['name']}.embedding_model", "old": op["embedding_model"],
                            "new": map_model(op["embedding_model"])})
            op["embedding_model"] = map_model(op["embedding_model"])
        schema = (op.get("output") or {}).get("schema") or {}
        if schema.get("clauses") == SCHEMA_OLD:
            schema["clauses"] = SCHEMA_NEW
            changes.append({"what": f"{op['name']}.output.schema.clauses", "old": SCHEMA_OLD, "new": SCHEMA_NEW})
    for name, old, new in REPLACEMENTS[node]:
        prompt = ops[name]["prompt"]
        if prompt.count(old) != 1:
            raise ValueError(f"node {node} {name}: expected the old text once, found {prompt.count(old)}: {old[:80]}")
        ops[name]["prompt"] = prompt.replace(old, new)
        changes.append({"what": f"{name}.prompt", "old": old, "new": new})
    for op in cfg["operations"]:  # nothing of the old format may remain in an LLM prompt
        if op.get("type") in LLM_TYPES:
            p = op.get("prompt", "")
            leftovers = [s for s in ("comma-separated", "commas", "empty string", "text_span:", "(text_span)", "'text_span'")
                         if s in p]
            if leftovers:
                raise ValueError(f"node {node} {op['name']}: old format wording left: {leftovers}")
            if (op.get("output") or {}).get("schema", {}).get("clauses") not in (None, SCHEMA_NEW):
                raise ValueError(f"node {node} {op['name']}: clauses schema not converted")
    cfg["bypass_cache"] = True
    return cfg, changes


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    plans = {}
    for p in frontier_points():
        node = p["node_id"]
        cfg = yaml.safe_load(open(SRC / p["file"].replace(".json", ".yaml")))
        new, changes = convert(node, cfg)
        d = OUT / str(node)
        d.mkdir(parents=True, exist_ok=True)
        yaml.safe_dump(new, open(d / "pipeline.yaml", "w"), sort_keys=False)
        json.dump(changes, open(d / "changes.json", "w"), indent=1)
        plans[node] = {"file": p["file"], "moar_cost": p["cost"], "moar_old_f1": p["f1"]}
        print(f"node {node}: {len(changes)} changes -> {d / 'pipeline.yaml'}")
    json.dump(plans, open(OUT / "plans.json", "w"), indent=1)


def pipeline_config(node, data_path, output_path):
    cfg = yaml.safe_load(open(OUT / str(node) / "pipeline.yaml"))
    cfg["datasets"]["contracts_data"]["path"] = str(data_path)
    cfg["pipeline"]["output"]["path"] = str(output_path)
    return cfg


def estimate_usd(node, plan):
    """MOAR's recorded cost on these 40 contracts, scaled by price where Gemini 2.5 became 3.5."""
    cfg = yaml.safe_load(open(OUT / str(node) / "pipeline.yaml"))
    models = {cfg.get("default_model")} | {op.get("model") for op in cfg["operations"] if op.get("model")}
    factor = 4.0 if any(str(m).startswith("gemini/gemini-3.5-flash-lite") for m in models) else 1.0
    return plan["moar_cost"] * factor * 1.15  # +15%: list output is a little longer


def check():
    from docetl.runner import DSLRunner
    plans = json.load(open(OUT / "plans.json"))
    rows, total = {}, 0.0
    for node, plan in plans.items():
        row = {"file": plan["file"], "estimated_usd": round(estimate_usd(node, plan), 4)}
        cfg = pipeline_config(node, DATA, OUT / node / "output.json")
        try:
            DSLRunner(cfg, max_threads=1)
            row["valid"], row["error"] = True, None
        except Exception as err:
            row["valid"], row["error"] = False, f"{type(err).__name__}: {err}"[:800]
        row["models"] = sorted({cfg.get("default_model")} | {op.get("model") for op in cfg["operations"] if op.get("model")})
        row["operators"] = [op["type"] for op in cfg["operations"]]
        rows[node] = row
        total += row["estimated_usd"]
        print(f"node {node:>3}: est ${row['estimated_usd']:.3f} valid={row['valid']} {row['models']} {row['operators']}"
              + (f" ERROR {row['error'][:150]}" if row["error"] else ""))
    json.dump({"checked_at": time.strftime("%Y-%m-%d %H:%M:%S"), "plans": rows, "estimated_total_usd": round(total, 4)},
              open(OUT / "check.json", "w"), indent=1)
    print(f"estimated total ${total:.2f}")


def run_plan(config, docs):
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


def smoke(nodes):
    docs = sorted(label_free_docs(), key=lambda d: len(d["document"]))[:2]
    for node in nodes:
        r = run_plan(pipeline_config(node, DATA, OUT / "smoke.json"), deepcopy(docs))
        entries = [c for row in r["output"] for c in (row.get("clauses") or [])]
        ok = all(isinstance(c, dict) and isinstance(c.get("text_spans"), list) for c in entries)
        res = {"node": node, "cost": r["cost"], "rows": len(r["output"]), "entries": len(entries), "all_lists": ok,
               "types": [c.get("clause_type") for c in entries][:8]}
        json.dump({**res, "output": r["output"]}, open(OUT / str(node) / "smoke.json", "w"), indent=1)
        print("SMOKE", json.dumps(res))


def keycheck():
    import os
    from litellm import completion
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key or len(key) < 20 or "your" in key.lower() or "xxx" in key.lower():
        raise SystemExit("OPENAI_API_KEY missing or looks like a placeholder")
    try:
        r = completion(model="gpt-4.1-mini", messages=[{"role": "user", "content": "Say ok."}], max_tokens=5)
        print("KEYCHECK ok:", repr(r.choices[0].message.content), "cost ~$0.00001")
    except Exception as err:
        raise SystemExit(f"KEYCHECK failed: {type(err).__name__}: {str(err)[:200]}")


def save_result(node, result, config):
    d = OUT / str(node)
    json.dump(result["output"], open(d / "output.json", "w"))
    plan = json.load(open(OUT / "plans.json"))[str(node)]
    json.dump({"node": int(node), "cost": result["cost"], "seconds": result["seconds"],
               "estimated_usd": round(estimate_usd(str(node), plan), 4)}, open(d / "run_meta.json", "w"), indent=1)
    print(f"node {node}: ${result['cost']:.4f} in {result['seconds']}s")


def todo_nodes(names, max_usd):
    checked = json.load(open(OUT / "check.json"))["plans"]
    plans = json.load(open(OUT / "plans.json"))
    todo = [n for n in (names or plans) if not (OUT / str(n) / "run_meta.json").exists()]
    bad = [n for n in todo if not checked[str(n)]["valid"]]
    if bad:
        raise SystemExit(f"invalid plans: {bad}")
    spent = sum(json.load(open(m))["cost"] for m in OUT.glob("*/run_meta.json"))
    est = sum(checked[str(n)]["estimated_usd"] for n in todo)
    print(f"spent ${spent:.2f}; {len(todo)} to run, estimated ${est:.2f}; cap ${max_usd:.2f}")
    if spent + est > max_usd:
        raise SystemExit("estimated total exceeds the cap")
    return sorted(todo, key=lambda n: checked[str(n)]["estimated_usd"])


app = modal.App("cuad-moar-rerun")


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


@app.function(image=_image, secrets=[modal.Secret.from_dotenv(REPO)], timeout=60 * 60, max_containers=8)
def run_plan_remote(config: dict, docs: list, node: str) -> dict:
    try:
        return {"node": node, **run_plan(config, docs)}
    except Exception as exc:
        import traceback
        return {"node": node, "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}"}


@app.local_entrypoint()
def modal_main(nodes_csv: str = "", max_usd: float = 8.0):
    names = [n for n in nodes_csv.split(",") if n]
    docs = label_free_docs()
    todo = todo_nodes(names, max_usd)
    configs = {n: pipeline_config(n, DATA, OUT / str(n) / "output.json") for n in todo}
    calls = [(deepcopy(configs[n]), docs, str(n)) for n in todo]
    failed = {}
    for result in run_plan_remote.starmap(calls, return_exceptions=True, order_outputs=False):
        if isinstance(result, BaseException):
            print(f"FAILED: {str(result)[:300]}")
            continue
        node = result.pop("node")
        if "error" in result:
            failed[node] = result["error"]
            (OUT / node / "error.txt").write_text(result["error"])
            print(f"FAILED {node}: {result['error'][:300]}")
        else:
            save_result(node, result, configs[node])
    print(f"failed: {sorted(failed) or 'none'}")


def score():
    from evaluation.cuad import evaluate_results
    plans = json.load(open(OUT / "plans.json"))
    rows = []
    for node, plan in plans.items():
        meta_path = OUT / node / "run_meta.json"
        if not meta_path.exists():
            continue
        meta = json.load(open(meta_path))
        m = evaluate_results(f"moar_{node}", str(OUT / node / "output.json"), str(GT), str(DATA))
        rows.append({"node": int(node), "file": plan["file"], "cost": meta["cost"], "f1": m["avg_f1"],
                     "f1_macro": m["f1_macro"], "precision": m["precision"], "recall": m["recall"],
                     "negative_task_accuracy": m["negative_task_accuracy"], "failed_contracts": m["failed_contracts"],
                     "missing_contracts": m["missing_contracts"], "unknown_clause_types": m["unknown_clause_types"]})
    json.dump(rows, open(OUT / "scores.json", "w"), indent=1)
    for r in sorted(rows, key=lambda r: r["cost"]):
        print(f"M{r['node']:<3} ${r['cost']:7.4f}  F1 {r['f1']:.3f}  P {r['precision']:.2f}  R {r['recall']:.2f}  "
              f"fail {r['failed_contracts'] + r['missing_contracts']}")


if __name__ == "__main__":
    command, rest = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    {"build": build, "check": check, "keycheck": keycheck, "score": score}.get(command, lambda: None)() \
        if command in ("build", "check", "keycheck", "score") else (smoke(rest) if command == "smoke" else
                                                                     (_ for _ in ()).throw(SystemExit(__doc__)))
