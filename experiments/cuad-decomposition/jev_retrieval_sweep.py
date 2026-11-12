"""Offline Jev-only sweep for "isolating subtasks + per-group Jev retrieval" on CUAD (no LLM calls).

For each config, one jev_retrieve per I3a group (3 groups) runs over the 40 label-free train contracts and
writes document_g{i}. Coverage of each group's annotated quotes is scored afterwards from the CSV labels
(labels never reach Jev). Jev's response cache stays on, so configs share identical requests; every request
(new or cached) is counted with its input tokens, giving both new spend and the reconstructed standalone cost.

  python experiments/cuad-decomposition/jev_retrieval_sweep.py run [--max-usd 5] [config ...]
  python experiments/cuad-decomposition/jev_retrieval_sweep.py report
"""
import json
import re
import sys
import threading
import time
from pathlib import Path
from statistics import median

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO), str(REPO / "experiments/reasoning")]
from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

DATA = REPO / "experiments/reasoning/data/train/cuad.json"
GT = REPO / "experiments/reasoning/data/CUAD-master_clauses.csv"
L1 = HERE / "runs" / "level1" / "configs.json"
L2 = HERE / "runs" / "level2"
OUT = HERE / "runs" / "jev_sweep"
PRICE = 0.042 / 1e6
I3A_COST = 0.828      # Level 1 actual cost of I3a: 3 groups, each reading the full contract
LONG_N = 40_000       # long-contract-only variant: retrieve only above this many characters


# ---- groups and targets (from prompts only, no labels) --------------------------------------------

def category_lines():
    """[(name, definition)] in the order of the CUAD prompt."""
    prompt = yaml.safe_load(open(REPO / "experiments/reasoning/pipelines/cuad.yaml"))["operations"][0]["prompt"]
    return [(m.group(1).strip(), m.group(2).strip())
            for m in re.finditer(r"(?m)^\s*\d{1,2}\.\s*([^:\n]+):\s*(.+)$", prompt)]


def groups():
    """{group index: [(name, definition)]} following I3a's three prompts."""
    pm = next(o for o in json.load(open(L1))["proposals"]["isolating_subtasks_1"]["operations"]
              if o["type"] == "parallel_map")
    out = {i: [] for i in (1, 2, 3)}
    for name, definition in category_lines():
        hits = [i for i, p in enumerate(pm["prompts"], 1) if name.lower() in p["prompt"].lower()]
        out[hits[0] if hits else 1].append((name, definition))
    return out


def key(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def per_category_targets(cats):
    return [{"name": key(n), "description": f"Passages of the contract containing the {n} provision: {d}"}
            for n, d in cats]


def broad_targets(cats, n_targets=4):
    """Consecutive categories of the group (prompt order is roughly topical) joined into n_targets targets."""
    size = -(-len(cats) // n_targets)
    out = []
    for i in range(0, len(cats), size):
        part = cats[i:i + size]
        names = ", ".join(n for n, _ in part)
        defs = " ".join(f"{n}: {d}" for n, d in part)
        out.append({"name": "_".join(key(n) for n, _ in part)[:60],
                    "description": f"Passages of the contract containing any of these provisions: {names}. {defs}"})
    return out


# ---- configs -------------------------------------------------------------------------------------

def configs():
    paragraph = json.load(open(L1))["proposals"]["jev"]["index"]
    out = {}
    # Order matters: the most generous config of each family runs first, so smaller ones are mostly cached.
    for idx_name, index in (("para", paragraph), ("cs", "contract_structure")):
        for targets in ("cat", "broad"):
            for cap in (12000, 6000, 3000):
                for dec in (128, 64):
                    out[f"{targets}_{idx_name}_c{cap // 1000}k_d{dec}"] = {
                        "targets": targets, "index_name": idx_name, "index": index, "cap": cap, "decisions": dec}
    return out


def broad_cap(cap, cats, n_targets=4):
    """Per-target cap for broad targets so the group's total cap matches the per-category config."""
    return int(cap * len(cats) / min(n_targets, len(cats)))


# ---- run -----------------------------------------------------------------------------------------

class Counter:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.requests = self.tokens = self.new_requests = self.new_tokens = 0


COUNT = Counter()


def patch_backend():
    import docetl.retrieval.jev_backend as jb
    from docetl.operations.utils.cache import cache

    original = jb.JevRetrievalBackend.decide

    def decide(self, payload):
        k = jb.CACHE_PREFIX + jb.hashlib.sha256(jb.json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        was_cached = (not self.bypass_cache) and cache.get(k) is not None
        raw = original(self, payload)
        tokens = (raw.get("usage") or {}).get("input_tokens", 0) if isinstance(raw, dict) else 0
        with COUNT.lock:
            COUNT.requests += 1
            COUNT.tokens += tokens
            if not was_cached:
                COUNT.new_requests += 1
                COUNT.new_tokens += tokens
        return raw

    jb.JevRetrievalBackend.decide = decide


class _Runner:
    config = {"bypass_cache": False}


def retrieve(docs, cfg):
    """Run one jev_retrieve per group over docs; returns docs with document_g1..3."""
    from rich.console import Console
    from docetl.operations.jev_retrieve import JevRetrieveOperation
    gs = groups()
    console = Console(quiet=True)
    for g, cats in gs.items():
        targets = per_category_targets(cats) if cfg["targets"] == "cat" else broad_targets(cats)
        cap = cfg["cap"] if cfg["targets"] == "cat" else broad_cap(cfg["cap"], cats)
        op_cfg = {"name": f"g{g}_jev", "type": "jev_retrieve", "text_key": "document", "output_key": f"document_g{g}",
                  "targets": targets, "evidence_chars": cap, "policy": "preserve_fill",
                  "max_decisions": cfg["decisions"], "regions_per_request": 4, "max_concurrency": 16,
                  "index": cfg["index"], "leaf_chars": 2400, "fanout": 8}
        op = JevRetrieveOperation(_Runner(), op_cfg, "jev", 8, console=console)
        docs, _ = op.execute(docs)
    return docs


def run(names, max_usd):
    from docetl.mcts.label_guard import strip_labels  # noqa: F401  (inputs are already label-free)
    patch_backend()
    docs = json.load(open(DATA))
    assert {k for d in docs for k in d} <= {"document", "name", "id"}
    OUT.mkdir(parents=True, exist_ok=True)
    ledger_path = OUT / "ledger.json"
    ledger = json.load(open(ledger_path)) if ledger_path.exists() else {}
    all_cfgs = configs()
    todo = names or list(all_cfgs)
    for name in todo:
        if name in ledger:
            continue
        spent = sum(v["new_cost"] for v in ledger.values())
        if spent > max_usd - 0.6:
            print(f"stopping before {name}: spent ${spent:.2f} of ${max_usd:.2f}")
            break
        cfg = all_cfgs[name] if name in all_cfgs else long_only_cfg(name, all_cfgs)
        COUNT.reset()
        t = time.time()
        subset = [d for d in docs if len(d["document"]) > LONG_N] if cfg.get("long_only") else docs
        out = retrieve([dict(d) for d in subset], cfg)
        json.dump([{k: d[k] for k in ("id", "name", "document_g1", "document_g2", "document_g3")} for d in out],
                  open(OUT / f"{name}.json", "w"))
        ledger[name] = {**{k: v for k, v in cfg.items() if k != "index"}, "docs_retrieved": len(subset),
                        "requests": COUNT.requests, "tokens": COUNT.tokens, "new_requests": COUNT.new_requests,
                        "new_cost": COUNT.new_tokens * PRICE, "standalone_cost": COUNT.tokens * PRICE,
                        "seconds": round(time.time() - t)}
        json.dump(ledger, open(ledger_path, "w"), indent=1)
        print(f"{name}: {COUNT.requests} requests ({COUNT.new_requests} new), new ${COUNT.new_tokens * PRICE:.3f}, "
              f"standalone ${COUNT.tokens * PRICE:.3f}, {time.time() - t:.0f}s; total new "
              f"${sum(v['new_cost'] for v in ledger.values()):.3f}")


def long_only_cfg(name, all_cfgs):
    base = name.removesuffix("_long")
    return {**all_cfgs[base], "long_only": True}


# ---- metrics -------------------------------------------------------------------------------------

def words(text):
    for t in (".", ",", ";", ":"):
        text = text.replace(t, "")
    return set(text.lower().replace("/", " ").split(" ")) - {""}


def coverage(rows, docs_by_name):
    """Strict and matchable quote coverage, kept share, against each group's annotated quotes."""
    from evaluation.cuad import _doc_key, canonical_clause_type, load_cuad_gold
    gold = load_cuad_gold(str(GT))
    gs = groups()
    group_of = {canonical_clause_type(n)[0]: g for g, cats in gs.items() for n, _ in cats}
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
    total = strict = match = 0
    kept_doc, kept_chars, full_chars = [], 0, 0
    for name, doc in docs_by_name.items():
        r = rows.get(name, {})
        texts = {g: r.get(f"document_g{g}", doc) for g in (1, 2, 3)}
        kept = sum(len(t) for t in texts.values())
        kept_doc.append(kept / (3 * len(doc)))
        kept_chars += kept
        full_chars += 3 * len(doc)
        passages = {g: [words(p) for p in texts[g].split("\n[...]\n")] for g in texts}
        normed = {g: norm(texts[g]) for g in texts}
        for cat, quotes in gold[_doc_key(name)].items():
            g = group_of.get(cat, 1)
            for q in quotes:
                total += 1
                strict += norm(q) in normed[g]
                qw = words(q)
                match += any(len(qw & p) >= 0.5 * len(qw) for p in passages[g]) if qw else 0
    return {"strict": strict / total, "matchable": match / total,
            "kept_median": median(kept_doc), "kept_total": kept_chars / full_chars, "quotes": total}


def report():
    docs = {d["name"]: d["document"] for d in json.load(open(DATA))}
    ledger = json.load(open(OUT / "ledger.json"))
    rows_out = []
    for name, led in ledger.items():
        rows = {r["name"]: r for r in json.load(open(OUT / f"{name}.json"))}
        m = coverage(rows, docs)
        jev = led["standalone_cost"]
        rows_out.append({"config": name, **{k: led[k] for k in ("targets", "index_name", "cap", "decisions")},
                         "long_only": bool(led.get("long_only")), **m,
                         "requests_per_contract": led["requests"] / 40, "jev_cost": jev,
                         "new_cost": led["new_cost"], "est_total": I3A_COST * m["kept_total"] + jev})
    # References: Level 2's R5 / R6 outputs, recomputed with the same metric code.
    for plan, reqs in (("R5_isolating_jev_per_group_35", 4703), ("R6_isolating_jev_per_group_70", 4706)):
        p = L2 / plan / "output.json"
        if p.exists():
            rows = {r["name"]: r for r in json.load(open(p))}
            m = coverage(rows, docs)
            rows_out.append({"config": plan.split("_")[0] + " (Level 2)", "targets": "cat", "index_name": "para",
                             "cap": None, "decisions": 64, "long_only": False, **m,
                             "requests_per_contract": reqs / 40, "jev_cost": reqs * 0.0001, "new_cost": 0,
                             "est_total": I3A_COST * m["kept_total"] + reqs * 0.0001})
    rows_out.append({"config": "full text (I3a)", "targets": "-", "index_name": "-", "cap": None, "decisions": None,
                     "long_only": False, "strict": 1.0, "matchable": 1.0, "kept_median": 1.0, "kept_total": 1.0,
                     "requests_per_contract": 0, "jev_cost": 0, "new_cost": 0, "est_total": I3A_COST})
    json.dump(rows_out, open(OUT / "report.json", "w"), indent=1)
    print(f"{'config':26s} {'strict':>6s} {'match':>6s} {'keptMed':>7s} {'keptTot':>7s} {'req/ct':>7s} "
          f"{'jev$':>6s} {'est$':>6s}")
    for r in sorted(rows_out, key=lambda r: r["est_total"]):
        print(f"{r['config']:26s} {r['strict']:6.1%} {r['matchable']:6.1%} {r['kept_median']:7.1%} "
              f"{r['kept_total']:7.1%} {r['requests_per_contract']:7.1f} {r['jev_cost']:6.3f} {r['est_total']:6.3f}")
    # Pareto set: matchable coverage vs estimated cost.
    front, best = [], -1
    for r in sorted(rows_out, key=lambda r: (r["est_total"], -r["matchable"])):
        if r["matchable"] > best:
            front.append(r["config"]); best = r["matchable"]
    print("pareto (matchable coverage vs est. cost):", front)
    print(f"total new Jev spend: ${sum(v['new_cost'] for v in ledger.values()):.3f}")


if __name__ == "__main__":
    cmd, rest = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    if cmd == "run":
        cap = float(rest[rest.index("--max-usd") + 1]) if "--max-usd" in rest else 5.0
        names = [a for a in rest if not a.startswith("--") and a != str(cap) and not re.fullmatch(r"[\d.]+", a)]
        run(names, cap)
    elif cmd == "report":
        report()
    else:
        raise SystemExit(__doc__)
