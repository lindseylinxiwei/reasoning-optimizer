"""Phase 3: run trimming variants of the Sustainability root pipeline and score them.

Each variant prepends one code_map that replaces tot_text_raw with a trimmed view, so the
filter, the activity map and the summary reduce all see the same reduced text (the same
placement as search plans 15/17). Prompts are unchanged.

The search ran on gemini-2.5-flash-lite, which is no longer available to new API users, so
everything here runs on gemini-3.5-flash-lite, including reruns of search plans 13/15/17/20
(their saved configs with every model swapped; plan 20's two gpt-5-nano cascade filters too).

Usage (from the repository root):
  python experiments/sustainability-decomposition/run_variants.py head_4000 head2000_bm25 bm25_c500 plan_15 plan_17 plan_20 plan_13 root
  python experiments/sustainability-decomposition/run_variants.py --score-only
"""
import json
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments/reasoning"))
sys.path.insert(0, str(HERE))

ROOT_YAML = REPO / "experiments/reasoning/pipelines/sustainability.yaml"
DATA = "experiments/reasoning/data/train_50/sustainability.json"
MODEL = "gemini/gemini-3.5-flash-lite"
SEARCH_MODELS = {"gemini/gemini-2.5-flash-lite", "azure/gpt-5-nano"}
RUNS = HERE / "runs"
SEARCH_LOGS = Path.home() / "Downloads/search_logs/sustainability"

# Shared helpers pasted into each code_map (code_map code must be self-contained).
HELPERS = r'''
import re
import tiktoken
_ENC = tiktoken.get_encoding("cl100k_base")
Q_ACT = ("about us company profile who we are our business business model core business principal "
         "activities we are a leading provider products services customers operations segments "
         "headquartered founded industry sector markets subsidiaries revenue")

def _bm25_tokens(text):
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip().split()

def _ranked_chunks(ids, size, start=0):
    from rank_bm25 import BM25Okapi
    bounds = [(s, min(s + size, len(ids))) for s in range(start, len(ids), size)]
    if not bounds:
        return []
    corpus = [_bm25_tokens(_ENC.decode(ids[s:e])) for s, e in bounds]
    if not any(corpus):
        return bounds
    scores = BM25Okapi(corpus).get_scores(_bm25_tokens(Q_ACT))
    return [bounds[i] for i in sorted(range(len(bounds)), key=lambda i: -scores[i])]

def _render(ids, kept):
    # Kept spans in document order, with a marker where text was skipped.
    kept = sorted(kept)
    parts = [_ENC.decode(ids[s:e]) for s, e in kept]
    return "\n[...]\n".join(parts)
'''

def head_code(n):
    return HELPERS + f'''
def transform(input_doc):
    ids = _ENC.encode(input_doc.get("tot_text_raw") or "", disallowed_special=())
    return {{"tot_text_raw": _ENC.decode(ids[:{n}])}}
'''


VARIANTS = {
    "head_1000": head_code(1000),
    "head_2000": head_code(2000),
    "head_4000": head_code(4000),
    "head2000_bm25": HELPERS + r'''
def transform(input_doc):
    ids = _ENC.encode(input_doc.get("tot_text_raw") or "", disallowed_special=())
    head, budget, size = 2000, 4000, 1000
    if len(ids) <= budget:
        return {"tot_text_raw": _ENC.decode(ids)}
    kept, used = [(0, head)], head
    for s, e in _ranked_chunks(ids, size, start=head):
        if used + (e - s) > budget:
            break
        kept.append((s, e))
        used += e - s
    return {"tot_text_raw": _render(ids, kept)}
''',
    "bm25_c500": HELPERS + r'''
def transform(input_doc):
    ids = _ENC.encode(input_doc.get("tot_text_raw") or "", disallowed_special=())
    budget, size = 4000, 500
    if len(ids) <= budget:
        return {"tot_text_raw": _ENC.decode(ids)}
    kept, used = [], 0
    for s, e in _ranked_chunks(ids, size):
        if used + (e - s) > budget:
            break
        kept.append((s, e))
        used += e - s
    return {"tot_text_raw": _render(ids, kept)}
''',
    "root": None,  # full text, no trimming
}

# Variants that reuse another variant's trim step and change some operators' models.
MODEL_OVERRIDES = {
    "head_1000_jev": ("head_1000", {"determine_economic_activity": "jev"}),
}

# Variants that replace the filter and the activity map with one Jev call answering both
# fields over the same trimmed text, followed by a code_filter on the boolean.
JEV_COMBINED = {"head_1000_jev_combined": "head_1000"}

COMBINED_PROMPT = """Answer two questions about the following company document.

Document text:
{{ input.tot_text_raw }}

Output field is_sustainability_report: determine if this document is a sustainability report or contains significant sustainability-related content.
A sustainability report typically includes information about:
- Environmental performance and initiatives
- Social responsibility and community impact
- Corporate governance and sustainability practices
- ESG (Environmental, Social, Governance) metrics
- Climate change initiatives
- Sustainable business practices
- Corporate social responsibility activities
True if this document is primarily a sustainability report or contains substantial sustainability-related content, false otherwise.

Output field economic_activity: determine the primary economic sector/activity of the company. Choose from these exact categories:
- manufacturing
- other
- information_communication
- finance_insurance
- agriculture
- energy_supply
- health_social
- trade_repair
- real_estate
- professional_scientific
- transportation
- construction
- accommodation_food
"""


def combine_with_jev(config):
    ops = {op["name"]: op for op in config["operations"]}
    enum = ops["determine_economic_activity"]["output"]["schema"]["economic_activity"]
    combined = {"name": "classify_report_jev", "type": "map", "model": "jev", "prompt": COMBINED_PROMPT,
                "output": {"schema": {"is_sustainability_report": "boolean", "economic_activity": enum}}}
    keep = {"name": "keep_sustainability_reports", "type": "code_filter",
            "code": "def transform(input_doc):\n    return bool(input_doc.get('is_sustainability_report'))\n"}
    config["operations"] = [op for op in config["operations"]
                            if op["name"] not in ("filter_sustainability_reports", "determine_economic_activity")]
    config["operations"] += [combined, keep]
    steps = config["pipeline"]["steps"][0]["operations"]
    i = steps.index("filter_sustainability_reports")
    steps[i:i + 2] = ["classify_report_jev", "keep_sustainability_reports"]

PLAN_NUMBERS = (13, 15, 17, 20)
# Existing outputs from the search run (gemini-2.5-flash-lite) on the same train_50 sample.
SEARCH_PLANS = {f"search_plan_{n}_2.5": SEARCH_LOGS / f"gemini_2.5_flash_lite_config_{n}.json"
                for n in PLAN_NUMBERS}
SEARCH_COSTS = {}
for row in json.load(open(SEARCH_LOGS / "evaluation_metrics.json")):
    SEARCH_COSTS[f"search_plan_{row['node_id']}_2.5"] = row["cost"]


def swap_models(node):
    if isinstance(node, dict):
        return {k: swap_models(v) for k, v in node.items()}
    if isinstance(node, list):
        return [swap_models(v) for v in node]
    return MODEL if node in SEARCH_MODELS else node


def build_config(name):
    out_dir = RUNS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    if name.startswith("plan_"):
        config = swap_models(yaml.safe_load(open(SEARCH_LOGS / f"gemini_2.5_flash_lite_config_{name[5:]}.yaml")))
        config["default_model"] = MODEL
        config["datasets"]["company_reports"]["path"] = DATA
        config["pipeline"]["output"]["path"] = str(out_dir / "output.json")
        yaml.safe_dump(config, open(out_dir / "pipeline.yaml", "w"), sort_keys=False)
        return config, out_dir
    config = yaml.safe_load(open(ROOT_YAML))
    config["default_model"] = MODEL
    config["datasets"]["company_reports"]["path"] = DATA
    config["pipeline"]["output"]["path"] = str(out_dir / "output.json")
    base, overrides = MODEL_OVERRIDES.get(name, (JEV_COMBINED.get(name, name), {}))
    if name in JEV_COMBINED:
        combine_with_jev(config)
    for op in config["operations"]:
        if op["name"] in overrides:
            op["model"] = overrides[op["name"]]
    code = VARIANTS[base]
    if code:
        op = {"name": f"trim_{base}", "type": "code_map", "code": code}
        config["operations"].insert(0, op)
        config["pipeline"]["steps"][0]["operations"].insert(0, op["name"])
    yaml.safe_dump(config, open(out_dir / "pipeline.yaml", "w"), sort_keys=False)
    return config, out_dir


def run(name):
    from docetl.runner import DSLRunner
    config, out_dir = build_config(name)
    start = time.time()
    cost = DSLRunner(config, max_threads=16).load_run_save()
    meta = {"variant": name, "cost": cost, "seconds": round(time.time() - start, 1), "model": MODEL}
    json.dump(meta, open(out_dir / "run_meta.json", "w"), indent=1)
    print(f"{name}: cost ${cost:.4f} in {meta['seconds']}s", flush=True)


def score_all():
    from evaluation.sustainability import evaluate_results
    gt_path = REPO / "experiments/reasoning/data/company_reports_gt.json"
    data_path = REPO / DATA
    rows = []
    names = list(VARIANTS) + list(MODEL_OVERRIDES) + list(JEV_COMBINED) + [f"plan_{n}" for n in PLAN_NUMBERS]
    outputs = {n: RUNS / n / "output.json" for n in names} | SEARCH_PLANS
    for name, path in outputs.items():
        if not path.exists():
            continue
        m = evaluate_results(name, str(path), str(gt_path), str(data_path))
        meta_path = RUNS / name / "run_meta.json"
        cost = json.load(open(meta_path))["cost"] if meta_path.exists() else SEARCH_COSTS.get(name)
        rows.append({"run": name, "cost": cost} | {k: v for k, v in m.items() if not isinstance(v, dict)})
    RUNS.mkdir(exist_ok=True)
    json.dump(rows, open(RUNS / "scores.json", "w"), indent=1)
    print(f"{'run':20s} {'cost':>8s} | {'activity':>8s} {'names':>6s} {'combined':>8s} {'act excl other':>14s} "
          f"| {'orig act':>8s} {'orig names':>10s} {'orig comb':>9s}")
    for r in sorted(rows, key=lambda r: (r["run"].startswith("search"), r["cost"] or 0)):
        cost = f"${r['cost']:.3f}" if r["cost"] is not None else "?"
        print(f"{r['run']:20s} {cost:>8s} | {r['economic_activity_accuracy']:8.2f} {r['company_name_accuracy']:6.2f} "
              f"{r['combined_score']:8.2f} {r['economic_activity_accuracy_excl_gt_other']:14.2f} "
              f"| {r['original_economic_activity_accuracy']:8.2f} {r['original_company_name_accuracy']:10.2f} "
              f"{r['original_combined_score']:9.2f}")


if __name__ == "__main__":
    load_dotenv(REPO / ".env")
    args = sys.argv[1:]
    if "--score-only" not in args:
        for name in args:
            run(name)
    score_all()
