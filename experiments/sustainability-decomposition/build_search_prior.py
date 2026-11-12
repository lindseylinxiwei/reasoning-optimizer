"""Build the search prior from the earlier data-view sweep (runs/view_sweep, runs/model_sweep).

The sweeps ran on train_50 (the 40 search reports + the 10 val reports). Their saved outputs are
rescored here on the 40 search reports only, offline and outside the search: output entries that
match one of the 10 extra reports better than any of the 40 are dropped. Cost on the 40 is the
run's cost x the 40 reports' share of input characters (per-report costs were not saved).

The prior holds only aggregate scores and costs (no labels): an evidence text for the expansion
prompts, how many configurations of each data rewrite the sweep tested, and the seed view.
Usage: python build_search_prior.py [out.json]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO), str(REPO / "experiments/reasoning")]
from evaluation.sustainability import evaluate_results, name_match_score  # noqa: E402

GT = REPO / "experiments/reasoning/data/company_reports_gt.json"
D40 = REPO / "experiments/reasoning/data/train/sustainability.json"
D50 = REPO / "experiments/reasoning/data/train_50/sustainability.json"
OUT = REPO / "experiments/reasoning/priors/sustainability_data_views.json"
# sweep technique -> search rewrite
DIRECTIVE = {"head_tail": "take_head_tail", "topk": "doc_chunking_topk", "jev": "jev_retrieval",
             "summarization": "doc_summarization"}
SEED_HEAD_WORDS = 500


def rescore(tmp: Path):
    gt = {g["id"]: g for g in json.load(open(GT))}
    d40, d50 = json.load(open(D40)), json.load(open(D50))
    ids40 = {d["id"] for d in d40}
    r40 = [gt[i] for i in ids40 if i in gt]
    r10 = [gt[d["id"]] for d in d50 if d["id"] not in ids40 and d["id"] in gt]
    share = sum(len(d["tot_text_raw"]) for d in d40) / sum(len(d["tot_text_raw"]) for d in d50)

    def best(name, reports):
        return max((name_match_score(name, r.get("GT company_name", "")) for r in reports), default=0)

    rows = []
    for sweep in ("view_sweep", "model_sweep"):
        for meta_path in sorted((HERE / "runs" / sweep).glob("*/run_meta.json")):
            res = json.load(open(meta_path.parent / "output.json"))
            for group in res:
                group["economic_activity_summary"] = [
                    e for e in group.get("economic_activity_summary") or []
                    if not (isinstance(e, dict) and (n := (e.get("company_name") or "").strip())
                            and best(n, r10) >= 0.5 and best(n, r10) > best(n, r40))]
            path = tmp / f"{sweep}_{meta_path.parent.name}.json"
            json.dump(res, open(path, "w"))
            m = evaluate_results(meta_path.parent.name, str(path), str(GT), str(D40))
            meta = json.load(open(meta_path))
            rows.append({"sweep": sweep, "plan": meta_path.parent.name, "model": meta.get("model"),
                         "cost": meta["cost"] * share, "activity": m["economic_activity_accuracy"],
                         "combined": m["combined_score"]})
    return rows


def evidence(rows) -> str:
    views = [r for r in rows if r["sweep"] == "view_sweep"]
    models = [r for r in rows if r["sweep"] == "model_sweep" and r["model"] != "jev"]
    lines = ["Earlier data-view sweep (run before this search on these 40 train reports; one view placed "
             "before the pipeline and read by all three operations; LLM steps on gemini-3.5-flash-lite; "
             "combined score, estimated cost on the 40 reports):"]
    for tech, name in DIRECTIVE.items():
        rs = sorted((r for r in views if r["plan"].startswith(tech)), key=lambda r: (-r["combined"], r["cost"]))
        if not rs:
            continue
        b = rs[0]
        lo, hi = min(r["combined"] for r in rs), max(r["combined"] for r in rs)
        clo, chi = min(r["cost"] for r in rs), max(r["cost"] for r in rs)
        lines.append(f"- {name}: {len(rs)} configurations, scores {lo:.3f}-{hi:.3f} at ${clo:.2f}-${chi:.2f}; "
                     f"best {b['combined']:.3f} at ${b['cost']:.3f}")
    if models:
        lo, hi = min(r["combined"] for r in models), max(r["combined"] for r in models)
        b = max(models, key=lambda r: (r["combined"], -r["cost"]))
        lines.append(f"- The smallest head cut (about 360 words) on every model: scores {lo:.3f}-{hi:.3f}; best "
                     f"{b['model'].replace('gemini/', '')} {b['combined']:.3f} at ${b['cost']:.3f}")
    lines.append("Not covered by the sweep: deterministic_doc_compression, doc_chunking, and all task rewrites "
                 "(prompt and operation changes). These numbers come from a view shared by all operations; a "
                 "rewrite applied to one operation, or combined with task rewrites, can behave differently. "
                 f"Each first-layer plan already has a child with a {SEED_HEAD_WORDS}-word head cut on every "
                 "operation.")
    return "\n".join(lines)


def main(out=OUT):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        rows = rescore(Path(tmp))
    counts = {}
    for r in rows:
        if r["sweep"] == "view_sweep":
            tech = next(t for t in DIRECTIVE if r["plan"].startswith(t))
            counts[DIRECTIVE[tech]] = counts.get(DIRECTIVE[tech], 0) + 1
    prior = {"source": "experiments/sustainability-decomposition/runs/{view_sweep,model_sweep}, rescored on "
                       "the 40 train reports by build_search_prior.py",
             "evidence_text": evidence(rows), "prior_counts": counts,
             "seed": {"directive": "take_head_tail", "head_words": SEED_HEAD_WORDS, "document_key": "tot_text_raw"}}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(prior, open(out, "w"), indent=1)
    print(prior["evidence_text"])
    print(counts, "->", out)


if __name__ == "__main__":
    main(*(sys.argv[1:2]))
