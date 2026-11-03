"""Phase 1: where do company-name and economic-activity cues appear in tot_text_raw?

Writes out/evidence.pkl (per-report spans, used by phase 2) and out/locate_evidence.json.
"""
import json
import pickle

import numpy as np

from common import OUT, evidence, load, tokenize_all

gt = load()
toks = tokenize_all(gt)

rows = []
ev_all = {}
for r in gt:
    ids, starts = toks[r["id"]]
    n = len(ids)
    ev = evidence(r, starts)
    ev_all[r["id"]] = ev
    row = {"id": r["id"], "in_sample": r["in_sample"], "tokens": n,
           "activity": r["GT economic_activity"], "name": r["GT company_name"]}
    for kind, spans in ev.items():
        firsts = [s for s, _ in spans]
        row[f"{kind}_count"] = len(spans)
        row[f"{kind}_first_tok"] = firsts[0] if firsts else None
        row[f"{kind}_first_frac"] = firsts[0] / n if firsts and n else None
        # Spread: share of 10 equal-length deciles of the document that contain a hit.
        row[f"{kind}_deciles"] = len({min(9, int(10 * s / n)) for s in firsts}) if firsts and n else 0
    rows.append(row)

pickle.dump(ev_all, open(OUT / "evidence.pkl", "wb"))


def summarize(sel, label):
    out = {"subset": label, "reports": len(sel)}
    toks_ = np.array([r["tokens"] for r in sel])
    out["tokens_median"] = int(np.median(toks_))
    out["tokens_p90"] = int(np.percentile(toks_, 90))
    out["tokens_max"] = int(toks_.max())
    for kind in ["name_strict", "name_partial", "cue", "lexicon"]:
        found = [r for r in sel if r[f"{kind}_count"] > 0]
        d = {"found_in_doc": len(found) / len(sel)}
        if found:
            ft = np.array([r[f"{kind}_first_tok"] for r in found])
            ff = np.array([r[f"{kind}_first_frac"] for r in found])
            for t in [250, 500, 1000, 2000, 4000, 8000]:
                d[f"first_within_{t}"] = float((ft < t).mean())
            d["first_tok_median"] = int(np.median(ft))
            d["first_frac_median"] = round(float(np.median(ff)), 4)
            d["mentions_median"] = int(np.median([r[f"{kind}_count"] for r in found]))
            d["deciles_with_hit_median"] = int(np.median([r[f"{kind}_deciles"] for r in found]))
        out[kind] = d
    return out


summary = [summarize(rows, "all_286"), summarize([r for r in rows if r["in_sample"]], "sample_40")]
json.dump({"summary": summary, "rows": rows}, open(OUT / "locate_evidence.json", "w"), indent=1)

for s in summary:
    print(f"\n=== {s['subset']} ({s['reports']} reports) tokens median {s['tokens_median']}, "
          f"p90 {s['tokens_p90']}, max {s['tokens_max']}")
    for kind in ["name_strict", "name_partial", "cue", "lexicon"]:
        d = s[kind]
        line = f"{kind:13s} found {d['found_in_doc']:.0%}"
        if "first_tok_median" in d:
            line += (f" | first hit median tok {d['first_tok_median']}"
                     f" | within 500/1k/2k/4k/8k: "
                     + "/".join(f"{d[f'first_within_{t}']:.0%}" for t in [500, 1000, 2000, 4000, 8000])
                     + f" | mentions median {d['mentions_median']}, deciles {d['deciles_with_hit_median']}/10")
        print(line)
