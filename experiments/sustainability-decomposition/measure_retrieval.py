"""Phase 2: offline comparison of text-reduction techniques. No LLM calls.

For each technique and report we compute the kept token intervals, then measure:
  - token share kept,
  - evidence recall: whether a company-name mention, a business-description cue, or a
    GT-activity lexicon term survives (among reports where the full text has one),
  - a classifier proxy: 5-fold cross-validated TF-IDF + logistic regression accuracy for
    GT economic_activity when trained and tested on the kept text only.

Run locate_evidence.py first. Writes out/retrieval_results.json and prints a table.
"""
import json
import pickle
import re
import sys
from collections import Counter

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from common import ENC, OUT, load, tokenize_all

# Query used by search plan 17 (doc_chunking_topk on the filter).
Q_SUST = ('"sustainability report" ESG GRI "Global Reporting Initiative" materiality '
          '"stakeholder engagement" environmental social governance "climate change" GHG '
          '"greenhouse gas" "Scope 1" "Scope 2" "Scope 3" "net zero" "carbon neutrality" '
          'sustainable CSR "corporate social responsibility"')
# Field-specific queries written for this study.
Q_ACT = ("about us company profile who we are our business business model core business principal "
         "activities we are a leading provider products services customers operations segments "
         "headquartered founded industry sector markets subsidiaries revenue")
Q_NAME = ("about this report reporting entity company name group limited ltd inc plc ag gmbh "
          "corporation holdings headquarters registered office legal name")
# Activity queries the search agent wrote for doc_chunking_topk (nodes 4, 9, 12).
Q_NODE4 = 'business overview company profile products services operations industry sector segment line of business main activities core business SASB GICS NAICS NACE SIC manufacturing factory production plant assembly automotive electronics machinery chemicals mining materials logistics transportation freight shipping warehousing courier airline rail port healthcare hospital clinic pharmaceutical biotech medical research information communication IT software telecom data center finance banking insurance investment asset management lending agriculture farming crops livestock agroindustrial biodiesel food processing energy electricity power utility oil gas renewable solar wind generation transmission distribution real estate property development leasing REIT professional scientific consulting engineering legal accounting construction building infrastructure civil engineering accommodation hospitality hotel restaurant retail wholesale trade repair aftermarket'
Q_NODE9 = 'business overview principal activities segments industry sector NAICS SIC manufacturing production factory plant assembly operations supply chain information communication ICT software SaaS cloud telecommunications data center finance insurance banking underwriting loans payments investment agriculture farm crop livestock forestry fishery energy supply utility electricity power generation transmission distribution oil gas renewable solar wind hydro health social healthcare hospital clinic medical services social care trade repair retail wholesale e-commerce after-sales service center real estate property development leasing brokerage REIT professional scientific consulting research engineering laboratory R&D transportation logistics shipping freight airline rail port warehousing construction building infrastructure civil engineering accommodation food hospitality hotel restaurant catering food service'
Q_NODE12 = '"company profile" "business overview" "principal activities" "operating segments" "products and services" industry sector NAICS SIC GICS segment revenue "what we do" our business manufacturing production plant factory logistics transport warehousing software IT telecom banking insurance agriculture farming energy power utility oil gas healthcare hospital clinical retail wholesale trade repair real estate property construction engineering accommodation hotel restaurant'
QUERIES = {"sust": Q_SUST, "act": Q_ACT, "name": Q_NAME, "act+name": Q_ACT + " " + Q_NAME,
           "node4": Q_NODE4, "node9": Q_NODE9, "node12": Q_NODE12}


def bm25_tokens(text):
    # Same preprocessing as DocETL's top_fts sampling.
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip().split()


def merge(intervals):
    out = []
    for s, e in sorted(i for i in intervals if i[1] > i[0]):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(i) for i in out]


class Doc:
    def __init__(self, r, ids, starts):
        self.r, self.ids, self.starts = r, ids, starts
        self.n = len(ids)
        self.text = r["tot_text_raw"] or ""
        self._chunks = {}
        self._scores = {}

    def chunks(self, size):
        if size not in self._chunks:
            bounds = [(s, min(s + size, self.n)) for s in range(0, self.n, size)] or [(0, 0)]
            self._chunks[size] = bounds
        return self._chunks[size]

    def scores(self, size, qname):
        key = (size, qname)
        if key not in self._scores:
            bounds = self.chunks(size)
            corpus = [bm25_tokens(ENC.decode(self.ids[s:e].tolist())) for s, e in bounds]
            if not any(corpus):
                self._scores[key] = np.zeros(len(bounds))
            else:
                self._scores[key] = BM25Okapi(corpus).get_scores(bm25_tokens(QUERIES[qname]))
        return self._scores[key]

    def tok_at_char(self, c):
        return int(np.searchsorted(self.starts, c, side="right") - 1)


# ---- techniques: each returns kept token intervals --------------------------------------

def t_full(d):
    return [(0, d.n)]


def t_head_tail_words(head_w, tail_w):
    def f(d):
        words = [m.span() for m in re.finditer(r"\S+", d.text)]
        if len(words) <= head_w + tail_w:
            return [(0, d.n)]
        head_end = d.tok_at_char(words[head_w - 1][1] - 1) + 1
        tail_start = d.tok_at_char(words[-tail_w][0])
        return [(0, head_end), (tail_start, d.n)]
    return f


def t_head_tail(head, tail=0):
    def f(d):
        return [(0, min(head, d.n))] + ([(max(0, d.n - tail), d.n)] if tail else [])
    return f


def top_chunks(d, size, qname, k):
    bounds = d.chunks(size)
    sc = d.scores(size, qname)
    order = np.argsort(-sc, kind="stable")[:k]
    return [bounds[i] for i in order]


def t_topk(size, qname, k):
    return lambda d: top_chunks(d, size, qname, k)


def t_budget(size, qname, budget, neighbors=0):
    """Top chunks until the token budget is used; neighbors widen each seed chunk."""
    def f(d):
        bounds = d.chunks(size)
        sc = d.scores(size, qname)
        kept, used = set(), 0
        for i in np.argsort(-sc, kind="stable"):
            group = [j for j in range(i - neighbors, i + neighbors + 1)
                     if 0 <= j < len(bounds) and j not in kept]
            cost = sum(bounds[j][1] - bounds[j][0] for j in group)
            if used + cost > budget and kept:
                break
            kept.update(group)
            used += cost
        return [bounds[j] for j in kept]
    return f


def t_head_plus(head, size, qname, budget):
    """Always keep the first `head` tokens, then fill the budget with retrieved chunks."""
    def f(d):
        kept = [(0, min(head, d.n))]
        used = kept[0][1]
        for s, e in [d.chunks(size)[i] for i in np.argsort(-d.scores(size, qname), kind="stable")]:
            if e <= head:
                continue
            s = max(s, head)
            if used + (e - s) > budget:
                break
            kept.append((s, e))
            used += e - s
        return kept
    return f


def t_per_field(size, budget):
    """Half the budget for the name query, half for the activity query."""
    def f(d):
        return (t_budget(size, "name", budget // 2)(d) + t_budget(size, "act", budget // 2)(d))
    return f


TECHNIQUES = {
    "full": t_full,
    # What search plans 15/17 (120+60 words) and 20 (120+100 words) keep.
    "words_head120_tail60": t_head_tail_words(120, 60),
    "words_head120_tail100": t_head_tail_words(120, 100),
}
for h in [250, 500, 1000, 2000, 4000, 8000, 16000]:
    TECHNIQUES[f"head_{h}"] = t_head_tail(h)
for h, t in [(1000, 500), (2000, 1000), (4000, 1000)]:
    TECHNIQUES[f"head_{h}_tail_{t}"] = t_head_tail(h, t)
# Plan 17 (12k chunks, k=8, sustainability query) and the agent's usual 10k-15k / k 5-10.
TECHNIQUES["plan17_c12000_k8_sust"] = t_topk(12000, "sust", 8)
for c, k in [(10000, 5), (15000, 10)]:
    for q in ["sust", "act"]:
        TECHNIQUES[f"topk_c{c}_k{k}_{q}"] = t_topk(c, q, k)
# The activity top-k configs from search nodes 4, 9 and 12.
TECHNIQUES["node4_c10000_k7"] = t_topk(10000, "node4", 7)
TECHNIQUES["node9_c12000_k7"] = t_topk(12000, "node9", 7)
TECHNIQUES["node12_c15000_k6"] = t_topk(15000, "node12", 6)
# Small chunks under a token budget.
for c in [500, 1000, 2000]:
    for q in QUERIES:
        for b in [2000, 4000, 8000]:
            TECHNIQUES[f"bm25_c{c}_{q}_B{b}"] = t_budget(c, q, b)
for c in [500, 1000]:
    for b in [4000, 8000]:
        TECHNIQUES[f"bm25_c{c}_act_B{b}_nbr1"] = t_budget(c, "act", b, neighbors=1)
for h in [1000, 2000]:
    for b in [4000, 8000]:
        TECHNIQUES[f"head{h}+bm25_c1000_act_B{b}"] = t_head_plus(h, 1000, "act", b)
for b in [4000, 8000]:
    TECHNIQUES[f"perfield_c1000_B{b}"] = t_per_field(1000, b)


# ---- measurement ------------------------------------------------------------------------

def covered(spans, kept):
    return any(any(ks <= s and e <= ke for ks, ke in kept) for s, e in spans)


def classifier_accuracy(texts, labels):
    counts = Counter(labels)
    y = [l if counts[l] >= 5 else "rare" for l in labels]
    model = make_pipeline(
        TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.9, max_features=50000),
        LogisticRegression(max_iter=2000, C=10),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    pred = cross_val_predict(model, texts, y, cv=cv)
    return float(np.mean(np.array(pred) == np.array(y)))


def main(selected=None):
    gt = load()
    toks = tokenize_all(gt)
    ev = pickle.load(open(OUT / "evidence.pkl", "rb"))
    docs = [Doc(r, *toks[r["id"]]) for r in gt]
    labels = [d.r["GT economic_activity"] for d in docs]

    results = []
    for name, fn in TECHNIQUES.items():
        if selected and name not in selected:
            continue
        per = []
        texts = []
        for d in docs:
            kept = merge(fn(d))
            k_tok = sum(e - s for s, e in kept)
            e = ev[d.r["id"]]
            name_spans = e["name_strict"] + e["name_partial"]
            per.append({
                "in_sample": d.r["in_sample"],
                "n": d.n, "kept": k_tok,
                "name": covered(name_spans, kept) if name_spans else None,
                "cue": covered(e["cue"], kept) if e["cue"] else None,
                "lexicon": covered(e["lexicon"], kept) if e["lexicon"] else None,
                "segments": len(kept),
            })
            # In document order, which is how a downstream prompt would see it.
            texts.append(" ... ".join(ENC.decode(d.ids[s:e].tolist()) for s, e in kept))
        row = {"technique": name}
        for label, sel in [("all", per), ("sample", [p for p in per if p["in_sample"]])]:
            n = np.array([p["n"] for p in sel])
            kept = np.array([p["kept"] for p in sel])
            row[f"{label}_token_share"] = float(kept.sum() / n.sum())
            row[f"{label}_kept_mean"] = float(kept.mean())
            row[f"{label}_kept_p90"] = float(np.percentile(kept, 90))
            row[f"{label}_kept_max"] = int(kept.max())
            for k in ["name", "cue", "lexicon"]:
                vals = [p[k] for p in sel if p[k] is not None]
                row[f"{label}_{k}_recall"] = float(np.mean(vals)) if vals else None
        row["all_classifier_acc"] = classifier_accuracy(texts, labels)
        results.append(row)
        print(f"{name:34s} share {row['all_token_share']:6.1%}  mean kept {row['all_kept_mean']:8.0f}  "
              f"name {row['all_name_recall']:.0%}  cue {row['all_cue_recall']:.0%}  "
              f"lex {row['all_lexicon_recall']:.0%}  clf {row['all_classifier_acc']:.0%}  "
              f"| sample name {row['sample_name_recall']:.0%}", flush=True)

    # Share of reports that exceed k * chunk_size, i.e. where top-k actually drops text.
    for c, k in [(12000, 8), (10000, 5), (15000, 10)]:
        over = [d for d in docs if d.n > c * k]
        print(f"chunk {c} k {k}: {len(over)}/{len(docs)} reports exceed k*chunk "
              f"({len(over)/len(docs):.0%}); sample {sum(d.r['in_sample'] for d in over)}/40")

    out = OUT / "retrieval_results.json"
    json.dump(results, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main(set(sys.argv[1:]) or None)
