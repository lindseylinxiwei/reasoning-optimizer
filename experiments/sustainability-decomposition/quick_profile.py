"""Label-free, LLM-free corpus profile meant to run in seconds before choosing a rewrite.

Inputs are only what an optimizer has up front: the documents, the text field, and the
operator's output schema. No ground truth is used to make the recommendation; it is
loaded afterwards only to check whether the profile was right.

Signals:
  1. Length: chars/4 token estimate, to know whether trimming matters at all.
  2. Document subject: per document, the capitalized phrase with the highest
     count x inverse document frequency (frequent here, rare in other documents). Where its
     first mention falls tells us whether identity fields (names) live in the head.
  3. Head/full agreement: for growing head windows, does the cheap answer computed from the
     head match the cheap answer computed from the full document? Done for the subject
     entity and for an enum vote over the schema's own label words. The smallest head
     where agreement is high is a candidate window size.

Usage: python quick_profile.py [path/to/dataset.json] [--all]
"""
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "reasoning/data/train/sustainability.json"
TEXT_KEY = "tot_text_raw"
# Taken from the pipeline's determine_economic_activity output schema.
ENUM_LABELS = ["manufacturing", "other", "information_communication", "finance_insurance",
               "agriculture", "energy_supply", "health_social", "trade_repair", "real_estate",
               "professional_scientific", "transportation", "construction", "accommodation_food"]
HEADS = [250, 500, 1000, 2000, 4000, 8000]  # in estimated tokens
CHARS_PER_TOKEN = 4  # measured 4.45-4.5 chars per cl100k token on this corpus

CAP_PHRASE = re.compile(r"\b[A-Z][A-Za-z0-9&'’.\-]+(?:[ \t]+[A-Z][A-Za-z0-9&'’.\-]+){0,3}")
STOP = {"The", "This", "Our", "We", "In", "For", "And", "Of", "To", "A", "As", "On", "By", "With",
        "At", "All", "It", "Its", "Is", "Table", "Contents", "Page", "Figure", "Note", "Notes"}


def phrases(text):
    out = []
    for m in CAP_PHRASE.finditer(text):
        words = [w.strip(".'’-") for w in m.group().split()]
        while words and words[0] in STOP:
            words.pop(0)
        if words and len(" ".join(words)) >= 3:
            out.append((" ".join(words).lower(), m.start()))
    return out


def label_terms():
    terms = {}
    for label in ENUM_LABELS:
        if label == "other":
            continue
        terms[label] = [t for t in label.split("_") if len(t) > 3]
    return terms


def enum_vote(text, terms):
    low = text.lower()
    counts = {lab: sum(len(re.findall(r"\b" + t, low)) for t in ts) for lab, ts in terms.items()}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else None


def profile(docs):
    t0 = time.time()
    texts = [d.get(TEXT_KEY) or "" for d in docs]
    n = len(texts)

    lengths = np.array([len(t) / CHARS_PER_TOKEN for t in texts])
    report = {"docs": n, "tokens_median": int(np.median(lengths)),
              "tokens_p90": int(np.percentile(lengths, 90)), "tokens_max": int(lengths.max())}

    # Document frequency of each phrase across documents.
    per_doc = [phrases(t) for t in texts]
    df = Counter()
    for ph in per_doc:
        df.update({p for p, _ in ph})

    subjects, first_pos, decile_spread = [], [], []
    for t, ph in zip(texts, per_doc):
        if not ph:
            subjects.append(None)
            continue
        counts = Counter(p for p, _ in ph)
        # A document's subject is named often in it and rarely elsewhere; generic headings
        # ("section", "adjusted") are frequent too but appear across many documents.
        rare = [p for p in counts if df[p] <= max(2, 0.1 * n)] or list(counts)
        subj = max(rare, key=lambda p: counts[p] * math.log((n + 1) / df[p]))
        first = min(pos for p, pos in ph if p == subj)
        subjects.append(subj)
        first_pos.append(first / CHARS_PER_TOKEN)
        L = max(1, len(t))
        decile_spread.append(len({min(9, int(10 * pos / L)) for p, pos in ph if p == subj}))

    fp = sorted(first_pos)
    report["subject_first_mention_median_tok"] = int(fp[len(fp) // 2]) if fp else None
    for h in HEADS:
        report[f"subject_first_within_{h}"] = round(sum(x < h for x in fp) / max(1, len(fp)), 3)
    report["subject_deciles_median"] = sorted(decile_spread)[len(decile_spread) // 2] if decile_spread else None

    # Head/full agreement for the subject entity and for the enum label vote.
    terms = label_terms()
    full_votes = [enum_vote(t, terms) for t in texts]
    for h in HEADS:
        heads = [t[: h * CHARS_PER_TOKEN] for t in texts]
        agree_subj = []
        for head, subj in zip(heads, subjects):
            if subj is None:
                continue
            hp = Counter(p for p, _ in phrases(head))
            rare = [p for p in hp if df[p] <= max(2, 0.1 * n)] or list(hp)
            top = max(rare, key=lambda p: hp[p] * math.log((n + 1) / max(1, df[p]))) if hp else None
            agree_subj.append(top == subj)
        report[f"subject_head_agree_{h}"] = round(sum(agree_subj) / max(1, len(agree_subj)), 3)
        votes = [enum_vote(x, terms) for x in heads]
        both = [(v, f) for v, f in zip(votes, full_votes) if f is not None]
        report[f"enum_vote_head_agree_{h}"] = round(sum(v == f for v, f in both) / max(1, len(both)), 3)

    report["seconds"] = round(time.time() - t0, 2)
    report["_subjects"] = subjects
    report["_full_votes"] = full_votes
    return report


def recommend(r):
    """Turn the profile into a plan without looking at labels."""
    lines = []
    if r["tokens_p90"] < 8000:
        return ["Documents are short; trimming will not save much. Leave text as is."]
    head_for_name = next((h for h in HEADS if r[f"subject_first_within_{h}"] >= 0.9), None)
    if head_for_name:
        lines.append(f"The document subject is introduced early (>=90% within {head_for_name} tokens) "
                     f"and repeated throughout (median {r['subject_deciles_median']}/10 deciles). "
                     f"Identity fields such as the company name: use a head window of ~{max(head_for_name, 500)} tokens.")
    else:
        lines.append("The document subject is not reliably introduced early; use retrieval for identity fields.")
    head_for_enum = next((h for h in HEADS if r[f"enum_vote_head_agree_{h}"] >= 0.8), None)
    if head_for_enum:
        lines.append(f"A head of {head_for_enum} tokens already agrees with the full-text label-word vote "
                     f"in >=80% of documents. Classification fields: start from a head window of that size.")
    else:
        lines.append("Label-word evidence is spread out (head never agrees with full text >=80%); "
                     "for classification fields use small-chunk retrieval with a field-specific query, "
                     "and keep the head as well.")
    return lines


def validate(r, docs):
    """Check the label-free guesses against GT. Not used by recommend()."""
    from common import GT_PATH, name_patterns, normalize
    gt = {g["id"]: g for g in json.load(open(GT_PATH))}

    def same_company(subject, name):
        # Typo-tolerant: the GT name's most distinctive word appears in the subject, or vice versa.
        strict, partial = name_patterns(name)
        subj = normalize(subject)[0]
        return bool(re.search(partial or strict, subj)) or subj in normalize(name)[0]

    ok = [bool(s) and same_company(s, gt[d["id"]]["GT company_name"])
          for d, s in zip(docs, r["_subjects"]) if d["id"] in gt]
    vote_ok = [v == gt[d["id"]]["GT economic_activity"] for d, v in zip(docs, r["_full_votes"]) if d["id"] in gt]
    return {"subject_is_gt_company": round(sum(ok) / max(1, len(ok)), 3),
            "full_text_label_vote_accuracy": round(sum(vote_ok) / max(1, len(vote_ok)), 3)}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--all" in sys.argv:
        docs = json.load(open(Path.home() / "moar-v2/moar_ground_truth_files/sustainability/company_reports_gt.json"))
    else:
        docs = json.load(open(args[0] if args else DEFAULT_DATA))
    r = profile(docs)
    print(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, indent=1))
    print("\nRecommendation (label-free):")
    for line in recommend(r):
        print(" -", line)
    print("\nValidation against GT (not used above):", validate(r, docs))
