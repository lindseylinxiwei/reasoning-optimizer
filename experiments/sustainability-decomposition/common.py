"""Shared loading, tokenization and evidence-finding for the offline study.

Token counts use tiktoken cl100k_base as a stand-in for the Gemini tokenizer.
"""
import json
import pickle
import re
from pathlib import Path

import numpy as np
import tiktoken

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

GT_PATH = Path.home() / "moar-v2/moar_ground_truth_files/sustainability/company_reports_gt.json"
SAMPLE_PATH = HERE.parent / "reasoning/data/train/sustainability.json"

ENUM = [
    "manufacturing", "other", "information_communication", "finance_insurance",
    "agriculture", "energy_supply", "health_social", "trade_repair", "real_estate",
    "professional_scientific", "transportation", "construction", "accommodation_food",
]

SUFFIXES = {
    "ltd", "limited", "inc", "corp", "corporation", "llc", "plc", "bv", "nv", "sa", "ag",
    "gmbh", "se", "co", "company", "group", "holding", "holdings", "the", "spa", "s.a.",
    "a/s", "asa", "ab", "oyj", "kgaa", "sas", "pjsc", "tbk", "berhad", "bhd",
}

# Hand-written cue lexicons per economic activity. They are a coarse proxy:
# generic words appear in most reports, so treat lexicon recall as an upper bound.
LEXICON = {
    "manufacturing": ["manufactur", "factory", "factories", "production plant", "assembly line",
                      "machinery", "steel", "automotive", "semiconductor", "chemicals", "packaging"],
    "information_communication": ["software", "telecommunication", "telecom", "it services",
                                  "cloud", "broadband", "mobile network", "publishing", "media",
                                  "internet", "data centre", "data center"],
    "finance_insurance": ["bank", "banking", "insurance", "insurer", "asset management", "lending",
                          "loans", "deposits", "financial services", "reinsurance", "credit card"],
    "agriculture": ["agricultur", "farm", "farming", "crops", "livestock", "plantation", "palm oil",
                    "dairy", "fisheries", "seeds", "harvest"],
    "energy_supply": ["electricity", "power generation", "utility", "utilities", "renewable energy",
                      "wind farm", "solar park", "gas distribution", "power plant", "grid operator",
                      "energy supplier"],
    "health_social": ["healthcare", "health care", "hospital", "pharmaceutical", "patients", "medical",
                      "clinic", "medicines", "care homes"],
    "trade_repair": ["retail", "retailer", "wholesale", "stores", "distributor", "e-commerce",
                     "supermarket", "shops"],
    "real_estate": ["real estate", "property portfolio", "properties", "reit", "tenants", "landlord",
                    "residential units", "lettable"],
    "professional_scientific": ["consulting", "consultancy", "engineering services", "legal services",
                                "accounting", "advisory", "research services", "audit"],
    "transportation": ["logistics", "shipping", "airline", "freight", "transport", "railway", "port",
                       "fleet", "aviation", "courier"],
    "construction": ["construction", "contractor", "civil engineering", "infrastructure projects",
                     "building projects"],
    "accommodation_food": ["hotel", "restaurant", "hospitality", "catering", "resort", "guests",
                           "food service"],
}

# Phrases that usually introduce a description of what the company does.
BUSINESS_CUES = [
    "about us", "company profile", "our business", "who we are", "business model", "core business",
    "principal activities", "nature of business", "our company", "we are a leading", "is a leading",
    "headquartered in", "our operations", "business overview", "company overview", "at a glance",
    "our activities", "business segments", "our products and services",
]

ENC = tiktoken.get_encoding("cl100k_base")


def load():
    gt = json.load(open(GT_PATH))
    sample_ids = {r["id"] for r in json.load(open(SAMPLE_PATH))}
    for r in gt:
        r["in_sample"] = r["id"] in sample_ids
    return gt


def tokenize_all(gt):
    """Return {id: (token_ids np.array, token_char_starts np.array)} cached on disk."""
    cache = OUT / "tokens.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    res = {}
    for r in gt:
        text = r["tot_text_raw"] or ""
        toks = ENC.encode(text, disallowed_special=())
        starts = np.zeros(len(toks), dtype=np.int64)
        # Byte-level tokens can split a character, so measure decoded prefix lengths in bytes
        # and map back to characters once.
        byte_to_char = np.cumsum([0] + [len(c.encode("utf-8")) for c in text])
        bpos = 0
        for i, t in enumerate(toks):
            starts[i] = bpos
            bpos += len(ENC.decode_single_token_bytes(t))
        starts = np.searchsorted(byte_to_char, starts, side="right") - 1
        res[r["id"]] = (np.array(toks, dtype=np.int32), starts)
    pickle.dump(res, open(cache, "wb"))
    return res


def char_to_tok(starts, c):
    return int(np.searchsorted(starts, c, side="right") - 1)


def normalize(text):
    """Lowercase, drop apostrophes, collapse doubled letters ("Accenture" -> "acenture").

    The GT names contain typos that are mostly dropped double letters, so both sides are
    normalized the same way. Returns the normalized text and, for each normalized
    character, its index in the original text.
    """
    low = text.lower()
    out, idx = [], []
    prev = ""
    for i, ch in enumerate(low):
        if ch in "'\u2019`":
            continue
        if ch == prev and ch.isalpha():
            continue
        out.append(ch)
        idx.append(i)
        prev = ch
    idx.append(len(low))
    return "".join(out), idx


def name_patterns(name):
    """Strict pattern for the cleaned full name, plus a partial pattern on its most distinctive word."""
    name = re.sub(r"\(.*?\)", " ", name)
    words = [w for w in re.split(r"[\s\-&,./()]+", normalize(name)[0]) if w]
    core = [w for w in words if w not in SUFFIXES] or words
    strict = r"(?<![a-z0-9])" + r"[\W_]*".join(re.escape(w) for w in core) + r"(?![a-z0-9])"
    long_words = sorted((w for w in core if len(w) >= 4), key=len, reverse=True)
    partial = None
    if long_words:
        partial = r"(?<![a-z0-9])" + re.escape(long_words[0]) + r"(?![a-z0-9])"
    return strict, partial


def find_spans(text_lower, pattern):
    if not pattern:
        return []
    return [(m.start(), m.end()) for m in re.finditer(pattern, text_lower)]


def term_pattern(term):
    # Prefix match for stems such as "manufactur"; word match otherwise.
    tail = r"" if term.endswith(("manufactur", "agricultur")) else r"(?![a-z0-9])"
    return r"(?<![a-z0-9])" + re.escape(term) + tail


def evidence(r, starts):
    """Evidence spans for one report, in token coordinates [start, end)."""
    text = r["tot_text_raw"] or ""
    low = text.lower()
    norm, norm_idx = normalize(text)

    def to_tok(spans):
        return [(char_to_tok(starts, s), char_to_tok(starts, e - 1) + 1) for s, e in spans]

    def norm_spans(pattern):
        return [(norm_idx[s], norm_idx[e - 1] + 1) for s, e in find_spans(norm, pattern)]

    strict, partial = name_patterns(r["GT company_name"])
    ev = {
        "name_strict": to_tok(norm_spans(strict)),
        "name_partial": to_tok(norm_spans(partial)),
        "cue": to_tok([sp for c in BUSINESS_CUES for sp in find_spans(low, term_pattern(c))]),
    }
    lex = LEXICON.get(r["GT economic_activity"], [])
    ev["lexicon"] = to_tok([sp for t in lex for sp in find_spans(low, term_pattern(t))])
    for k in ev:
        ev[k].sort()
    return ev
