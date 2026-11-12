"""Adaptive parameter grid for the MCTS search.

When the agent instantiates a rewrite that has a main parameter, the search runs one child
per selected value of that parameter. Which values run is decided here, from one shared
results table:

- Probe: the first time a rewrite is applied to an operator under a parent model, run the
  agent's value and its neighbours on the ladder (size parameters), or the fixed probe
  values (models, checkers).
- Probe cheap, apply expensive: under an expensive parent model, when the rewrite was
  already tried on the same operator under a cheaper model, run only the best value from
  that context.
- Refine: later applications run untried values between the two best values, or one step
  past the best value when it sits at the edge of what has been tried.
- Prune: scores within `threshold` of the best score are a tie, and a tie goes to the
  cheaper value. Only the cheapest close value is kept; other close values, and worse
  values that cost more than the kept one, are dropped. A value dropped in 2 contexts is
  dropped everywhere for that rewrite and operator.
- Finished: when nothing new is left to try, run only the kept value.
- In the tree: after all values of one application have run, only the variants that are
  Pareto-optimal among themselves under the same tie-to-cheaper rule stay as nodes; the others
  are logged as "pruned" (results kept here, files kept on disk) and leave the tree.

Ladders come from label-free statistics of the 40 train reports (see the design doc).
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Label-free corpus statistics of the 40 Sustainability train reports (defaults; a run replaces them
# with its own documents' statistics through configure_corpus).
MEDIAN_DOC_TOKENS = 33_093
TOKENS_PER_WORD = 1.39
CHARS_PER_TOKEN = 4

DEFAULT_NOISE_THRESHOLD = 0.05

# Shown to the agent when it instantiates a rewrite with a size parameter. Label-free: computed
# from tot_text_raw of the 40 train reports only.
CORPUS_STATS_TEXT = (
    "Document statistics (40 train reports, text only): median 23.5k words (33k tokens), 10th "
    "percentile 9.5k words, 90th percentile 61k words, longest 252k words (405k tokens); about 1.4 "
    "tokens per word. Every report is longer than 8k words; a 16k-word head leaves 32% of reports "
    "whole and a 32k-word head 65%. The text is PDF extraction: 72% of lines are under 40 "
    "characters, 28% hold only numbers (tables), about 3% look like headings, so section "
    "boundaries are unreliable. About 32% of lines repeat within a report (headers, footers)."
)


def corpus_stats_text() -> str:
    return CORPUS_STATS_TEXT


def configure_corpus(docs: Sequence[dict], key: Optional[str] = None) -> str:
    """Replace the default (Sustainability) statistics with label-free ones computed from this
    run's input documents. key: the document field; default the field with the longest text.
    Updates MEDIAN_DOC_TOKENS, TOKENS_PER_WORD and CORPUS_STATS_TEXT, returns the text."""
    global MEDIAN_DOC_TOKENS, TOKENS_PER_WORD, CORPUS_STATS_TEXT
    docs = [d for d in docs if isinstance(d, dict)]
    if not docs:
        return CORPUS_STATS_TEXT
    if key is None:
        lengths: Dict[str, int] = {}
        for d in docs:
            for k, v in d.items():
                if isinstance(v, str):
                    lengths[k] = lengths.get(k, 0) + len(v)
        if not lengths:
            return CORPUS_STATS_TEXT
        key = max(lengths, key=lengths.get)
    texts = [str(d.get(key) or "") for d in docs]
    words = sorted(len(t.split()) for t in texts)
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        sample = texts[: min(len(texts), 20)]
        tok = sum(len(enc.encode(t, disallowed_special=())) for t in sample)
        tpw = tok / max(1, sum(len(t.split()) for t in sample))
    except Exception:
        tpw = TOKENS_PER_WORD
    n = len(words)

    def pct(q: float) -> int:
        return words[min(n - 1, int(q * (n - 1) + 0.5))]

    def k(w: float) -> str:
        return f"{w / 1000:.1f}k" if w >= 1000 else str(int(w))

    med = pct(0.5)
    lines = [l for t in texts for l in t.splitlines() if l.strip()]
    short = sum(len(l) < 40 for l in lines) / max(1, len(lines))
    numeric = sum(bool(l.strip()) and all(c in "0123456789.,%$()-+ /" for c in l.strip()) for l in lines) / max(1, len(lines))
    whole = [(h, sum(w <= h for w in words) / n) for h in (2000, 8000, 16000, 32000)]
    MEDIAN_DOC_TOKENS = max(1, int(med * tpw))
    TOKENS_PER_WORD = round(tpw, 2)
    CORPUS_STATS_TEXT = (
        f"Document statistics ({n} input documents, field {key!r}, text only): median {k(med)} words "
        f"({k(med * tpw)} tokens), 10th percentile {k(pct(0.1))} words, 90th percentile {k(pct(0.9))} words, "
        f"shortest {k(words[0])}, longest {k(words[-1])} words ({k(words[-1] * tpw)} tokens); about {tpw:.2f} "
        f"tokens per word. A head of " + ", ".join(f"{k(h)} words keeps {f:.0%}" for h, f in whole)
        + f" of documents whole. {short:.0%} of non-empty lines are under 40 characters and {numeric:.0%} "
        f"hold only numbers."
    )
    return CORPUS_STATS_TEXT


def grid_prompt(spec: "ParamSpec") -> str:
    ladder = ", ".join(str(v) for v in spec.ladder)
    return (
        f"{corpus_stats_text()}\n\nThe parameter {spec.param} of this rewrite will be run at several "
        f"values from the ladder [{ladder}]: your value and its neighbours on the ladder. Choose the "
        f"starting value from the statistics above and the operation's task; your value is snapped "
        f"to the nearest ladder value."
    )
PRUNE_CONTEXTS_FOR_GLOBAL = 2


@dataclass(frozen=True)
class ParamSpec:
    directive: str
    param: str
    ladder: Tuple[Any, ...]                       # ordered from cheapest to most expensive
    get: Callable[[Any], Any]                     # rewrite -> current value
    set: Callable[[Any, Any], Any]                # (rewrite, value) -> new rewrite (copy)
    sized: bool = True                            # True: agent picks the start, probe = neighbours
    probe: Tuple[Any, ...] = ()                   # fixed probe values when sized is False
    kept_tokens: Optional[Callable[[Any], float]] = None   # value -> tokens kept per document

    def snap(self, value: Any) -> Any:
        """Nearest ladder value (by ratio for numbers)."""
        if value in self.ladder:
            return value
        if isinstance(value, (int, float)) and all(isinstance(v, (int, float)) for v in self.ladder):
            return min(self.ladder, key=lambda v: abs((v / max(value, 1e-9)) - 1) if value else v)
        return self.ladder[0]


def _copy_set(attr: str) -> Callable[[Any, Any], Any]:
    def setter(rewrite, value):
        return rewrite.model_copy(update={attr: value}, deep=True)
    return setter


def _topk_get(rw) -> int:
    return int(rw.chunk_size) * int(rw.topk_config.k)


def _topk_set(rw, budget):
    k = max(1, round(budget / max(1, int(rw.chunk_size))))
    topk = rw.topk_config.model_copy(update={"k": k})
    return rw.model_copy(update={"topk_config": topk}, deep=True)


def _gate_get(rw):
    return (rw.checker, rw.checker_model if rw.checker == "llm" else None)


def _gate_set(rw, value):
    checker, model = value
    return rw.model_copy(update={"checker": checker, "checker_model": model}, deep=True)


GRIDS: Dict[str, ParamSpec] = {
    spec.directive: spec
    for spec in [
        ParamSpec(
            "take_head_tail", "head_words", (500, 1000, 2000, 4000, 8000, 16000),
            get=lambda rw: int(rw.head_words), set=_copy_set("head_words"),
            kept_tokens=lambda v: v * TOKENS_PER_WORD,
        ),
        ParamSpec(
            "doc_chunking_topk", "chunk_size*k", (1000, 2000, 4000, 8000, 16000),
            get=_topk_get, set=_topk_set, kept_tokens=lambda v: v,
        ),
        ParamSpec(
            "jev_retrieval", "evidence_chars", (4000, 8000, 16000, 32000, 64000),
            get=lambda rw: int(rw.evidence_chars), set=_copy_set("evidence_chars"),
            kept_tokens=lambda v: v / CHARS_PER_TOKEN,
        ),
        ParamSpec(
            "doc_chunking", "chunk_size", (2000, 4000, 8000, 16000),
            get=lambda rw: int(rw.chunk_size), set=_copy_set("chunk_size"),
        ),
        ParamSpec(
            "doc_summarization", "model",
            ("gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.6-flash"),
            get=lambda rw: str(rw.model).replace("gemini/", ""), set=_copy_set("model"),
            sized=False, probe=("gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.6-flash"),
        ),
        ParamSpec(
            "presence_gate", "checker",
            (("jev", None), ("llm", "gemini-3.1-flash-lite"), ("llm", "gemini-3.5-flash-lite")),
            get=_gate_get, set=_gate_set,
            sized=False, probe=(("jev", None), ("llm", "gemini-3.1-flash-lite")),
        ),
    ]
}


def _key(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


@dataclass
class Result:
    value: Any
    cost: Optional[float]
    score: Optional[float]      # None when the child failed


def prune(results: Sequence[Result], threshold: float) -> Tuple[Optional[Any], set]:
    """Return (kept value, pruned value keys) for one context under the tie-to-cheaper rule.

    Repeated values are averaged. Failed values (score None) are pruned."""
    agg: Dict[str, List[Result]] = {}
    for r in results:
        agg.setdefault(_key(r.value), []).append(r)
    pruned = {k for k, rs in agg.items() if all(r.score is None for r in rs)}
    scored = []
    for k, rs in agg.items():
        ok = [r for r in rs if r.score is not None]
        if ok:
            scored.append((k, rs[0].value,
                           sum(r.cost or 0.0 for r in ok) / len(ok),
                           sum(r.score for r in ok) / len(ok)))
    if not scored:
        return None, pruned
    best = max(s for _, _, _, s in scored)
    close = [x for x in scored if x[3] >= best - threshold]
    kept = min(close, key=lambda x: (x[2], -x[3]))
    for k, _, cost, score in scored:
        if k == kept[0]:
            continue
        if score >= best - threshold or cost > kept[2]:
            pruned.add(k)
    return kept[1], pruned


@dataclass
class ParamTable:
    """Shared results table, persisted as JSONL so a resumed search keeps it."""

    path: Optional[str] = None
    threshold: float = DEFAULT_NOISE_THRESHOLD
    rows: List[dict] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @classmethod
    def load(cls, path: Optional[str], threshold: float = DEFAULT_NOISE_THRESHOLD) -> "ParamTable":
        table = cls(path=path, threshold=threshold)
        if path and Path(path).exists():
            with open(path) as f:
                table.rows = [json.loads(line) for line in f if line.strip()]
        return table

    def record(self, directive: str, op: str, parent_model: str, value: Any,
               cost: Optional[float], score: Optional[float], node_id: str = "") -> None:
        row = {"directive": directive, "op": op, "parent_model": parent_model, "value": value,
               "cost": cost, "score": score, "node_id": node_id}
        row = json.loads(json.dumps(row))      # same form in memory as after a reload
        with self._lock:
            self.rows.append(row)
            if self.path:
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a") as f:
                    f.write(json.dumps(row) + "\n")

    # ---- queries -------------------------------------------------------------------------

    def _results(self, directive: str, op: str, parent_model: Optional[str] = None) -> List[Result]:
        return [Result(_value(r["value"]), r["cost"], r["score"]) for r in self.rows
                if r["directive"] == directive and r["op"] == op
                and (parent_model is None or r["parent_model"] == parent_model)]

    def contexts(self, directive: str, op: str) -> List[str]:
        return sorted({r["parent_model"] for r in self.rows
                       if r["directive"] == directive and r["op"] == op})

    def globally_pruned(self, directive: str, op: str) -> set:
        counts: Dict[str, int] = {}
        for ctx in self.contexts(directive, op):
            _, pruned = prune(self._results(directive, op, ctx), self.threshold)
            for k in pruned:
                counts[k] = counts.get(k, 0) + 1
        return {k for k, n in counts.items() if n >= PRUNE_CONTEXTS_FOR_GLOBAL}

    def choose(self, spec: ParamSpec, op: str, parent_model: str, agent_value: Any,
               model_cost_rank: Optional[Dict[str, float]] = None,
               expensive_parent: bool = False) -> List[Any]:
        """Values to run for this application of the rewrite."""
        with self._lock:
            ladder = list(spec.ladder)
            keys = [_key(v) for v in ladder]
            dead = self.globally_pruned(spec.directive, op)
            here = self._results(spec.directive, op, parent_model)

            if not here:
                cheaper = self._cheaper_context(spec, op, parent_model, model_cost_rank)
                if expensive_parent and cheaper is not None:
                    return self._best_values(spec, op, cheaper, dead, n=1) or [spec.snap(agent_value)]
                return [v for v in self._probe(spec, agent_value) if _key(v) not in dead] \
                    or [spec.snap(agent_value)]

            kept, pruned = prune(here, self.threshold)
            tried = {_key(r.value) for r in here}
            blocked = tried | pruned | dead
            if kept is None:            # every value failed here: try one untried probe value
                rest = [v for v, k in zip(ladder, keys) if k not in blocked]
                return rest[:1]
            alive = [r for r in here if r.score is not None and _key(r.value) not in pruned | dead]
            ranked = sorted({_key(r.value): r for r in alive}.values(), key=lambda r: -r.score)
            anchors = [kept] + [r.value for r in ranked if _key(r.value) != _key(kept)][:1]
            idx = sorted(keys.index(_key(a)) for a in anchors if _key(a) in keys)
            new: List[Any] = []
            if len(idx) == 2:
                new = [ladder[i] for i in range(idx[0] + 1, idx[1]) if keys[i] not in blocked]
            if not new and spec.sized and _key(kept) in keys:
                i = keys.index(_key(kept))
                tried_idx = [keys.index(k) for k in tried if k in keys]
                if i == max(tried_idx) and i + 1 < len(ladder) and keys[i + 1] not in blocked:
                    new = [ladder[i + 1]]
                elif i == min(tried_idx) and i - 1 >= 0 and keys[i - 1] not in blocked:
                    new = [ladder[i - 1]]
            new = _middle(new, 2)
            return new if new else [kept]

    # ---- helpers -------------------------------------------------------------------------

    def _probe(self, spec: ParamSpec, agent_value: Any) -> List[Any]:
        if not spec.sized:
            return list(spec.probe)
        ladder = list(spec.ladder)
        i = ladder.index(spec.snap(agent_value))
        lo, hi = max(0, i - 1), min(len(ladder), i + 2)
        if hi - lo < 3:
            lo, hi = (0, min(3, len(ladder))) if lo == 0 else (max(0, len(ladder) - 3), len(ladder))
        return ladder[lo:hi]

    def _cheaper_context(self, spec, op, parent_model, rank) -> Optional[str]:
        if not rank:
            return None
        mine = rank.get(parent_model)
        cands = [c for c in self.contexts(spec.directive, op)
                 if mine is not None and rank.get(c) is not None and rank[c] < mine
                 and prune(self._results(spec.directive, op, c), self.threshold)[0] is not None]
        return min(cands, key=lambda c: rank[c]) if cands else None

    def _best_values(self, spec, op, ctx, dead, n) -> List[Any]:
        results = self._results(spec.directive, op, ctx)
        kept, pruned = prune(results, self.threshold)
        if kept is None:
            return []
        others = sorted((r for r in results if r.score is not None
                         and _key(r.value) not in pruned | dead | {_key(kept)}),
                        key=lambda r: -r.score)
        out = [kept] + [r.value for r in others]
        uniq = list({_key(v): v for v in out}.values())
        return uniq[:n]


def _value(v: Any) -> Any:
    return tuple(v) if isinstance(v, list) else v


def _middle(values: List[Any], n: int) -> List[Any]:
    if len(values) <= n:
        return values
    mid = len(values) // 2
    return values[max(0, mid - n // 2): max(0, mid - n // 2) + n]


def estimate_child_cost(spec: Optional[ParamSpec], value: Any, parent_cost: float,
                        target_share: float = 1.0) -> float:
    """Rough pipeline cost of a child: a trimming rewrite scales the target operator's share of
    the parent cost by the fraction of tokens it keeps; anything else costs about the parent."""
    if spec is None or spec.kept_tokens is None or not parent_cost:
        return float(parent_cost or 0.0)
    frac = min(1.0, spec.kept_tokens(value) / MEDIAN_DOC_TOKENS)
    return float(parent_cost) * (1.0 - target_share + target_share * frac)
