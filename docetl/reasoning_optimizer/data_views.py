"""Shared data views for data decomposition rewrites.

A data view replaces a long text field once, at the start of the pipeline, with a trimmed
version that every downstream operator reads (prompts unchanged). Each technique has a
config schema, a builder for its DocETL operators, and an offline estimate of how much text
it keeps (checked before any paid run).

Techniques:
  head_tail      first/last N tokens (code_map)
  topk           split -> topk (fts or embedding) per document -> code_reduce that joins the
                 retrieved chunks in document order
  jev            jev_retrieve with output_key = text_key (structural index, no embeddings)
  summarization  LLM map that rewrites the text field as a summary
"""
import json
from statistics import median
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from docetl.reasoning_optimizer.instantiate_schemas import JevRetrievalTargetSchema

MAX_KEPT_SHARE = 0.30  # a view must keep at most this share of the median document
CHARS_PER_TOKEN = 4.5  # measured on the Sustainability reports (cl100k)


class HeadTailView(BaseModel):
    rationale: str
    head_tokens: int = Field(..., description="Tokens kept from the start of the document")
    tail_tokens: int = Field(0, description="Tokens kept from the end (0 for none)")

    @field_validator("head_tokens", "tail_tokens")
    @classmethod
    def nonnegative(cls, v):
        if v < 0:
            raise ValueError("token counts must be >= 0")
        return v


class TopKView(BaseModel):
    rationale: str
    chunk_tokens: int = Field(..., description="Chunk size in tokens")
    k: int = Field(..., description="Chunks kept per document")
    method: str = Field(..., description='"fts" (BM25) or "embedding" (gemini-embedding-001)')
    query: str = Field(..., description="Retrieval query describing the evidence the pipeline needs")

    @field_validator("method")
    @classmethod
    def method_ok(cls, v):
        if v not in ("fts", "embedding"):
            raise ValueError('method must be "fts" or "embedding"')
        return v

    @field_validator("chunk_tokens", "k")
    @classmethod
    def positive(cls, v):
        if v < 1:
            raise ValueError("chunk_tokens and k must be positive")
        return v


class JevView(BaseModel):
    rationale: str
    targets: List[JevRetrievalTargetSchema]
    evidence_chars: int
    policy: str = "preserve_fill"
    max_decisions: int = 64
    regions_per_request: int = 4


class SummaryView(BaseModel):
    rationale: str
    summary_words: int = Field(..., description="Target summary length in words")
    focus: str = Field(..., description="What the summary must preserve for the downstream operators")


VIEW_TYPES = {"head_tail": HeadTailView, "topk": TopKView, "jev": JevView, "summarization": SummaryView}


# ---- sweeps: what the agent proposes ------------------------------------------------------
# One sweep per technique: fixed parameters, each with a rationale grounded in the corpus
# profile, plus LEVELS values of the main dimension (how much text is kept), from aggressive to
# generous, roughly doubling. Comparisons that change exactly one parameter are derived in code.

LEVELS = 5


def _check_levels(levels):
    if len(levels) != LEVELS:
        raise ValueError(f"give exactly {LEVELS} levels")
    if any(b <= a for a, b in zip(levels, levels[1:])):
        raise ValueError("levels must increase from aggressive to generous")
    return levels


class HeadTailSweep(BaseModel):
    tail_tokens: int = Field(..., description="Tail tokens, fixed across levels (0 for none)")
    tail_rationale: str
    head_levels: List[int] = Field(..., description=f"{LEVELS} head sizes in tokens, increasing")
    levels_rationale: str

    @field_validator("head_levels")
    @classmethod
    def levels_ok(cls, v):
        return _check_levels(v)

    def configs(self):
        return {f"head_tail_{i}": HeadTailView(rationale=self.levels_rationale, head_tokens=h, tail_tokens=self.tail_tokens)
                for i, h in enumerate(self.head_levels, 1)}


class TopKSweep(BaseModel):
    chunk_tokens: int = Field(..., description="Chunk size in tokens, fixed across levels")
    chunk_rationale: str
    method: str = Field(..., description='"fts" or "embedding", fixed across levels')
    method_rationale: str
    query: str
    query_rationale: str
    k_levels: List[int] = Field(..., description=f"{LEVELS} values of k (chunks kept), increasing")
    levels_rationale: str

    @field_validator("k_levels")
    @classmethod
    def levels_ok(cls, v):
        return _check_levels(v)

    def configs(self):
        base = dict(rationale=self.levels_rationale, chunk_tokens=self.chunk_tokens, method=self.method, query=self.query)
        out = {f"topk_{i}": TopKView(**base, k=k) for i, k in enumerate(self.k_levels, 1)}
        mid = self.k_levels[LEVELS // 2]
        other = "fts" if self.method == "embedding" else "embedding"
        out[f"topk_3_{other}"] = TopKView(**{**base, "method": other, "k": mid,
                                             "rationale": f"Level 3 with method={other} instead of {self.method}"})
        # Same text budget, chunks half the size (twice as many chunks).
        out["topk_3_half_chunks"] = TopKView(**{**base, "chunk_tokens": max(50, self.chunk_tokens // 2), "k": mid * 2,
                                                "rationale": "Level 3 budget with chunks half the size"})
        return out


class JevSweep(BaseModel):
    targets: List[JevRetrievalTargetSchema] = Field(..., description="Evidence targets, fixed across levels")
    targets_rationale: str
    regions_per_request: int = Field(..., description="1-8, fixed across levels")
    regions_rationale: str
    cap_levels: List[int] = Field(..., description=f"{LEVELS} evidence_chars caps per target, increasing")
    decision_levels: List[int] = Field(..., description=f"{LEVELS} max_decisions values (16-256), one per cap level")
    levels_rationale: str

    @field_validator("cap_levels")
    @classmethod
    def levels_ok(cls, v):
        return _check_levels(v)

    @field_validator("decision_levels")
    @classmethod
    def decisions_ok(cls, v):
        if len(v) != LEVELS or any(not 16 <= d <= 256 for d in v):
            raise ValueError(f"give {LEVELS} max_decisions values between 16 and 256")
        return v

    def configs(self):
        base = dict(rationale=self.levels_rationale, targets=self.targets, regions_per_request=self.regions_per_request)
        out = {f"jev_{i}": JevView(**base, evidence_chars=c, max_decisions=d, policy="preserve_fill")
               for i, (c, d) in enumerate(zip(self.cap_levels, self.decision_levels), 1)}
        mid = LEVELS // 2
        out["jev_3_adaptive"] = JevView(**{**base, "rationale": "Level 3 with policy=adaptive"},
                                        evidence_chars=self.cap_levels[mid], max_decisions=self.decision_levels[mid],
                                        policy="adaptive")
        return out


class SummarySweep(BaseModel):
    focus: str = Field(..., description="What every summary must preserve, fixed across levels")
    focus_rationale: str
    word_levels: List[int] = Field(..., description=f"{LEVELS} summary lengths in words, increasing")
    levels_rationale: str

    @field_validator("word_levels")
    @classmethod
    def levels_ok(cls, v):
        return _check_levels(v)

    def configs(self):
        return {f"summarization_{i}": SummaryView(rationale=self.levels_rationale, summary_words=w, focus=self.focus)
                for i, w in enumerate(self.word_levels, 1)}


SWEEP_TYPES = {"head_tail": HeadTailSweep, "topk": TopKSweep, "jev": JevSweep, "summarization": SummarySweep}


def _head_tail_code(text_key, head, tail):
    return f'''
import tiktoken
_ENC = tiktoken.get_encoding("cl100k_base")

def transform(input_doc):
    ids = _ENC.encode(input_doc.get("{text_key}") or "", disallowed_special=())
    if len(ids) <= {head + tail}:
        return {{"{text_key}": _ENC.decode(ids)}}
    head = _ENC.decode(ids[:{head}])
    tail = _ENC.decode(ids[-{tail}:]) if {tail} > 0 else ""
    return {{"{text_key}": head + ("\\n[...]\\n" + tail if tail else "")}}
'''


_JOIN_CODE = '''
def transform(items):
    items = sorted(items, key=lambda d: d["{split}_chunk_num"])
    return {{"{text_key}": "\\n[...]\\n".join(d["{text_key}_chunk"] for d in items)}}
'''


def view_ops(technique: str, config: BaseModel, text_key: str, model: str,
             index_spec: Optional[dict] = None, prefix: str = "view") -> List[dict]:
    """DocETL operators that replace `text_key` with the view; run them before the pipeline."""
    if technique == "head_tail":
        return [{"name": f"{prefix}_head_tail", "type": "code_map",
                 "code": _head_tail_code(text_key, config.head_tokens, config.tail_tokens)}]
    if technique == "topk":
        split = f"{prefix}_split"
        topk = {"name": f"{prefix}_topk", "type": "topk", "method": config.method, "k": config.k,
                "keys": [f"{text_key}_chunk"], "query": config.query, "stratify_key": [f"{split}_id"]}
        if config.method == "embedding":
            topk["embedding_model"] = "gemini/gemini-embedding-001"
            # Embed the whole chunk (DocETL embeds only 1000 chars by default), within Gemini's ~2048-token limit.
            topk["embedding_max_chars"] = min(8000, int(config.chunk_tokens * CHARS_PER_TOKEN * 1.3))
        return [
            {"name": split, "type": "split", "split_key": text_key, "method": "token_count",
             "method_kwargs": {"num_tokens": config.chunk_tokens, "model": model}},
            topk,
            {"name": f"{prefix}_join", "type": "code_reduce", "reduce_key": f"{split}_id", "pass_through": True,
             "code": _JOIN_CODE.format(split=split, text_key=text_key)},
        ]
    if technique == "jev":
        return [{"name": f"{prefix}_jev", "type": "jev_retrieve", "text_key": text_key, "output_key": text_key,
                 "targets": [t.model_dump() for t in config.targets], "evidence_chars": config.evidence_chars,
                 "policy": config.policy, "max_decisions": config.max_decisions,
                 "regions_per_request": config.regions_per_request,
                 "index": index_spec or {"levels": [{"kind": "blank_line"}]}}]
    if technique == "summarization":
        return [{"name": f"{prefix}_summary", "type": "map", "model": model,
                 "prompt": (f"Summarize the following document in about {config.summary_words} words. "
                            f"{config.focus}\nUse only information stated in the document.\n\n"
                            f"Document:\n{{{{ input.{text_key} }}}}"),
                 "output": {"schema": {text_key: "string"}}}]
    raise ValueError(f"Unknown technique {technique!r}")


def kept_chars(technique: str, config: BaseModel, doc_chars: int) -> float:
    """Offline estimate of the characters a view keeps for a document of doc_chars."""
    if technique == "head_tail":
        kept = (config.head_tokens + config.tail_tokens) * CHARS_PER_TOKEN
    elif technique == "topk":
        kept = config.chunk_tokens * config.k * CHARS_PER_TOKEN
    elif technique == "jev":
        kept = config.evidence_chars * len(config.targets)
    else:
        kept = config.summary_words * 6.5
    return min(doc_chars, kept)


def check_view(technique: str, config: BaseModel, doc_lengths: List[int],
               max_share: float = MAX_KEPT_SHARE, min_share: float = 0.0) -> dict:
    """Reject a view that keeps too much (or, if min_share is set, too little) of the median document."""
    shares = [kept_chars(technique, config, n) / max(1, n) for n in doc_lengths]
    result = {"kept_share_median": round(median(shares), 4),
              "kept_chars_median": int(median(kept_chars(technique, config, n) for n in doc_lengths))}
    if result["kept_share_median"] > max_share:
        raise ValueError(f"keeps {result['kept_share_median']:.0%} of the median document "
                         f"(limit {max_share:.0%}); it would not reduce the text much")
    if result["kept_share_median"] < min_share:
        raise ValueError(f"keeps only {result['kept_share_median']:.0%} of the median document "
                         f"(minimum {min_share:.0%}); too little for this pipeline")
    return result


TECHNIQUE_TEXT = {
    "head_tail": "Keep the first head_tokens tokens and optionally the last tail_tokens tokens of each document.",
    "topk": ("Split each document into chunks of chunk_tokens tokens, keep the k chunks that best match `query` "
             "(method 'fts' = BM25 keyword matching, free; method 'embedding' = semantic similarity with "
             "gemini-embedding-001, about $0.15 per million tokens embedded), and join them in document order. "
             "Kept text is about chunk_tokens x k tokens per document. Smaller chunks localize evidence better "
             "but lose surrounding context; the query should describe the evidence, not the answer."),
    "jev": ("Jev (a cheap classifier) navigates an index of each document and keeps passages that score as "
            "matching named evidence targets, up to evidence_chars characters per target."),
    "summarization": ("An LLM (gemini-3.5-flash-lite) reads each full document once and writes a summary of "
                      "about summary_words words that replaces the text. Costs one full read per document."),
}


def proposal_prompt(technique: str, ops: List[dict], profile: dict, extra: str = "",
                    max_share: float = MAX_KEPT_SHARE, level_guidance: Optional[str] = None) -> str:
    llm_ops = [{k: op[k] for k in ("name", "type", "prompt", "output") if k in op}
               for op in ops if op.get("type") in ("map", "filter", "reduce")]
    lengths = profile.get("chars", {})
    return (
        f"You are choosing data decomposition settings for a document pipeline. One data view replaces the long "
        f"text field once, before the first operator, and every operator below reads the same trimmed text.\n\n"
        f"Operators that will read the view:\n{json.dumps(llm_ops, indent=2)}\n\n"
        f"Corpus profile (no labels): {json.dumps({k: v for k, v in profile.items() if k != 'excerpts'})}\n"
        f"Document lengths in characters: {json.dumps(lengths)} (about {CHARS_PER_TOKEN} characters per token)\n\n"
        f"Technique: {technique}. {TECHNIQUE_TEXT[technique]}\n{extra}\n"
        f"Propose ONE sweep for this technique as a {SWEEP_TYPES[technique].__name__} object:\n"
        f"- Fixed parameters (everything except the levels) stay the same across the sweep, so the levels differ "
        f"in one dimension only: how much text is kept. Give a rationale for EACH fixed parameter that refers to "
        f"the numbers in the profile (document and section lengths, heading counts) and to what the operators "
        f"need.\n"
        f"- {LEVELS} levels of the main dimension, "
        + (level_guidance or "increasing and roughly doubling, from very small (cheapest, e.g. around 500 tokens "
           "or the smallest useful amount) to generous")
        + f". Explain in levels_rationale how you chose the range.\n"
        f"- Every level must keep at most {max_share:.0%} of the median document, otherwise it does not "
        f"reduce the text.\n"
    )
