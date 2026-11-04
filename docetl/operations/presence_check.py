"""presence_check: cheap yes/no presence decisions (Jev) that gate a later, expensive operation.

For every document and every check ({key, question}), Jev answers "does this text contain
what the question describes?" as a probability. Long documents are cut into chunks that fit
one Jev request; a check is present if any chunk scores >= threshold, and a check that is
already present is not asked again in later chunks. Questions are bundled 32 per request
(Jev's limit). Output per document:
  <output_prefix><key>  bool for each check
  <any_key>             bool, true if any check is present (use as a gated_map gate_key)
  <diagnostics_key>     max probability per check, requests and chunks used

The threshold is a recall knob: a gate should rarely say "absent" for something present,
because the gated operation never sees it again. Cost = Jev input tokens x price.
"""
import math
import re
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field, field_validator

from docetl.operations.base import BaseOperation
from docetl.operations.utils import RichLoopBar
from docetl.retrieval.jev_backend import JevRetrievalBackend

JEV_MAX_QUESTIONS = 32
JEV_MAX_REQUEST_BYTES = 28000  # same request budget the vendored Jev retrieval uses
_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class PresenceCheck(BaseModel):
    key: str
    question: str = Field(..., min_length=1)

    @field_validator("key")
    @classmethod
    def check_key(cls, key):
        if not _KEY.match(key):
            raise ValueError(f"check key {key!r} must be an identifier")
        return key


def chunk_text(text: str, chunk_chars: int, overlap: int) -> list[tuple[int, int]]:
    """(start, end) chunks of at most chunk_chars, cut at a paragraph or sentence end when possible."""
    if len(text) <= chunk_chars:
        return [(0, len(text))]
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        if end < len(text):
            window = text[start + chunk_chars // 2:end]
            cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if cut >= 0:
                end = start + chunk_chars // 2 + cut + 1
        chunks.append((start, end))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


class PresenceCheckOperation(BaseOperation):
    class schema(BaseOperation.schema):
        type: str = "presence_check"
        text_key: str
        checks: list[PresenceCheck]
        context: str = ""  # short task description shown to Jev with every chunk
        threshold: float = Field(0.3, gt=0, lt=1)
        chunk_chars: int = Field(16000, ge=1000, le=24000)
        overlap_chars: int = Field(400, ge=0, le=4000)
        output_prefix: str | None = None  # default: "<name>_"
        any_key: str | None = None  # default: "<name>_any"
        diagnostics_key: str | None = None  # default: "_<name>_presence"
        jev_model: str = "jev-1.13.0"
        input_price_per_million: float = 0.042
        max_concurrency: int = Field(8, ge=1)

        @field_validator("checks")
        @classmethod
        def check_checks(cls, checks):
            if not checks:
                raise ValueError("presence_check needs at least one check")
            if len({c.key for c in checks}) != len(checks):
                raise ValueError("presence_check keys must be unique")
            return checks

    def _keys(self):
        cfg = self.config
        prefix = cfg.get("output_prefix") or f"{cfg['name']}_"
        return (prefix, cfg.get("any_key") or f"{cfg['name']}_any",
                cfg.get("diagnostics_key") or f"_{cfg['name']}_presence")

    def _state(self, chunk, i, n):
        context = self.config.get("context") or ""
        head = f"Task context: {context}\n\n" if context else ""
        return f"{head}Document text (part {i + 1} of {n}):\n{chunk}"

    def _check_document(self, text, checks, backend):
        cfg = self.config
        threshold = cfg.get("threshold", 0.3)
        best = {c["key"]: 0.0 for c in checks}
        spans = chunk_text(text, cfg.get("chunk_chars", 16000), cfg.get("overlap_chars", 400)) if text else []
        requests = 0
        for i, (s, e) in enumerate(spans):
            open_checks = [c for c in checks if best[c["key"]] < threshold]
            if not open_checks:
                break
            state = self._state(text[s:e], i, len(spans))
            if len(state.encode()) > JEV_MAX_REQUEST_BYTES:
                raise ValueError(f"presence_check request is {len(state.encode())} bytes; lower chunk_chars "
                                 f"(limit {JEV_MAX_REQUEST_BYTES} bytes incl. context)")
            for b in range(0, len(open_checks), JEV_MAX_QUESTIONS):
                bundle = open_checks[b:b + JEV_MAX_QUESTIONS]
                payload = {"model": cfg.get("jev_model", "jev-1.13.0"), "state": state,
                           "questions": {c["key"]: {"type": "noul", "instructions": c["question"]} for c in bundle}}
                raw = backend.decide(payload)
                requests += 1
                answers = raw.get("answers") if isinstance(raw, dict) else None
                for c in bundle:
                    answer = (answers or {}).get(c["key"])
                    p = answer.get("noul") if isinstance(answer, dict) else None
                    if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
                        raise ValueError(f"presence_check: invalid Jev answer for {c['key']!r}")
                    best[c["key"]] = max(best[c["key"]], float(p))
        return best, {"chunks": len(spans), "requests": requests}

    def execute(self, input_data: list[dict]) -> tuple[list[dict], float]:
        cfg = self.config
        text_key = cfg["text_key"]
        checks = [c if isinstance(c, dict) else c.model_dump() for c in cfg["checks"]]
        threshold = cfg.get("threshold", 0.3)
        prefix, any_key, diagnostics_key = self._keys()
        backend = JevRetrievalBackend(
            input_price_per_million=cfg.get("input_price_per_million", 0.042),
            bypass_cache=cfg.get("bypass_cache", False) or self.runner.config.get("bypass_cache", False),
            max_concurrency=cfg.get("max_concurrency", 8))

        def one(doc):
            text = doc.get(text_key) or ""
            if not isinstance(text, str):
                raise TypeError(f"presence_check: {text_key!r} must be a string")
            best, info = self._check_document(text, checks, backend)
            flags = {f"{prefix}{k}": p >= threshold for k, p in best.items()}
            return {**doc, **flags, any_key: any(flags.values()),
                    diagnostics_key: {"max_probability": {k: round(p, 4) for k, p in best.items()}, **info}}

        results = [None] * len(input_data)
        with ThreadPoolExecutor(max_workers=min(len(input_data), 8) or 1) as executor:
            futures = [executor.submit(one, doc) for doc in input_data]
            for i in RichLoopBar(range(len(futures)), desc=f"Processing {cfg['name']} (presence_check)",
                                 console=self.console):
                results[i] = futures[i].result()
        present = sum(r[any_key] for r in results)
        self.console.log(f"presence_check {cfg['name']}: {len(checks)} checks, {present}/{len(results)} documents "
                         f"with any present, {backend.calls} Jev calls, {backend.cache_hits} cached, ${backend.cost:.4f}")
        return results, backend.cost
