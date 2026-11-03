"""jev_retrieve: keep only the passages of a long text field that matter for named targets.

Uses the vendored Jev retrieval method (docetl/retrieval/jev): build a hierarchical index
of each document, let Jev score regions against each target (adaptive navigation,
threshold 0.3, 128 decisions per target by default), optionally top up to the character
cap from the same scores (preserve-then-fill), and write the selected literal passages,
in document order, to output_key. Downstream operators read output_key instead of the
full text.

Index (how each document is split into a hierarchy of regions):
  an IndexSpec dict (default: paragraphs) - declarative splitters applied identically to every
             document, see docetl/retrieval/index_spec.py; no model calls
  "contract_structure"  the handover's contract-heading parser (ARTICLE / SECTION n.n)

Targets: Jev allows 32 questions per request and every region in a request is asked about
every target in it, so targets are split into bundles of at most 32 // regions_per_request.
Targets in a bundle share requests; each extra bundle navigates with its own requests.

Cost reported to DocETL = Jev input tokens x price.
A retrieval failure fails the operation instead of passing partial evidence downstream.
"""
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from docetl.operations.base import BaseOperation
from docetl.operations.utils import RichLoopBar
from docetl.retrieval.index_spec import IndexSpec, build_spec_index
from docetl.retrieval.jev.navigation import Task
from docetl.retrieval.jev.retrieval import PRESETS
from docetl.retrieval.jev.run_navigation import run_bundle
from docetl.retrieval.jev.semantic_index import build_index
from docetl.retrieval.jev.shared_capacity import _effective_shared_config
from docetl.retrieval.jev.shared_scorer import SharedScorer
from docetl.retrieval.jev.top_up import preserve_then_fill
from docetl.retrieval.jev_backend import JevRetrievalBackend

JEV_MAX_QUESTIONS = 32


class RetrievalTarget(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)


def target_bundles(targets, regions_per_request):
    """Consecutive groups of targets that fit one Jev request with regions_per_request regions."""
    size = max(1, JEV_MAX_QUESTIONS // regions_per_request)
    return [targets[i:i + size] for i in range(0, len(targets), size)]


class JevRetrieveOperation(BaseOperation):
    class schema(BaseOperation.schema):
        type: str = "jev_retrieve"
        text_key: str
        output_key: str | None = None  # default: "<text_key>_jev"
        targets: list[RetrievalTarget]
        evidence_chars: int = Field(6000, gt=0, description="Character cap per target (not tokens)")
        policy: Literal["adaptive", "preserve_fill"] = "preserve_fill"
        index: Literal["contract_structure"] | IndexSpec = Field(
            default_factory=lambda: IndexSpec(levels=[{"kind": "blank_line"}]),
            description="IndexSpec, or the named contract parser")
        regions_per_request: int = Field(4, ge=1, le=JEV_MAX_QUESTIONS,
                                         description="Sibling regions Jev scores per request")
        leaf_chars: int = Field(2400, ge=32, description="contract_structure only; an IndexSpec has its own")
        fanout: int = Field(4, ge=2, description="contract_structure only; an IndexSpec has its own")
        threshold: float = Field(0.3, ge=0, le=1)
        max_decisions: int = Field(128, gt=0, description="Controller decisions per target")
        workers: int = Field(4, ge=1, le=4, description="Concurrent decisions per target")
        max_concurrency: int = Field(8, ge=1, description="Concurrent Jev requests across documents")
        jev_model: str = "jev-1.13.0"
        input_price_per_million: float = 0.042
        separator: str = "\n[...]\n"
        diagnostics_key: str | None = None  # default: "_<name>_retrieval"

        @field_validator("targets")
        @classmethod
        def check_targets(cls, targets):
            if not targets:
                raise ValueError("jev_retrieve needs at least one target")
            if len({t.name for t in targets}) != len(targets):
                raise ValueError("jev_retrieve target names must be unique")
            return targets

    def _index(self, text):
        cfg = self.config
        index = cfg.get("index") or {"levels": [{"kind": "blank_line"}]}
        if index == "contract_structure":
            return build_index(text, policy="source-structure", fanout=cfg.get("fanout", 4),
                               leaf_chars=cfg.get("leaf_chars", 2400))
        spec = index if isinstance(index, IndexSpec) else IndexSpec(**index)
        return build_spec_index(text, spec)

    def _navigate(self, index, tasks, backend):
        """Run each target bundle over the shared index; return {target name: packet}."""
        cfg = self.config
        model = cfg.get("jev_model", "jev-1.13.0")
        regions = cfg.get("regions_per_request", 4)
        packets = {}
        for bundle in target_bundles(tasks, regions):
            scorer = SharedScorer(bundle, backend, model=model)
            config = replace(PRESETS["baseline_03_128"], model=model,
                             evidence_chars=cfg.get("evidence_chars", 6000),
                             threshold=cfg.get("threshold", 0.3),
                             max_calls=cfg.get("max_decisions", 128), workers=cfg.get("workers", 4),
                             max_questions=regions)
            config = _effective_shared_config(config, scorer, bundle)
            packets.update(run_bundle(index, bundle, scorer, config,
                                      outer_workers=min(4, len(bundle)))["results"])
        return packets

    def _retrieve(self, text, tasks, backend):
        """Return (evidence intervals, diagnostics) for one document."""
        cfg = self.config
        cap = cfg.get("evidence_chars", 6000)
        if len(text) <= cap:
            return [(0, len(text))], {"passthrough": True, "source_chars": len(text)}
        packets = self._navigate(self._index(text), tasks, backend)

        intervals, per_target = [], {}
        for task in tasks:
            packet = packets[task.name]
            if "controller_error" in packet:
                raise RuntimeError(f"Jev retrieval failed for target {task.name!r}: {packet['controller_error']}")
            if any(e.get("stage") == "failure" for e in packet.get("trace", [])):
                detail = backend.errors[0] if backend.errors else "see trace"
                raise RuntimeError(f"Jev scoring call failed during retrieval for target {task.name!r}: {detail}")
            final = preserve_then_fill(text, packet) if cfg.get("policy", "preserve_fill") == "preserve_fill" else packet
            for ev in final["evidence"]:
                if text[ev["start"]:ev["end"]] != ev["text"]:
                    raise RuntimeError("Jev retrieval evidence does not match the source text")
                intervals.append((ev["start"], ev["end"]))
            per_target[task.name] = {
                "evidence_chars": final["evidence_chars"],
                "original_evidence_chars": packet["evidence_chars"],
                "controller_calls": packet.get("controller_calls"),
                "status": packet.get("retrieval_status"),
                **({"shortfall_chars": final["shortfall_chars"]} if "shortfall_chars" in final else {}),
            }
        return intervals, {"passthrough": False, "source_chars": len(text), "targets": per_target}

    @staticmethod
    def _merge(intervals):
        merged = []
        for s, e in sorted(intervals):
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return merged

    def execute(self, input_data: list[dict]) -> tuple[list[dict], float]:
        cfg = self.config
        text_key = cfg["text_key"]
        output_key = cfg.get("output_key") or f"{text_key}_jev"
        diagnostics_key = cfg.get("diagnostics_key") or f"_{cfg['name']}_retrieval"
        tasks = [Task(t["name"], t["description"]) for t in cfg["targets"]]
        backend = JevRetrievalBackend(
            input_price_per_million=cfg.get("input_price_per_million", 0.042),
            bypass_cache=cfg.get("bypass_cache", False) or self.runner.config.get("bypass_cache", False),
            max_concurrency=cfg.get("max_concurrency", 8))
        separator = cfg.get("separator", "\n[...]\n")

        def one(doc):
            text = doc.get(text_key) or ""
            if not isinstance(text, str):
                raise TypeError(f"jev_retrieve: {text_key!r} must be a string")
            intervals, diagnostics = self._retrieve(text, tasks, backend)
            merged = self._merge(intervals)
            rendered = separator.join(text[s:e] for s, e in merged)
            diagnostics["evidence_chars"] = sum(e - s for s, e in merged)
            return {**doc, output_key: rendered, diagnostics_key: diagnostics}

        results = [None] * len(input_data)
        with ThreadPoolExecutor(max_workers=min(len(input_data), 8) or 1) as executor:
            futures = [executor.submit(one, doc) for doc in input_data]
            for i in RichLoopBar(range(len(futures)), desc=f"Processing {cfg['name']} (jev_retrieve)",
                                 console=self.console):
                results[i] = futures[i].result()
        if not math.isfinite(backend.cost):
            raise RuntimeError("jev_retrieve cost is not finite")
        bundles = len(target_bundles(tasks, cfg.get("regions_per_request", 4)))
        self.console.log(f"jev_retrieve {cfg['name']}: {len(tasks)} targets in {bundles} bundle(s), "
                         f"{backend.calls} Jev calls, {backend.cache_hits} cached, "
                         f"{backend.input_tokens} input tokens, ${backend.cost:.4f}")
        return results, backend.cost
