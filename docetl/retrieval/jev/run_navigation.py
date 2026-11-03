"""Offline-verifiable, shared-scorer document-bundle navigation entry point.

The reusable helper performs retrieval only. Paid baseline comparisons and final
extraction belong to the parent campaign launcher; this CLI cannot independently
authorize them. Neither import nor --verify reads credentials.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import time

try:
    from .navigation import NavigationConfig, Navigator, Task, _task
    from .semantic_index import build_index
except ImportError:
    from navigation import NavigationConfig, Navigator, Task, _task
    from semantic_index import build_index


_STAT_KEYS = ("network_calls", "cache_hits", "singleflight_waits", "question_count", "request_bytes")


def _snapshot(scorer):
    stats = getattr(scorer, "stats", None)
    values = stats() if callable(stats) else stats
    if values is None:
        return None
    if not isinstance(values, dict):
        raise ValueError("Scorer stats must return a dictionary")
    # Keep shared network counts separate from controller attempts. A provider
    # without counters remains unavailable, never silently replaced by zero.
    snapshot = {}
    for name in _STAT_KEYS:
        value = values.get(name)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("Invalid scorer counter: " + name)
        snapshot[name] = value
    return snapshot


def _provider_info(scorer):
    backend = getattr(scorer, "backend", None)
    kind = getattr(scorer, "kind", "injected_scorer")
    backend_kind = getattr(backend, "kind", None)
    marker = backend_kind or kind
    simulation = getattr(scorer, "simulation_only", getattr(backend, "simulation_only", None))
    if simulation is None:
        if marker.startswith("offline_") or marker.startswith("scripted_"):
            simulation = True
        elif marker.startswith("live_"):
            simulation = False
    return {"scorer": kind, "backend": backend_kind, "simulation_only": simulation,
            "execution_mode": "offline_simulation" if simulation is True else
                              "live_provider" if simulation is False else "injected_unclassified",
            "model_predictions": False if simulation is True else None}


def run_bundle(index, tasks, scorer, config=None, *, outer_workers=4):
    """Navigate one document for several targets with one shared scorer object.

    SharedScorer must provide thread-safe score(payload, context) and stats().
    It owns single-flight/cache policy and the globally bounded backend pool;
    this helper limits active target controllers, not paid provider concurrency.
    Result ordering follows the supplied target order, never finish order.
    An individual failed controller is retained instead of replaced/retried.
    """
    if type(outer_workers) is not int or not 1 <= outer_workers <= 4:
        raise ValueError("outer_workers must be 1..4")
    config = config or NavigationConfig()
    if isinstance(tasks, dict):
        tasks = [Task(name, description) for name, description in tasks.items()]
    tasks = tuple(_task(task) for task in tasks)
    if not tasks or len({task.name for task in tasks}) != len(tasks):
        raise ValueError("Supply distinct nonempty extraction targets")
    index.validate()
    source_hash = hashlib.sha256(index.text.encode("utf-8")).hexdigest()
    index_json = index.to_json()
    index_hash = hashlib.sha256(index_json.encode("utf-8")).hexdigest()
    before = _snapshot(scorer)
    started = time.monotonic()

    def one(task):
        try:
            return Navigator(scorer, config).retrieve(index, task)
        except Exception as exc:
            # A controller failure is observable and partial. Do not dispatch a
            # replacement or manufacture a completed empty extraction.
            return {"task": asdict(task), "source_sha256": source_hash, "evidence": [],
                    "evidence_chars": 0, "retrieval_status": "partial", "retrieval_partial": True,
                    "controller_error": type(exc).__name__, "controller_calls": 0,
                    "decision_calls": 0, "pending": [{"node": index.root_id, "start": 0,
                    "end": len(index.text), "reason": "controller_error"}],
                    "issues": [{"reason": "controller_error", "error_type": type(exc).__name__}],
                    "empty_evidence_does_not_prove_absence": True, "trace": []}

    with ThreadPoolExecutor(max_workers=min(outer_workers, len(tasks))) as executor:
        futures = [executor.submit(one, task) for task in tasks]
        results = [future.result() for future in futures]
    seconds = time.monotonic() - started
    after = _snapshot(scorer)
    delta = None
    if before is not None and after is not None:
        delta = {name: after[name] - before[name] if before[name] is not None and after[name] is not None else None
                 for name in _STAT_KEYS}
        if any(value is not None and value < 0 for value in delta.values()):
            raise ValueError("Scorer counters decreased during bundle execution")
    if hashlib.sha256(index.to_json().encode("utf-8")).hexdigest() != index_hash:
        raise RuntimeError("Reusable source index mutated during navigation")
    return {"schema_version": 1, "stage": "retrieval_only", "source_sha256": source_hash,
            "index_sha256": index_hash, "index_policy": index.policy,
            "task_order": [task.name for task in tasks], "tasks": [asdict(task) for task in tasks],
            "config": asdict(config), "outer_workers": outer_workers,
            "providers": _provider_info(scorer),
            "results": {task.name: result for task, result in zip(tasks, results)},
            "controller_attempts": sum(result["controller_calls"] for result in results),
            "network_stats_before": before, "network_stats_after": after,
            "network_stats_delta": delta,
            "partial_targets": [task.name for task, result in zip(tasks, results) if result["retrieval_partial"]],
            "bundle_wall_seconds": seconds,
            "accounting_note": "Controller attempts and shared network counts are distinct. "
                "Provider requests, usage, costs and unknown-charge holds belong to the campaign gate's "
                "separate ledger/wire artifacts; this bundle supplies no inferred monetary charge.",
            "timing_note": "Wall time covers concurrent target navigation with the already-built index. "
                "It excludes index construction, final extraction, serialization and saving."}


def save_bundle(bundle, destination):
    """Write a new attributable artifact; never replace a saved experiment."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(bundle, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return path


def verify_bundle():
    """Offline synthetic integration demo, explicitly excluded from model results."""
    class NegativeFixture:
        kind = "offline_scripted_not_model_predictions"
        supports_concurrent_calls = True

        def score(self, payload, context):
            return {name: 0.0 for name in payload["questions"]}

    index = build_index("SECTION ONE\n\nSynthetic example text.\n\nSECTION TWO\n\nMore synthetic text.", leaf_chars=40)
    tasks = [Task(f"Synthetic target {i}", f"Extract synthetic target {i}.") for i in range(8)]
    bundle = run_bundle(index, tasks, NegativeFixture(), NavigationConfig(max_questions=4))
    assert bundle["providers"]["simulation_only"] is True
    assert len(bundle["results"]) == 8
    assert all(not result["evidence"] and not result["retrieval_partial"] for result in bundle["results"].values())
    bundle["verification"] = "Passed synthetic integration only; no provider, annotation or credential access."
    return bundle


