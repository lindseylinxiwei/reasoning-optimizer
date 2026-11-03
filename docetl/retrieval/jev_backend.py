"""Jev transport for the vendored retrieval controller (docetl.retrieval.jev).

SharedScorer calls backend.decide(payload) with a complete Jev request (model, structured
state, Noul questions) and expects the raw response envelope back. This backend reuses
DocETL's Jev HTTP transport, bounds concurrent requests across all documents of one
operation, caches responses like the Jev map/filter backend, and accounts input tokens.
"""
import hashlib
import json
import math
import threading

from docetl.operations.utils.cache import cache
from docetl.operations.utils.jev import request_jev

CACHE_PREFIX = "jev-retrieval-v1:"


class JevRetrievalBackend:
    supports_concurrent_calls = True

    def __init__(self, *, input_price_per_million=0.042, timeout=60, bypass_cache=True,
                 max_concurrency=4, request_fn=None):
        if not math.isfinite(input_price_per_million) or input_price_per_million < 0:
            raise ValueError("Jev input price must be finite and nonnegative")
        self.price = input_price_per_million
        self.timeout = timeout
        self.bypass_cache = bypass_cache
        self.request_fn = request_fn or request_jev
        self._slots = threading.BoundedSemaphore(max_concurrency)
        self._lock = threading.Lock()
        self.calls = 0          # physical requests sent (SharedScorer.stats reads this name)
        self.cache_hits = 0
        self.input_tokens = 0
        self.cost = 0.0
        self.errors = []        # messages of failed requests; the controller only keeps the type

    def decide(self, payload):
        key = CACHE_PREFIX + hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if not self.bypass_cache:
            cached = cache.get(key)
            if cached is not None:
                with self._lock:
                    self.cache_hits += 1
                return cached
        try:
            with self._slots:
                raw = self.request_fn(payload, self.timeout)
            usage = raw.get("usage") if isinstance(raw, dict) else None
            tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
            if type(tokens) is not int or tokens < 0:
                raise ValueError("Jev response missing input token usage")
        except Exception as exc:
            with self._lock:
                self.errors.append(f"{type(exc).__name__}: {exc}")
            raise
        with self._lock:
            self.calls += 1
            self.input_tokens += tokens
            self.cost += tokens * self.price / 1_000_000
        cache.set(key, raw)
        return raw
