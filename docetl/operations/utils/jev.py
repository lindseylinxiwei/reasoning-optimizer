"""Jev classification backend. No generation, truncation, or LLM fallback."""
import http.client
import socket
import ssl
import hashlib
import json
import math
import os
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPSHandler, Request, build_opener, urlopen

from .cache import cache
from .llm import LLMResult


def build_questions(schema):
    questions = {}
    for field, kind in schema.items():
        if kind in ("bool", "boolean"):
            questions[field] = {
                "type": "noul",
                "instructions": f"Following the task instructions in the state, should the output field {field!r} be true?",
            }
        elif isinstance(kind, str) and re.fullmatch(r"enum\[.*\]", kind):
            labels = [label.strip() for label in kind[5:-1].split(",")]
            if not all(labels) or len(labels) != len(set(labels)) or not 2 <= len(labels) <= 255:
                raise ValueError(f"Invalid Jev enum for {field}")
            questions[field] = {
                "type": "choice",
                "instructions": f"Following the task instructions in the state, select the value of output field {field!r}.",
                "criteria": dict.fromkeys(labels),
            }
        else:
            raise ValueError(f"Jev supports boolean and enum outputs only; {field}: {kind}")
    if not questions:
        raise ValueError("Jev requires a nonempty output schema")
    return questions


def _ipv4_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None):
    host, port = address
    last_error = None
    for _, _, _, _, sockaddr in socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM):
        try:
            return socket.create_connection(sockaddr, timeout, source_address)
        except OSError as error:
            last_error = error
    if last_error:
        raise last_error
    raise OSError("No IPv4 address available for gateway")


class _IPv4HTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_connection = _ipv4_connection


class _IPv4HTTPSHandler(HTTPSHandler):
    def https_open(self, request):
        return self.do_open(_IPv4HTTPSConnection, request, context=self._context)


def request_jev(payload, timeout):
    proxy = os.environ.get("JEV_PROXY_URL")
    key = os.environ.get("JEV_PROXY_TOKEN" if proxy else "TYPESAFE_API_KEY")
    if not key:
        raise ValueError("Configure JEV_PROXY_URL and JEV_PROXY_TOKEN for the Netlify bridge, or TYPESAFE_API_KEY and optionally TYPESAFE_BASE_URL for direct/gateway access")
    base = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai").rstrip("/")
    endpoint = proxy or base + ("/systemone" if base.endswith("/v1") else "/v1/systemone")
    request = Request(endpoint, data=json.dumps(payload).encode(), headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"}, method="POST")
    open_request = (build_opener(_IPv4HTTPSHandler(context=ssl.create_default_context())).open
                    if os.environ.get("JEV_FORCE_IPV4") == "1" else urlopen)
    for attempt in range(3):
        try:
            with open_request(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            status = exc.code
            exc.close()
            if attempt < 2 and (status == 429 or 500 <= status < 600):
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Jev HTTP {status}; check credentials, gateway setup, and context limit. No document was truncated.") from None
        except (URLError, TimeoutError):
            raise RuntimeError("Jev connection failed or timed out") from None
    raise RuntimeError("Jev retries exhausted")


def call_jev(model, messages, schema, *, timeout, bypass_cache, validation_config=None):
    if len(messages) != 1 or messages[0].get("role") != "user" or not isinstance(messages[0].get("content"), str):
        raise ValueError("Jev map/filter expects one rendered text prompt")
    payload = {"model": "jev-1.13.0" if model == "jev" else model,
               "state": messages[0]["content"], "questions": build_questions(schema)}
    endpoint = os.environ.get("JEV_PROXY_URL") or os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai/v1")
    digest = hashlib.sha256(json.dumps([endpoint, payload], sort_keys=True).encode()).hexdigest()
    cache_key = "jev-classification-v1:" + digest
    result = None if bypass_cache else cache.get(cache_key)
    cached = result is not None
    if not cached:
        result = request_jev(payload, timeout)
    if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
        raise ValueError("Malformed Jev response")
    output = {}
    for field, question in payload["questions"].items():
        answer = result["answers"].get(field)
        if not isinstance(answer, dict) or answer.get("type") != question["type"]:
            raise ValueError(f"Missing or invalid Jev answer: {field}")
        if question["type"] == "noul":
            probability = answer.get("noul")
            if type(probability) not in (int, float) or not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"Invalid Jev probability: {field}")
            output[field] = probability >= 0.5
        else:
            choice = answer.get("choice")
            if not isinstance(choice, str) or choice not in question["criteria"]:
                raise ValueError(f"Unknown Jev choice: {field}")
            output[field] = choice
    usage = result.get("usage", {})
    tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
    if type(tokens) is not int or tokens < 0:
        raise ValueError("Jev response missing input token usage")
    price = float(os.environ.get("JEV_INPUT_PRICE_PER_MILLION", "0.042"))
    if not math.isfinite(price) or price < 0:
        raise ValueError("Jev input price must be finite and nonnegative")
    if not cached:
        cache.set(cache_key, result)
    valid = True
    if validation_config:
        output, valid = validation_config["validation_fn"](output)
    return LLMResult(response=output, total_cost=0.0 if cached else tokens * price / 1_000_000, validated=valid)
