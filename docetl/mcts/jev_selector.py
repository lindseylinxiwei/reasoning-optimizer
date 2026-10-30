"""TypeSafe Jev adapter for MOAR's directive-selection step (stdlib only)."""

import json
import math
import os
import time
from itertools import combinations
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_JEV_MODEL = "jev-1.13.0"
# https://docs.typesafe.ai/models, checked 2026-09-22. Output tokens are free.
DEFAULT_INPUT_PRICE = 0.042
PAIR_TYPES = {
    "chunk_header_summary": ("split", "gather"),
    "map_reduce_fusion": ("map", "reduce"),
    "operator_fusion": None,
}
MULTI_TARGET = {
    "doc_summarization", "doc_compression", "deterministic_doc_compression",
    "change model", "change model acc", "change model cost",
}


class JevSelectionError(RuntimeError):
    """No usable selection was returned; never silently switch to another model."""


class JevAPIError(JevSelectionError):
    """Fatal provider/configuration error; stop the experiment instead of retrying nodes."""


class JevSelector:
    def __init__(self, model=DEFAULT_JEV_MODEL, input_price=DEFAULT_INPUT_PRICE,
                 timeout=60.0, api_key=None):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise ValueError("Jev selection requires TYPESAFE_API_KEY")
        if not math.isfinite(input_price) or input_price < 0:
            raise ValueError("Jev input price must be finite and nonnegative")
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError("Jev timeout must be positive and finite")
        self.model = model
        self.input_price = input_price
        self.timeout = timeout

    def _request(self, payload):
        request = Request(
            "https://api.typesafe.ai/v1/systemone",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(3):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except HTTPError as exc:
                status = exc.code
                retry_after = exc.headers.get("Retry-After", "") if exc.headers else ""
                exc.close()
                if attempt < 2 and (status == 429 or 500 <= status < 600):
                    try:
                        delay = min(30.0, max(0.0, float(retry_after)))
                    except ValueError:
                        delay = 2 ** attempt
                    time.sleep(delay)
                    continue
                # Do not echo request bodies, credentials, or provider error bodies.
                raise JevAPIError(f"TypeSafe returned HTTP {status}") from None
            except (URLError, TimeoutError):
                raise JevAPIError("TypeSafe connection failed or timed out") from None
            except (ValueError, UnicodeError):
                raise JevAPIError("TypeSafe returned invalid JSON") from None

    def _choose(self, state, candidates, instruction, on_call):
        if not candidates:
            raise JevSelectionError("No candidates for Jev selection")
        if len(candidates) == 1:
            return next(iter(candidates))
        if len(candidates) > 255:
            raise JevAPIError(
                "Jev Choice supports at most 255 options; candidates were not truncated"
            )
        started = time.monotonic()
        result = self._request({
            "model": self.model, "state": state,
            "questions": {"selection": {"type": "choice",
                "instructions": instruction, "criteria": candidates}},
        })
        usage = result.get("usage", {}) if isinstance(result, dict) else {}
        input_tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
        if type(input_tokens) is not int or input_tokens < 0:
            raise JevAPIError("TypeSafe response missing valid input token usage")
        answers = result.get("answers", {})
        answer = answers.get("selection", {}) if isinstance(answers, dict) else {}
        choice = answer.get("choice") if isinstance(answer, dict) else None
        # Record paid calls even if their selection fails validation afterwards.
        on_call({
            "model": result.get("model", self.model), "usage": usage,
            "estimated_cost_usd": input_tokens * self.input_price / 1_000_000,
            "input_price_per_million": self.input_price,
            "latency_seconds": time.monotonic() - started,
            "question": instruction, "answer": answer,
            "selected_option": candidates.get(choice) if isinstance(choice, str) else None,
        })
        if not isinstance(choice, str) or choice not in candidates:
            raise JevAPIError("TypeSafe returned an unknown or missing choice")
        return choice

    def select(self, *, messages, action_options, directives, pipeline, goal, on_call):
        """Return the existing {directive, operators} selection contract.

        First choose the directive. Then choose a target (or a typed pair).
        Variable-size target lists use sequential Choice calls with a stop option;
        this avoids reducing them to singletons or enumerating a powerset.
        """
        op_configs = {op["name"]: op for op in pipeline["operations"]}
        order = []
        for step in pipeline.get("pipeline", {}).get("steps", []):
            for ref in step.get("operations", []):
                name = ref if isinstance(ref, str) else next(iter(ref))
                if name in op_configs and name not in order:
                    order.append(name)
        order.extend(name for name in op_configs if name not in order)
        eligible = {}
        for op, directive in action_options:
            if op in op_configs and directive in directives:
                eligible.setdefault(directive, set()).add(op)
        targets = {}
        for name, ops in eligible.items():
            ordered = [op for op in order if op in ops]
            if name in PAIR_TYPES:
                types = PAIR_TYPES[name]
                pairs = [list(pair) for pair in combinations(ordered, 2)
                         if types is None or tuple(op_configs[x].get("type")
                                                   for x in pair) == types]
                if pairs:
                    targets[name] = pairs
            else:
                targets[name] = [[op] for op in ordered]
        names = sorted(targets)
        state = {"messages": messages, "optimization_goal": goal}
        choices = {f"d{i}": {"directive": name,
                    "description": directives[name].to_string_for_plan(),
                    "eligible_operators": sorted(eligible[name])}
                   for i, name in enumerate(names)}
        selected = self._choose(state, choices,
            "Choose the next rewrite directive for the stated optimization goal. "
            "Use pipeline evidence and search history, balancing exploration and improvement.",
            on_call)
        name = choices[selected]["directive"]
        target_state = dict(state, selected_directive=choices[selected])
        if name == "arbitrary_rewrite":
            return {"directive": name, "operators": []}
        target_choices = {f"t{i}": {"operators": group,
                          "configs": [op_configs[x] for x in group]}
                          for i, group in enumerate(targets[name])}
        selected = self._choose(target_state, target_choices,
            "Choose the target operator(s), in pipeline order, for the selected directive.",
            on_call)
        operators = list(target_choices[selected]["operators"])
        if name in MULTI_TARGET or name.startswith("change to "):
            while len(operators) < len(eligible[name]):
                remaining = {f"o{i}": {"operator": op, "config": op_configs[op]}
                             for i, op in enumerate(order)
                             if op in eligible[name] and op not in operators}
                remaining["stop"] = "The selected operators are sufficient; add no more."
                choice = self._choose(dict(target_state, selected_operators=operators),
                    remaining, "Add one more target for this same rewrite, or stop.", on_call)
                if choice == "stop":
                    break
                operators.append(remaining[choice]["operator"])
            operators = [op for op in order if op in operators]
        return {"directive": name, "operators": operators}
