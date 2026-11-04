import json
import re
from copy import deepcopy
from statistics import median
from typing import Dict, List, Type

from jinja2 import Environment
from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import PresenceGateInstantiateSchema

from .agent_utils import _extract_json_from_text, agent_completion, is_together_model
from .base import (
    AVAILABLE_MODELS,
    MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS,
    Directive,
    DirectiveTestCase,
    available_models_info,
)
from .jev_retrieval import _load_texts

ABSENT_NOTE = ("\n\nOnly the categories listed above apply to this document; the other categories were checked "
               "and are absent, so do not output them.")

def _empty_value(type_str: str):
    t = str(type_str).strip().lower()
    if t.startswith("list"):
        return []
    if t in ("str", "string", "text"):
        return ""
    if t.startswith("{") or t.startswith("dict"):
        return {}
    return None


def _fill_code(schema: Dict) -> str:
    empty = {k: _empty_value(v) for k, v in schema.items() if _empty_value(v) is not None}
    return (
        "def transform(doc):\n"
        f"    empty = {empty!r}\n"
        "    return {k: v for k, v in empty.items() if doc.get(k) is None}\n"
    )


def _any_code(fields: List[str], any_key: str) -> str:
    return (
        "def transform(doc):\n"
        f"    return {{{any_key!r}: any(bool(doc.get(k)) for k in {fields!r})}}\n"
    )


def _gate_explanation() -> str:
    return (
        "How the presence gate works:\n"
        "- A checker answers, for every document and every check, whether the text CONTAINS the thing the check "
        "describes (yes/no). The target map then runs only on what is present:\n"
        "  granularity 'category': each check has a prompt_fragment (the verbatim part of the map prompt asking "
        "for that output). The fragment is included in the prompt only when its check is present, so the LLM "
        "never sees, and cannot invent answers for, absent categories. Categories without a check stay in the "
        "prompt unchanged.\n"
        "  granularity 'document': one check; documents where it is absent skip the map entirely.\n"
        "  skip_when_none: documents where no check is present skip the map (outputs become empty values).\n"
        "- checker 'jev': the Jev classifier scores each check 0-1 on chunks of chunk_chars characters (up to "
        "32 checks per request); present = any chunk scores >= threshold. Cost: input tokens only, about "
        "$0.042 per million, i.e. roughly $0.0005 per 40k-character document per 32 checks. Jev reads only the "
        "chunk plus the context and the question, so each question must stand on its own.\n"
        "- checker 'llm': one call of checker_model over the whole text answering all checks; costs a full "
        "read of the document with that model.\n"
        "- A wrong 'absent' loses that output for good; a wrong 'present' only costs what the map cost before. "
        "So questions should be broad enough to catch every phrasing, and the threshold low (0.2-0.4).\n"
    )


class PresenceGateDirective(Directive):
    name: str = Field(default="presence_gate", description="The name of the directive")
    formal_description: str = Field(
        default="Map => PresenceCheck -> Map (only present parts)",
        description="Inserts a cheap presence check before a map and runs the map only on what is present",
    )
    nl_description: str = Field(
        default="Task decomposition into a cheap presence decision and the expensive generation: before the "
        "target map, a checker (the Jev classifier, or a cheap LLM) decides per document whether each output "
        "category is present in the text at all. The map then asks only about the present categories (their "
        "instructions are removed from the prompt for that document) and is skipped for documents where nothing "
        "is present. Absent categories get no answer instead of an invented one."
    )
    when_to_use: str = Field(
        default="When a map extracts or answers several things (or one thing) that are often absent from a "
        "document, e.g. contract clauses a contract may not contain, adverse events a report may not mention, and "
        "false answers on absent items hurt precision or extraction is expensive. Not useful when nearly every "
        "document contains every output, or when deciding presence is as hard as the extraction itself."
    )

    instantiate_schema_type: Type[BaseModel] = Field(default=PresenceGateInstantiateSchema)

    example: str = Field(
        default="""
        Original Map Operation:
        - name: extract_clauses
          type: map
          prompt: |
            Extract the text spans for each category from this contract: {{ input.contract_text }}
            1. Governing Law: Which state/country's law governs the contract?
            2. Anti-Assignment: Is consent or notice required if the contract is assigned to a third party?
          output:
            schema:
              clauses: "list[{clause_type: string, text_spans: list[string]}]"

        PresenceGateInstantiateSchema:
        {
          "name": "clause_presence",
          "document_key": "contract_text",
          "granularity": "category",
          "checker": "jev",
          "context": "Deciding which kinds of clauses a commercial contract contains.",
          "checks": [
            {"key": "governing_law", "question": "Does the text state which state's or country's law governs the agreement?",
             "prompt_fragment": "1. Governing Law: Which state/country's law governs the contract?"},
            {"key": "anti_assignment", "question": "Does the text restrict assigning or transferring the agreement, e.g. requiring the other party's consent or notice?",
             "prompt_fragment": "2. Anti-Assignment: Is consent or notice required if the contract is assigned to a third party?"}
          ],
          "threshold": 0.3,
          "chunk_chars": 16000,
          "skip_when_none": true
        }

        Resulting Pipeline:
        - name: clause_presence
          type: presence_check
          text_key: contract_text
          checks: [...keys and questions as above...]
          threshold: 0.3
        - name: extract_clauses
          type: gated_map
          gate_key: clause_presence_any
          prompt: |
            Extract the text spans for each category from this contract: {{ input.contract_text }}
            {% if input.clause_presence_governing_law %}1. Governing Law: ...{% endif %}
            {% if input.clause_presence_anti_assignment %}2. Anti-Assignment: ...{% endif %}
            (+ a note that unlisted categories are absent)
        - name: extract_clauses_fill_skipped
          type: code_map     # skipped documents get empty outputs ([] / "") instead of None
        """
    )

    test_cases: List[DirectiveTestCase] = Field(
        default_factory=lambda: [
            DirectiveTestCase(
                name="contract_clause_presence",
                description="Many clause categories, most of them absent from any one contract",
                input_config={
                    "name": "extract_clauses",
                    "type": "map",
                    "prompt": "Extract spans for: 1. Non-Compete 2. Audit Rights 3. Insurance\n{{ input.contract_text }}",
                    "output": {"schema": {"clauses": "list[{clause_type: string, text_spans: list[string]}]"}},
                },
                target_ops=["extract_clauses"],
                expected_behavior="Should insert a presence check with one check per category and wrap each category's line in a condition",
                should_pass=True,
            ),
        ]
    )

    def __eq__(self, other):
        return isinstance(other, PresenceGateDirective)

    def __hash__(self):
        return hash("PresenceGateDirective")

    def to_string_for_instantiate(self, original_op: Dict, profile: Dict = None) -> str:
        profile = profile or {}
        excerpts = "\n\n".join(f"--- Sample document ({len(t)} chars), start ---\n{t[:1500]}"
                               for t in profile.get("excerpts", []))
        return (
            f"You are applying the presence_gate directive to this map operation:\n{json.dumps(original_op, indent=2)}\n\n"
            f"Directive: {self.name}\n{self.nl_description}\n\n"
            f"{_gate_explanation()}\n"
            f"Document lengths (chars): {json.dumps(profile.get('chars', {}))}\n\n"
            f"Models available for checker 'llm':\n{available_models_info()}\n\n"
            f"Sample documents (no labels):\n{excerpts}\n\n"
            f"Output a PresenceGateInstantiateSchema:\n"
            f"- name: identifier for the presence-check operator\n"
            f"- document_key: the text field the map reads (from {{{{ input.KEY }}}})\n"
            f"- granularity: 'category' if the map asks for several outputs that are independently present or absent; "
            f"'document' if it asks one thing that many documents simply do not contain\n"
            f"- checker, checker_model: 'jev' unless presence needs reasoning a classifier cannot do\n"
            f"- context: one or two sentences on the task, shown with every chunk\n"
            f"- checks: one per output category worth gating. Gate categories that are often absent; leave out "
            f"categories nearly every document has (they stay in the prompt). Each question asks whether the text "
            f"contains the thing, broad enough to catch any phrasing. prompt_fragment must be copied verbatim from "
            f"the map prompt (exact characters, usually one whole numbered item), one fragment per check, no overlaps.\n"
            f"- threshold, chunk_chars: Jev settings (defaults 0.3 and 16000)\n"
            f"- skip_when_none: skip the map for documents where no check is present\n\n"
            f"Example:\n{self.example}\n\n"
            f"Output only the PresenceGateInstantiateSchema object."
        )

    def llm_instantiate(self, original_op: Dict, agent_llm: str, texts: List[str], message_history: list = []):
        cost = 0.0
        lengths = sorted(len(t) for t in texts)
        profile = {}
        if lengths:
            profile["chars"] = {"n": len(lengths), "min": lengths[0], "median": int(median(lengths)), "max": lengths[-1]}
            order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
            profile["excerpts"] = [texts[order[len(order) // 2]], texts[order[-1]]]
        message_history.append({"role": "user", "content": self.to_string_for_instantiate(original_op, profile)})
        for _ in range(MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS):
            resp, call_cost = agent_completion(model=agent_llm, messages=message_history,
                                               response_format=PresenceGateInstantiateSchema)
            cost += call_cost
            content = resp.choices[0].message.content
            message_history.append({"role": "assistant", "content": content})
            try:
                parsed = _extract_json_from_text(content) if is_together_model(agent_llm) else json.loads(content)
                rewrite = PresenceGateInstantiateSchema(**parsed)
                self.apply([original_op], original_op["name"], rewrite)  # validates fragments and the template
                return rewrite, message_history, cost
            except Exception as err:
                message_history.append({"role": "user", "content": f"Validation error: {err}\nPlease try again."})
        raise Exception(f"Failed to instantiate directive after {MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS} attempts.")

    @staticmethod
    def _gate_prompt(prompt: str, rewrite: PresenceGateInstantiateSchema, prefix: str) -> str:
        spans = []
        for c in rewrite.checks:
            n = prompt.count(c.prompt_fragment)
            if n != 1:
                raise ValueError(f"prompt_fragment of check {c.key!r} occurs {n} times in the prompt; it must be "
                                 f"copied verbatim and occur exactly once")
            start = prompt.index(c.prompt_fragment)
            spans.append((start, start + len(c.prompt_fragment), c.key))
        spans.sort()
        for (s1, e1, k1), (s2, e2, k2) in zip(spans, spans[1:]):
            if s2 < e1:
                raise ValueError(f"prompt fragments of {k1!r} and {k2!r} overlap")
        out, pos = [], 0
        for s, e, key in spans:
            out += [prompt[pos:s], f"{{% if input.{prefix}{key} %}}{prompt[s:e]}{{% endif %}}"]
            pos = e
        out.append(prompt[pos:])
        return "".join(out) + ABSENT_NOTE

    def apply(self, ops_list: List[Dict], target_op: str, rewrite: PresenceGateInstantiateSchema) -> List[Dict]:
        new_ops_list = deepcopy(ops_list)
        pos = [i for i, op in enumerate(new_ops_list) if op["name"] == target_op][0]
        target = new_ops_list[pos]
        if target.get("type") != "map":
            raise ValueError(f"presence_gate applies to map operations, not {target.get('type')!r}")
        key = rewrite.document_key
        prompt = target.get("prompt", "")
        if not re.search(rf"\b[A-Za-z_][A-Za-z0-9_]*\.{re.escape(key)}(?![A-Za-z0-9_])", prompt):
            raise ValueError(f"The prompt of {target_op!r} does not reference document_key {key!r}")
        if rewrite.granularity == "document" and not rewrite.skip_when_none:
            raise ValueError("granularity 'document' gates by skipping the map, so skip_when_none must be true")
        names = {op["name"] for op in new_ops_list}
        if rewrite.name in names:
            raise ValueError(f"operator name {rewrite.name!r} already exists")

        prefix, any_key = f"{rewrite.name}_", f"{rewrite.name}_any"
        fields = [f"{prefix}{c.key}" for c in rewrite.checks]
        if rewrite.checker == "jev":
            checker_ops = [{
                "name": rewrite.name, "type": "presence_check", "text_key": key,
                "checks": [{"key": c.key, "question": c.question} for c in rewrite.checks],
                "context": rewrite.context, "threshold": rewrite.threshold, "chunk_chars": rewrite.chunk_chars,
                "output_prefix": prefix, "any_key": any_key,
            }]
        else:
            model = rewrite.checker_model if "/" in rewrite.checker_model else f"gemini/{rewrite.checker_model}"
            if model.split("/", 1)[1] not in AVAILABLE_MODELS:
                raise ValueError(f"checker_model must be one of {AVAILABLE_MODELS}")
            questions = "\n".join(f"- {prefix}{c.key}: {c.question}" for c in rewrite.checks)
            checker_ops = [
                {"name": rewrite.name, "type": "map", "model": model,
                 "prompt": (f"{rewrite.context}\n\nDocument:\n{{{{ input.{key} }}}}\n\nFor each field below, answer "
                            f"true if the document contains what the question describes, otherwise false. When "
                            f"unsure, answer true.\n{questions}"),
                 "output": {"schema": {f: "boolean" for f in fields}}},
                {"name": f"{rewrite.name}_any_flag", "type": "code_map", "code": _any_code(fields, any_key)},
            ]

        if rewrite.granularity == "category":
            target["prompt"] = self._gate_prompt(prompt, rewrite, prefix)
        Environment().parse(target["prompt"])  # the rewritten prompt must still be a valid template

        tail = []
        if rewrite.skip_when_none:
            target["type"] = "gated_map"
            target["gate_key"] = any_key if rewrite.granularity == "category" else fields[0]
            schema = (target.get("output") or {}).get("schema") or {}
            if any(_empty_value(v) is not None for v in schema.values()):
                tail = [{"name": f"{target_op}_fill_skipped", "type": "code_map", "code": _fill_code(schema)}]

        new_ops_list[pos:pos + 1] = checker_ops + [target] + tail
        return new_ops_list

    def instantiate(self, operators: List[Dict], target_ops: List[str], agent_llm: str,
                    message_history: list = [], **kwargs):
        assert len(target_ops) == 1, "presence_gate directive requires exactly one target op"
        target_op_config = [op for op in operators if op["name"] == target_ops[0]][0]
        prompt_text = target_op_config.get("prompt", "")
        keys = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\.([A-Za-z_][A-Za-z0-9_]*)", prompt_text)
        candidates = {k: _load_texts(kwargs.get("input_file_path"), k) for k in set(keys)}
        texts = max(candidates.values(), key=lambda t: sum(map(len, t)), default=[])
        rewrite, message_history, call_cost = self.llm_instantiate(target_op_config, agent_llm, texts, message_history)
        return self.apply(operators, target_ops[0], rewrite), message_history, call_cost
