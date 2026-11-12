"""jev_fusion: merge two consecutive Jev classifications into one Jev call.

Jev sends the rendered prompt once as its state and asks one question per output field, so two
Jev operations that read the same input can share a single call: the fused map outputs the union
of their fields, and a filter becomes a code_filter on its boolean field after the fused map.
The rewrite is deterministic (no agent call): the prompts are merged mechanically.
"""
import re
from copy import deepcopy
from typing import Dict, List, Optional, Type

from jinja2 import Environment, TemplateSyntaxError
from pydantic import BaseModel, Field

from .base import Directive, DirectiveTestCase
from .jev_classify import JEV_MODEL, is_jev_op

_FIELD_RE = re.compile(r"input\.([A-Za-z_][A-Za-z0-9_]*)|input\[['\"]([^'\"]+)['\"]\]")
_SECTION = "Output field "
# A Jev filter and map keep only these keys; LLM-only options were removed by jev_classify.
_KEEP_KEYS = ("model", "skip_on_error", "timeout", "bypass_cache", "drop_keys")


def _fields_read(prompt: str) -> set:
    return {a or b for a, b in _FIELD_RE.findall(prompt or "")}


def _is_classification(op: Dict) -> bool:
    schema = (op.get("output") or {}).get("schema") or {}
    return bool(schema) and all(
        str(k).strip() in ("bool", "boolean") or re.fullmatch(r"enum\[.*\]", str(k).strip())
        for k in schema.values())


def _split_prompt(prompt: str):
    """(task text, output-field sections, input block). The input block is every line that
    renders input, plus a short label line right before it (e.g. 'Document text:')."""
    lines = (prompt or "").splitlines()
    input_idx = {i for i, l in enumerate(lines) if "{{" in l or "{%" in l}
    for i in sorted(input_idx):
        j = i - 1
        while j >= 0 and not lines[j].strip():
            j -= 1
        if j >= 0 and j not in input_idx and lines[j].strip().endswith(":") and _SECTION not in lines[j] \
                and len(lines[j].strip()) <= 40:
            input_idx.add(j)
    body = [l for i, l in enumerate(lines) if i not in input_idx]
    block = "\n".join(l for i, l in enumerate(lines) if i in input_idx).strip()
    text = "\n".join(body).strip()
    k = text.find(_SECTION)
    if k < 0:
        return text, "", block
    return text[:k].strip(), text[k:].strip(), block


class JevFusionInstantiateSchema(BaseModel):
    """Nothing to choose: the fused operation is built from the two target operations."""
    fused_name: Optional[str] = Field(default=None, description="Name of the fused operation (optional)")


class JevFusionDirective(Directive):
    name: str = Field(default="jev_fusion", description="The name of the directive")
    formal_description: str = Field(
        default="Filter/Map (Jev) -> Filter/Map (Jev) => Map (Jev) [-> Code Filter]",
        description="Merges two consecutive Jev classifications into one Jev call",
    )
    nl_description: str = Field(
        default="Merges two consecutive operations that both run on Jev and read the same input into ONE Jev "
        "call. Jev reads the rendered prompt once (its state) and answers one question per output field, so the "
        "fused operation outputs the fields of both operations while the document is read and billed once. A "
        "filter becomes a code_filter on its boolean field right after the fused map. Deterministic: the prompts "
        "are merged mechanically (no agent call)."
    )
    when_to_use: str = Field(
        default="For cost: two consecutive Jev operations (e.g. a Jev filter followed by a Jev classification map) "
        "that read the same input fields. Saves one Jev call per document. Not usable when the second operation "
        "reads an output of the first, or when either is not on Jev (use jev_classify first)."
    )
    instantiate_schema_type: Type[BaseModel] = Field(default=JevFusionInstantiateSchema)
    example: str = Field(
        default="""
        Before: filter_sustainability_reports (Jev, is_sustainability_report: boolean) ->
                determine_economic_activity (Jev, economic_activity: enum[...]), both reading {{ input.tot_text_raw }}
        After:  filter_sustainability_reports_determine_economic_activity_jev (Jev map with both fields, one call)
                -> keep_is_sustainability_report (code_filter on is_sustainability_report)
        """
    )
    test_cases: List[DirectiveTestCase] = Field(default_factory=list)

    def __eq__(self, other):
        return isinstance(other, JevFusionDirective)

    def __hash__(self):
        return hash("JevFusionDirective")

    # ---- checks --------------------------------------------------------------------------------

    @staticmethod
    def check(ops_list: List[Dict], target_ops: List[str]) -> Optional[str]:
        """Why these two operations cannot be fused, or None."""
        if len(target_ops) != 2:
            return "jev_fusion takes exactly two operations"
        names = [o.get("name") for o in ops_list]
        if any(t not in names for t in target_ops):
            return f"unknown operation in {target_ops}"
        i, j = names.index(target_ops[0]), names.index(target_ops[1])
        if j != i + 1:
            return "the two operations must be consecutive, in pipeline order"
        a, b = ops_list[i], ops_list[j]
        for op in (a, b):
            if op.get("type") not in ("filter", "map"):
                return f"{op['name']} is a {op.get('type')}; jev_fusion fuses filters and maps"
            if not is_jev_op(op):
                return f"{op['name']} does not run on Jev (apply jev_classify first)"
            if not _is_classification(op):
                return f"{op['name']} has outputs that are not boolean or enum"
        fa, fb = _fields_read(a.get("prompt")), _fields_read(b.get("prompt"))
        if not fa or fa != fb:
            return f"the two operations read different inputs ({sorted(fa)} vs {sorted(fb)})"
        a_out = set((a.get("output") or {}).get("schema") or {})
        if fb & a_out:
            return f"{b['name']} reads {sorted(fb & a_out)}, an output of {a['name']}"
        b_out = set((b.get("output") or {}).get("schema") or {})
        if a_out & b_out:
            return f"both operations output {sorted(a_out & b_out)}"
        if str(a.get("model")) != str(b.get("model")):
            return "the two operations use different Jev models"
        return None

    # ---- rewrite -------------------------------------------------------------------------------

    @staticmethod
    def fused_prompt(a: Dict, b: Dict) -> str:
        ta, sa, ia = _split_prompt(a.get("prompt"))
        tb, sb, _ = _split_prompt(b.get("prompt"))
        fields = list(((a.get("output") or {}).get("schema") or {})) + list(((b.get("output") or {}).get("schema") or {}))
        parts = [f"This document is used for {len(fields)} separate decisions. Answer each output field on its "
                 f"own, following its section below; the decisions do not depend on each other.", ""]
        for task, sections, op in ((ta, sa, a), (tb, sb, b)):
            own = ", ".join(((op.get("output") or {}).get("schema") or {}))
            if task:
                parts += [f"Task for {own}: {task}", ""]
            if sections:
                parts += [sections, ""]
        parts.append(ia or "Document text:\n{{ input.tot_text_raw }}")
        return "\n".join(parts).strip() + "\n"

    def apply(self, ops_list: List[Dict], target_ops: List[str],
              rewrite: Optional[JevFusionInstantiateSchema] = None) -> List[Dict]:
        err = self.check(ops_list, target_ops)
        if err:
            raise ValueError(err)
        new_ops = deepcopy(ops_list)
        names = [o["name"] for o in new_ops]
        i = names.index(target_ops[0])
        a, b = new_ops[i], new_ops[i + 1]
        schema = {**a["output"]["schema"], **b["output"]["schema"]}
        name = (rewrite.fused_name if rewrite and rewrite.fused_name else f"{a['name']}_{b['name']}_jev")
        fused = {"name": name, "type": "map", "prompt": self.fused_prompt(a, b), "output": {"schema": schema}}
        for k in _KEEP_KEYS:
            if k in a:
                fused[k] = a[k]
        fused["model"] = a.get("model") or JEV_MODEL
        try:
            Environment().parse(fused["prompt"])
        except TemplateSyntaxError as e:
            raise ValueError(f"fused prompt is not valid Jinja: {e}")
        replacement = [fused]
        for op in (a, b):         # each filter keeps documents only where its boolean is true
            if op["type"] == "filter":
                field = next(iter(op["output"]["schema"]))
                replacement.append({"name": f"keep_{field}", "type": "code_filter",
                                    "code": f"def transform(input_doc):\n    return bool(input_doc.get({field!r}))\n"})
        return new_ops[:i] + replacement + new_ops[i + 2:]

    # ---- search interface (no agent call) ---------------------------------------------------------

    def to_string_for_instantiate(self, *args, **kwargs) -> str:
        return ""

    def llm_instantiate(self, *args, **kwargs):
        return JevFusionInstantiateSchema(), kwargs.get("message_history", []), 0.0

    def instantiate(self, operators: List[Dict], target_ops: List[str], agent_llm: str = "",
                    message_history: list = [], **kwargs):
        rewrite = JevFusionInstantiateSchema()
        return self.apply(operators, target_ops, rewrite), message_history, 0.0
