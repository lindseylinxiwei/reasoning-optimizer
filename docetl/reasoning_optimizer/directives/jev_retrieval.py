"""jev_retrieval rewrite: Jev on fixed chunks, neighbours as context (operator jev_chunk_retrieve).

Every 500-token chunk of the document is scored once by Jev for the evidence targets; the best-scored chunks
are kept up to a share of the document (2.5%, 5%, 10% or 20%). This replaced the region-tree navigation, which
is kept unregistered as a ghost in jev_region_retrieval.py. On the Sustainability emissions test set (97
reports), chunk scoring kept the most evidence at every share (71% of targets at 2.5%, 91% at 5%, 97% at 10%).
"""
import json
import re
from copy import deepcopy
from typing import Dict, List, Optional, Type

from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import JEV_RETRIEVAL_SHARES, JevRetrievalInstantiateSchema

from .agent_utils import _extract_json_from_text, agent_completion, is_together_model
from .base import MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS, Directive, DirectiveTestCase

# Template references to a field: input.key / item.key / doc.key and input["key"].
_ATTR = r"(\b[A-Za-z_][A-Za-z0-9_]*)\.{key}(?![A-Za-z0-9_])"
_ITEM = r"""(\b[A-Za-z_][A-Za-z0-9_]*)\[(["']){key}\2\]"""


def _rewrite_references(text: str, key: str, new_key: str) -> tuple[str, int]:
    text, n1 = re.subn(_ATTR.format(key=re.escape(key)), rf"\1.{new_key}", text)
    text, n2 = re.subn(_ITEM.format(key=re.escape(key)), rf"\1[\2{new_key}\2]", text)
    return text, n1 + n2


def _load_texts(input_file_path: Optional[str], key: str) -> List[str]:
    """Texts of `key` from the (label-free) input file, or [] if unavailable."""
    if not input_file_path:
        return []
    try:
        with open(input_file_path) as f:
            data = json.load(f)
    except Exception:
        return []
    return [d[key] for d in data if isinstance(d, dict) and isinstance(d.get(key), str)]


def _length_summary(texts: List[str]) -> str:
    if not texts:
        return "unknown (no input file)"
    n = sorted(len(t) for t in texts)
    pick = lambda q: n[min(len(n) - 1, int(q * (len(n) - 1)))]  # noqa: E731
    return (f"{len(n)} documents; characters: median {pick(0.5):,}, 10th percentile {pick(0.1):,}, "
            f"90th percentile {pick(0.9):,}, longest {n[-1]:,} (about 4 characters per token)")


class JevRetrievalDirective(Directive):
    name: str = Field(default="jev_retrieval", description="The name of the directive")
    formal_description: str = Field(
        default="LLM_Op => JevChunkRetrieve -> LLM_Op",
        description="Inserts a jev_chunk_retrieve operator before an LLM operation so it reads only the best chunks",
    )
    nl_description: str = Field(
        default="Data decomposition with Jev retrieval: before the target operation, a jev_chunk_retrieve operator "
        "cuts each document into 500-token chunks, has the Jev classifier score every chunk for named evidence "
        "targets (each request shows a run of consecutive chunks plus one neighbouring chunk on each side as "
        "context), and keeps the best-scored chunks up to a share of the document, in document order. The target "
        "operation's prompt is pointed at the kept chunks; other operations keep the full text."
    )
    when_to_use: str = Field(
        default="When an LLM operation reads long documents but only needs sparse evidence (a small part of each "
        "document, in one or a few places) whose position differs from document to document, so a fixed head/tail "
        "window cannot capture it (e.g. an emissions table, a particular contract clause). Not useful when the "
        "answer depends on content spread across the whole document (summaries, counts or aggregates over all "
        "sections), or when the evidence always sits at a fixed position such as the opening pages (keeping the "
        "head of the document is cheaper there)."
    )

    instantiate_schema_type: Type[BaseModel] = Field(default=JevRetrievalInstantiateSchema)

    example: str = Field(
        default="""
        Original Map Operation (emissions extraction from sustainability reports):
        - name: extract_emissions
          type: map
          prompt: |
            Extract the company's total Scope 1 and Scope 2 emissions from this report: {{ input.tot_text_raw }}
          output:
            schema:
              scope_1: "string"
              scope_2: "string"

        JevRetrievalInstantiateSchema:
        {
          "name": "retrieve_emissions_evidence",
          "document_key": "tot_text_raw",
          "targets": [
            {"name": "scope_1_2_emissions", "description": "Passages or tables reporting the company's total Scope 1 (direct) and Scope 2 (purchased electricity, steam, heating or cooling) greenhouse gas emissions, with values, units and years."}
          ],
          "share": 0.05
        }

        Resulting Pipeline:
        - name: retrieve_emissions_evidence
          type: jev_chunk_retrieve
          text_key: tot_text_raw
          output_key: tot_text_raw_jev_extract_emissions
          targets: [...as above...]
          share: 0.05
          chunk_tokens: 500
        - name: extract_emissions
          type: map
          prompt: |
            Extract the company's total Scope 1 and Scope 2 emissions from this report: {{ input.tot_text_raw_jev_extract_emissions }}
          ...
        """
    )

    test_cases: List[DirectiveTestCase] = Field(
        default_factory=lambda: [
            DirectiveTestCase(
                name="emissions_table_extraction",
                description="Extract figures from a table whose position differs from report to report",
                input_config={
                    "name": "extract_emissions",
                    "type": "map",
                    "prompt": "Extract total Scope 1 and Scope 2 emissions from this report: {{ input.tot_text_raw }}",
                    "output": {"schema": {"scope_1": "string", "scope_2": "string"}},
                },
                target_ops=["extract_emissions"],
                expected_behavior="Should insert a jev_chunk_retrieve over tot_text_raw with an emissions evidence target and point the map prompt at the kept chunks",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="adverse_reaction_filter",
                description="Filter medical reports mentioning an adverse reaction somewhere in a long text",
                input_config={
                    "name": "has_adverse_reaction",
                    "type": "filter",
                    "prompt": "Does this report describe an adverse drug reaction? {{ input.fulltext }}",
                    "output": {"schema": {"has_reaction": "boolean"}},
                },
                target_ops=["has_adverse_reaction"],
                expected_behavior="Should keep chunks describing patient reactions or side effects before the filter",
                should_pass=True,
            ),
        ]
    )

    # Shared data view: the same trimmed text for every operator (see data_views.py).
    view_technique: str = "jev"

    def view_ops(self, config, text_key, model, **kwargs):
        from docetl.reasoning_optimizer.data_views import view_ops
        return view_ops("jev", config, text_key, model, **kwargs)

    def __eq__(self, other):
        return isinstance(other, JevRetrievalDirective)

    def __hash__(self):
        return hash("JevRetrievalDirective")

    def to_string_for_instantiate(self, original_op: Dict, lengths: str = "unknown") -> str:
        shares = ", ".join(f"{s:g}" for s in JEV_RETRIEVAL_SHARES)
        return (
            f"You are configuring Jev retrieval in front of this operation:\n{json.dumps(original_op, indent=2)}\n\n"
            f"Directive: {self.name}\n{self.nl_description}\n\n"
            "How it works: each document is cut into 500-token chunks. Jev is a classifier: one request shows a run "
            "of consecutive chunks (up to 28 KB, at most 32 questions) plus one neighbouring chunk on each side, and "
            "asks for every chunk and target 'does this chunk contain evidence for the target?', scored 0-1. Every "
            "chunk is scored once; a chunk's score is its best score over the targets. The best-scored chunks are "
            "kept, up to `share` of the document, and joined in document order. Cost: input tokens only, about "
            "$0.042 per million, so scoring a whole document costs about its length in tokens x $0.04 per million "
            "(times about the number of targets when chunks per request must shrink to fit 32 questions).\n\n"
            f"Document lengths of the field: {lengths}\n\n"
            "Output a JevRetrievalInstantiateSchema:\n"
            "- name: a descriptive name for the jev_chunk_retrieve operator\n"
            "- document_key: the long text field the operation reads (from {{ input.KEY }} references)\n"
            "- targets: 1-4 evidence targets. Each description defines which passages to keep, phrased as the "
            "evidence the operation needs (e.g. 'passages or tables reporting Scope 1 and Scope 2 emissions with "
            "values and units'), never as the answer. One target is usually enough; add one only for a distinct "
            "kind of evidence.\n"
            f"- share: share of each document to keep, one of {shares}. Smaller shares are cheaper downstream but "
            "can miss evidence; choose from how sparse the evidence is relative to the document lengths above.\n\n"
            f"Example:\n{self.example}\n\n"
            "Output only the JevRetrievalInstantiateSchema object."
        )

    def llm_instantiate(self, original_op: Dict, agent_llm: str, texts: List[str], message_history: list = []):
        message_history.append({"role": "user", "content": self.to_string_for_instantiate(original_op, _length_summary(texts))})
        cost = 0.0
        for _ in range(MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS):
            resp, call_cost = agent_completion(model=agent_llm, messages=message_history,
                                               response_format=JevRetrievalInstantiateSchema)
            cost += call_cost
            content = resp.choices[0].message.content
            try:
                parsed = _extract_json_from_text(content) if is_together_model(agent_llm) else json.loads(content)
                rewrite = JevRetrievalInstantiateSchema(**parsed)
                self.apply([original_op], original_op["name"], rewrite)  # fails on an unread document_key
                message_history.append({"role": "assistant", "content": content})
                return rewrite, message_history, cost
            except Exception as err:
                message_history.append({"role": "assistant", "content": content})
                message_history.append({"role": "user", "content": f"Validation error: {err}\nPlease try again."})
        raise Exception(f"Failed to instantiate directive after {MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS} attempts.")

    def apply(self, ops_list: List[Dict], target_op: str, rewrite: JevRetrievalInstantiateSchema) -> List[Dict]:
        new_ops_list = deepcopy(ops_list)
        pos = [i for i, op in enumerate(new_ops_list) if op["name"] == target_op][0]
        target = new_ops_list[pos]
        suffix = f"_jev_{target_op}"
        key = rewrite.document_key[: -len(suffix)] if rewrite.document_key.endswith(suffix) else rewrite.document_key
        new_key = f"{key}{suffix}"
        retrieve_op = {
            "name": rewrite.name,
            "type": "jev_chunk_retrieve",
            "text_key": key,
            "output_key": new_key,
            "targets": [t.model_dump() for t in rewrite.targets],
            "share": rewrite.share,
            "chunk_tokens": 500,
        }
        # The operation already reads a Jev view of this field: replace that view (other share or targets)
        # instead of retrieving from retrieved text.
        existing = [i for i, op in enumerate(new_ops_list[:pos])
                    if op.get("type") == "jev_chunk_retrieve" and op.get("output_key") == new_key]
        if existing:
            new_ops_list[existing[-1]] = {**retrieve_op, "name": new_ops_list[existing[-1]]["name"]}
            return new_ops_list

        replaced = 0
        for field, value in list(target.items()):
            if "prompt" in field and isinstance(value, str):
                target[field], n = _rewrite_references(value, key, new_key)
                replaced += n
        if replaced == 0:
            raise ValueError(f"The prompt of {target_op!r} does not reference document_key {key!r}")
        new_ops_list.insert(pos, retrieve_op)
        return new_ops_list

    def instantiate(self, operators: List[Dict], target_ops: List[str], agent_llm: str,
                    message_history: list = [], **kwargs):
        assert len(target_ops) == 1, "jev_retrieval directive requires exactly one target op"
        target_op_config = [op for op in operators if op["name"] == target_ops[0]][0]
        prompt_text = " ".join(v for k, v in target_op_config.items() if "prompt" in k and isinstance(v, str))
        keys = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\.([A-Za-z_][A-Za-z0-9_]*)", prompt_text)
        candidates = {k: _load_texts(kwargs.get("input_file_path"), k) for k in set(keys)}
        texts = max(candidates.values(), key=lambda t: sum(map(len, t)), default=[])
        rewrite, message_history, call_cost = self.llm_instantiate(target_op_config, agent_llm, texts, message_history)
        return self.apply(operators, target_ops[0], rewrite), message_history, call_cost
