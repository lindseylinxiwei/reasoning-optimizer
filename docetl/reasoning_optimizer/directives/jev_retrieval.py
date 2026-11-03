import json
import re
from copy import deepcopy
from statistics import median
from typing import Dict, List, Optional, Type

from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import (
    JevIndexDesignSchema,
    JevRetrievalInstantiateSchema,
)
from docetl.retrieval.index_spec import (
    MAX_HEADING_LINE_SHARE,
    MIN_SECTION_CHARS,
    IndexSpec,
    check_spec,
    corpus_profile,
)

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


def _validation_sample(texts: List[str], n: int = 10) -> List[str]:
    """Up to n documents spread over the length distribution."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    picks = sorted({order[int(f * (len(order) - 1))] for f in [k / max(1, n - 1) for k in range(n)]}) if order else []
    return [texts[i] for i in picks]


def _jev_explanation() -> str:
    return (
        "How Jev retrieval works:\n"
        "- Each document is split into a hierarchy of regions (the index). Jev is a classifier: one Jev request "
        "shows it the text of up to `regions_per_request` sibling regions (about 7k tokens at most) and asks, for "
        "every (region, target) pair, 'does this region contain text matching the target?', scored 0-1.\n"
        "- A navigation step (decision) is one such request for the children of a region. Regions scoring >= 0.3 "
        "are kept as evidence and promising regions are opened further, until the cap is reached, max_decisions is "
        "used up, or nothing promising is left. Each target navigates separately; identical requests are shared "
        "between targets.\n"
        "- Jev allows 32 questions per request and every region in a request is asked about every target in "
        "it. Targets are therefore split automatically into bundles of 32 // regions_per_request targets; "
        "targets in a bundle share requests, and each extra bundle navigates with its own requests (more cost).\n"
        "- Cost: input tokens only, about $0.0001 per request, so at most about max_decisions x $0.0001 per "
        "target per document.\n"
    )


class JevRetrievalDirective(Directive):
    name: str = Field(default="jev_retrieval", description="The name of the directive")
    formal_description: str = Field(
        default="LLM_Op => JevRetrieve -> LLM_Op",
        description="Inserts a jev_retrieve operator before an LLM operation so it reads only retrieved passages",
    )
    nl_description: str = Field(
        default="Data decomposition with Jev retrieval: before the target operation, a jev_retrieve operator "
        "builds a hierarchical index of each long document, lets the Jev classifier score regions against named "
        "evidence targets and navigate into the promising ones, and keeps only the selected literal passages (up to "
        "a character cap per target, in document order). The target operation's prompt is pointed at those "
        "passages; other operations keep the full text."
    )
    when_to_use: str = Field(
        default="When an LLM operation reads long documents but only needs sparse evidence (a small part of each "
        "document, in one or a few places) whose position differs from document to document, so a fixed head/tail "
        "window cannot capture it (e.g. a particular contract clause, the passage describing an event). Not useful "
        "when the answer depends on content spread across the whole document (summaries, counts or aggregates over "
        "all sections), when documents are short relative to the cap, or when the evidence always sits at a fixed "
        "position such as the opening pages (keeping the head of the document is cheaper there)."
    )

    instantiate_schema_type: Type[BaseModel] = Field(default=JevRetrievalInstantiateSchema)

    example: str = Field(
        default="""
        Original Map Operation (Contract clause extraction):
        - name: extract_termination_terms
          type: map
          prompt: |
            Extract the termination conditions and notice periods from this contract: {{ input.contract_text }}
          output:
            schema:
              termination_conditions: "list[string]"
              notice_period: "string"

        JevRetrievalInstantiateSchema:
        {
          "name": "retrieve_termination_evidence",
          "document_key": "contract_text",
          "targets": [
            {"name": "termination", "description": "Provisions allowing a party to terminate the agreement, including conditions, exceptions and consequences."},
            {"name": "notice_period", "description": "Clauses stating how much notice must be given before termination or non-renewal."}
          ],
          "evidence_chars": 6000,
          "policy": "preserve_fill"
        }

        Resulting Pipeline:
        - name: retrieve_termination_evidence
          type: jev_retrieve
          text_key: contract_text
          output_key: contract_text_jev_extract_termination_terms
          targets: [...as above...]
          evidence_chars: 6000
          policy: preserve_fill
          max_decisions: 128
          regions_per_request: 4
          index:                      # designed in the first call
            levels:
              - - {kind: heading, style: article_section}
              - {kind: blank_line}
            leaf_chars: 3000
            min_chars: 200
            fanout: 6
        - name: extract_termination_terms
          type: map
          prompt: |
            Extract the termination conditions and notice periods from this contract: {{ input.contract_text_jev_extract_termination_terms }}
          ...
        """
    )

    test_cases: List[DirectiveTestCase] = Field(
        default_factory=lambda: [
            DirectiveTestCase(
                name="contract_clause_extraction",
                description="Extract a specific clause whose position differs from contract to contract",
                input_config={
                    "name": "extract_termination_terms",
                    "type": "map",
                    "prompt": "Extract the termination conditions from this contract: {{ input.contract_text }}",
                    "output": {"schema": {"termination_conditions": "list[string]"}},
                },
                target_ops=["extract_termination_terms"],
                expected_behavior="Should insert a jev_retrieve over contract_text with a termination evidence target and point the map prompt at the retrieved field",
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
                expected_behavior="Should retrieve passages describing patient reactions or side effects before the filter",
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

    def index_design_prompt(self, original_op: Dict, profile: Dict) -> str:
        stats = {k: v for k, v in profile.items() if k != "excerpts"}
        excerpts = "\n\n".join(
            f"--- Sample document ({e['chars']} chars), start ---\n{e['start']}\n"
            f"--- same document, middle ---\n{e['middle']}" for e in profile.get("excerpts", []))
        return (
            f"You are designing how documents are split into a hierarchy of regions for Jev retrieval.\n\n"
            f"Target {original_op.get('type', 'operation')} operation:\n{json.dumps(original_op, indent=2)}\n\n"
            f"{_jev_explanation()}\n"
            f"Corpus profile of the text field (no labels):\n{json.dumps(stats, indent=2)}\n\n"
            f"Sample excerpts:\n{excerpts}\n\n"
            f"Design one index for the whole corpus. It is applied identically to every document with "
            f"deterministic string rules (no code). Output a JevIndexDesignSchema:\n"
            f"- levels: splitters from outer to inner, each one of\n"
            f"    {{\"kind\": \"heading\", \"style\": NAME}}  split before every line of a named style (see 'styles' "
            f"in the profile: measured counts, section sizes and real example lines for this corpus)\n"
            f"    {{\"kind\": \"heading\", \"pattern\": REGEX}}  only if no named style fits: lines whose start matches REGEX\n"
            f"    {{\"kind\": \"page_break\"}} or {{\"kind\": \"page_break\", \"style\": \"page_number_line\"}}  form feeds, or page-number lines\n"
            f"    {{\"kind\": \"blank_line\"}}  paragraphs\n"
            f"    {{\"kind\": \"fixed\", \"chars\": N}}  pieces of about N characters\n"
            f"  A level that does not split a region is skipped for that region.\n"
            f"- leaf_chars: regions longer than this are cut into leaves at sentence boundaries.\n"
            f"- min_chars: segments shorter than this are merged into a neighbor (avoids tiny regions).\n"
            f"- fanout: maximum children per node; larger sets are grouped into contiguous blocks.\n\n"
            f"Guidelines:\n"
            f"- Follow the structure the documents actually have (from the profile and excerpts): real section "
            f"headings as outer levels if they exist, paragraphs or fixed pieces otherwise.\n"
            f"- Regions should be units in which the evidence for the operation appears together, so a region "
            f"Jev keeps is useful on its own. Leaves of roughly 1000-4000 characters work well for Jev requests.\n"
            f"- Prefer named styles. Use a style as a heading level only if its example_lines are real headings "
            f"in this corpus (not page numbers, table rows or running headers).\n"
            f"- Each heading level must match at most {MAX_HEADING_LINE_SHARE:.0%} of non-empty lines and give "
            f"sections of at least {MIN_SECTION_CHARS} characters on average (share_of_lines_median and "
            f"section_chars_median in the profile); otherwise the design is rejected.\n"
            f"- Make the levels match the rationale (if you describe a paragraph level, include it).\n"
            f"- Prefer few levels; an empty list means fixed leaves only."
        )

    def retrieval_config_prompt(self, original_op: Dict, profile: Dict, spec: IndexSpec, stats: Dict) -> str:
        return (
            f"You are configuring Jev retrieval in front of this operation:\n{json.dumps(original_op, indent=2)}\n\n"
            f"Directive: {self.name}\n{self.nl_description}\n\n"
            f"{_jev_explanation()}\n"
            f"Document lengths (chars): {json.dumps(profile.get('chars', {}))}\n"
            f"Index already designed: {spec.model_dump_json()}\n"
            f"Index built on sample documents: {json.dumps(stats)}\n\n"
            f"Output a JevRetrievalInstantiateSchema:\n"
            f"- name: a descriptive name for the jev_retrieve operator\n"
            f"- document_key: the long text field the operation reads (from {{{{ input.KEY }}}} references)\n"
            f"- targets: 1-32 evidence targets. Each description defines which passages to keep, phrased as the "
            f"evidence the operation needs (e.g. 'passages describing the company's products and business "
            f"segments'), never as the answer. One target per distinct kind of information is usually enough. "
            f"Each target adds questions to every request, and every 32 // regions_per_request targets form an "
            f"extra bundle with its own requests.\n"
            f"- evidence_chars: character cap per target, well below the median document length, or the operator "
            f"keeps most of the text and saves nothing. Documents shorter than the cap pass through unchanged.\n"
            f"- policy: 'preserve_fill' (default) fills up to the cap from already-computed scores; 'adaptive' keeps "
            f"only regions scoring >= 0.3 and may return less.\n"
            f"- max_decisions: navigation steps per target (16-256). Lower it when evidence is easy to find; raise "
            f"it for long documents with hard-to-find evidence.\n"
            f"- regions_per_request: siblings scored per request (1-8). More regions per request explores faster "
            f"per decision but each request is larger, and fewer targets fit in one bundle.\n\n"
            f"Example:\n{self.example}\n\n"
            f"Output only the JevRetrievalInstantiateSchema object."
        )

    @staticmethod
    def check_index(spec: IndexSpec, samples: List[str]) -> Dict:
        return check_spec(spec, samples)

    def to_string_for_instantiate(self, original_op: Dict, profile: Optional[Dict] = None) -> str:
        """Prompt of the first call (index design); the second call's prompt is retrieval_config_prompt."""
        return self.index_design_prompt(original_op, profile or {})

    @staticmethod
    def _ask(agent_llm, message_history, schema_type, validate, cost):
        for _ in range(MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS):
            resp, call_cost = agent_completion(model=agent_llm, messages=message_history, response_format=schema_type)
            cost[0] += call_cost
            content = resp.choices[0].message.content
            try:
                parsed = _extract_json_from_text(content) if is_together_model(agent_llm) else json.loads(content)
                result = validate(schema_type(**parsed))
                message_history.append({"role": "assistant", "content": content})
                return result
            except Exception as err:
                message_history.append({"role": "assistant", "content": content})
                message_history.append({"role": "user", "content": f"Validation error: {err}\nPlease try again."})
        raise Exception(f"Failed to instantiate directive after {MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS} attempts.")

    def llm_instantiate(self, original_op: Dict, agent_llm: str, texts: List[str], message_history: list = []):
        """Two agent calls: (1) design the index, (2) configure retrieval on top of it."""
        cost = [0.0]
        profile = corpus_profile(texts) if texts else {}
        samples = _validation_sample(texts)

        message_history.append({"role": "user", "content": self.index_design_prompt(original_op, profile)})

        def check_design(design):
            levels = [level.model_dump(exclude_none=True) for level in design.levels]
            spec = IndexSpec(levels=levels, leaf_chars=design.leaf_chars,
                             min_chars=design.min_chars, fanout=design.fanout)
            return spec, self.check_index(spec, samples)

        spec, stats = self._ask(agent_llm, message_history, JevIndexDesignSchema, check_design, cost)

        message_history.append({"role": "user", "content": self.retrieval_config_prompt(original_op, profile, spec, stats)})

        def check_config(rewrite):
            self.apply([original_op], original_op["name"], rewrite, spec)  # fails on an unread document_key
            return rewrite

        rewrite = self._ask(agent_llm, message_history, JevRetrievalInstantiateSchema, check_config, cost)
        return rewrite, spec, message_history, cost[0]

    def apply(self, ops_list: List[Dict], target_op: str, rewrite: JevRetrievalInstantiateSchema,
              index_spec: Optional[IndexSpec] = None) -> List[Dict]:
        new_ops_list = deepcopy(ops_list)
        pos = [i for i, op in enumerate(new_ops_list) if op["name"] == target_op][0]
        target = new_ops_list[pos]
        key = rewrite.document_key
        new_key = f"{key}_jev_{target_op}"

        replaced = 0
        for field, value in list(target.items()):
            if "prompt" in field and isinstance(value, str):
                target[field], n = _rewrite_references(value, key, new_key)
                replaced += n
        if replaced == 0:
            raise ValueError(f"The prompt of {target_op!r} does not reference document_key {key!r}")

        retrieve_op = {
            "name": rewrite.name,
            "type": "jev_retrieve",
            "text_key": key,
            "output_key": new_key,
            "targets": [t.model_dump() for t in rewrite.targets],
            "evidence_chars": rewrite.evidence_chars,
            "policy": rewrite.policy,
            "max_decisions": rewrite.max_decisions,
            "regions_per_request": rewrite.regions_per_request,
            "index": (index_spec or IndexSpec(levels=[{"kind": "blank_line"}])).model_dump(exclude_none=True),
        }
        new_ops_list.insert(pos, retrieve_op)
        return new_ops_list

    def instantiate(self, operators: List[Dict], target_ops: List[str], agent_llm: str,
                    message_history: list = [], **kwargs):
        assert len(target_ops) == 1, "jev_retrieval directive requires exactly one target op"
        target_op_config = [op for op in operators if op["name"] == target_ops[0]][0]
        prompt_text = " ".join(v for k, v in target_op_config.items() if "prompt" in k and isinstance(v, str))
        keys = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\.([A-Za-z_][A-Za-z0-9_]*)", prompt_text)
        # Profile the longest text field the operation reads.
        candidates = {k: _load_texts(kwargs.get("input_file_path"), k) for k in set(keys)}
        texts = max(candidates.values(), key=lambda t: sum(map(len, t)), default=[])
        rewrite, spec, message_history, call_cost = self.llm_instantiate(
            target_op_config, agent_llm, texts, message_history)
        return self.apply(operators, target_ops[0], rewrite, spec), message_history, call_cost
