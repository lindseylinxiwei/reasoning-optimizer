import json
import re
from copy import deepcopy
from typing import Dict, List, Type

from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import (
    JEV_CONTEXT_TOKENS,
    JEV_MAX_PROMPT_TOKENS,
    JevClassifyInstantiateSchema,
)

from .agent_utils import _extract_json_from_text, agent_completion, is_together_model
from .base import MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS, Directive, DirectiveTestCase

JEV_MODEL = "jev"
JEV_PRICE_PER_MILLION_INPUT = 0.042
# Operator options Jev rejects or ignores; removed from the target operator.
_UNSUPPORTED_KEYS = ("gleaning", "tools", "scratchpad", "batch_prompt", "calibrate", "pdf_url_key",
                     "litellm_completion_kwargs", "num_retries_on_validate_failure")

# Separate rewrites that shorten what an operation reads. Shown to the agent so it can bring a
# long input under Jev's limit; jev_classify itself never changes the input.
SHORTENING_REWRITES = (
    "take_head_tail (keep the start and/or end of the document)",
    "doc_chunking_topk or jev_retrieval (keep only the passages relevant to the task)",
    "doc_summarization or deterministic_doc_compression (condense the document)",
    "doc_chunking, on a map (split the document, run the map on each chunk, combine the chunk answers "
    "with a reduce; the per-chunk map can then be switched to Jev)",
)


def is_jev_op(op: Dict) -> bool:
    model = str(op.get("model") or "")
    return model == JEV_MODEL or model.startswith("jev-")


def jev_compatible(op: Dict) -> bool:
    """A filter or map whose outputs are all boolean or enum."""
    if op.get("type") not in ("filter", "map") or not op.get("prompt") or is_jev_op(op):
        return False
    schema = (op.get("output") or {}).get("schema") or {}
    return bool(schema) and all(
        str(k).strip() in ("bool", "boolean") or re.fullmatch(r"enum\[.*\]", str(k).strip())
        for k in schema.values())


class JevClassifyDirective(Directive):
    name: str = Field(default="jev_classify", description="The name of the directive")
    formal_description: str = Field(
        default="Filter/Map (LLM) => Filter/Map (Jev)",
        description="Runs a filter or map whose outputs are all boolean/enum with the Jev classifier instead of an LLM",
    )
    nl_description: str = Field(
        default="Switches a classification operation (a filter, or a map whose output fields are all boolean or "
        "enum) from an LLM to Jev, a classifier that answers each output field directly, with no text "
        f"generation. Jev charges only for input (${JEV_PRICE_PER_MILLION_INPUT}/M tokens). The prompt is "
        "rewritten into the form Jev reads (one section per output field); nothing else in the pipeline changes. "
        f"Jev reads at most {JEV_CONTEXT_TOKENS // 1000}K tokens per call and does not truncate: on a longer "
        "input the operation fails."
    )
    when_to_use: str = Field(
        default="For cost: a filter or a map whose outputs are all boolean or enum labels (document type, topic, "
        f"sector, yes/no checks), and whose input fits in {JEV_CONTEXT_TOKENS // 1000}K tokens. If the input can "
        "be longer (e.g. full long documents), first shorten it with a separate rewrite: "
        + "; ".join(SHORTENING_REWRITES) + ". Not usable on operations with string, number or list outputs."
    )

    instantiate_schema_type: Type[BaseModel] = Field(default=JevClassifyInstantiateSchema)

    example: str = Field(
        default="""
        Original Filter (after a take_head_tail rewrite keeps the first 2000 words of tot_text_raw):
        - name: filter_sustainability_reports
          type: filter
          model: gemini-3.5-flash-lite
          litellm_completion_kwargs: {temperature: 0.0}
          output: {schema: {is_sustainability_report: boolean}}
          prompt: |
            Analyze the following document and determine if it is a sustainability report.
            Document text: {{ input.tot_text_raw }}
            Return true if this document is primarily a sustainability report, false otherwise.

        JevClassifyInstantiateSchema:
        {
          "prompt": "Decide whether a company document is a sustainability report.\\n\\nOutput field is_sustainability_report: true if the document is primarily a sustainability, ESG or corporate responsibility report, or contains substantial content on environmental performance, social responsibility, governance practices, ESG metrics or climate initiatives; false for other documents such as annual financial reports without such content, product brochures or press releases.\\n\\nDocument text:\\n{{ input.tot_text_raw }}",
          "rationale": "Document type is a classification a classifier can make, and the earlier head cut keeps the input near 2,800 tokens, well under Jev's limit."
        }

        Resulting operation:
        - name: filter_sustainability_reports
          type: filter
          model: jev
          output: {schema: {is_sustainability_report: boolean}}
          prompt: (the prompt above)
        """
    )

    test_cases: List[DirectiveTestCase] = Field(
        default_factory=lambda: [
            DirectiveTestCase(
                name="sector_classification",
                description="A map that assigns one sector label per document",
                input_config={
                    "name": "determine_sector",
                    "type": "map",
                    "prompt": "Determine the primary sector of the company: {{ input.text }}",
                    "output": {"schema": {"sector": "enum[energy, finance, other]"}},
                    "model": "gemini-3.5-flash-lite",
                },
                target_ops=["determine_sector"],
                expected_behavior="Should write a prompt with an 'Output field sector:' section describing energy, "
                "finance and other",
                should_pass=True,
            ),
        ]
    )

    def __eq__(self, other):
        return isinstance(other, JevClassifyDirective)

    def __hash__(self):
        return hash("JevClassifyDirective")

    def to_string_for_instantiate(self, original_op: Dict) -> str:
        schema = (original_op.get("output") or {}).get("schema") or {}
        return (
            "You are an expert at optimizing document processing pipelines for cost and accuracy.\n\n"
            f"Target operation:\n{json.dumps(original_op, indent=2)}\n\n"
            f"Directive: {self.name}\n"
            "Switch this operation from an LLM to Jev, a document classifier. Jev is not a language model, so "
            "the prompt has to be adapted. Read these facts first:\n\n"
            "How Jev works\n"
            "- Jev receives the rendered prompt as one block of text (its 'state') plus one question per output "
            "field, built automatically from the output schema:\n"
            "    boolean field X: \"Following the task instructions in the state, should the output field 'X' be true?\" "
            "(true when Jev's probability is at least 0.5)\n"
            "    enum field X: \"Following the task instructions in the state, select the value of output field 'X'.\" "
            "(Jev picks one of the labels exactly as written in the schema; it sees no description of the labels "
            "other than what the prompt says)\n"
            "- It generates no text: it cannot explain, reason step by step, write JSON or follow output-format "
            "instructions. It only produces the boolean or label.\n"
            f"- Cost: ${JEV_PRICE_PER_MILLION_INPUT} per million input tokens, nothing for output.\n"
            "- Not supported (removed automatically): gleaning, tools, batching, calibration, temperature and "
            "other LLM sampling options. The output schema stays exactly as it is.\n\n"
            "Context length\n"
            f"- Jev reads at most {JEV_CONTEXT_TOKENS // 1000}K tokens per call, prompt and input together, and does "
            "not truncate: on any document whose rendered prompt is longer, the operation fails.\n"
            "- Check how long the text this operation reads can be: the document statistics (if given below) "
            "describe the raw documents; earlier operations in the pipeline may already shorten the field it "
            "reads (a head/tail cut, top-k chunks, Jev retrieval, a summary, chunks from a split).\n"
            f"- If the input can exceed the limit, the context has to be shortened by a separate rewrite, applied "
            f"to this operation before switching it to Jev:\n"
            + "".join(f"    {r}\n" for r in SHORTENING_REWRITES)
            + "  This rewrite does not shorten anything itself. Say in the rationale why the input fits.\n\n"
            "Prompt\n"
            "Rewrite the prompt into the state Jev reads:\n"
            "- A short statement of the task.\n"
            + "".join(
                f"- A section starting exactly with 'Output field {f}:' that states the criterion"
                + (" for true and for false.\n" if str(k).strip() in ("bool", "boolean") else
                   ", then every label with one line on when it applies (labels written exactly as in the "
                   "schema).\n")
                for f, k in schema.items())
            + "- The input, read exactly as the original prompt reads it (same input fields, no new ones).\n"
            "- Keep every instruction from the original that defines the decision (definitions, criteria, edge "
            "cases); drop instructions about reasoning, explanation or answer format.\n"
            f"- Keep the prompt without the input under {JEV_MAX_PROMPT_TOKENS} tokens.\n\n"
            f"Example:\n{self.example}\n\n"
            "Output only the JevClassifyInstantiateSchema object: prompt, rationale."
        )

    def llm_instantiate(self, original_op: Dict, agent_llm: str, message_history: list = []):
        message_history.extend([{"role": "user", "content": self.to_string_for_instantiate(original_op)}])
        for _ in range(MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS):
            resp, call_cost = agent_completion(
                model=agent_llm, messages=message_history, response_format=JevClassifyInstantiateSchema)
            try:
                content = resp.choices[0].message.content
                parsed = _extract_json_from_text(content) if is_together_model(agent_llm) else json.loads(content)
                schema = JevClassifyInstantiateSchema(**parsed)
                schema.check_against(original_op)
                message_history.append({"role": "assistant", "content": content})
                return schema, message_history, call_cost
            except Exception as err:
                message_history.append({"role": "user", "content": f"Validation error: {err}\nPlease try again."})
        raise Exception(f"Failed to instantiate directive after {MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS} attempts.")

    def apply(self, ops_list: List[Dict], target_op: str, rewrite: JevClassifyInstantiateSchema) -> List[Dict]:
        new_ops = deepcopy(ops_list)
        op = [o for o in new_ops if o["name"] == target_op][0]
        if not jev_compatible(op):
            raise ValueError(f"{target_op} is not a filter/map with only boolean/enum outputs")
        rewrite.check_against(op)
        for k in _UNSUPPORTED_KEYS:
            op.pop(k, None)
        if isinstance(op.get("output"), dict):
            op["output"].pop("mode", None)
        op["model"] = JEV_MODEL
        op["prompt"] = rewrite.prompt
        return new_ops

    def instantiate(self, operators: List[Dict], target_ops: List[str], agent_llm: str,
                    message_history: list = [], **kwargs):
        assert len(target_ops) == 1, "jev_classify takes exactly one target operation"
        target = [op for op in operators if op["name"] == target_ops[0]][0]
        if not jev_compatible(target):
            raise ValueError(f"{target_ops[0]} is not a filter/map with only boolean/enum outputs")
        rewrite, message_history, call_cost = self.llm_instantiate(target, agent_llm, message_history)
        return self.apply(operators, target_ops[0], rewrite), message_history, call_cost
