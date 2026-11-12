import json
import os
from copy import deepcopy
from typing import Dict, List, Type

from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import ChangeModelInstantiateSchema
from .agent_utils import agent_completion, is_together_model, _extract_json_from_text

MODEL_STATS = {
    "cuad": {
        # Gemini sweep on the original pipeline, train split (40 contracts), 2026-11-10;
        # acc = avg_f1 of the official-match CUAD evaluator (the search reward); cost in USD.
        "gemini-3.1-flash-lite": {"acc": 0.5480314960629922, "cost": 0.196732},
        "gemini-3.5-flash-lite": {"acc": 0.5594622543950362, "cost": 0.281492},
        "gemini-3.7-flash": {"acc": 0.5969230769230769, "cost": 1.065943},
        "gemini-3.8-flash": {"acc": 0.5855175477565526, "cost": 1.131685},
        "gemini-3-flash-preview": {"acc": 0.5027322404371585, "cost": 1.217525},
        "gemini-3.6-flash": {"acc": 0.5818699572852397, "cost": 1.218827},
        "gemini-4-argon": {"acc": 0.6312659303313509, "cost": 1.983882},
        "gemini-3.5-flash": {"acc": 0.6069605568445475, "cost": 2.844681},
        "gemini-3.1-pro-preview": {"acc": 0.6239151398264223, "cost": 4.581762},
    },
    "medec": {
        "gpt-5": {"acc": 0.7724715715510082, "cost": 1.008255},
        "gpt-5-mini": {"acc": 0.6665255825243509, "cost": 0.120811},
        "gpt-5-nano": {"acc": 0.6419064045434841, "cost": 0.062694},
        "gpt-4.1": {"acc": 0.6582786810593204, "cost": 0.099804},
        "gpt-4.1-mini": {"acc": 0.5130606013669332, "cost": 0.019399},
        "gpt-4.1-nano": {"acc": 0.5011682624970316, "cost": 0.004886},
        "gpt-4o": {"acc": 0.6638914672336493, "cost": 0.135778},
        "gpt-4o-mini": {"acc": 0.6038406827744512, "cost": 0.008168},
        "gemini-2.5-pro": {"acc": 0.7482653668131851, "cost": 0.689124},
        "gemini-2.5-flash-lite": {"acc": 0.3601489578058821, "cost": 0.005106},
    },
    "biodex": {
        "gpt-5": {"acc": 0.35962962962962963, "cost": 10.06},
        "gpt-5-mini": {"acc": 0.3256666666666666, "cost": 2.24},
        "gpt-5-nano": {"acc": 0.2913333333333333, "cost": 0.45},
        "gpt-4.1": {"acc": 0.273, "cost": 15.41},
        "gpt-4.1-mini": {"acc": 0.24600000000000002, "cost": 3.08},
        "gpt-4.1-nano": {"acc": 0.15433333333333335, "cost": 0.77},
        "gpt-4o": {"acc": 0.206, "cost": 17.57},
        "gpt-4o-mini": {"acc": 0.27033333333333326, "cost": 1.05},
        "gemini-2.5-flash-lite": {"acc": 0.19866666666666666, "cost": 0.71},
    },
    "sustainability": {
        # Gemini sweep on the original pipeline, train split (40 docs), 2026-11-08;
        # acc = combined_score (the search reward); cost in USD for the 40 documents.
        "gemini-3.1-flash-lite": {"acc": 0.7222222222222222, "cost": 1.621415},
        "gemini-3.5-flash-lite": {"acc": 0.7297297297297297, "cost": 2.027031},
        "gemini-3-flash-preview": {"acc": 0.7567567567567568, "cost": 3.377226},
        "gemini-3.8-flash": {"acc": 0.7571428571428571, "cost": 4.721345},
        "gemini-3.6-flash": {"acc": 0.7714285714285715, "cost": 5.177072},
        "gemini-3.7-flash": {"acc": 0.7361111111111112, "cost": 5.482474},
        "gemini-3.5-flash": {"acc": 0.7714285714285715, "cost": 9.51586},
        "gemini-4-argon": {"acc": 0.8, "cost": 12.311158},
        "gemini-3.1-pro-preview": {"acc": 0.7162162162162162, "cost": 20.52302},
    },
    "blackvault": {
        "gpt-5": {"acc": 2.58974358974359, "cost": 2.607966},
        "gpt-5-mini": {"acc": 7.682926829268292, "cost": 0.412403},
        "gpt-5-nano": {"acc": 4.703703703703703, "cost": 0.111214},
        "gpt-4.1": {"acc": 8.764705882352942, "cost": 2.449074},
        "gpt-4.1-nano": {"acc": 0.967741935483871, "cost": 0.125311},
        "gpt-4o": {"acc": 9.875, "cost": 2.934132},
        "gpt-4o-mini": {"acc": 6.5625, "cost": 0.174887},
        "gemini-2.5-pro": {"acc": 16.863636363636363, "cost": 3.129470},
    },
    "game_reviews": {
        "gpt-5": {"acc": 0.7240131578947369, "cost": 6.560770},
        "gpt-5-mini": {"acc": 0.7276842105263159, "cost": 1.370671},
        "gpt-5-nano": {"acc": 0.7291378138974889, "cost": 0.464833},
        "gpt-4.1": {"acc": 0.7164704162366701, "cost": 8.729372},
        "gpt-4.1-mini": {"acc": 0.5850613521325596, "cost": 1.735382},
        "gpt-4.1-nano": {"acc": 0.5001233000745384, "cost": 0.445612},
        "gpt-4o": {"acc": 0.6224749823619792, "cost": 10.044458},
        "gpt-4o-mini": {"acc": 0.4558171816143953, "cost": 0.602058},
        "gemini-2.5-pro": {"acc": 0.72196783625731, "cost": 10.528458},
        "gemini-2.5-flash-lite": {"acc": 0.6054982297865206, "cost": 0.497641},
    },
    
}

FRONTIER_MODELS = {
    "cuad": {
        # Pareto frontier of the 9-model sweep under avg_f1.
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite",
        "gemini-3.7-flash",
        "gemini-4-argon",
    },
    "medec": {
        "gpt-4.1-nano",
        "gpt-4o-mini",
        "gpt-5-nano",
        "gpt-4.1",
        "gpt-5-mini",
        "gemini-2.5-pro",
        "gpt-5",
    },
    "biodex": {
        "gpt-5-nano",
        "gpt-5",
        "gpt-5-mini",
    },
    "sustainability": {
        # Pareto frontier of the 9-model sweep under combined_score.
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite",
        "gemini-3-flash-preview",
        "gemini-3.8-flash",
        "gemini-3.6-flash",
        "gemini-4-argon",
    },
    "blackvault": {
        "gpt-5-nano",
        "gpt-4o-mini",
        "gpt-5-mini",
        "gpt-4.1",
        "gpt-4o",
        "gemini-2.5-pro",
    },
    "game_reviews": {
        "gpt-4.1-nano",
        "gpt-5-nano"
    }
}


MODEL_COSTS = {
    "cuad": {
        # Gemini sweep on the original pipeline, train split (40 contracts), 2026-11-10.
        "gemini-3.1-flash-lite": 0.196732,
        "gemini-3.5-flash-lite": 0.281492,
        "gemini-3.7-flash": 1.065943,
        "gemini-3.8-flash": 1.131685,
        "gemini-3-flash-preview": 1.217525,
        "gemini-3.6-flash": 1.218827,
        "gemini-4-argon": 1.983882,
        "gemini-3.5-flash": 2.844681,
        "gemini-3.1-pro-preview": 4.581762,
    },
    "medec": {
        "gpt-5": 1.008255,
        "gpt-5-mini": 0.120811,
        "gpt-5-nano": 0.062694,
        "gpt-4.1": 0.099804,
        "gpt-4.1-mini": 0.019399,
        "gpt-4.1-nano": 0.004886,
        "gpt-4o": 0.135778,
        "gpt-4o-mini": 0.008168,
        "gemini-2.5-pro": 0.689124,
        "gemini-2.5-flash-lite": 0.005106,
    },
    "biodex": {
        "gpt-5": 10.06,
        "gpt-5-mini": 2.24,
        "gpt-5-nano": 0.45,
        "gpt-4.1": 15.41,
        "gpt-4.1-mini": 3.08,
        "gpt-4.1-nano": 0.77,
        "gpt-4o": 17.57,
        "gpt-4o-mini": 1.05,
        "gemini-2.5-flash-lite": 0.71,
    },
    "sustainability": {
        # Gemini sweep on the original pipeline, train split (40 docs), 2026-11-08.
        "gemini-3.1-flash-lite": 1.621415,
        "gemini-3.5-flash-lite": 2.027031,
        "gemini-3-flash-preview": 3.377226,
        "gemini-3.8-flash": 4.721345,
        "gemini-3.6-flash": 5.177072,
        "gemini-3.7-flash": 5.482474,
        "gemini-3.5-flash": 9.51586,
        "gemini-4-argon": 12.311158,
        "gemini-3.1-pro-preview": 20.52302,
    },
    "blackvault": {
        "gpt-5": 2.607966,
        "gpt-5-mini": 0.412403,
        "gpt-5-nano": 0.111214,
        "gpt-4.1": 2.449074,
        "gpt-4.1-nano": 0.125311,
        "gpt-4o": 2.934132,
        "gpt-4o-mini": 0.174887,
        "gemini-2.5-pro": 3.129470,
    },
    "game_reviews": {
        "gpt-5": 6.560770,
        "gpt-5-mini": 1.370671,
        "gpt-5-nano": 0.464833,
        "gpt-4.1": 8.729372,
        "gpt-4.1-mini": 1.735382,
        "gpt-4.1-nano": 0.445612,
        "gpt-4o": 10.044458,
        "gpt-4o-mini": 0.602058,
        "gemini-2.5-pro": 10.528458,
        "gemini-2.5-flash-lite": 0.497641,
    }
}

first_layer_yaml_paths = {
    "cuad": [
        "gemini_3.1_flash_lite_config.yaml",
        "gemini_3.5_flash_lite_config.yaml",
        "gemini_3.7_flash_config.yaml",
        "gemini_4_argon_config.yaml",
    ],
    "medec": [
        "gpt_4.1_nano_config.yaml",
        "gpt_4o_mini_config.yaml",
        "gpt_5_nano_config.yaml",
        "gpt_4.1_config.yaml",
        "gpt_5_mini_config.yaml",
        "gemini_2.5_pro_config.yaml",
        "gpt_5_config.yaml",
    ],
    "biodex": [
        "gpt_5_nano_config.yaml",
        "gpt_5_config.yaml",
        "gpt_5_mini_config.yaml",
    ],
    "sustainability": [
        "gemini_3.1_flash_lite_config.yaml",
        "gemini_3.5_flash_lite_config.yaml",
        "gemini_3_flash_preview_config.yaml",
        "gemini_3.8_flash_config.yaml",
        "gemini_3.6_flash_config.yaml",
        "gemini_4_argon_config.yaml",
    ],
    "blackvault": [
        "gpt_5_nano_config.yaml",
        "gpt_4o_mini_config.yaml",
        "gpt_5_mini_config.yaml",
        "gpt_4.1_config.yaml",
        "gpt_4o_config.yaml",
        "gemini_2.5_pro_config.yaml",
    ],
    "game_reviews": [
        "gpt_4.1_nano_config.yaml",
        "gpt_5_nano_config.yaml",
    ]
}

first_layer_yaml_paths_pruned = {
    "cuad": [
        "gemini_3.1_flash_lite_config.yaml",
        "gemini_3.5_flash_lite_config.yaml",
        "gemini_3.7_flash_config.yaml",
        "gemini_4_argon_config.yaml",
    ],
    "medec": [
        "gpt_4o_mini_config.yaml",
        "gpt_5_nano_config.yaml",
        "gpt_4.1_config.yaml",
        "gpt_5_mini_config.yaml",
        "gemini_2.5_pro_config.yaml",
        "gpt_5_config.yaml",
    ],
    "biodex": [
        "gpt_5_nano_config.yaml",
        "gpt_5_mini_config.yaml",
        "gpt_5_config.yaml",
    ],
    "sustainability": [
        "gemini_3.1_flash_lite_config.yaml",
        "gemini_3.5_flash_lite_config.yaml",
        "gemini_3_flash_preview_config.yaml",
        "gemini_3.8_flash_config.yaml",
        "gemini_3.6_flash_config.yaml",
        "gemini_4_argon_config.yaml",
    ],
    "blackvault": [
        "gpt_4o_mini_config.yaml",
        "gpt_5_mini_config.yaml",
        "gpt_4.1_config.yaml",
        "gpt_4o_config.yaml",
        "gemini_2.5_pro_config.yaml",
    ],
    "game_reviews": [
        "gpt_4.1_nano_config.yaml",
        "gpt_5_nano_config.yaml",
    ]
}


def get_cheaper_models(current_model: str, dataset: str) -> List[str]:
    """Get list of models that are cheaper than the current model for a given dataset."""
    if dataset not in MODEL_COSTS:
        return []
    
    current_cost = MODEL_COSTS[dataset].get(current_model)
    if current_cost is None:
        return []
    
    cheaper_models = []
    for model, cost in MODEL_COSTS[dataset].items():
        if cost < current_cost:
            cheaper_models.append(model)
    
    # Sort by cost (cheapest first)
    cheaper_models.sort(key=lambda x: MODEL_COSTS[dataset][x])
    return cheaper_models

from .base import (
    AVAILABLE_MODELS,
    available_models_info,
    MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS,
    Directive,
    DirectiveTestCase,
)

# The tables above are historical measurements on GPT / Gemini 2.5 models, kept for reference.
# Only models in AVAILABLE_MODELS can be proposed or used by the search, so drop the rest here.
MODEL_STATS = {ds: {m: v for m, v in t.items() if m in AVAILABLE_MODELS} for ds, t in MODEL_STATS.items()}
MODEL_COSTS = {ds: {m: v for m, v in t.items() if m in AVAILABLE_MODELS} for ds, t in MODEL_COSTS.items()}
FRONTIER_MODELS = {ds: {m for m in t if m in AVAILABLE_MODELS} for ds, t in FRONTIER_MODELS.items()}


def create_model_specific_directives(current_model: str):
    """Create model-specific directives for the current model."""
    directive = ChangeModelCostDirective(target_model=current_model)
    directive.name = f"change to {current_model}"
    directive.nl_description = f"Rewrites an operator to use the {current_model} model to optimize expenses while maintaining adequate performance."
    directive.allowed_model_list = [current_model]  # Only allow this specific model
    
    return directive


class ChangeModelCostDirective(Directive):
    name: str = Field(
        default="change model cost", description="The name of the directive"
    )
    formal_description: str = Field(
        default="Op => Op* (same operation with a more cost-effective model choice)"
    )
    nl_description: str = Field(
        default="Rewrites an operator to use a more cost-effective LLM model to optimize expenses. Prioritizes cost savings while maintaining adequate performance, typically suggesting cheaper models like gemini-3.1-flash-lite or gemini-3.5-flash-lite for simpler tasks."
    )
    when_to_use: str = Field(
        default="When cost optimization is the primary concern and the task can be performed adequately by a less expensive model. Suitable for simple extraction, basic classification, or high-volume processing where budget constraints are important."
    )
    instantiate_schema_type: Type[BaseModel] = ChangeModelInstantiateSchema

    example: str = Field(
        default=(
            "Original Op (MapOpConfig):\n"
            "- name: extract_names\n"
            "  type: map\n"
            "  prompt: |\n"
            "    Extract person names from the following text.\n"
            "    Text: {{ input.text }}\n"
            "  output:\n"
            "    schema:\n"
            '      names: "list[str]"\n'
            "  model: gemini-3.1-pro-preview\n"
            "\n"
            "Example InstantiateSchema:\n"
            "{\n"
            '  "model": "gemini-3.5-flash-lite"\n'
            "}"
        ),
    )

    allowed_model_list: List[str] = Field(
        default=AVAILABLE_MODELS,
        description="The allowed list of models to choose from",
    )
    
    target_model: str = Field(
        default="",
        description="The specific target model for this directive instance",
    )

    model_info: str = Field(default_factory=available_models_info)

    test_cases: List[DirectiveTestCase] = Field(
        default_factory=lambda: [
            DirectiveTestCase(
                name="cost_optimization_simple_extraction",
                description="Should suggest cheaper model for simple extraction task",
                input_config={
                    "name": "extract_names",
                    "type": "map",
                    "prompt": "Extract person names from: {{ input.text }}",
                    "output": {"schema": {"names": "list[str]"}},
                    "model": "gemini-3.1-pro-preview",
                },
                target_ops=["extract_names"],
                expected_behavior="Should recommend gemini-3.1-flash-lite or gemini-3.5-flash-lite for cost savings on simple task",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="high_volume_processing_cost_optimization",
                description="Should optimize model for high-volume document processing to reduce costs",
                input_config={
                    "name": "batch_document_summary",
                    "type": "map",
                    "prompt": "Summarize this document in 2-3 sentences: {{ input.document }}",
                    "output": {"schema": {"summary": "str"}},
                    "model": "gemini-3.1-pro-preview",
                },
                target_ops=["batch_document_summary"],
                expected_behavior="Should recommend cheaper model like gemini-3.1-flash-lite or gemini-3.5-flash-lite for high-volume simple summarization",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="basic_classification_cost_savings",
                description="Should recommend cost-effective model for basic classification",
                input_config={
                    "name": "classify_sentiment",
                    "type": "map",
                    "prompt": "Classify the sentiment of this text as positive, negative, or neutral: {{ input.text }}",
                    "output": {"schema": {"sentiment": "string"}},
                    "model": "gemini-3.1-pro-preview",
                },
                target_ops=["classify_sentiment"],
                expected_behavior="Should recommend cheaper model for basic sentiment classification to reduce costs",
                should_pass=True,
            ),
        ]
    )

    def __eq__(self, other):
        return isinstance(other, ChangeModelCostDirective) and self.target_model == other.target_model

    def __hash__(self):
        return hash(f"ChangeModelCostDirective_{self.target_model}")

    def to_string_for_instantiate(self, original_op: Dict, dataset: str) -> str:
        """
        Generate a prompt for an agent to instantiate this directive for cost optimization.

        Args:
            original_op (str): The YAML or string representation of the original operation.

        Returns:
            str: The agent prompt for instantiating the directive.
        """
        return (
            f"You are an expert at choosing the most cost-effective model for a given task while maintaining adequate performance.\n\n"
            f"Original Operation:\n"
            f"{str(original_op)}\n\n"
            f"Directive: {self.name}\n"
            f"Your task is to instantiate this directive by suggesting the cheapest model that meets the requirements for executing the original operation.\n\n"
            f"COST OPTIMIZATION STRATEGY:\n"
            f"• Choose the cheapest model that meets the task requirements\n"
            f"• For tasks requiring ultra-long context (1M+ context window), use gemini-3.8-flash series or gemini models\n"
            f"• For tasks that can work with document samples or fit within 272k context window, use gemini-3.1-flash-lite\n"
            f"• For simple tasks (extraction, basic classification, straightforward summarization), use the cheapest available model like gemini-3.1-flash-lite or gemini-3.5-flash-lite\n"
            f"• For high-volume processing where cost accumulates quickly, prioritize the most economical option\n"
            f"• Only use expensive models if the task absolutely requires capabilities not available in cheaper alternatives\n"
            f"• Consider document length and context requirements when selecting models\n\n"
            f"You have a list of allowed models to choose from: {str(self.allowed_model_list)}.\n\n"
            f"Consider the information about the allowed models: \n {self.model_info}\n"
            f"You have a list of model statistics on the task with the original query pipeline: \n {str(MODEL_STATS.get(dataset, {}))}\n"
            f"Your response should include the cheapest model choice that meets the operation requirements."
            f"Ensure that your chosen model is in the list of allowed models."
            f"Example:\n"
            f"{self.example}\n\n"
            f"Please output only the ChangeModelInstantiateSchema as JSON."
        )

    def llm_instantiate(
        self,
        global_default_model: str,
        original_op: Dict,
        agent_llm: str,
        dataset: str,
        message_history: list = [],
    ):
        """
        Use LLM to instantiate this directive for cost optimization.

        Args:
            original_op (Dict): The original operation.
            agent_llm (str): The LLM model to use.
            message_history (List, optional): Conversation history for context.

        Returns:
            ChangeModelInstantiateSchema: The structured output from the LLM.
        """

        # If target_model is specified, use it directly without LLM call
        if self.target_model:
            schema = ChangeModelInstantiateSchema(model=self.target_model)
            return schema, message_history, 0.0
        
        # Otherwise, use LLM to choose the model
        message_history.extend(
            [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant for document processing pipelines focused on cost optimization.",
                },
                {
                    "role": "user",
                    "content": self.to_string_for_instantiate(original_op, dataset),
                },
            ]
        )
        for _ in range(MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS):
            resp, call_cost = agent_completion(
                model=agent_llm,
                messages=message_history,
                response_format=ChangeModelInstantiateSchema,
            )
            try:
                content = resp.choices[0].message.content
                if is_together_model(agent_llm):
                    parsed_res = _extract_json_from_text(content)
                else:
                    parsed_res = json.loads(content)
                schema = ChangeModelInstantiateSchema(**parsed_res)
                orig_model = global_default_model
                if "model" in original_op:
                    orig_model = original_op.get("model")
                # Validate the model is in the allowed model list
                ChangeModelInstantiateSchema.validate_diff_model_in_list(
                    orig_model=orig_model,
                    model=schema.model,
                    list_of_model=self.allowed_model_list,
                )
                message_history.append(
                    {"role": "assistant", "content": content}
                )
                return schema, message_history, call_cost
            except Exception as err:
                error_message = f"Validation error: {err}\nPlease try again."
                message_history.append({"role": "user", "content": error_message})

        raise Exception(
            f"Failed to instantiate directive after {MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS} attempts."
        )

    def apply(
        self,
        global_default_model: str,
        ops_list: List[Dict],
        target_op: str,
        rewrite: ChangeModelInstantiateSchema,
    ) -> List[Dict]:
        """
        Apply the directive to the pipeline config by changing the model of the target operator.
        """
        # Create a copy of the pipeline config
        new_ops_list = deepcopy(ops_list)

        # Find position of the target op to modify
        pos_to_replace = [
            i for i, op in enumerate(ops_list) if op["name"] == target_op
        ][0]

        # Add change model configuration to the target operator
        target_operator = new_ops_list[pos_to_replace]
        target_operator["model"] = rewrite.model

        return new_ops_list

    def instantiate(
        self,
        operators: List[Dict],
        target_ops: List[str],
        agent_llm: str,
        message_history: list = [],
        global_default_model: str = None,
        dataset: str = None,
        **kwargs,
    ):
        """
        Instantiate the directive for a list of operators.
        """
        new_ops_list = deepcopy(operators)
        inst_error = 0
        for target_op in target_ops:
            target_op_config = [op for op in operators if op["name"] == target_op][0]
            # Instantiate the directive
            try:
                rewrite, message_history, call_cost = self.llm_instantiate(
                    global_default_model,
                    target_op_config,
                    agent_llm,
                    dataset,
                    message_history,
                )
                print(rewrite)
            except Exception:
                inst_error += 1
            new_ops_list = self.apply(
                global_default_model, new_ops_list, target_op, rewrite
            )

        if inst_error == len(target_ops):
            print("CHANGE MODEL COST ERROR")
            return None, message_history
        return new_ops_list, message_history, call_cost
