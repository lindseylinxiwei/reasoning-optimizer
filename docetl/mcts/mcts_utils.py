"""
Utility functions for MCTS implementation.

This module contains helper functions and utilities used by the MCTS algorithm
but not core to the MCTS structure itself.
"""
import json
import math
import os
import re
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

import litellm
import yaml
from pydantic import BaseModel

from docetl.reasoning_optimizer.directives import (
    ALL_DIRECTIVES,
    ALL_COST_DIRECTIVES,
    DIRECTIVE_GROUPS,
    MULTI_INSTANCE_DIRECTIVES,
    Directive,
    get_all_directive_strings,
    get_all_cost_directive_strings,
)
from docetl.reasoning_optimizer.directives.base import AVAILABLE_MODELS
from docetl.reasoning_optimizer.directives.change_model_cost import MODEL_STATS
from docetl.reasoning_optimizer.load_data import load_input_doc
from docetl.mcts.label_guard import label_fields, strip_labels
from docetl.reasoning_optimizer.op_descriptions import *

# Maximum number of tokens we will allow in the prompt we send to the model.
# The Azure GPT-5 family allows 272,000 tokens.
MAX_CONTEXT_TOKENS = 270_000


class ExpandResponseFormat(BaseModel):
    directive: str
    operators: List[str]
    reason: str = ""


def format_action_options(action_options, choice_counts=None, prior_counts=None) -> str:
    """One line per (operator, rewrite). With choice_counts (rewrite family -> times the agent chose it
    in this search), rewrites chosen least come first and each line says how often it was chosen.
    prior_counts (family -> runs in an earlier sweep) count toward the order and are named."""
    if choice_counts is None:
        return "".join(f"Operator: {op}, Rewrite directive: {name}\n" for op, name in action_options)
    prior_counts = prior_counts or {}
    fam = {name: rewrite_family(name) for _, name in action_options}
    total = {name: choice_counts.get(f, 0) + prior_counts.get(f, 0) for name, f in fam.items()}
    lines = []
    for op, name in sorted(action_options, key=lambda o: (total[o[1]], o[1], o[0])):
        n, p = choice_counts.get(fam[name], 0), prior_counts.get(fam[name], 0)
        here = "not tried yet in this search" if n == 0 else f"chosen {n} time{'s' if n != 1 else ''} in this search"
        note = f"{p} configurations tested in the earlier data-view sweep; {here}" if p else here
        lines.append(f"Operator: {op}, Rewrite directive: {name} ({note})\n")
    return "".join(lines)


EXPLORATION_NOTE = (
    "\n\nExploration turn: so that the search learns about rewrites it has rarely tried, the valid choices "
    "above are limited to the least-chosen rewrites that apply to this plan. Pick the one most likely to help "
    "this plan. If none of them can work on this plan, answer with directive \"none\", operators [], and a "
    "one-sentence reason in the reason field; you will then get the full list.\n")


def rewrite_family(name: str) -> str:
    """Model switches count as one rewrite ('change model'); other rewrites by name."""
    return "change model" if name.startswith("change to ") else name


LLM_OP_TYPES = {"map", "filter", "reduce", "parallel_map", "resolve", "extract", "equijoin"}
REDUCE_REWRITES = {"reduce_gleaning", "reduce_chaining", "swap_with_code", "hierarchical_reduce"}


def fits_plan_structure(ops: List[Dict], op_name: str, name: str) -> bool:
    """False when a rewrite cannot apply to this plan's shape, so it is not offered (and does not
    sit forever among the least-chosen rewrites on exploration turns)."""
    names = [o.get("name") for o in ops]
    if op_name not in names:
        return True
    i = names.index(op_name)
    kind = ops[i].get("type")
    if name in REDUCE_REWRITES:
        return kind == "reduce"
    if name == "cascade_filtering":
        return kind == "filter"
    if name == "chunk_header_summary":
        kinds = {o.get("type") for o in ops}
        return "split" in kinds and "gather" in kinds
    if name == "operator_fusion":
        if kind not in LLM_OP_TYPES:
            return False
        return any(0 <= j < len(ops) and ops[j].get("type") in LLM_OP_TYPES for j in (i - 1, i + 1))
    if name == "map_reduce_fusion":
        return kind == "map" and any(o.get("type") == "reduce" for o in ops[i + 1:])
    return True


class ShorteningChoice(BaseModel):
    directive: str
    reason: str


def count_tokens(messages):
    """Count estimated tokens in messages list."""
    # messages should be a list of dicts, each with a "content" key
    total_chars = sum(
        len(m.get("content", "")) for m in messages if isinstance(m, dict)
    )
    return max(1, total_chars // 4)


def trim_history(history: list, keep_system_first: bool = True) -> list:
    """Trim the conversation history in-place so its estimated token count
    (via ``count_tokens``) does not exceed ``MAX_CONTEXT_TOKENS``.

    We always keep the very first system message and the first user message so the 
    assistant retains the global instructions and the initial query context. After 
    that we drop the oldest messages until the budget is satisfied. Returns the 
    trimmed history list.
    """

    # Determine starting index to preserve the initial system message and first user message
    start_idx = 0
    if keep_system_first and history:
        if history[0].get("role") == "system":
            start_idx = 1
            # Find the first user message after the system message
            for i in range(1, len(history)):
                if history[i].get("role") == "user":
                    start_idx = i + 1
                    break
        elif history[0].get("role") == "user":
            # If first message is user, keep it and find the next user message
            start_idx = 1
            for i in range(1, len(history)):
                if history[i].get("role") == "user":
                    start_idx = i + 1
                    break

    # Drop oldest messages (just after the preserved block) until within limit
    while len(history) > start_idx + 1 and count_tokens(history) > MAX_CONTEXT_TOKENS:
        history.pop(start_idx)

    return history


def get_directive_group(directive_name: str) -> str:
    """
    Get the group name for a directive.
    
    Args:
        directive_name: Name of the directive
        
    Returns:
        Group name if found, None otherwise
    """
    for group_name, directives in DIRECTIVE_GROUPS.items():
        for directive in directives:
            if directive.name == directive_name:
                return group_name
    return None


def get_excluded_directives_for_operation(node, op_name: str) -> set:
    """Get compression directives to exclude for code_map and extract operations."""
    op_type = node.op_dict[op_name].get("type")
    compression_exclusions = set()
    if op_type in ["code_map", "extract"]:
        compression_exclusions = set(DIRECTIVE_GROUPS.get("compression", []))
    return compression_exclusions


def is_action_applicable(node, action: Directive) -> bool:
    """Check if an action is applicable to a node."""
    return True


def update_pipeline(orig_config, new_ops_list, target_ops):
    """
    Update the pipeline configuration with new operations.

    Args:
        orig_config (dict): The original pipeline configuration
        new_ops_list (list): List of new operations to add
        target_ops (list): List of target operation names to replace

    Returns:
        dict: Updated pipeline configuration
    """
    if new_ops_list is not None:
        op_names = [op.get("name") for op in new_ops_list if "name" in op]

    # Update the pipeline steps to use the new operation names
    if "pipeline" in orig_config and "steps" in orig_config["pipeline"]:
        for step in orig_config["pipeline"]["steps"]:
            if "operations" in step:
                new_ops = []
                for op in step["operations"]:
                    if op == target_ops[0]:
                        new_ops.extend(op_names)
                step["operations"] = new_ops

    return orig_config


def fix_models(parsed_yaml):
    """Give bare Gemini model names the LiteLLM "gemini/" provider prefix."""
    
    def get_model_type(model_name):
        """Determine model type from model name."""
        if model_name.startswith("gemini"):
            return "gemini"
        else:
            return "unknown"
    
    def traverse(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if (key == "model" or key == "default_model") and isinstance(value, str):
                    model_type = get_model_type(value)
                    
                    if model_type == "gemini" and not value.startswith("gemini/"):
                        obj[key] = f"gemini/{value}"
                else:
                    traverse(value)
        elif isinstance(obj, list):
            for item in obj:
                traverse(item)

    traverse(parsed_yaml)


def is_fully_explored(node, max_children_multiplier: float = 1.0) -> bool:
    """Check if a node has been fully explored based on visit count."""
    if node.parent is None:
        return bool(node.children)   # without a first layer the root is expanded directly
    if getattr(node, "no_actions", False) and node.children:
        return True     # nothing left to add here; descend into its children
    allowed_children = max(2, 1 + math.floor(math.sqrt(float(node.visits)) * max_children_multiplier))
    return len(node.children) >= allowed_children


def should_continue_search(iteration: int, max_iterations: int, start_time: float, 
                          max_time: Optional[float] = None) -> bool:
    """Determine if search should continue based on iteration count and time."""
    if iteration >= max_iterations:
        return False
    
    if max_time is not None:
        elapsed_time = time.time() - start_time
        if elapsed_time >= max_time:
            return False
    
    return True


def calculate_ucb1(node, parent_visits: int, exploration_constant: float = math.sqrt(2)) -> float:
    """Calculate UCB1 value for node selection."""
    if node.visits == 0:
        return float('inf')
    
    exploitation = node.value / node.visits
    exploration = exploration_constant * math.sqrt(math.log(parent_visits) / node.visits)
    return exploitation + exploration


def print_tree_visits_and_values(node=None, depth=0, file_handle=None):
    """Print tree structure with visit counts and values."""
    if node is None:
        return
        
    indent = "  " * depth
    node_info = f"{indent}Node ID: {node.get_id()}, Visits: {node.visits}, Value: {node.value:.4f}"
    
    if file_handle:
        file_handle.write(node_info + "\n")
    else:
        print(node_info)
        
    for child in node.children:
        print_tree_visits_and_values(child, depth + 1, file_handle)


def log_tree_to_file(root_node, iteration_num, output_dir="./outputs"):
    """Log the tree structure to a file."""
    log_file_path = os.path.join(output_dir, f"mcts_tree_iteration_{iteration_num}.txt")
    
    with open(log_file_path, "w") as f:
        f.write(f"MCTS Tree Structure - Iteration {iteration_num}\n")
        f.write("=" * 50 + "\n")
        print_tree_visits_and_values(root_node, file_handle=f)


def create_expansion_prompt_acc(node, action_options, input_query, available_actions, action_cost_changes, action_accuracy_changes, action_counts, sample_input, root_node, yaml_file_path, dataset=None, node_accuracies=None, choice_counts=None, prior_counts=None) -> tuple[str, str]:
    """Create expansion prompt for accuracy optimization."""
    
    ### DEBUG 
    print("memo: ")
    print(node.get_memo_for_llm(root_node, node_accuracies))

    availabel_actions_str = format_action_options(action_options, choice_counts, prior_counts)

    print("availabel_actions_str: ")
    print(availabel_actions_str)

    action_stats = []
    for action in available_actions:
        cost_change = action_cost_changes.get(action, 0)
        accuracy_change = action_accuracy_changes.get(action, 0)
        count = action_counts.get(action, 0)
        
        if count > 0:
            avg_cost_change = cost_change / count
            avg_accuracy_change = accuracy_change / count
            action_stats.append(
                f"- {action.name}: {int(count)} successful runs, avg change in cost: {avg_cost_change:+.2f}, avg change in accuracy: {avg_accuracy_change:+.4f}"
            )
        else:
            action_stats.append(
                f"- {action.name}: not run successfully yet"
            )

    action_stats_str = "\n".join(action_stats)

    print("action_stats_str: ")
    print(action_stats_str)

    input_schema = load_input_doc(yaml_file_path, exclude_fields=label_fields(dataset))

    user_message = f"""
    I have a set of operations used to process long documents, along with a list of possible rewrite directives aimed at improving the quality of the query result.
    Given a query pipeline made up of these operations, recommend one specific rewrite directive (specify by its name) that would improve accuracy and specify which operators (specify by their names) in the pipeline the directive should be applied to.
    Make sure that your chosen directive is in the provided list of rewrite directives.

    Pipeline:
    Pipelines in DocETL are the core structures that define the flow of data processing. A pipeline consists of five main components: \n
    - Default Model: The language model to use for the pipeline. Limit your choice of model to {AVAILABLE_MODELS} \n
    - System Prompts: A description of your dataset and the "persona" you'd like the LLM to adopt when analyzing your data. \n
    - Datasets: The input data sources for your pipeline. \n
    - Operators: The processing steps that transform your data. \n
    - Pipeline Specification: The sequence of steps and the output configuration. \n

    Operators:
    Operators form the building blocks of data processing pipelines. Below is the list of operators:
    {op_map.to_string()}\n
    {op_extract.to_string()}\n
    {op_parallel_map.to_string()}\n
    {op_filter.to_string()}\n
    {op_reduce.to_string()}\n
    {op_split.to_string()}\n
    {op_gather.to_string()}\n
    {op_unnest.to_string()}\n
    {op_sample.to_string()}\n
    {op_resolve.to_string()}\n

    Rewrite directives:
    {get_all_directive_strings()}\n

    Your valid choice of operation and rewrite directive combination. Only choose one of these:\n
    {availabel_actions_str}

    Action Performance History:
    Based on plans already run in this search, here's how each action has performed (average change vs the plan it was applied to):\n
    {action_stats_str}

    Note: These statistics come from applying actions to other plans in this search, not the current one. Use this as general guidance about action effectiveness, but consider that performance may vary for this plan's models and structure.

    Model Performance Reference:
    If you are considering a model change directive, here are model statistics on this specific dataset: \n {str(MODEL_STATS.get(dataset, {}))}
    These show the accuracy (acc) and cost for each model. Only reference this when evaluating model change options.

    Selection Strategy:
    Consider the current query pipeline and which directive can best improve the accuracy.

    {node.get_memo_for_llm(root_node, node_accuracies)}

    Make sure you read every rewrite directive carefully.
    Make sure you only choose from the valid choices above and avoid already used combinations or approaches too similar to what has already been tried in the current optimization path.

    Input document schema with token statistics: {input_schema} \n
    Input data sample: {json.dumps(strip_labels(sample_input, dataset), indent=2)[:5000]} \n
    The original query in YAML format using our operations: {input_query} \n
    The original query result: {json.dumps(strip_labels(node.sample_result, dataset), indent=2)[:3000]} \n
    """
    
    # Create a condensed version for message history (without full operator/directive descriptions)
    condensed_user_message = f"""
    Recommend one specific rewrite directive for accuracy optimization.
    
    Valid choices:
    {availabel_actions_str}
    
    Action Performance History:
    {action_stats_str}
    
    Current pipeline: {input_query} 
    """
    
    return user_message, condensed_user_message


def create_expansion_prompt_cost(node, action_options, input_query, available_actions, action_cost_changes, action_accuracy_changes, action_counts, sample_input, root_node, yaml_file_path, dataset=None, node_accuracies=None, choice_counts=None, prior_counts=None) -> tuple[str, str]:
    """Create expansion prompt for cost optimization."""

    ### DEBUG 
    print("memo: ")
    print(node.get_memo_for_llm(root_node, node_accuracies))
    print("***"*50)

    availabel_actions_str = format_action_options(action_options, choice_counts, prior_counts)

    print("availabel_actions_str: ")
    print(availabel_actions_str)
    action_stats = []
    for action in available_actions:
        cost_change = action_cost_changes.get(action, 0)
        accuracy_change = action_accuracy_changes.get(action, 0)
        count = action_counts.get(action, 0)
        
        if count > 0:
            avg_cost_change = cost_change / count
            avg_accuracy_change = accuracy_change / count
            action_stats.append(
                f"- {action.name}: {int(count)} successful runs, avg change in cost: {avg_cost_change:+.2f}, avg change in accuracy: {avg_accuracy_change:+.4f}"
            )
        else:
            action_stats.append(
                f"- {action.name}: not run successfully yet"
            )

    action_stats_str = "\n".join(action_stats)

    print("action_stats_str: ")
    print(action_stats_str)

    input_schema = load_input_doc(yaml_file_path, exclude_fields=label_fields(dataset))

    user_message = f"""
    I have a set of operations used to process long documents, along with a list of possible rewrite directives designed to improve the cost effectiveness of the pipeline, while maintaining similar or better accuracy.
    Given a query pipeline composed of these operations, recommend one specific rewrite directive (identified by its name from the provided list) that would improve cost effectiveness. Also, specify which operator(s) (by name) in the pipeline the directive should be applied to.
    Make sure your recommended directive is selected from the provided list.

    Pipeline:
    Pipelines in DocETL are the core structures that define the flow of data processing. A pipeline consists of five main components: \n
    - Default Model: The language model to use for the pipeline. Limit your choice of model to {AVAILABLE_MODELS} \n
    - System Prompts: A description of your dataset and the "persona" you'd like the LLM to adopt when analyzing your data. \n
    - Datasets: The input data sources for your pipeline. \n
    - Operators: The processing steps that transform your data. \n
    - Pipeline Specification: The sequence of steps and the output configuration. \n

    Operators:
    Operators form the building blocks of data processing pipelines. Below is the list of operators:
    {op_map.to_string()}\n
    {op_extract.to_string()}\n
    {op_parallel_map.to_string()}\n
    {op_filter.to_string()}\n
    {op_reduce.to_string()}\n
    {op_split.to_string()}\n
    {op_gather.to_string()}\n
    {op_unnest.to_string()}\n
    {op_sample.to_string()}\n
    {op_resolve.to_string()}\n

    Rewrite directives:
    {get_all_cost_directive_strings()}\n

    Your valid choice of operation and rewrite directive combination. Only choose one of these:\n
    {availabel_actions_str}

    Action Performance History:
    Based on plans already run in this search, here's how each action has performed (average change vs the plan it was applied to):\n
    {action_stats_str}

    Note: These statistics come from applying actions to other plans in this search, not the current one. Use this as general guidance about action effectiveness, but consider that performance may vary for this plan's models and structure.

    Model Performance Reference:
    If you are considering a model change directive, here are model statistics on this specific dataset: \n {str(MODEL_STATS.get(dataset, {}))}
    These show the accuracy (acc) and cost for each model. Only reference this when evaluating model change options.

    Selection Strategy:
    Consider the current query pipeline and which directive can best improve cost effectiveness.

    {node.get_memo_for_llm(root_node, node_accuracies)}

    Make sure you only choose from the valid choices above and avoid already used combinations or approaches too similar to what has already been tried in the current optimization path.

    Input document schema with token statistics: {input_schema} \n
    Input data sample: {json.dumps(strip_labels(sample_input, dataset), indent=2)[:5000]} \n
    The original query in YAML format using our operations: {input_query} \n
    The original query result: {json.dumps(strip_labels(node.sample_result, dataset), indent=2)[:3000]} \n
    """
    
    # Create a condensed version for message history (without full operator/directive descriptions)
    condensed_user_message = f"""
    Recommend one specific rewrite directive for cost optimization.
    
    Valid choices:
    {availabel_actions_str}
    
    Action Performance History:
    {action_stats_str}
    
    Current pipeline: {input_query}
    """
    
    return user_message, condensed_user_message

def disallowed_models(ops, default_model=None) -> list:
    """Model names in a pipeline outside the allowed lists: LLMs ("model", default_model) must be in
    AVAILABLE_MODELS or Jev; embedding models must be the Gemini embedding model."""
    from docetl.reasoning_optimizer.directives.base import AVAILABLE_MODELS
    allowed = set(AVAILABLE_MODELS)
    found = []

    def ok(m):
        m = str(m)
        return m == "jev" or m.startswith("jev-") or m.split("/", 1)[-1] in allowed

    from docetl.reasoning_optimizer.instantiate_schemas import ALLOWED_EMBEDDING_MODELS

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "model" and isinstance(v, str) and not ok(v):
                    found.append(v)
                if k == "embedding_model" and isinstance(v, str) and v not in ALLOWED_EMBEDDING_MODELS:
                    found.append(v)
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(ops)
    if default_model and not ok(default_model):
        found.append(default_model)
    return sorted(set(found))
