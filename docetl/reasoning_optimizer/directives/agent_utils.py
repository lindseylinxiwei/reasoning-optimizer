import json
import os
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

from litellm import completion as litellm_completion, model_cost
from pydantic import BaseModel, Field
from rich import print as rprint
from .base import MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS
from docetl.operations.utils.llm import count_tokens


KIMI_K2_MODEL = "together_ai/moonshotai/Kimi-K2-Thinking"
QWEN2_5_7B_MODEL = "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo"

TOGETHER_MODELS = {KIMI_K2_MODEL, QWEN2_5_7B_MODEL}

# Cost per million tokens for Together AI models
# Reference: https://www.together.ai/pricing
TOGETHER_MODEL_COSTS = {
    KIMI_K2_MODEL: {
        "input_cost_per_token": 0.60 / 1_000_000,  # $0.60 per 1M input tokens
        "output_cost_per_token": 2.00 / 1_000_000,  # $2.00 per 1M output tokens
    },
    QWEN2_5_7B_MODEL: {
        "input_cost_per_token": 0.30 / 1_000_000,  # $0.30 per 1M input tokens
        "output_cost_per_token": 0.30 / 1_000_000,  # $0.30 per 1M output tokens
    },
}

# Default cost for unknown Together AI models
TOGETHER_DEFAULT_COSTS = {
    "input_cost_per_token": 1.00 / 1_000_000,  # $1.00 per 1M input tokens (conservative)
    "output_cost_per_token": 1.00 / 1_000_000,  # $1.00 per 1M output tokens (conservative)
}


def is_together_model(model: str) -> bool:
    return model in TOGETHER_MODELS or model.startswith("together_ai/")


def _extract_json_from_text(text: str) -> dict:
    # Try to find JSON in code blocks first
    code_block_pattern = r'```(?:json)?\s*\n?([\s\S]*?)\n?```'
    matches = re.findall(code_block_pattern, text)
    if matches:
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    
    # Try to find raw JSON object
    json_pattern = r'\{[\s\S]*\}'
    matches = re.findall(json_pattern, text)
    if matches:
        # Try the largest match first (most likely to be complete)
        for match in sorted(matches, key=len, reverse=True):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # Last resort: try parsing the whole text
    return json.loads(text)


def _normalize_together_model(model: str) -> str:
    if model.startswith("together_ai/"):
        return model
    # Model should already have the prefix
    raise ValueError(f"Together AI model must start with 'together_ai/' prefix, got: {model}")


# Context window limits for Together AI models
TOGETHER_MODEL_CONTEXT_LIMITS = {
    KIMI_K2_MODEL: 262144,  # 256K context
    QWEN2_5_7B_MODEL: 131072,  # 128K context
}
TOGETHER_DEFAULT_CONTEXT_LIMIT = 32768

# Minimum output tokens to reserve for response
MIN_OUTPUT_TOKENS = 1000


def _get_together_context_limit(model: str) -> int:
    """Get the context window limit for a Together AI model."""
    return TOGETHER_MODEL_CONTEXT_LIMITS.get(model, TOGETHER_DEFAULT_CONTEXT_LIMIT)


def _truncate_messages_for_context(
    messages: List[Dict], 
    max_input_tokens: int,
    model: str = "gpt-4.1-mini"
) -> List[Dict]:
    """
    Truncate messages to fit within the input token limit.
    Preserves system message and latest user message, truncates middle content.
    
    Args:
        messages: List of message dicts
        max_input_tokens: Maximum tokens allowed for input
        model: Model for token counting
    
    Returns:
        Truncated list of messages
    """
    if not messages:
        return messages
    
    # Calculate current token count
    total_tokens = sum(estimate_token_count(msg.get("content", ""), model) for msg in messages)
    
    if total_tokens <= max_input_tokens:
        return messages
    
    rprint(f"[dim]⚠️ Input too long ({total_tokens} tokens), truncating to fit {max_input_tokens} tokens[/dim]")
    
    # Make a copy to avoid modifying original
    messages = [m.copy() for m in messages]
    
    # Keep system message (first) and latest message (last)
    truncated = []
    system_msg = None
    if messages[0].get("role") == "system":
        system_msg = messages[0]
        remaining = messages[1:]
    else:
        remaining = messages
    
    # Always keep the latest message
    latest_msg = remaining[-1] if remaining else None
    middle_msgs = remaining[:-1] if remaining else []
    
    # Calculate tokens for preserved messages
    system_tokens = estimate_token_count(system_msg.get("content", ""), model) if system_msg else 0
    latest_tokens = estimate_token_count(latest_msg.get("content", ""), model) if latest_msg else 0
    
    # Available tokens for middle messages
    available_for_middle = max_input_tokens - system_tokens - latest_tokens - 500  # buffer
    
    # Add middle messages from most recent backwards until we hit the limit
    kept_middle = []
    current_tokens = 0
    for msg in reversed(middle_msgs):
        msg_tokens = estimate_token_count(msg.get("content", ""), model)
        if current_tokens + msg_tokens <= available_for_middle:
            kept_middle.insert(0, msg)
            current_tokens += msg_tokens
        else:
            # Try to truncate this message to fit remaining space
            remaining_space = available_for_middle - current_tokens
            if remaining_space > 500:  # Only truncate if meaningful space left
                content = msg.get("content", "")
                # Estimate chars per token and truncate
                chars_per_token = len(content) / max(1, msg_tokens)
                target_chars = int(remaining_space * chars_per_token * 0.9)  # 90% to be safe
                truncated_content = content[:target_chars] + "\n... [content truncated due to context limit]"
                truncated_msg = msg.copy()
                truncated_msg["content"] = truncated_content
                kept_middle.insert(0, truncated_msg)
            break
    
    # Reconstruct messages
    if system_msg:
        truncated.append(system_msg)
    truncated.extend(kept_middle)
    if latest_msg:
        truncated.append(latest_msg)
    
    new_total = sum(estimate_token_count(msg.get("content", ""), model) for msg in truncated)
    rprint(f"[dim]📝 Truncated messages from {total_tokens} to {new_total} tokens ({len(messages)} -> {len(truncated)} messages)[/dim]")
    
    return truncated


def _together_completion(
    model: str,
    messages: List[Dict],
    response_format: Optional[Type[BaseModel]] = None,
    temperature: float = 1.0,
    max_tokens: int = 16000,
) -> Tuple[Any, float]:
    """
    Call Together AI API via litellm for Kimi K2 and other Together models.
    
    Note: Kimi K2 does not support response_format for schema enforcement,
    so we inject the schema into the prompt instead.
    
    Returns:
        Tuple of (response_object, cost)
    """
    # Normalize model name to have together_ai/ prefix
    litellm_model = _normalize_together_model(model)
    
    # Make a copy of messages to avoid modifying the original
    messages = [m.copy() for m in messages]
    
    # Add schema instruction to system message if response_format is provided
    # (Kimi doesn't support native response_format like OpenAI)
    if response_format is not None:
        schema_instruction = f"\n\nYou MUST respond with a valid JSON object matching this schema:\n{json.dumps(response_format.model_json_schema(), indent=2)}\n\nDo not include any text before or after the JSON. Only output the JSON object."
        
        # Find or create system message
        has_system = any(m.get("role") == "system" for m in messages)
        if has_system:
            for m in messages:
                if m.get("role") == "system":
                    m["content"] = m["content"] + schema_instruction
                    break
        else:
            messages.insert(0, {"role": "system", "content": schema_instruction})
    
    # Get context limit and calculate token budget
    context_limit = _get_together_context_limit(model)
    
    # Reserve tokens for output (use requested max_tokens but cap at reasonable limit)
    output_token_budget = min(max_tokens, context_limit // 2)  # At most half the context for output
    output_token_budget = max(output_token_budget, MIN_OUTPUT_TOKENS)  # At least MIN_OUTPUT_TOKENS
    
    # Calculate max input tokens
    max_input_tokens = context_limit - output_token_budget - 100  # 100 token safety buffer
    
    # Truncate messages if input is too long
    messages = _truncate_messages_for_context(messages, max_input_tokens)
    
    # Recalculate actual input tokens after truncation
    input_tokens = sum(estimate_token_count(msg.get("content", "")) for msg in messages)
    
    # Final check: adjust output tokens if still needed
    available_tokens = context_limit - input_tokens - 100
    effective_max_tokens = max(MIN_OUTPUT_TOKENS, min(max_tokens, available_tokens))
    
    if effective_max_tokens < max_tokens:
        rprint(f"[dim]⚠️ Setting max_tokens to {effective_max_tokens} ({input_tokens} input tokens, {context_limit} context limit)[/dim]")
    
    # Call via litellm (no response_format since Kimi doesn't support it)
    response = litellm_completion(
        model=litellm_model,
        messages=messages,
        temperature=temperature,
        max_tokens=effective_max_tokens,
    )
    
    # Get token usage from response
    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0
    
    # Get model-specific pricing, fall back to default if not found
    model_costs = TOGETHER_MODEL_COSTS.get(model, TOGETHER_DEFAULT_COSTS)
    
    # Calculate cost using model-specific pricing
    cost = (
        input_tokens * model_costs["input_cost_per_token"] +
        output_tokens * model_costs["output_cost_per_token"]
    )
    
    # Extract model name for display
    model_display_name = model.split("/")[-1] if "/" in model else model
    rprint(f"[dim]🤖 Together AI ({model_display_name}) usage: {input_tokens} input, {output_tokens} output tokens, cost: ${cost:.4f}[/dim]")
    
    return response, cost


def agent_completion(
    model: str,
    messages: List[Dict],
    response_format: Optional[Type[BaseModel]] = None,
    temperature: Optional[float] = None,
    max_tokens: int = 16000,
    **kwargs
) -> Tuple[Any, float]:
    """
    Unified completion function that routes to appropriate backend.
    
    For Kimi K2 / Together AI models: Uses litellm with together_ai/ prefix.
    For other models: Uses litellm with Azure.
    
    Args:
        model: Model identifier.
               For Together AI: "together_ai/moonshotai/Kimi-K2-Thinking"
               For Azure: "gpt-4.1", "gpt-4o", etc.
        messages: List of message dicts with role and content
        response_format: Optional Pydantic model for structured output
                        (injected into prompt for Kimi since it doesn't support native schema)
        temperature: Temperature for sampling (default: None for Azure, 1.0 for Kimi)
        max_tokens: Maximum tokens in response
        **kwargs: Additional arguments passed to litellm
    
    Returns:
        Tuple of (response_object, cost) where response_object has .choices[0].message.content
    """
    if is_together_model(model):
        # Use Together AI via litellm for Kimi K2
        temp = temperature if temperature is not None else 1.0
        response, cost = _together_completion(
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=temp,
            max_tokens=max_tokens,
        )
        
        # Add cost to response's hidden params for consistency
        if not hasattr(response, '_hidden_params'):
            response._hidden_params = {}
        response._hidden_params["response_cost"] = cost
        
        return response, cost
    
    else:
        # Use litellm for Azure/OpenAI models
        litellm_kwargs = {
            "model": model,
            "messages": messages,
            "api_key": os.environ.get("AZURE_API_KEY"),
            "api_base": os.environ.get("AZURE_API_BASE"),
            "api_version": os.environ.get("AZURE_API_VERSION"),
            "azure": True,
        }
        
        if response_format is not None:
            litellm_kwargs["response_format"] = response_format
        
        if temperature is not None:
            litellm_kwargs["temperature"] = temperature
            
        # Add any additional kwargs
        litellm_kwargs.update(kwargs)
        
        response = litellm_completion(**litellm_kwargs)
        cost = response._hidden_params.get("response_cost", 0.0)
        
        return response, cost


def parse_response_content(response, response_format: Optional[Type[BaseModel]] = None):
    """
    Parse response content, handling both litellm and Together AI responses.
    
    Args:
        response: Response object from agent_completion
        response_format: Optional Pydantic model to validate against
    
    Returns:
        Parsed dict or Pydantic model instance
    """
    content = response.choices[0].message.content
    
    # Try to extract JSON from the content
    try:
        parsed = _extract_json_from_text(content)
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Failed to parse JSON from response: {e}\nContent: {content[:500]}")
    
    if response_format is not None:
        return response_format(**parsed)
    
    return parsed


class AgentDecision(BaseModel):
    """Schema for agent decision-making in agentic loops."""

    action: Literal["read_next_docs", "read_operator_doc", "output_schema"] = Field(
        ..., description="The action the agent wants to take"
    )
    reasoning: str = Field(
        ...,
        description="Explanation of why the agent chose this action and what they learned from current samples",
    )
    operator_name: Optional[str] = Field(
        None,
        description="For read_operator_doc action: the operator name to read documentation for (e.g., 'map', 'filter', 'reduce')",
    )



class ReadNextDocTool:
    """Tool for iteratively reading documents from input data."""

    def __init__(
        self,
        input_data: List[Dict],
        context_window: int = 32000,
        docs_per_iteration: int = 3,
    ):
        self.input_data = input_data
        self.current_index = 0
        self.context_window = context_window
        self.total_docs = len(input_data)
        self.docs_per_iteration = docs_per_iteration

    def read_next_doc(self) -> Optional[Dict]:
        """Read the next document from the input data."""
        if self.current_index >= len(self.input_data):
            return None

        doc = self.input_data[self.current_index]
        self.current_index += 1
        return doc

    def read_next_docs(self, count: int = None) -> List[Dict]:
        """Read the next N documents from the input data."""
        if count is None:
            count = self.docs_per_iteration
        docs = []
        for _ in range(count):
            if self.current_index >= len(self.input_data):
                break
            docs.append(self.input_data[self.current_index])
            self.current_index += 1
        return docs

    def has_more_docs(self) -> bool:
        """Check if there are more documents to read."""
        return self.current_index < len(self.input_data)

    def get_remaining_count(self) -> int:
        """Get the number of remaining documents."""
        return len(self.input_data) - self.current_index

    def reset(self) -> None:
        """Reset the iterator to the beginning."""
        self.current_index = 0


def estimate_token_count(text: str, model: str = "gpt-4.1-mini") -> int:
    """Use proper token counting instead of rough estimation."""
    return count_tokens(text, model)


def truncate_message_content(messages: List[Dict], max_tokens: int) -> List[Dict]:
    """
    Truncate message content to fit within token limits.
    Preserves system message and latest user message, truncates middle content.
    """
    if not messages:
        return messages

    # Calculate total token count
    total_tokens = sum(estimate_token_count(msg.get("content", "")) for msg in messages)

    if total_tokens <= max_tokens:
        return messages

    # Keep system message and latest user message
    truncated_messages = []
    if messages[0].get("role") == "system":
        truncated_messages.append(messages[0])
        remaining_messages = messages[1:]
    else:
        remaining_messages = messages

    # Always keep the latest message
    if remaining_messages:
        truncated_messages.append(remaining_messages[-1])
        middle_messages = remaining_messages[:-1]
    else:
        middle_messages = []

    # Calculate available tokens for middle messages
    system_tokens = (
        estimate_token_count(truncated_messages[0].get("content", ""))
        if truncated_messages
        else 0
    )
    latest_tokens = (
        estimate_token_count(truncated_messages[-1].get("content", ""))
        if len(truncated_messages) > 1
        else 0
    )
    available_tokens = (
        max_tokens - system_tokens - latest_tokens - 1000
    )  # Buffer for response

    # Add middle messages until we hit the limit
    current_tokens = 0
    for msg in reversed(middle_messages):  # Add most recent first
        msg_tokens = estimate_token_count(msg.get("content", ""))
        if current_tokens + msg_tokens <= available_tokens:
            current_tokens += msg_tokens
            truncated_messages.insert(-1, msg)  # Insert before the latest message
        else:
            break

    return truncated_messages


class AgenticDirectiveRunner:
    """
    Utility class for running agentic directives that iteratively process documents.
    Manages context windows, document iteration, and decision-making loops.
    """

    def __init__(
        self,
        input_data: List[Dict],
        agent_llm: str = "gpt-4.1-mini",
        validation_func: Optional[callable] = None,
        enable_operator_docs: bool = False,
    ):
        self.input_data = input_data
        self.agent_llm = agent_llm
        self.context_window = self._get_model_context_window(agent_llm)
        self.enable_operator_docs = enable_operator_docs
        # Double the max iterations if operator docs are enabled to allow more exploration
        self.docs_per_iteration = 3
        self.max_iterations = 6 if enable_operator_docs else 3
        self.doc_reader = ReadNextDocTool(
            input_data, self.context_window, self.docs_per_iteration
        )
        self.message_history = []
        self.validation_func = validation_func
        self.docs_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            ),
            "docs",
            "operators",
        )

    def _get_model_context_window(self, model: str) -> int:
        """Get the context window size for the given model."""
        model_cost_info = model_cost.get(model, {})
        if not model_cost_info:
            # Try stripping the first part before the /
            split_model = model.split("/")
            if len(split_model) > 1:
                model_cost_info = model_cost.get("/".join(split_model[1:]), {})

        if not model_cost_info:
            model_cost_info = model_cost.get(model.split("/")[-1], {})

        return model_cost_info.get("max_input_tokens", 32768)

    def _read_operator_doc(self, operator_name: str) -> Optional[str]:
        """
        Read the documentation for a specific operator.

        Args:
            operator_name: Name of the operator (e.g., 'map', 'filter', 'reduce')

        Returns:
            The markdown documentation content or None if not found
        """
        doc_file = os.path.join(self.docs_path, f"{operator_name}.md")
        if os.path.exists(doc_file):
            try:
                with open(doc_file, "r") as f:
                    content = f.read()
                return content
            except Exception as e:
                return f"Error reading documentation for {operator_name}: {str(e)}"
        else:
            # Try alternative names (e.g., 'parallel-map' for 'parallel_map')
            alt_name = operator_name.replace("_", "-")
            doc_file = os.path.join(self.docs_path, f"{alt_name}.md")
            if os.path.exists(doc_file):
                try:
                    with open(doc_file, "r") as f:
                        content = f.read()
                    return content
                except Exception as e:
                    return f"Error reading documentation for {operator_name}: {str(e)}"
            return f"Documentation not found for operator: {operator_name}"

    def _truncate_doc_to_tokens(self, doc: Dict, max_tokens: int) -> str:
        """
        Truncate document content to fit within the specified token limit.

        Args:
            doc: The document dictionary to truncate
            max_tokens: Maximum number of tokens to allow

        Returns:
            Truncated document string
        """
        doc_str = json.dumps(doc, indent=2)
        doc_tokens = estimate_token_count(doc_str, self.agent_llm)

        if doc_tokens <= max_tokens:
            return doc_str

        # If document is too long, truncate it
        # Estimate characters per token (rough approximation)
        chars_per_token = len(doc_str) / doc_tokens
        target_chars = int(max_tokens * chars_per_token)

        truncated = doc_str[:target_chars]
        return truncated + "... [truncated]"

    def run_agentic_loop(
        self, system_prompt: str, initial_user_message: str, response_schema: BaseModel
        ):
        """
        Run an agentic loop where the agent analyzes input data for directive instantiation.

        Args:
            system_prompt: System message for the agent
            initial_user_message: Initial user message with task description
            response_schema: Pydantic schema for the expected response

        Returns:
            Tuple of (parsed_response, message_history)
        """
        call_cost = 0.0
       
        # Initialize message history
        self.message_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_user_message},
        ]

        max_iterations = min(
            self.max_iterations, len(self.input_data)
        )  # Conservative limit for analysis

        rprint(
            f"[blue]🤖 Determining rewrite instantiation with {len(self.input_data)} documents available[/blue]"
        )

        for iteration in range(max_iterations):
            # Calculate remaining context
            current_tokens = sum(
                estimate_token_count(msg.get("content", ""), self.agent_llm)
                for msg in self.message_history
            )
            remaining_tokens = self.context_window - current_tokens - 2000  # Buffer

            # Add context info for analysis
            context_info = f"""
Analysis Progress:
- Remaining context window: ~{remaining_tokens} tokens
- Documents analyzed: {self.doc_reader.current_index}/{self.doc_reader.total_docs}
- Documents remaining: {self.doc_reader.get_remaining_count()}

Analyze the input samples to understand patterns, edge cases, and data characteristics that will help you complete your task effectively.
"""

            # Create action guidance
            if self.enable_operator_docs:
                action_guidance = f"""
Choose your next action:
- read_next_docs: If you need more examples to understand data patterns, edge cases, or to gather more information for your analysis (reads ~{self.docs_per_iteration} documents at once)
- read_operator_doc: If you need to understand how a specific operator works, its parameters, or see examples (specify operator_name like 'map', 'filter', 'reduce')
- output_schema: If you have sufficient examples to complete your task based on the patterns and insights you've gathered from the data

Focus on quality over quantity - a few diverse, informative examples are better than many similar ones.
"""
            else:
                action_guidance = f"""
Choose your next action:
- read_next_docs: If you need more examples to understand data patterns, edge cases, or to gather more information for your analysis (reads ~{self.docs_per_iteration} documents at once)
- output_schema: If you have sufficient examples to complete your task based on the patterns and insights you've gathered from the data

Focus on quality over quantity - a few diverse, informative examples are better than many similar ones.
"""

            # Update the latest user message with context info
            if self.message_history[-1]["role"] == "user":
                self.message_history[-1]["content"] += context_info + action_guidance

            # Truncate messages if needed
            self.message_history = truncate_message_content(
                self.message_history, self.context_window - 2000
            )

            rprint(
                f"[yellow]🧠 Iteration {iteration + 1}/{max_iterations}: Asking {self.agent_llm} agent to decide next action (tokens: {remaining_tokens} remaining)[/yellow]"
            )

            # Get structured agent decision
            response, step_cost = agent_completion(
                model=self.agent_llm,
                messages=self.message_history,
                response_format=AgentDecision,
            )
            call_cost += step_cost

            try:
                content = response.choices[0].message.content
                if is_together_model(self.agent_llm):
                    decision_json = _extract_json_from_text(content)
                else:
                    decision_json = json.loads(content)
                decision = AgentDecision(**decision_json)
            except Exception as e:
                raise Exception(f"Failed to parse agent decision: {str(e)}")

            self.message_history.append(
                {"role": "assistant", "content": response.choices[0].message.content}
            )

            # Handle agent's decision
            if decision.action == "read_operator_doc":
                # Agent wants to read operator documentation
                if not self.enable_operator_docs:
                    user_message = "Operator documentation reading is not enabled for this directive."
                    self.message_history.append(
                        {"role": "user", "content": user_message}
                    )
                elif not decision.operator_name:
                    user_message = "Please specify which operator documentation you want to read (e.g., 'map', 'filter', 'reduce')."
                    self.message_history.append(
                        {"role": "user", "content": user_message}
                    )
                else:
                    rprint(
                        f"[blue]📖 Agent reading documentation for operator: {decision.operator_name}[/blue]"
                    )
                    doc_content = self._read_operator_doc(decision.operator_name)
                    user_message = f"Documentation for '{decision.operator_name}' operator:\n\n{doc_content}\n\nAnalyze this documentation to understand how to use this operator effectively."
                    self.message_history.append(
                        {"role": "user", "content": user_message}
                    )

            elif decision.action == "read_next_docs":
                # Agent wants to analyze more data
                next_docs = self.doc_reader.read_next_docs()
                if not next_docs:
                    # No more documents - force output
                    rprint(
                        "[red]📄 No more documents available. Proceeding with schema generation.[/red]"
                    )
                    user_message = "No more documents available. Based on the samples you've analyzed, please complete your task."
                    self.message_history.append(
                        {"role": "user", "content": user_message}
                    )
                    break
                else:
                    rprint(
                        f"[green]📄 Agent reading {len(next_docs)} documents (up to {self.doc_reader.current_index}/{len(self.input_data)})[/green]"
                    )
                    docs_content = "\n".join(
                        [
                            f"Sample {self.doc_reader.current_index - len(next_docs) + i + 1}:\n{self._truncate_doc_to_tokens(doc, 1000)}"
                            for i, doc in enumerate(next_docs)
                        ]
                    )
                    user_message = f"{docs_content}\n\nAnalyze these samples for patterns, edge cases, and characteristics that will help with your task."
                    self.message_history.append(
                        {"role": "user", "content": user_message}
                    )

            elif decision.action == "output_schema":
                # Agent is ready to create improved prompt
                rprint(
                    f"[cyan]✨ Agent ready to generate final schema after analyzing {self.doc_reader.current_index} documents[/cyan]"
                )
                schema_prompt = f"""Based on your analysis of the input samples, complete your task using the patterns and insights you've gathered from the data.

Provide your response as a JSON object matching this schema: {response_schema.model_json_schema()}"""
                self.message_history.append({"role": "user", "content": schema_prompt})
                break

        # Get the final schema response with validation and retries
        rprint("[magenta]🔧 Generating final rewrite schema...[/magenta]")


        error_message = ""

        for attempt in range(MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS):
            schema_response, step_cost = agent_completion(
                model=self.agent_llm,
                messages=self.message_history,
                response_format=response_schema,
            )
            call_cost += step_cost

            try:
                content = schema_response.choices[0].message.content
                if is_together_model(self.agent_llm):
                    parsed_response = _extract_json_from_text(content)
                else:
                    parsed_response = json.loads(content)
                schema_instance = response_schema(**parsed_response)

                # Add any additional validation if provided
                if self.validation_func:
                    self.validation_func(schema_instance)

                rprint(
                    f"[green]✅ Schema validation passed on attempt {attempt + 1}[/green]"
                )
                self.message_history.append(
                    {
                        "role": "assistant",
                        "content": schema_response.choices[0].message.content,
                    }
                )
                return schema_instance, self.message_history, call_cost

            except Exception as err:
                error_message = f"Validation error: {err}\nPlease try again with a corrected response."
                rprint(
                    f"[red]❌ Schema validation failed on attempt {attempt + 1}: {str(err)}[/red]"
                )

                if attempt < MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS - 1:
                    rprint(
                        f"[yellow]🔄 Retrying schema generation (attempt {attempt + 2}/{MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS})[/yellow]"
                    )
                    self.message_history.append(
                        {
                            "role": "assistant",
                            "content": schema_response.choices[0].message.content,
                        }
                    )
                    self.message_history.append(
                        {"role": "user", "content": error_message}
                    )

        raise Exception(
            f"Failed to generate valid schema after {MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS} attempts. Error: {error_message}"
        )

       

   