import json
import os
from typing import Any, Dict, List, Literal, Optional, Tuple

from litellm import completion, model_cost
from pydantic import BaseModel, Field
from rich import print as rprint

from docetl.operations.utils.llm import count_tokens

# Configuration for how many documents to read at once
DOCS_TO_READ_PER_ITERATION = 3


class AgentDecision(BaseModel):
    """Schema for agent decision-making in agentic loops."""

    action: Literal["read_next_docs", "output_schema"] = Field(
        ..., description="The action the agent wants to take"
    )
    reasoning: str = Field(
        ...,
        description="Explanation of why the agent chose this action and what they learned from current samples",
    )


class ReadNextDocTool:
    """Tool for iteratively reading documents from input data."""

    def __init__(self, input_data: List[Dict], context_window: int = 32000):
        self.input_data = input_data
        self.current_index = 0
        self.context_window = context_window
        self.total_docs = len(input_data)

    def read_next_doc(self) -> Optional[Dict]:
        """Read the next document from the input data."""
        if self.current_index >= len(self.input_data):
            return None

        doc = self.input_data[self.current_index]
        self.current_index += 1
        return doc

    def read_next_docs(self, count: int = DOCS_TO_READ_PER_ITERATION) -> List[Dict]:
        """Read the next N documents from the input data."""
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
    ):
        self.input_data = input_data
        self.agent_llm = agent_llm
        self.context_window = self._get_model_context_window(agent_llm)
        self.doc_reader = ReadNextDocTool(input_data, self.context_window)
        self.message_history = []
        self.validation_func = validation_func

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

    def run_agentic_loop(
        self, system_prompt: str, initial_user_message: str, response_schema: BaseModel
    ) -> Tuple[Any, List[Dict]]:
        """
        Run an agentic loop where the agent analyzes input data for directive instantiation.

        Args:
            system_prompt: System message for the agent
            initial_user_message: Initial user message with task description
            response_schema: Pydantic schema for the expected response

        Returns:
            Tuple of (parsed_response, message_history)
        """
        # Initialize message history
        self.message_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_user_message},
        ]

        max_iterations = min(3, len(self.input_data))  # Conservative limit for analysis

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
            action_guidance = f"""
Choose your next action:
- read_next_docs: If you need more examples to understand data patterns, edge cases, or to gather more information for your analysis (reads ~{DOCS_TO_READ_PER_ITERATION} documents at once)
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
            response = completion(
                model=self.agent_llm,
                messages=self.message_history,
                response_format=AgentDecision,
                api_key=os.environ.get("AZURE_API_KEY"),
                api_base=os.environ.get("AZURE_API_BASE"),
                api_version=os.environ.get("AZURE_API_VERSION"),
                azure=True,
            )

            try:
                decision_json = json.loads(response.choices[0].message.content)
                decision = AgentDecision(**decision_json)
            except Exception as e:
                raise Exception(f"Failed to parse agent decision: {str(e)}")

            self.message_history.append(
                {"role": "assistant", "content": response.choices[0].message.content}
            )

            # Handle agent's decision
            if decision.action == "read_next_docs":
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
                            f"Sample {self.doc_reader.current_index - len(next_docs) + i + 1}:\n{json.dumps(doc, indent=2)}"
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

        from .base import MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS

        error_message = ""

        for attempt in range(MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS):
            schema_response = completion(
                model=self.agent_llm,
                messages=self.message_history,
                response_format=response_schema,
                api_key=os.environ.get("AZURE_API_KEY"),
                api_base=os.environ.get("AZURE_API_BASE"),
                api_version=os.environ.get("AZURE_API_VERSION"),
                azure=True,
            )

            try:
                parsed_response = json.loads(schema_response.choices[0].message.content)
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
                return schema_instance, self.message_history

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
