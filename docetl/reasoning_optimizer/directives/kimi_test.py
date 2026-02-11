# from together import Together

# client = Together()
# stream = client.chat.completions.create(
#     model="moonshotai/Kimi-K2-Thinking",
#     messages=[
#         {
#             "role": "user",
#             "content": "Which number is bigger, 9.11 or 9.9? Think carefully.",
#         }
#     ],
#     stream=True,
#     max_tokens=500,
# )

# for chunk in stream:
#     if chunk.choices:
#         delta = chunk.choices[0].delta

#         # Show reasoning tokens if present
#         if hasattr(delta, "reasoning") and delta.reasoning:
#             print(delta.reasoning, end="", flush=True)

#         # Show content tokens if present
#         if hasattr(delta, "content") and delta.content:
#             print(delta.content, end="", flush=True)

from litellm import completion 
import os

from pydantic import BaseModel, Field
class AgentTest(BaseModel):
    """Schema for agent decision-making in agentic loops."""

    answer: float = Field(
        ..., description="The answer to the question"
    )
    reasoning: str = Field(
        ...,
        description="Explanation of why the agent gave the answer",
    )
os.environ["TOGETHERAI_API_KEY"] = "tgp_v1_4HF8Bq4TXJrgT-bsQWJuVXdfKd3Fbhc_gixmNBztFU4"

messages = [{"role": "user", "content": "What is the answer of 9.11 + 9.9?"}]

response = completion(model="together_ai/moonshotai/Kimi-K2-Thinking", messages=messages)
print(response.choices[0].message.content)
cost = response.usage
print(f"Cost: {cost}")