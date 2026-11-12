"""Effective input length per model: the longest input (cl100k tokens) a model reads before its
accuracy drops, used as the suggested chunk size in the chunking rewrites.

Published effective-length numbers (RULER, NoLiMa) do not cover these models, so the table is
measured on CUAD clause extraction from the first-layer sweep on the 40 train contracts
(experiments/cuad-decomposition/measure_effective_length.py): the longest length bin whose pooled
recall stays at or above 85% of the model's recall on contracts up to 8K tokens. Bins are 8K and
16K; values above 16K are not claimed (only 8 train contracts are longer).
"""
from typing import Optional

EFFECTIVE_TOKENS = {
    "gemini-3.1-flash-lite": 8000,
    "gemini-3.1-pro-preview": 8000,
    "gemini-3.5-flash": 16000,
    "gemini-3.5-flash-lite": 16000,
    "gemini-3.6-flash": 16000,
    "gemini-3.7-flash": 16000,
    "gemini-3.8-flash": 16000,
    "gemini-3-flash-preview": 16000,
    "gemini-4-argon": 16000,
}
DEFAULT_RANGE = "8000-16000"


def effective_tokens(model: Optional[str]) -> Optional[int]:
    if not model:
        return None
    return EFFECTIVE_TOKENS.get(str(model).split("/")[-1])


def chunk_size_hint(model: Optional[str], topk: bool = False) -> str:
    """One line for the agent's instantiation prompt. topk: the model reads k chunks in one call,
    so the hint bounds chunk_size * k rather than one chunk."""
    n = effective_tokens(model)
    if n is None:
        return (f"Choose a token count (typically {DEFAULT_RANGE}); no effective length is measured "
                f"for the model that reads the chunks ({model or 'unknown'})")
    name = str(model).split("/")[-1]
    why = (f"{n} tokens is the effective length of {name}, the model that reads the chunks (the longest "
           f"input before its recall drops, measured on long contracts)")
    if topk:
        return (f"The reduce reads the k selected chunks in one call, so keep chunk_size * k at or below "
                f"{n} tokens: {why}. Pick chunk_size and k together (e.g. {n // 4} x 4 or {n // 2} x 2)")
    return f"Suggested: {n} tokens; {why}. Chunks longer than this lose recall"
