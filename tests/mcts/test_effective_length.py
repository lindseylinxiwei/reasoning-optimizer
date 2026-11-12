from docetl.reasoning_optimizer.directives import DocumentChunkingDirective, DocumentChunkingTopKDirective
from docetl.reasoning_optimizer.effective_length import chunk_size_hint, effective_tokens

OP = {"name": "extract", "type": "map", "prompt": "{{ input.document }}"}


def test_lookup_ignores_provider_prefix():
    assert effective_tokens("gemini/gemini-3.1-flash-lite") == effective_tokens("gemini-3.1-flash-lite") == 8000
    assert effective_tokens("gpt-5") is None and effective_tokens(None) is None


def test_hint_uses_the_model_that_reads_the_chunks():
    line = DocumentChunkingDirective().to_string_for_instantiate({**OP, "model": "gemini/gemini-4-argon"})
    assert "Suggested: 16000 tokens" in line and "10000-15000" not in line


def test_topk_hint_bounds_chunk_size_times_k():
    text = DocumentChunkingTopKDirective().to_string_for_instantiate({**OP, "model": "gemini-3.1-flash-lite"})
    assert "chunk_size * k at or below 8000" in text and "typically 5-10" not in text


def test_unknown_model_falls_back_to_a_range():
    assert "typically 8000-16000" in chunk_size_hint("gpt-5")
