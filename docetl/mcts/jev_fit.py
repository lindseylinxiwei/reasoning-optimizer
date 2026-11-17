"""Does an operation's input fit in Jev's context? A static bound from the pipeline, no data.

Walks the operations before the target and tracks an upper bound (tokens) for every field they
write. Raw dataset fields and anything not recognised are unbounded. Recognised writers:
  code_map from take_head_tail     (head_words + tail_words) words
  jev_retrieve                     evidence_chars characters
  jev_chunk_retrieve               max_chars characters if set; a share of the document is unbounded
  LLM map / parallel_map           generated output fields
  split (token_count)              num_tokens per chunk
  gather without a 'middle'        chunk * (1 + peripheral head/tail chunks); a 'middle' section
                                   includes every earlier chunk, so it is unbounded
The bound is conservative: words and characters are converted with margins.
"""
from __future__ import annotations

import re
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

TOKENS_PER_WORD_SAFE = 1.6        # corpus average is 1.39 (cl100k); margin for number-heavy text
CHARS_PER_TOKEN_SAFE = 3.0        # usual English is ~4
LLM_FIELD_TOKENS = 8_000          # generated fields (summaries, extractions)
JEV_FIT_TOKENS = 28_000           # under Jev's 32K, margin for its own tokenizer

# Rewrites that can bring an operation's input under the limit while leaving a filter/map that
# Jev can run on (doc_chunking_topk turns the map into a reduce; compression saves ~15%).
# jev_retrieval keeps a share of each document, which has no fixed upper bound; when it is the shortening
# rewrite, cap_chunk_retrieval gives its jev_chunk_retrieve a max_chars that fits the budget.
SHORTENING_FOR_JEV = ("take_head_tail", "jev_retrieval", "doc_summarization", "doc_chunking")


def _fields_read(prompt: str) -> List[str]:
    return sorted(set(re.findall(r"input\.([A-Za-z_][A-Za-z0-9_]*)", prompt or "")) |
                  set(re.findall(r"input\[['\"]([^'\"]+)['\"]\]", prompt or "")))


def _prompt_tokens(prompt: str) -> int:
    import tiktoken
    text = re.sub(r"\{\{.*?\}\}", "", prompt or "", flags=re.S)
    return len(tiktoken.get_encoding("cl100k_base").encode(text, disallowed_special=()))


def _code_map_writes(op: Dict) -> Dict[str, Optional[int]]:
    code = op.get("code") or ""
    keys = set(re.findall(r"return\s*\{\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*:", code))
    head = re.search(r"words\[:(\d+)\]", code)
    if head:            # take_head_tail: keeps head_words (+ tail_words) words of one field
        tail = re.search(r"words\[-(\d+):\]", code)
        words = int(head.group(1)) + (int(tail.group(1)) if tail else 0)
        return {k: int(words * TOKENS_PER_WORD_SAFE) for k in keys}
    return {k: None for k in keys}


def input_token_bound(ops: List[Dict], target: str) -> Tuple[Optional[int], str]:
    """(upper bound on the target's rendered prompt in tokens, or None if unbounded; reason)."""
    bounds: Dict[str, Optional[int]] = {}
    origin: Dict[str, str] = {}
    target_op = None
    for op in ops:
        if op.get("name") == target:
            target_op = op
            break
        kind = op.get("type")
        writes: Dict[str, Optional[int]] = {}
        if kind == "code_map":
            writes = _code_map_writes(op)
        elif kind == "jev_retrieve" and op.get("output_key"):
            writes = {op["output_key"]: int(int(op.get("evidence_chars", 0)) / CHARS_PER_TOKEN_SAFE) or None}
        elif kind == "jev_chunk_retrieve" and op.get("output_key"):
            cap = op.get("max_chars")
            writes = {op["output_key"]: int(int(cap) / CHARS_PER_TOKEN_SAFE) if cap else None}
        elif kind in ("map", "parallel_map"):
            schema = (op.get("output") or {}).get("schema") or {}
            writes = {f: LLM_FIELD_TOKENS for f in schema}
        elif kind == "split" and op.get("split_key"):
            n = (op.get("method_kwargs") or {}).get("num_tokens") if op.get("method") == "token_count" else None
            writes = {f"{op['split_key']}_chunk": int(n) if n else None}
        elif kind == "gather" and op.get("content_key"):
            ck = op["content_key"]
            periph = op.get("peripheral_chunks") or {}
            base = bounds.get(ck)
            if base is None or any("middle" in (periph.get(side) or {}) for side in ("previous", "next")):
                writes = {f"{ck}_rendered": None}
            else:
                extra = sum(int(((periph.get(side) or {}).get(part) or {}).get("count", 0))
                            for side in ("previous", "next") for part in ("head", "tail"))
                writes = {f"{ck}_rendered": base * (1 + extra)}
        for f, b in writes.items():
            bounds[f] = b
            origin[f] = op.get("name", kind)
    if target_op is None:
        return None, f"operation {target!r} not found"
    total, parts = _prompt_tokens(target_op.get("prompt", "")), []
    for f in _fields_read(target_op.get("prompt", "")):
        b = bounds.get(f)
        if b is None:
            src = f"written by {origin[f]} with no size limit" if f in origin else "the raw document field"
            return None, (f"{target} reads {f!r} ({src}); its length is not bounded by the pipeline, so it "
                          f"can exceed Jev's {JEV_FIT_TOKENS // 1000}K budget")
        total += b
        parts.append(f"{f} <= {b} tokens (from {origin[f]})")
    return total, f"{target} reads " + ", ".join(parts) + f"; prompt plus input <= {total} tokens"


def fits_jev(ops: List[Dict], target: str) -> Tuple[bool, Optional[int], str]:
    bound, why = input_token_bound(ops, target)
    return (bound is not None and bound <= JEV_FIT_TOKENS), bound, why


JEV_PROMPT_HEADROOM = 1_000      # the Jev switch rewrites the prompt (adds output labels) after the cap is set


def cap_chunk_retrieval(ops: List[Dict], target: str, headroom: int = JEV_PROMPT_HEADROOM) -> List[Dict]:
    """Give every uncapped jev_chunk_retrieve whose output the target reads a max_chars, so the target's input
    fits Jev's budget: the room left after the prompt, the other fields it reads and `headroom` tokens for the
    Jev switch's longer prompt, split evenly. Returns the ops unchanged when nothing is uncapped or another
    field the target reads is unbounded (a cap cannot help)."""
    ops = deepcopy(ops)
    target_op = next((o for o in ops if o.get("name") == target), None)
    if target_op is None:
        return ops
    read = set(_fields_read(target_op.get("prompt", "")))
    uncapped = [o for o in ops if o.get("type") == "jev_chunk_retrieve" and o.get("output_key") in read
                and not o.get("max_chars")]
    if not uncapped:
        return ops
    for o in uncapped:
        o["max_chars"] = 1              # counts as 0 tokens: measures everything else
    bound, _ = input_token_bound(ops, target)
    room = JEV_FIT_TOKENS - headroom - bound if bound is not None else 0
    for o in uncapped:
        if room > 0:
            o["max_chars"] = int(room / len(uncapped) * CHARS_PER_TOKEN_SAFE)
        else:
            o.pop("max_chars")
    return ops
