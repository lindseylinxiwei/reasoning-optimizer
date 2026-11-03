"""Shared payload capacity helper extracted unchanged from the campaign runner."""
from dataclasses import asdict, replace
import json

def _effective_shared_config(config, scorer, tasks):
    """Reserve the EXACT shared-query overhead, without sampling/truncating text."""
    questions = min(config.max_questions, 32 // len(tasks))
    if questions < 1:
        raise ValueError("A query bundle cannot exceed 32 targets")
    maximum_delta = 0
    for task in tasks:
        for count in range(1, questions+1):
            single = {"model": config.model, "state": {"task": {
                "type": "extraction", "target": asdict(task), "objective":
                "The description defines the extraction target. Find all matching original text, "
                "including conditions and exceptions; keep distinct occurrences."}},
                "questions": {f"C{i}": {"type": "noul", "instructions": f"Does C{i} contain text to extract?"}
                              for i in range(count)}}
            _, shared_wire = scorer.payload(single)
            single_bytes = len(json.dumps(single, ensure_ascii=False, allow_nan=False).encode("utf-8"))
            maximum_delta = max(maximum_delta, len(shared_wire.encode("utf-8"))-single_bytes)
    byte_cap = min(config.max_request_bytes, scorer.max_request_bytes-maximum_delta)
    if byte_cap <= 0:
        raise ValueError("Shared target definitions leave no source capacity")
    return replace(config, max_questions=questions, max_request_bytes=byte_cap)
