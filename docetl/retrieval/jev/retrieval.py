"""Retrieval-only API. Callers supply an index and scorer; no provider setup.

The default is the new fill-cap policy. The comparison preset keeps the reported
adaptive/0.3/128 policy. Controller decisions are not physical API-call budgets.
"""
from dataclasses import replace

try:
    from .navigation import NavigationConfig
    from .run_navigation import run_bundle
except ImportError:
    from navigation import NavigationConfig
    from run_navigation import run_bundle


PRESETS = {
    "baseline_03_128": NavigationConfig(variant="adaptive", threshold=0.3,
        max_calls=128, evidence_chars=48000, max_questions=4),
    "fill_cap_128": NavigationConfig(variant="fill_cap", threshold=None,
        max_calls=128, evidence_chars=48000, max_questions=4),
    "fill_cap_512": NavigationConfig(variant="fill_cap", threshold=None,
        max_calls=512, evidence_chars=48000, max_questions=4),
}


def retrieve_document(index, tasks, scorer, *, preset="fill_cap_512",
                      evidence_chars=None, max_decisions=None, workers=None,
                      outer_workers=4):
    """Retrieve from the same reusable source/topic index used by the baseline.

    Pass a scorer with score(payload, context), optionally the existing shared
    scorer. No embedding generation or final extraction is performed here.
    A finite max_decisions applies per target, independently of the evidence cap.
    An injected live scorer would need its own separately authorized cost guard.
    """
    if preset not in PRESETS:
        raise ValueError("Unknown retrieval preset")
    changes = {}
    for name, value in (("evidence_chars", evidence_chars),
                        ("max_calls", max_decisions), ("workers", workers)):
        if value is not None:
            changes[name] = value
    config = replace(PRESETS[preset], **changes)
    return run_bundle(index, tasks, scorer, config, outer_workers=outer_workers)
