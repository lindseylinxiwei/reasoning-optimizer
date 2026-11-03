"""Run the handover's tests unchanged: they import the vendored modules by flat name."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "docetl/retrieval/jev"), str(Path(__file__).resolve().parent)]

# These two need the original research archive, which the handover does not ship.
ARCHIVE_ONLY = {
    "test_historical_baseline_bytes_unchanged",
    "test_original_policy_outputs_match_frozen_engine",
}


def pytest_collection_modifyitems(items):
    for item in items:
        if item.name in ARCHIVE_ONLY:
            item.add_marker(pytest.mark.skip(reason="requires the original MOAR-Jev research archive"))
