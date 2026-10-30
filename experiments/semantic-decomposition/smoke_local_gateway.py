"""One tiny live Jev call using credentials supplied by netlify dev:exec."""
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from docetl.operations.utils.jev import call_jev

# Prefer Netlify's injected gateway credentials over the deployed bridge URL.
os.environ.pop("JEV_PROXY_URL", None)
if not os.getenv("TYPESAFE_API_KEY") or not os.getenv("TYPESAFE_BASE_URL"):
    raise SystemExit("Missing gateway credentials. Run this through run_local_gateway.sh.")
started = time.monotonic()
try:
    result = call_jev(
        "jev",
        [{"role": "user", "content": "This is a sustainability report from a company operating solar power plants. Set is_sustainability_report to true or false and economic_activity to energy_supply or other."}],
        {"is_sustainability_report": "boolean", "economic_activity": "enum[energy_supply, other]"},
        timeout=60, bypass_cache=True,
    )
except Exception as error:
    print(f"Gateway smoke test failed: {error}", file=sys.stderr)
    raise SystemExit(1)
print("Live Jev response:", result.response)
print(f"Elapsed: {time.monotonic() - started:.2f}s")
print(f"Estimated provider cost: ${result.total_cost:.8f}")
print("Local gateway smoke test succeeded.")
