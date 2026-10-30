"""Execute the prepared two-operator pipeline inside netlify dev:exec."""
import os
from pathlib import Path
import sys
import time
import json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from docetl.runner import DSLRunner
from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=False)
os.environ.pop("JEV_PROXY_URL", None)
folder = Path(__file__).resolve().parent
started = time.monotonic()
runner = DSLRunner.from_yaml(str(folder / "sustainability_train_classifiers.yaml"), max_threads=1)
runner.load_run_save()
output = json.loads((folder / "results_train_classifiers.json").read_text())
print(f"Live pipeline completed: {len(output)} documents passed the filter and received sector labels; {time.monotonic()-started:.2f}s elapsed.")
