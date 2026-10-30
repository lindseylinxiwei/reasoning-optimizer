"""Run identical inputs through Jev+Gemini and Gemini-only, with a call ledger."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import threading
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("DOCETL_HOME_DIR", "/private/tmp/jev-comparison-cache")
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)
os.environ.pop("JEV_PROXY_URL", None)
os.environ["TYPESAFE_BASE_URL"] = "https://api.typesafe.ai"

import yaml
import litellm
from docetl.runner import DSLRunner
from docetl.operations.utils import api, jev

GEMINI = "gemini/gemini-3.8-flash"
MODEL_RATES = {
    "gemini/gemini-3.8-flash": (0.75, 0.075, 3.75),
    "gemini/gemini-3.1-flash-lite": (0.25, 0.025, 1.50),
    "gemini/gemini-3.5-flash-lite": (0.30, 0.03, 2.50),
}
PRICING = {
    "currency": "USD",
    "basis": "Published standard paid-tier token rates; not an invoice; free credits/tier not deducted",
    "as_of": "2026-09-28",
    "jev_input_per_million": 0.042,
    "gemini_input_per_million": 0.75,
    "gemini_cached_input_per_million": 0.075,
    "gemini_output_including_thinking_per_million": 3.75,
    "sources": ["https://docs.typesafe.ai/models", "https://ai.google.dev/gemini-api/docs/pricing"],
}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def main():
    global GEMINI
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir")
    parser.add_argument("--variant", choices=["both", "jev_gemini", "gemini_only"], default="both")
    parser.add_argument("--gemini-model", choices=list(MODEL_RATES), default=GEMINI)
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--pipeline-file", type=Path)
    args = parser.parse_args()
    GEMINI = args.gemini_model
    input_rate, cached_rate, output_rate = MODEL_RATES[GEMINI]
    PRICING.update(gemini_input_per_million=input_rate,
                   gemini_cached_input_per_million=cached_rate,
                   gemini_output_including_thinking_per_million=output_rate)
    for key in ("GEMINI_API_KEY", "TYPESAFE_API_KEY"):
        if not os.environ.get(key) and (key != "TYPESAFE_API_KEY" or args.variant != "gemini_only"):
            raise SystemExit(f"Missing {key} in repository .env")
    folder = Path(__file__).resolve().parent
    out = Path(args.output_dir) if args.output_dir else folder / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if (out / "experiment.json").exists():
        raise SystemExit(f"Refusing to overwrite existing experiment: {out}")
    out.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load((args.pipeline_file or folder / "sustainability_train_pipeline.yaml").read_text())
    dataset_path = args.input_file or Path(base["datasets"]["company_reports"]["path"])
    data = json.loads(dataset_path.read_text())
    save(out / "input.json", data)
    save(out / "pricing.json", PRICING)
    # Register known metadata so DocETL never falls back to a 32K truncation limit.
    litellm.register_model({GEMINI: {
        "max_input_tokens": 1048576, "max_output_tokens": 65536,
        "input_cost_per_token": input_rate/1e6, "output_cost_per_token": output_rate/1e6,
        "litellm_provider": "gemini", "mode": "chat",
        "supports_function_calling": True, "supports_response_schema": True,
    }})
    experiment = {
        "input_count": len(data), "input_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "threads": args.threads, "gemini_model": GEMINI, "jev_model": "jev-1.13.0",
        "cache": "DocETL bypass enabled; fresh intermediate directories; provider cache usage recorded",
        "timing": "perf_counter around DSLRunner.load_run_save; includes data loading, calls, retries, and result saving; excludes installation, imports, and runner construction",
        "litellm_version": importlib.metadata.version("litellm"), "variants": {},
    }
    save(out / "experiment.json", experiment)
    original_completion = api.completion
    original_request = jev.request_jev
    original_run_op = DSLRunner._run_operation
    variants = ["jev_gemini", "gemini_only"] if args.variant == "both" else [args.variant]
    for variant in variants:
        dest = out / variant
        dest.mkdir(exist_ok=True)
        if (dest / "calls.jsonl").exists():
            raise SystemExit(f"Refusing to overwrite an existing run: {dest}")
        config = copy.deepcopy(base)
        config["default_model"] = GEMINI
        config["bypass_cache"] = True
        config.pop("intermediate_dir", None)
        config["datasets"]["company_reports"]["path"] = str(out / "input.json")
        config["pipeline"]["output"]["path"] = str(dest / "output.json")
        config["pipeline"]["output"].pop("intermediate_dir", None)
        for op in config["operations"]:
            op["model"] = "jev" if variant == "jev_gemini" and op["type"] in ("filter", "map") else GEMINI
            op["bypass_cache"] = True
            op["max_batch_size"] = args.threads
            op["timeout"] = 240
            op["max_retries_per_timeout"] = 0
            op["enable_observability"] = True
        (dest / "pipeline.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        calls, operations = [], []
        lock = threading.Lock()
        current_op = {"name": None}

        def record(row):
            row["operator"] = current_op["name"]
            with lock:
                row["call_index"] = len(calls) + 1
                calls.append(row)
                with (dest / "calls.jsonl").open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            print(f"CALL {variant} {row['operator']} {row['provider']} {row['status']} {row.get('elapsed_seconds', 0):.2f}s", flush=True)

        def gemini_call(*a, **kw):
            if kw.get("model", a[0] if a else "") != GEMINI:
                raise RuntimeError("Unexpected LLM model; comparison only allows configured Gemini")
            kw["num_retries"] = 0
            kw["timeout"] = 180
            start = time.perf_counter()
            try:
                response = original_completion(*a, **kw)
            except Exception as e:
                record({"provider": "gemini", "status": "error", "error_type": type(e).__name__, "elapsed_seconds": time.perf_counter()-start, "cost_usd": None})
                raise
            usage = response.usage.model_dump()
            prompt = usage.get("prompt_tokens")
            completion = usage.get("completion_tokens")
            if not isinstance(prompt, int) or not isinstance(completion, int):
                raise RuntimeError("Missing Gemini usage; refusing to report zero cost")
            details = usage.get("prompt_tokens_details") or {}
            cached = details.get("cached_tokens", 0) or 0
            cost = ((prompt-cached)*input_rate + cached*cached_rate + completion*output_rate)/1e6
            response._completion_cost = cost
            record({"provider": "gemini", "requested_model": GEMINI, "returned_model": response.model, "status": "success", "usage": usage,
                    "elapsed_seconds": time.perf_counter()-start, "cost_usd": cost, "response": response.model_dump()})
            return response

        def jev_call(payload, timeout):
            start = time.perf_counter()
            try:
                response = original_request(payload, timeout)
            except Exception as e:
                record({"provider": "jev", "status": "error", "error_type": type(e).__name__, "elapsed_seconds": time.perf_counter()-start, "cost_usd": None})
                raise
            usage = response["usage"]
            cost = usage["input_tokens"] * 0.042 / 1e6
            record({"provider": "jev", "requested_model": payload["model"], "returned_model": response.get("model"), "status": "success", "usage": usage,
                    "elapsed_seconds": time.perf_counter()-start, "cost_usd": cost, "response": response})
            return response

        def run_op(runner, op_config, input_data, *a, **kw):
            current_op["name"] = op_config["name"]
            start = time.perf_counter()
            result = original_run_op(runner, op_config, input_data, *a, **kw)
            output = result[0] if kw.get("return_instance") else result
            save(dest / (op_config["name"] + ".json"), output)
            operations.append({"name": op_config["name"], "type": op_config["type"], "input_count": len(input_data) if input_data is not None else None, "output_count": len(output), "wall_seconds": time.perf_counter()-start})
            return result

        runner = DSLRunner(config, max_threads=args.threads)
        status, error = "success", None
        start = time.perf_counter()
        try:
            with patch.object(api, "completion", gemini_call), patch.object(jev, "request_jev", jev_call), patch.object(DSLRunner, "_run_operation", run_op):
                runner.load_run_save()
        except Exception as e:
            status, error = "failed", type(e).__name__
            import traceback
            (dest / "failure.txt").write_text(traceback.format_exc())
            print(f"Run failed: {error}", flush=True)
        elapsed = time.perf_counter()-start
        totals = {p: sum(row.get("cost_usd") or 0 for row in calls if row["provider"]==p) for p in ("jev", "gemini")}
        summary = {"status": status, "error_type": error, "wall_clock_seconds": elapsed, "cost_usd": totals,
                   "total_cost_usd": sum(totals.values()), "engine_cost_usd": runner.total_cost,
                   "calls_by_provider": {p:sum(row["provider"]==p for row in calls) for p in totals},
                   "operations": operations, "failed_calls": sum(row["status"]=="error" for row in calls),
                   "billing_note": PRICING["basis"]}
        save(dest / "metrics.json", summary)
        experiment["variants"][variant] = summary
        save(out / "experiment.json", experiment)
        print(json.dumps({"variant": variant, **summary}), flush=True)
    print("RESULT_DIRECTORY=" + str(out), flush=True)


if __name__ == "__main__":
    main()
