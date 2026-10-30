"""Run candidate extraction -> Jev selection -> source copying on JSON records."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from pipeline import MODELS, LocalExtractor, JevSelector, extract_candidates, select_and_copy

ROOT = Path(__file__).resolve().parents[3]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n")


def overlap_stages(records, extract, finish, enabled=False):
    """One extraction on the caller thread overlaps one selection on a worker.

    Wait for the previous selection before submitting another: at most one API
    selection is active, and no unbounded queue or concurrent ledger writes exist.
    """
    if not enabled:
        for record in records:
            yield finish(extract(record))
        return
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = None
        for record in records:
            item = extract(record)
            if pending is not None:
                yield pending.result()
            pending = pool.submit(finish, item)
        if pending is not None:
            yield pending.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-field", default="tot_text_raw")
    parser.add_argument("--extractor", choices=MODELS, default="gliner25")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=320)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overlap-api", action="store_true", help="Overlap next-document extraction with one selection worker")
    parser.add_argument("--extract-only", action="store_true", help="Save candidates without classifier calls")
    args = parser.parse_args()
    if not 0 <= args.overlap < args.chunk_size <= 384:
        parser.error("Require 0 <= overlap < chunk-size <= 384 for consistent extraction windows")
    if args.batch_size < 1 or (args.batch_size > 1 and args.extractor != "gliner"):
        parser.error("Positive batch size required; batching currently supports gliner only")
    records = json.loads(args.input.read_text())
    if not records or len({str(r['id']) for r in records}) != len(records):
        parser.error("Input must contain records with unique IDs")
    if any(not isinstance(r.get(args.text_field), str) or not r[args.text_field].strip() for r in records):
        parser.error("Each record must contain a nonempty text field")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    # Load environment before lazy ML imports; keep cache separate from the earlier experiments.
    os.environ.setdefault("HF_HOME", "/private/tmp/gliner-smoke-hf")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    selector = None if args.extract_only else JevSelector(args.output_dir / "calls.jsonl")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    save(args.output_dir / "input.json", [{"id": r["id"], "text": r[args.text_field]} for r in records])
    import torch
    torch.set_num_threads(4)
    started = time.perf_counter()
    extractor = LocalExtractor(args.extractor, args.device, args.threshold)
    load_seconds = time.perf_counter()-started
    save(args.output_dir / "config.json", {
        **{k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        "model_id": extractor.model_id, "revision": extractor.revision,
        "load_seconds": load_seconds, "cpu_threads": 4,
        "selector_model": None if args.extract_only else "jev-1.13.0",
        "selector_payload_format": "Single question unchanged; overflow uses shared full text with per-question candidate names; offsets local only",
        "invalid_span_policy": "Record and exclude spans failing exact source-offset validation",
        "selector_input_price_usd_per_million": 0.042,
        "selector_context_policy": "Hierarchical batches of 254 + NONE; shared-state parallel questions; conservative cl100k_base budgets: state + longest question <=24000, request <=48000; no truncation",
        "local_compute_cost": "USD 0 by user-requested accounting convention",
        "versions": {p:importlib.metadata.version(p) for p in ("torch", "gliner", "gliner2", "transformers")}})
    results = []
    started = time.perf_counter()
    def extract_record(record):
        document_started = time.perf_counter()
        text = record[args.text_field]
        row = {"id": record["id"], "document_sha256": hashlib.sha256(text.encode()).hexdigest(),
               "document_start_seconds": document_started-started,
               "stage_cost_usd": {"extraction": 0.0, "selection": 0.0, "copying": 0.0}}
        t = time.perf_counter()
        try:
            row.update(extract_candidates(text, extractor, args.chunk_size, args.overlap, args.batch_size))
        except Exception as exc:
            row.update(status="error", error_type=type(exc).__name__, error=str(exc))
        finally:
            row["extraction_wall_seconds"] = time.perf_counter()-t
        with (args.output_dir / "candidates.jsonl").open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False)+"\n")
        return text, row, document_started, time.perf_counter()

    def finish_record(item):
        text, row, document_started, extraction_finished = item
        row["selection_queue_seconds"] = time.perf_counter()-extraction_finished
        def timed_selector(text, candidates):
            selected_at = time.perf_counter()
            try:
                decision = selector(text, candidates)
                row["stage_cost_usd"]["selection"] = decision["cost_usd"]
                return decision
            finally:
                row["selection_wall_seconds"] = time.perf_counter()-selected_at
        if row.get("status") != "error":
            try:
                t = time.perf_counter()
                if selector:
                    row.update(select_and_copy(text, row["candidates"], timed_selector))
                else:
                    row["status"] = "extracted_only"
                row["selection_and_copy_wall_seconds"] = time.perf_counter()-t
                row.setdefault("selection_wall_seconds", 0.0)
                row["copying_wall_seconds"] = row["selection_and_copy_wall_seconds"]-row["selection_wall_seconds"]
            except Exception as exc:
                row.update(status="error", error_type=type(exc).__name__, error=str(exc))
                attempts = [json.loads(line) for line in (args.output_dir / "calls.jsonl").read_text().splitlines()] if (args.output_dir / "calls.jsonl").exists() else []
                attempts = [x for x in attempts if x["document_sha256"] == row["document_sha256"]]
                row["stage_cost_usd"]["selection"] = sum(x["cost_usd"] for x in attempts) if all(x.get("cost_usd") is not None for x in attempts) else None
        row["document_wall_seconds"] = time.perf_counter()-document_started
        row["document_completed_seconds"] = time.perf_counter()-started
        return row

    for row in overlap_stages(records, extract_record, finish_record,
                              enabled=args.overlap_api and selector is not None):
        results.append(row)
        save(args.output_dir / "results.json", results)
        print(f"{row['id']}: {row['status']}; candidates={len(row.get('candidates', []))}", flush=True)
    ledger = [json.loads(line) for line in (args.output_dir / "calls.jsonl").read_text().splitlines()] if (args.output_dir / "calls.jsonl").exists() else []
    save(args.output_dir / "metrics.json", {
        "load_seconds": load_seconds, "pipeline_wall_seconds": time.perf_counter()-started,
        "mean_document_wall_seconds": sum(r["document_wall_seconds"] for r in results)/len(results),
        "documents": len(results), "errors": sum(r["status"]=="error" for r in results),
        "abstentions": sum(r["status"]=="abstained" for r in results),
        "selector_api_cost_usd": sum(r.get("cost_usd") or 0 for r in ledger),
        "calls_with_unknown_cost": sum(r.get("cost_usd") is None for r in ledger),
        "stage_wall_seconds": {
            "extraction": sum(r.get("extraction_wall_seconds", 0) for r in results),
            "selection": sum(r.get("selection_wall_seconds", 0) for r in results),
            "copying": sum(r.get("copying_wall_seconds", 0) for r in results)},
        "stage_cost_usd": {"extraction": 0.0, "selection": sum(r.get("cost_usd") or 0 for r in ledger), "copying": 0.0},
        "cost_note": "Local extraction, loading and copying cost USD 0 by user convention. Jev uses returned usage at USD 0.042/M input. Unknown billed usage is flagged, not treated as a free call."})
    if any(r["status"]=="error" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
