"""Evaluate saved Flash-Lite runs and append a comparison to the existing report."""
import argparse
import importlib.util
import json
from pathlib import Path
from evaluate_comparison import EVALUATOR, select_ground_truth


def read(path):
    return json.loads(path.read_text())


def build_report(root):
    inputs = read(root / "input.json")
    gt_path = root / "evaluation/ground_truth.json"
    gt = {r["id"]: r for r in select_ground_truth(inputs, read(gt_path))}
    ids = set(gt)
    assert len(ids) == 11
    assert all(r["GT document_category"] == "Sustainability Report" for r in gt.values())
    spec = importlib.util.spec_from_file_location("moar_eval", EVALUATOR)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    sources = [
        ("Jev + Gemini 3.8 Flash", root, "jev_gemini"),
        ("Gemini 3.8 Flash", root, "gemini_only"),
        ("Gemini 3.1 Flash-Lite", root / "flash_lite_3_1", "gemini_only"),
        ("Gemini 3.5 Flash-Lite", root / "flash_lite_3_5", "gemini_only"),
    ]
    results = {}
    for label, run, variant in sources:
        assert read(run / "input.json") == inputs, "Inputs differ"
        dest = run / variant
        metrics = read(dest / "metrics.json")
        assert metrics["status"] == "success" and metrics["failed_calls"] == 0
        calls = [json.loads(s) for s in (dest / "calls.jsonl").read_text().splitlines()]
        assert all(r["status"] == "success" for r in calls)
        assert abs(sum(r["cost_usd"] for r in calls) - metrics["total_cost_usd"]) < 1e-10
        assert abs(metrics["engine_cost_usd"] - metrics["total_cost_usd"]) < 1e-10
        filters = read(dest / "filter_sustainability_reports.json")
        maps = read(dest / "determine_economic_activity.json")
        retained = {r["id"] for r in filters}
        pred = {r["id"]: r["economic_activity"] for r in maps}
        assert len(retained) == len(filters) and len(pred) == len(maps)
        assert retained == set(pred) and retained <= ids
        correct = {i for i in pred if pred[i] == gt[i]["GT economic_activity"]}
        quality = evaluator.evaluate_results(label, str(dest / "output.json"), str(gt_path), str(root / "input.json"))
        usage = []
        for provider in ("jev", "gemini"):
            rows = [r for r in calls if r["provider"] == provider]
            if not rows:
                continue
            usage.append({
                "provider": provider, "calls": len(rows),
                "input_tokens": sum(r["usage"].get("prompt_tokens", r["usage"].get("input_tokens", 0)) for r in rows),
                "output_tokens": sum(r["usage"].get("completion_tokens", r["usage"].get("output_tokens", 0)) for r in rows),
                "cached_tokens": sum((r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0 for r in rows),
                "thinking_tokens": sum((r["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0 for r in rows) if provider == "gemini" else None,
            })
        result = {
            "artifact_directory": str(dest.relative_to(root)), "metrics": metrics,
            "final_output_quality": quality, "usage": usage,
            "filter_retained": len(retained), "map_correct": len(correct),
            "filter_recall": len(retained)/len(ids),
            "conditional_map_accuracy": len(correct)/len(pred),
            "joint_correctness": len(correct)/len(ids),
            "per_document": [{"id": i, "company": gt[i]["GT company_name"], "gt_sector": gt[i]["GT economic_activity"], "predicted_sector": pred.get(i), "retained": i in retained, "sector_correct": i in correct} for i in sorted(ids)],
        }
        results[label] = result
        if run != root:
            (run / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    section = ["## Flash-Lite comparison", "",
        "Both added runs use the selected Flash-Lite model for every operator (filter, map, and reduce). They reuse the exact 11-document input snapshot, prompts, schemas, configured max_threads=4, temperature, and cache-bypass policy from the original run. Provider-default thinking settings are used, so reasoning effort is not equalized across models. No new Jev calls were made.", "",
        "### Cost, pipeline wall time, and intermediate classification quality", "",
        "| Pipeline | Total cost (USD) | Wall time (s) | Filter recall | Conditional map accuracy | Joint correctness |",
        "|---|---:|---:|---:|---:|---:|"]
    for label, r in results.items():
        section.append(f"| {label} | ${r['metrics']['total_cost_usd']:.6f} | {r['metrics']['wall_clock_seconds']:.2f} | {r['filter_retained']}/11 ({r['filter_recall']:.2%}) | {r['map_correct']}/{r['filter_retained']} ({r['conditional_map_accuracy']:.2%}) | {r['map_correct']}/11 ({r['joint_correctness']:.2%}) |")
    section += ["", "Recall is retained GT-positive documents / 11. Conditional map accuracy excludes filtered-out documents. Joint correctness counts documents both retained and assigned the correct sector / 11. All labels are matched by document ID and unchanged from the original ground truth.", "",
        "### Final-output quality (unchanged MOAR evaluator)", "",
        "| Pipeline | Sector accuracy | Company name accuracy | Combined score |",
        "|---|---:|---:|---:|"]
    for label, r in results.items():
        q = r["final_output_quality"]
        section.append(f"| {label} | {q['economic_activity_accuracy']:.2%} | {q['company_name_accuracy']:.2%} | {q['combined_score']:.2%} |")
    section += ["", "Final-output scores use company-name fuzzy matching and exact sector matching, with denominator 11. They do not measure factual accuracy of sustainability findings. Name matching is not one-to-one, and these scores should not be interpreted as standalone classification accuracy.", "",
        "### Token usage", "",
        "| Pipeline | Provider | Calls | Input tokens | Cached input | Output tokens | Thinking tokens (included in output) |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for label, r in results.items():
        for u in r["usage"]:
            thinking = f"{u['thinking_tokens']:,}" if u['thinking_tokens'] is not None else "N/A"
            section.append(f"| {label} | {u['provider']} | {u['calls']} | {u['input_tokens']:,} | {u['cached_tokens']:,} | {u['output_tokens']:,} | {thinking} |")
    section += ["", "Gemini 3.1 Flash-Lite rates: $0.25 input, $0.025 cached input, $1.50 output per million tokens. Gemini 3.5 Flash-Lite rates: $0.30 input, $0.03 cached input, $2.50 output per million tokens. Output billing includes thinking. Costs use returned token usage at standard paid-tier rates, not invoices; no free credits deducted. [Official pricing](https://ai.google.dev/gemini-api/docs/pricing).", "",
        "### Per-document classification", "",
        "| ID | GT company | GT sector | 3.1 Flash-Lite | 3.5 Flash-Lite |", "|---|---|---|---|---|"]
    a = results['Gemini 3.1 Flash-Lite']['per_document']
    b = results['Gemini 3.5 Flash-Lite']['per_document']
    for x, y in zip(a, b):
        assert x['id'] == y['id']
        values = ['Dropped' if not z['retained'] else z['predicted_sector'] + (' (correct)' if z['sector_correct'] else ' (incorrect)') for z in (x,y)]
        section.append(f"| {x['id']} | {x['company']} | {x['gt_sector']} | {values[0]} | {values[1]} |")
    section += ["", "### Interpretation and artifacts", "",
        "In this run, 3.1 Flash-Lite was the lowest-cost and fastest full pipeline, while both Lite models tied on the reported quality metrics. Both retained 11/11 documents and classified 7/11 sectors correctly. Their MOAR combined score was 45.45%. Both Lite runs reported zero thinking tokens; the 3.8 Flash baseline reported 9,186. Default thinking behavior therefore differs materially across these comparisons.", "",
        "These are single runs on a small, all-positive short-document subset. Latency includes data loading, calls, intermediate recording, and saving; excludes runtime installation, imports, and runner construction. The two new runs were sequential; the earlier results were reused, not rerun simultaneously. Filter decisions can change downstream work, and provider-default thinking can also change latency and cost. This subset cannot measure filter false-positive rate or establish full-dataset quality.", "",
        "- [3.1 Flash-Lite output](flash_lite_3_1/gemini_only/output.json), [metrics and evaluation](flash_lite_3_1/evaluation.json)",
        "- [3.5 Flash-Lite output](flash_lite_3_5/gemini_only/output.json), [metrics and evaluation](flash_lite_3_5/evaluation.json)",
        "- Each run directory includes its input snapshot, pricing, experiment metadata, exact pipeline YAML, per-call ledger, and operator outputs.", ""]
    path = root / "summary.md"
    previous = path.read_text().split("\n## Flash-Lite comparison")[0]
    path.write_text(previous.rstrip() + "\n\n" + "\n".join(section))
    (root / "evaluation/flash_lite_comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({name: {"cost": r['metrics']['total_cost_usd'], "seconds": r['metrics']['wall_clock_seconds'], "retained": r['filter_retained'], "map_correct": r['map_correct'], "quality": r['final_output_quality']} for name, r in results.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    build_report(parser.parse_args().run_dir.resolve())
