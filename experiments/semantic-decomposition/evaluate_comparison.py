"""Evaluate saved comparison outputs using the unchanged MOAR evaluator (no API calls)."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "experiments/reasoning/evaluation/sustainability.py"
DEFAULT_GT = ROOT / "experiments/reasoning/data/company_reports_gt.json"


def select_ground_truth(records, ground_truth):
    """Select original labels by document ID, preserving input order; fail on ambiguity."""
    if not records:
        raise ValueError("Cannot evaluate an empty input dataset")
    ids = [str(row["id"]) for row in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate IDs in the input dataset")
    selected_ids = set(ids)
    by_id = {}
    for row in ground_truth:
        key = str(row["id"])
        if key in selected_ids:
            if key in by_id:
                raise ValueError(f"Duplicate ground-truth ID: {key}")
            for field in ("GT company_name", "GT economic_activity"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"Missing {field} for ID {key}")
            by_id[key] = row
    missing = selected_ids - by_id.keys()
    if missing:
        raise ValueError(f"Missing ground-truth IDs: {sorted(missing)}")
    return [by_id[key] for key in ids]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    input_path = run / "input.json"
    records = json.loads(input_path.read_text())
    gt_path = args.ground_truth.resolve()
    subset = select_ground_truth(records, json.loads(gt_path.read_text()))
    experiment = json.loads((run / "experiment.json").read_text())
    variants = list(experiment["variants"])
    if not variants:
        raise ValueError("No variants to evaluate")
    for variant in variants:
        if experiment["variants"][variant]["status"] != "success":
            raise ValueError(f"Cannot evaluate unsuccessful variant: {variant}")
        if not (run / variant / "output.json").is_file():
            raise FileNotFoundError(run / variant / "output.json")
    spec = importlib.util.spec_from_file_location("moar_sustainability_eval", EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dest = run / "evaluation"
    dest.mkdir(exist_ok=True)
    subset_path = dest / "ground_truth.json"
    save(subset_path, subset)
    results = {}
    for variant in variants:
        results[variant] = module.evaluate_results(
            variant, str(run / variant / "output.json"), str(subset_path), str(input_path)
        )
        save(dest / f"{variant}_metrics.json", results[variant])
    report = {
        "evaluator": str(EVALUATOR),
        "evaluator_sha256": hashlib.sha256(EVALUATOR.read_bytes()).hexdigest(),
        "ground_truth_source": str(gt_path),
        "ground_truth_source_sha256": hashlib.sha256(gt_path.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "document_ids": [row["id"] for row in records],
        "denominator": len(records),
        "ground_truth_policy": "Original labels selected by input ID; fuzzy-match candidates restricted to these documents; no relabeling",
        "scoring": "Unchanged MOAR evaluator: name similarity >= 0.5, exact sector match, combined_score = (correct names + correct sectors) / (2 * input count)",
        "limitations": [
            "Does not judge the factual quality of key findings",
            "Matches generated company names, not document IDs; matching is not one-to-one",
            "missing_companies counts unmatched generated names, not input documents omitted by the pipeline",
            "total_companies_processed is the input denominator, not the number of generated company summaries",
        ],
        "variants": results,
    }
    save(dest / "evaluation.json", report)
    lines = ["# Sustainability evaluation", "", f"Input documents / scoring denominator: {len(records)}.", "",
             "Unchanged MOAR scoring code, with ground truth restricted by ID to the shared input snapshot. Original labels are preserved.", "",
             "| Variant | Sector accuracy | Company name accuracy | Combined score |",
             "|---|---:|---:|---:|"]
    for name, metrics in results.items():
        lines.append(f"| {name} | {metrics['economic_activity_accuracy']:.4%} | {metrics['company_name_accuracy']:.4%} | {metrics['combined_score']:.4%} |")
    lines += ["", "## Interpretation", ""] + [f"- {note}" for note in report["limitations"]]
    (dest / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(results, indent=2))
    print(f"Saved evaluation to {dest}")


if __name__ == "__main__":
    main()
