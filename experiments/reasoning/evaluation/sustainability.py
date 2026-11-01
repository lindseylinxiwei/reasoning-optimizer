import json
import re
import pandas as pd
import numpy as np
from collections import defaultdict

def evaluate_results_original(method_name, results_file, ground_truth_file, original_json_file=None):
    """
    Original MOAR evaluator, kept unchanged for reference and for reproducing earlier
    results. evaluate_results() below replaces it; see that docstring for what changed.

    Evaluate sustainability analysis results against ground truth.
    
    Args:
        method_name: Name of the method being evaluated
        results_file: Path to DocETL results JSON file
        ground_truth_file: Path to ground truth data (should be the original dataset)
    
    Returns:
        Dictionary containing evaluation metrics
    """
    
    # Read the DocETL results JSON file
    with open(results_file, "r") as f:
        docetl_results = json.load(f)
    
    # Read the ground truth file (original company reports dataset)
    with open(ground_truth_file, "r") as f:
        ground_truth_data = json.load(f)

    
    if original_json_file:
        with open(original_json_file, "r") as f:
            original_json_content = json.load(f)
        orig_total_processed = len(original_json_content)
    else: 
        orig_total_processed = 0
    
    # Create ground truth mapping by ID
    gt_by_id = {item["id"]: item for item in ground_truth_data}
    
    # Evaluation metrics
    metrics = {
        "economic_activity_accuracy": 0.0,
        "company_name_accuracy": 0.0,
        "total_companies_processed": 0,
        "economic_activity_distribution": defaultdict(int),
        "missing_companies": 0,
        "avg_findings_length": 0.0,
        "combined_score": 0.0
    }
    
    total_processed = 0
    economic_activity_correct = 0
    company_name_correct = 0
    all_findings_lengths = []
    
    # Process each economic activity group in results
    for activity_result in docetl_results:
        if "economic_activity_summary" not in activity_result:
            continue
            
        predicted_activity = activity_result.get("economic_activity", "unknown")

        for company_summary in activity_result["economic_activity_summary"]:
            if isinstance(company_summary, str):
                continue
            company_name = company_summary.get("company_name", "").strip()
            key_findings = company_summary.get("key_findings", "")
            if not company_name:
                continue
                
            total_processed += 1
            
            # Track findings length
            if key_findings:
                all_findings_lengths.append(len(key_findings))
            
            # Find matching ground truth by company name (fuzzy matching)
            gt_match = None
            best_match_score = 0
            best_match_company_name = None
            for gt_item in ground_truth_data:
                gt_company_name = gt_item.get("GT company_name", "").strip()
                if not gt_company_name:
                    continue
                    
                # Simple similarity check
                similarity = calculate_name_similarity(company_name.lower(), gt_company_name.lower())
                if similarity > best_match_score:  
                    best_match_score = similarity
                    if similarity >= 0.5: gt_match = gt_item
                    best_match_company_name = gt_company_name
            
            if gt_match:
                # Check economic activity accuracy (exact match, no normalization)
                gt_economic_activity = gt_match.get("GT economic_activity", "").strip()
                
                if gt_economic_activity == predicted_activity:
                    economic_activity_correct += 1
                    
                # else:
                #     print(f"Economic activity mismatch: GT {gt_economic_activity} != {predicted_activity}")
    
                # Check company name accuracy (if we found a match, name is reasonably accurate)
                company_name_correct += 1
                
                # Track economic activity distribution
                if gt_economic_activity:
                    metrics["economic_activity_distribution"][gt_economic_activity] += 1
            else:
                metrics["missing_companies"] += 1
    
    # Calculate final metrics
    if orig_total_processed > 0: total_processed = orig_total_processed
    if total_processed > 0:
        metrics["economic_activity_accuracy"] = economic_activity_correct / total_processed
        metrics["company_name_accuracy"] = company_name_correct / total_processed
        metrics["total_companies_processed"] = total_processed
        
        if all_findings_lengths:
            metrics["avg_findings_length"] = sum(all_findings_lengths) / len(all_findings_lengths)
    
    # Additional analysis metrics
    metrics["total_economic_activities"] = len(metrics["economic_activity_distribution"])
    metrics["most_common_activity"] = max(metrics["economic_activity_distribution"].items(), 
                                        key=lambda x: x[1])[0] if metrics["economic_activity_distribution"] else None
    metrics["combined_score"] = (economic_activity_correct + company_name_correct) / (total_processed * 2)
    
    return metrics

def calculate_name_similarity(name1, name2):
    """Calculate simple similarity between two company names with typo handling"""
    from difflib import SequenceMatcher
    
    # Remove common corporate suffixes and clean names
    suffixes = ["ltd", "limited", "inc", "corp", "corporation", "llc", "plc", "bv", "nv", "sa", "ag"]
    
    def clean_name(name):
        if not name:
            return ""
        words = name.lower().split()
        words = [w for w in words if w not in suffixes]
        return " ".join(words)
    
    clean1 = clean_name(name1)
    clean2 = clean_name(name2)
    
    if not clean1 or not clean2:
        return 0.0
    
    # Check if one name is contained in the other (for cases like "deloitte" vs "deloitte touche...")
    if clean1 in clean2 or clean2 in clean1:
        shorter = min(len(clean1), len(clean2))
        longer = max(len(clean1), len(clean2))
        return shorter / longer
    
    # Word-level similarity with typo tolerance
    words1 = clean1.split()
    words2 = clean2.split()
    
    if not words1 or not words2:
        return 0.0
    
    matched = 0
    used_words = set()
    
    for word1 in words1:
        best_match = 0
        best_word = None
        
        for word2 in words2:
            if word2 in used_words:
                continue
            
            # Exact match or similar enough (handles typos)
            similarity = SequenceMatcher(None, word1, word2).ratio()
            if similarity > best_match:
                best_match = similarity
                best_word = word2
        
        # Accept matches above 80% similarity
        if best_match >= 0.8:
            matched += 1
            used_words.add(best_word)
    
    # Return ratio of matched words to total unique words
    total_words = len(set(words1 + words2))
    return matched / total_words if total_words > 0 else 0.0


# Corporate-form words ignored when comparing names. Compared after punctuation is removed,
# so "S.p.A." and "N.V." become "s p a" and "n v" and are dropped word by word.
NAME_STOPWORDS = {
    "ltd", "limited", "inc", "incorporated", "corp", "corporation", "llc", "plc", "bv", "nv", "sa",
    "ag", "se", "spa", "gmbh", "co", "company", "group", "holding", "holdings", "the", "kgaa",
    "ab", "asa", "oyj", "as", "tbk", "berhad", "bhd", "pjsc", "sas", "s", "p", "a", "n", "v",
}


def _name_words(name):
    """Lowercase, drop apostrophes, split on punctuation, collapse doubled letters.

    The GT names contain typos that are mostly dropped double letters ("Acenture",
    "Swis Re"), so both sides are normalized the same way.
    """
    name = re.sub(r"['’`]", "", name.lower())
    words = [re.sub(r"([a-z])\1+", r"\1", w) for w in re.split(r"[^\w]+|_", name) if w]
    core = [w for w in words if w not in NAME_STOPWORDS]
    return tuple(core or words)


def _name_variants(name):
    """The name without parentheticals, plus each parenthetical on its own.

    Outputs such as "NIPPON EXPRESS HOLDINGS, INC. (NIPPON EXPRESS GROUP)" carry an
    alternative name in parentheses.
    """
    variants = [re.sub(r"\(.*?\)", " ", name)] + re.findall(r"\((.*?)\)", name)
    return [v for v in (_name_words(x) for x in variants) if v]


def _contains(longer, shorter):
    n = len(shorter)
    return any(longer[i:i + n] == shorter for i in range(len(longer) - n + 1))


def name_match_score(output_name, gt_name):
    """Score in [0, 1] for whether output_name refers to the company gt_name; >= 0.5 is a match.

    1.0  same core name ("Enpro Industries, Inc." vs "Enpro Industries")
    0.9  one core name is a word sequence inside the other ("Enpro Industries, Inc." vs "Enpro")
    0.85 same letters once spaces are removed ("ALCAR GO LOGISTIC S" vs "Allcargo Logistics")
    0.8  one name is the acronym of the other ("LGIM")
    0.7  the GT name's first core word (>= 3 letters) appears in the output
    else the original word-level typo similarity, capped at 0.65
    """
    best = 0.0
    gt_variants = _name_variants(gt_name)
    for out in _name_variants(output_name):
        for gt in gt_variants:
            if out == gt:
                return 1.0
            # The first core word is usually the distinctive one ("XYZ Energy", "Volvo Car").
            # A shorter output must keep it, so "Energy" does not match "XYZ Energy".
            lead = gt[0]
            if _contains(out, gt) or (_contains(gt, out) and lead in out):
                best = max(best, 0.9)
                continue
            joined_out, joined_gt = "".join(out), "".join(gt)
            similar_length = min(len(joined_out), len(joined_gt)) >= 0.7 * max(len(joined_out), len(joined_gt))
            if len(joined_gt) >= 4 and (
                joined_gt in joined_out or (joined_out in joined_gt and similar_length)
            ):
                best = max(best, 0.85)
                continue
            if len(lead) >= 3 and lead in out:
                best = max(best, 0.7)
                continue
            # Acronyms: "LGIM" vs "Legal & General Investment Management".
            initials_out, initials_gt = "".join(w[0] for w in out), "".join(w[0] for w in gt)
            if (len(gt) == 1 and len(joined_gt) >= 3 and joined_gt == initials_out) or (
                len(out) == 1 and len(joined_out) >= 3 and joined_out == initials_gt
            ):
                best = max(best, 0.8)
                continue
            # Fall back to the original word-level typo matching. Containment was handled
            # above, so skip the original's containment length ratio ("energy" in "xyz energy").
            text_out, text_gt = " ".join(out), " ".join(gt)
            if text_out not in text_gt and text_gt not in text_out:
                best = max(best, min(0.65, calculate_name_similarity(text_out, text_gt)))
    return best


def evaluate_results(method_name, results_file, ground_truth_file, original_json_file=None):
    """
    Evaluate sustainability analysis results against ground truth (corrected version).

    Differences from evaluate_results_original, which is kept for reference:
      1. Name matching tolerates full legal names, corporate suffixes, parenthetical
         alternatives and the GT typos (see name_match_score). The original used a length
         ratio, so "Enpro Industries, Inc." failed against "Enpro" and the report counted as
         wrong for both name and activity.
      2. Output entries are matched only against the reports that were in the input
         (original_json_file), not against all GT records.
      3. Each input report counts once. Duplicate entries for the same company cannot
         inflate accuracy, and a report listed under several activities is correct only if
         every listing has the GT activity.
      4. The denominator is the number of input reports, so reports the pipeline drops
         (e.g. in a filter) count as misses. Without original_json_file, every GT record is a
         candidate and the denominator is matched reports plus unmatched output entries.

    The returned keys match the original (economic_activity_accuracy is MOAR's search
    metric for this workload, combined_score is used for the reported Pareto frontier).
    Diagnostics and the original evaluator's scores are added for comparison.
    """
    with open(results_file, "r") as f:
        docetl_results = json.load(f)
    with open(ground_truth_file, "r") as f:
        ground_truth_data = json.load(f)
    gt_by_id = {item["id"]: item for item in ground_truth_data}

    if original_json_file:
        with open(original_json_file, "r") as f:
            input_ids = [item["id"] for item in json.load(f)]
        reports = [gt_by_id[i] for i in input_ids if i in gt_by_id]
    else:
        reports = list(ground_truth_data)

    credited = {}  # report id -> list of predicted activities
    findings_lengths = []
    unmatched_entries = 0
    for activity_result in docetl_results:
        predicted_activity = activity_result.get("economic_activity", "unknown")
        for entry in activity_result.get("economic_activity_summary", []) or []:
            if not isinstance(entry, dict):
                continue
            company_name = (entry.get("company_name") or "").strip()
            if not company_name:
                continue
            scored = [(name_match_score(company_name, r.get("GT company_name", "")), r) for r in reports]
            best_score = max((s for s, _ in scored), default=0.0)
            if best_score < 0.5:
                unmatched_entries += 1
                continue
            # Several reports can share a company name; prefer one not yet credited.
            tied = [r for s, r in scored if s == best_score]
            target = next((r for r in tied if r["id"] not in credited), tied[0])
            credited.setdefault(target["id"], []).append(predicted_activity)
            if entry.get("key_findings"):
                findings_lengths.append(len(entry["key_findings"]))

    if original_json_file:
        total = len(reports)
    else:
        total = len(credited) + unmatched_entries

    activity_correct = 0
    distribution = defaultdict(int)
    for report in reports:
        predictions = credited.get(report["id"])
        if not predictions:
            continue
        gt_activity = report.get("GT economic_activity", "").strip()
        distribution[gt_activity] += 1
        if set(predictions) == {gt_activity}:
            activity_correct += 1

    not_other = [r for r in reports if r.get("GT economic_activity", "").strip() != "other"]
    metrics = {
        "economic_activity_accuracy": activity_correct / total if total else 0.0,
        "company_name_accuracy": len(credited) / total if total else 0.0,
        "combined_score": (activity_correct + len(credited)) / (2 * total) if total else 0.0,
        "total_companies_processed": total,
        "economic_activity_distribution": distribution,
        "missing_companies": len(reports) - len(credited),
        "avg_findings_length": float(np.mean(findings_lengths)) if findings_lengths else 0.0,
        "total_economic_activities": len(distribution),
        "most_common_activity": max(distribution.items(), key=lambda x: x[1])[0] if distribution else None,
        # Diagnostics
        "unmatched_output_entries": unmatched_entries,
        "reports_with_conflicting_activities": sum(len(set(p)) > 1 for p in credited.values()),
        "economic_activity_accuracy_excl_gt_other": (
            sum(set(credited.get(r["id"], [])) == {r["GT economic_activity"].strip()} for r in not_other)
            / len(not_other) if not_other else 0.0
        ),
    }
    original = evaluate_results_original(method_name, results_file, ground_truth_file, original_json_file)
    for key in ("economic_activity_accuracy", "company_name_accuracy", "combined_score"):
        metrics[f"original_{key}"] = original[key]
    return metrics


def main():
    """
    Main function to run sustainability evaluation.
    Update the file paths below to match your actual data files.
    """
    # Example file paths - update these to match your actual files
    method_name = "sustainability_analysis"
    results_file = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/experiments/reasoning/outputs/sustain_gpt4mini.json"
    ground_truth_file = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/experiments/reasoning/data/company_reports_gt.json"
    
    try:
        # Run evaluation
        metrics = evaluate_results(method_name, results_file, ground_truth_file)
        
        # Print results
        print(f"\n=== Evaluation Results for {method_name} ===")
        print(f"Total companies processed: {metrics['total_companies_processed']}")
        print(f"Economic activity accuracy: {metrics['economic_activity_accuracy']:.3f}")
        print(f"Company name accuracy: {metrics['company_name_accuracy']:.3f}")
        print(f"Missing companies: {metrics['missing_companies']}")
        print(f"Average findings length: {metrics['avg_findings_length']:.1f}")
        print(f"Total economic activities: {metrics['total_economic_activities']}")
        print(f"Most common activity: {metrics['most_common_activity']}")
        print(f"Combined score: {metrics['combined_score']:.3f}")
                
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please update the file paths in the main function to point to your actual data files.")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
    except Exception as e:
        print(f"Error during evaluation: {e}")


if __name__ == "__main__":
    main()

