import json
import re
import pandas as pd
import numpy as np
from Levenshtein import distance
import random

def evaluate_results_original(method_name, results_file, ground_truth_file, original_json_file):
    """The original MOAR CUAD evaluator, kept unchanged for reference. See CUAD_EVAL.md."""

    # Read the DocETL results JSON file
    with open(results_file, "r") as f:
        docetl_results = json.load(f)
        docetl_results = pd.DataFrame(docetl_results)

    with open(original_json_file, "r") as f:
        original_json_content = json.load(f)
    
    original_json_content_cleaned = {}
    for item in original_json_content:
        filename = item["name"].split("/")[-1].upper().rstrip(".TXT").replace(".", "").replace(",", "").replace(" ", "").replace("_", "").replace("-", "").replace("'", "").replace(r'[^a-zA-Z0-9]$', '')
        original_json_content_cleaned[filename] = item

    # Variables to track total clause length for calculating average
    all_text_spans = []

    # Read the ground truth CSV file
    ground_truth_df = pd.read_csv(ground_truth_file)
    ground_truth_df.columns = ground_truth_df.columns.map(
        lambda x: x.replace(" ", "_").lower()
    )

    # Sort ground truth dataframe
    ground_truth_df["filename"] = ground_truth_df["filename"].apply(
        lambda x: x.upper()
        .replace(".", "")
        .replace(",", "")
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("'", "")
        .replace(r'[^a-zA-Z0-9]$', '')
    )

    # Sort DocETL results
    filename_key = "name" if "name" in list(docetl_results.columns) else "filename"
    docetl_results["filename"] = docetl_results[filename_key].apply(
        lambda x: x.split("/")[-1]
        .upper()
        .rstrip(".TXT")
        .replace(".", "")
        .replace(",", "")
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("'", "")
        .replace(r'[^a-zA-Z0-9]$', '')
    )


    # Function to calculate precision and recall
    def calculate_metrics(metric, predicted_results, ground_truth):
        true_positives = 0
        false_positives = 0 
        false_negatives = 0
        true_negatives = 0

        # For all predicted results, strip the non-alphanumeric characters
        def clean_text(text):
            if isinstance(text, list):
                text = ' '.join(str(t) for t in text)
            elif not isinstance(text, str):
                text = str(text)
            cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            return '' if len(cleaned) < 1 else cleaned
            
        predicted_results_cleaned = {k: clean_text(v) for k, v in predicted_results.items()}
        ground_truth_cleaned = {k: clean_text(v) for k, v in ground_truth.items()}

        # print(len(ground_truth_cleaned.keys()))
        # print("___________________________________")
        # print(len(original_json_content_cleaned.keys()))

        # Calculate metrics across all filenames
        #print(f"----- {metric} -----")
        
        for filename in original_json_content_cleaned:
            if filename not in predicted_results_cleaned:
                # print(f"Filename not in predicted results: {filename}")
                pred = ""
            else:
                pred = predicted_results_cleaned[filename]
            if filename not in ground_truth_cleaned:
                truth = "ground truth not found"
            else: 
                truth = ground_truth_cleaned[filename]
            
            # Calculate Jaccard similarity between truth and pred
            def get_jaccard_sim(str1, str2):
                words1 = set(str1.lower().split())
                words2 = set(str2.lower().split())
                intersection = words1.intersection(words2)
                union = words1.union(words2)
                return len(intersection) / len(union) if union else 0

            if truth != "" and pred != "":
                #print(f"True positive (correct span)")
                if get_jaccard_sim(truth, pred) > 0.15:
                    true_positives += 1
                else:
                    if method_name == "gemini_gemini_opt":
                        print(f"False positive (incorrect span) for {method_name}, {pred}")
                    false_positives += 1
            elif truth == "" and pred != "":
                if method_name == "gemini_gemini_opt":
                    print(f"False positive (no span) for {method_name}, {pred}")
                false_positives += 1
            elif truth != "" and pred == "":
                #print(f"False negative (no span) for {method_name}, {truth}")
                false_negatives += 1
            elif truth == "" and pred == "":
                true_negatives += 1
                #print(f"True negative")

        # Calculate precision and recall
        total_predictions = true_positives + false_positives
        total_ground_truth = true_positives + false_negatives

        if total_predictions == 0:
            precision = 0.0
        else:
            precision = true_positives / total_predictions

        if total_ground_truth == 0:
            recall = float("nan")
        else:
            recall = true_positives / total_ground_truth

        return precision, recall

    # List of metrics to evaluate
    metrics = [
        "document_name", "parties", "agreement_date", "effective_date", 
        "expiration_date", "renewal_term", "notice_to_terminate_renewal",
        "governing_law", "most_favored_nation", "non_compete", "exclusivity",
        "no_solicit_of_customers", "competitive_restriction_exception",
        "no_solicit_of_employees", "non_disparagement", "termination_for_convenience",
        "right_of_first_refusal", "change_of_control", "anti_assignment",
        "revenue_profit_sharing", "price_restriction", "minimum_commitment",
        "volume_restriction", "ip_ownership_assignment", "joint_ip_ownership",
        "license_grant", "non_transferable_license", "affiliate_ip_license_licensor",
        "affiliate_ip_license_licensee", "unlimited_license",
        "irrevocable_or_perpetual_license", "source_code_escrow",
        "post_termination_services", "audit_rights", "uncapped_liability",
        "cap_on_liability", "liquidated_damages", "warranty_duration",
        "insurance", "covenant_not_to_sue", "third_party_beneficiary",
    ]

    # Reindex the dataframes and join them
    docetl_results = docetl_results.sort_values(by="filename").reset_index()

    # print(f"Number of documents in results: {len(docetl_results)}")

    # Find closest matching filename for each docetl result
    matched_filenames = []
    for docetl_filename in docetl_results["filename"]:
        closest_match = max(
            ground_truth_df["filename"],
            key=lambda x: sum(a == b for a, b in zip(x, docetl_filename))
        )
        matched_filenames.append(closest_match)
    

    ground_truth_df = ground_truth_df[ground_truth_df["filename"].isin(matched_filenames)]
    ground_truth_df = ground_truth_df.sort_values(by="filename").reset_index()
    docetl_results = docetl_results.sort_values(by="filename").reset_index()
    # Drop any of the metric cols from docetl_results if they exist
    existing_metrics = [col for col in metrics if col in docetl_results.columns]
    if existing_metrics:
        docetl_results = docetl_results.drop(columns=existing_metrics)

    # Merge the dataframes on the index
    merged_df = pd.merge(docetl_results, ground_truth_df, left_index=True, right_index=True, how="inner")
    # print(merged_df.head())

    # Calculate precision and recall for DocETL results
    docetl_metrics = {}
    for metric in metrics:
        if metric in merged_df.columns:
            predicted_clauses = dict(zip(merged_df["filename_x"], merged_df["clauses"]))
            predicted_results = {}
            for k, clauses in predicted_clauses.items():
                if isinstance(clauses, dict):
                    clauses = clauses.get("clauses", [])

                if not clauses:
                    predicted_results[k] = ""
                    continue
                
                # Print out problematic c values
                for c in clauses:
                    if not (isinstance(c, dict) and "clause_type" in c and "text_span" in c):
                        pass
                        #print(f"Problematic clause entry: {c} (type: {type(c)})")
                clauses = [{
                    "clause_type": c["clause_type"].lower().strip().replace(" ", "_").replace("-", "_"),
                    "text_span": c["text_span"]
                } for c in clauses if isinstance(c, dict) and "clause_type" in c and "text_span" in c]
                clause_types = [c["clause_type"] for c in clauses]

                if len(clause_types) == 0:
                    predicted_results[k] = ""
                    continue
                
                closest_match = min(
                    clause_types,
                    key=lambda x: distance(x, metric)
                )
                
                # If the closest match doesn't share any words with the metric, set the closest match to ""
                metric_words = metric.split('_')
                closest_match_words = closest_match.split('_')
                if not any(word in closest_match_words for word in metric_words):
                    predicted_results[k] = ""
                    continue
                
                match_for_clause_type = [c["text_span"] for c in clauses if c["clause_type"] == closest_match][0]
                predicted_results[k] = match_for_clause_type

            precision, recall = calculate_metrics(
                metric, predicted_results, merged_df[["filename_x", metric]].set_index("filename_x")[metric].to_dict()
            )
            docetl_metrics[metric] = {"precision": precision, "recall": recall}
            
            # Track all extracted text spans for average length calculation
            for key, text_span in predicted_results.items():
                if text_span and isinstance(text_span, str) and len(text_span.strip()) > 0:
                    all_text_spans.append(text_span)
        else:
            docetl_metrics[metric] = {"precision": np.nan, "recall": np.nan}

    # Calculate average clause length
    avg_clause_length = 0
    if len(all_text_spans) > 0:
        avg_clause_length = sum(len(span) for span in all_text_spans) / len(all_text_spans)

    # Calculate average metrics
    precisions = [v["precision"] for v in docetl_metrics.values()]
    recalls = [v["recall"] for v in docetl_metrics.values()]

    avg_precision = np.nanmean(precisions)
    avg_recall = np.nanmean(recalls)
    nan_fraction = (np.isnan(precisions) | np.isnan(recalls)).mean()

    # Calculate F1 score for non-nan values
    f1_scores = [
        (
            2 * (p * r) / (p + r)
            if not (np.isnan(p) or np.isnan(r)) and (p + r) != 0
            else 0 if (p + r) == 0 else np.nan
        )
        for p, r in zip(precisions, recalls)
    ]
    avg_f1 = np.nanmean(f1_scores)

    return {
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "nan_fraction": nan_fraction,
        "avg_f1": avg_f1,
        "avg_clause_length": avg_clause_length,
        "per_metric": docetl_metrics
    }



# ---------------------------------------------------------------------------------------
# Corrected evaluator: official CUAD matching (see CUAD_EVAL.md)
# ---------------------------------------------------------------------------------------

EVALUATOR_VERSION = "cuad-official-match-v1"
OFFICIAL_IOU_THRESH = 0.5
GOLD_PART_SEPARATOR = "<omitted>"

# (metric key, ground-truth CSV column, extra aliases). The metric keys are the ones the
# original evaluator reported; the CSV columns are the exact headers of CUAD-master_clauses.csv.
CUAD_CATEGORIES = [
    ("document_name", "Document Name", []),
    ("parties", "Parties", []),
    ("agreement_date", "Agreement Date", []),
    ("effective_date", "Effective Date", []),
    ("expiration_date", "Expiration Date", []),
    ("renewal_term", "Renewal Term", []),
    ("notice_to_terminate_renewal", "Notice Period To Terminate Renewal", []),
    ("governing_law", "Governing Law", []),
    ("most_favored_nation", "Most Favored Nation", []),
    ("non_compete", "Non-Compete", []),
    ("exclusivity", "Exclusivity", []),
    ("no_solicit_of_customers", "No-Solicit Of Customers", ["non_solicit_of_customers"]),
    ("competitive_restriction_exception", "Competitive Restriction Exception",
     ["competitive_restriction_exceptions"]),
    ("no_solicit_of_employees", "No-Solicit Of Employees", ["non_solicit_of_employees"]),
    ("non_disparagement", "Non-Disparagement", ["no_disparagement"]),
    ("termination_for_convenience", "Termination For Convenience", []),
    ("right_of_first_refusal", "Rofr/Rofo/Rofn",
     ["right_of_first_refusal_offer_or_negotiation", "right_of_first_refusal_offer_negotiation",
      "rights_of_first_refusal_offer_negotiation"]),
    ("change_of_control", "Change Of Control", []),
    ("anti_assignment", "Anti-Assignment", []),
    ("revenue_profit_sharing", "Revenue/Profit Sharing", ["revenue_sharing"]),
    ("price_restriction", "Price Restrictions", []),
    ("minimum_commitment", "Minimum Commitment", ["minimum_commitments"]),
    ("volume_restriction", "Volume Restriction", ["volume_restrictions"]),
    ("ip_ownership_assignment", "Ip Ownership Assignment", []),
    ("joint_ip_ownership", "Joint Ip Ownership", []),
    ("license_grant", "License Grant", ["license_grants"]),
    ("non_transferable_license", "Non-Transferable License", []),
    ("affiliate_ip_license_licensor", "Affiliate License-Licensor", []),
    ("affiliate_ip_license_licensee", "Affiliate License-Licensee", []),
    ("unlimited_license", "Unlimited/All-You-Can-Eat-License", ["unlimited_all_you_can_eat_license"]),
    ("irrevocable_or_perpetual_license", "Irrevocable Or Perpetual License", []),
    ("source_code_escrow", "Source Code Escrow", []),
    ("post_termination_services", "Post-Termination Services", []),
    ("audit_rights", "Audit Rights", []),
    ("uncapped_liability", "Uncapped Liability", []),
    ("cap_on_liability", "Cap On Liability", []),
    ("liquidated_damages", "Liquidated Damages", []),
    ("warranty_duration", "Warranty Duration", []),
    ("insurance", "Insurance", []),
    ("covenant_not_to_sue", "Covenant Not To Sue", []),
    ("third_party_beneficiary", "Third Party Beneficiary", []),
]
CUAD_METRICS = [key for key, _, _ in CUAD_CATEGORIES]


def _type_key(text):
    """lowercase, drop a leading list number such as '24) ', and join words with '_'."""
    text = re.sub(r"^\s*\d+\s*[).:\-]\s*", "", str(text).lower())
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


_TYPE_NAMES = {}
for _key, _column, _aliases in CUAD_CATEGORIES:
    for _name in [_key, _column, *_aliases]:
        _TYPE_NAMES[_type_key(_name)] = _key
# Longest names first, so "right_of_first_refusal_offer_or_negotiation_..." is not read as
# "right_of_first_refusal" followed by a description.
_TYPE_NAMES_BY_LENGTH = sorted(_TYPE_NAMES, key=len, reverse=True)


def _singular(key):
    """Each word in singular form ('beneficiaries' -> 'beneficiary', 'rights' -> 'right')."""
    def one(word):
        if len(word) > 4 and word.endswith("ies"):
            return word[:-3] + "y"
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            return word[:-1]
        return word
    return "_".join(one(w) for w in key.split("_"))


# The same names with every word singular, for plural/singular variants ("Third Party Beneficiaries").
_TYPE_NAMES_SINGULAR = {}
for _name, _key in _TYPE_NAMES.items():
    if _TYPE_NAMES_SINGULAR.setdefault(_singular(_name), _key) != _key:
        raise ValueError(f"Ambiguous singular clause type name: {_name}")


def canonical_clause_type(clause_type):
    """Map a predicted clause_type to a metric key, or None.

    Exact match on the normalized name, the CSV header or an alias, also with every word made
    singular ("Third Party Beneficiaries" -> third_party_beneficiary). Otherwise, a known name
    followed by more words ("Governing Law: Which state's law ...", the prompt line copied with its
    question) maps to that name, the longest one that fits. No nearest-name guessing.
    Returns (key, how) with how in {"exact", "plural", "prefix"}.
    """
    key = _type_key(clause_type)
    if key in _TYPE_NAMES:
        return _TYPE_NAMES[key], "exact"
    if _singular(key) in _TYPE_NAMES_SINGULAR:
        return _TYPE_NAMES_SINGULAR[_singular(key)], "plural"
    for name in _TYPE_NAMES_BY_LENGTH:
        if key.startswith(name + "_"):
            return _TYPE_NAMES[name], "prefix"
    return None, None


def official_jaccard(gt, pred):
    """Exact copy of get_jaccard from the official CUAD evaluate.py."""
    remove_tokens = [".", ",", ";", ":"]
    for token in remove_tokens:
        gt = gt.replace(token, "")
        pred = pred.replace(token, "")
    gt = gt.lower()
    pred = pred.lower()
    gt = gt.replace("/", " ")
    pred = pred.replace("/", " ")

    gt_words = set(gt.split(" "))
    pred_words = set(pred.split(" "))

    intersection = gt_words.intersection(pred_words)
    union = gt_words.union(pred_words)
    return len(intersection) / len(union)


def official_task_counts(gold, predictions, parties=False):
    """Official many-to-many matching for one (contract, category) task.

    TP: gold answers matched by at least one prediction. FN: gold answers matched by none.
    FP: predictions matching no gold answer (every prediction on a negative task).
    For Parties, a prediction that contains the gold string also matches.
    """
    def match(answer, pred):
        return official_jaccard(answer, pred) >= OFFICIAL_IOU_THRESH or (parties and answer in pred)

    rows = [[match(answer, pred) for pred in predictions] for answer in gold]
    tp = sum(any(row) for row in rows)
    fn = len(gold) - tp
    fp = sum(not any(row[i] for row in rows) for i in range(len(predictions)))
    return tp, fp, fn


def _prf(tp, fp, fn):
    """Precision, recall, F1 from pooled counts. None where the denominator is zero, except
    precision, which is 0 when there are no predictions (as in the handover's summaries)."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return precision, recall, f1


def _doc_key(name):
    """Normalized filename: no directory, no .pdf/.txt, uppercase letters and digits only."""
    name = str(name).split("/")[-1]
    name = re.sub(r"\.(pdf|txt)$", "", name, flags=re.IGNORECASE)
    return re.sub(r"[^A-Z0-9]", "", name.upper())


def load_cuad_gold(ground_truth_file, categories=None):
    """{doc key: {metric key: [gold answer strings]}} from CUAD-master_clauses.csv.

    The CSV joins the parts of a multi-part answer with "<omitted>"; the official QA data lists
    each part as its own answer, so the cells are split on that marker.
    """
    import ast

    df = pd.read_csv(ground_truth_file)
    wanted = [c for c in CUAD_CATEGORIES if categories is None or c[0] in categories]
    missing = [column for _, column, _ in wanted if column not in df.columns]
    if missing:
        raise ValueError(f"CUAD ground truth is missing columns: {missing}")
    gold = {}
    for _, row in df.iterrows():
        key = _doc_key(row["Filename"])
        if key in gold:
            raise ValueError(f"Duplicate contract in CUAD ground truth: {row['Filename']}")
        entry = {}
        for metric, column, _ in wanted:
            cell = row[column]
            spans = ast.literal_eval(cell) if isinstance(cell, str) and cell.strip() else []
            entry[metric] = [part for span in spans for part in span.split(GOLD_PART_SEPARATOR)]
        gold[key] = entry
    return gold


def _resolve_categories(categories):
    if categories is None:
        return list(CUAD_METRICS)
    resolved = []
    for name in categories:
        key, how = canonical_clause_type(name)
        if key is None or how != "exact":
            raise ValueError(f"Unknown CUAD category: {name!r}")
        resolved.append(key)
    return list(dict.fromkeys(resolved))


def predictions_from_clauses(clauses, categories):
    """{metric key: [quotes]} from one output row's `clauses`, plus parsing diagnostics.

    Each entry {clause_type, text_span} or {clause_type, text_spans} contributes its spans (a string,
    or every string in a list) to the category its clause_type maps to. Repeated entries of a type add more quotes.
    Exactly empty strings are dropped and exact duplicates removed, keeping order. Strings are not
    split on commas or otherwise repaired, and quotes need not occur in the source.
    """
    preds = {c: [] for c in categories}
    info = {"unknown_types": [], "prefix_types": [], "malformed_entries": 0}
    if isinstance(clauses, dict):
        clauses = clauses.get("clauses", [])
    for entry in clauses:
        if not isinstance(entry, dict) or "clause_type" not in entry:
            info["malformed_entries"] += 1
            continue
        key, how = canonical_clause_type(entry["clause_type"])
        if key is None:
            info["unknown_types"].append(str(entry["clause_type"]))
            continue
        if how == "prefix":
            info["prefix_types"].append(str(entry["clause_type"]))
        if key not in preds:
            continue  # a category outside the scored subset
        span = entry["text_spans"] if "text_spans" in entry else entry.get("text_span")
        values = span if isinstance(span, list) else [span]
        for value in values:
            if value is None:
                continue
            if not isinstance(value, str):
                info["malformed_entries"] += 1
                continue
            if value != "":
                preds[key].append(value)
    return {k: list(dict.fromkeys(v)) for k, v in preds.items()}, info


def evaluate_results(method_name, results_file, ground_truth_file, original_json_file, categories=None):
    """Score a CUAD run with the official CUAD answer matching, pooled over all tasks.

    A task is one (input contract, category). The input contracts come from original_json_file
    (all of them, including any the pipeline dropped); gold answers from the CSV.
    Headline: avg_f1 = pooled (micro) F1 over all tasks. Also macro F1 over categories.
    categories: optional subset of metric keys or CSV names, e.g. the 8 types of earlier work.
    """
    from collections import Counter

    cats = _resolve_categories(categories)
    gold = load_cuad_gold(ground_truth_file, cats)

    with open(results_file) as f:
        outputs = json.load(f)
    if isinstance(outputs, dict):
        outputs = [outputs]

    if original_json_file:
        with open(original_json_file) as f:
            inputs = json.load(f)
    else:  # no input list: score the contracts present in the output
        inputs = [{"name": o.get("name") or o.get("filename"), "id": o.get("id")} for o in outputs]

    # Input contracts, joined to the gold by normalized filename. Fail loudly on any gap.
    docs, id_to_key, unmatched = {}, {}, []
    for item in inputs:
        key = _doc_key(item["name"])
        if key not in gold:
            unmatched.append(item["name"])
            continue
        docs[key] = item["name"]
        if item.get("id") is not None:
            id_to_key[item["id"]] = key
    if unmatched:
        raise ValueError(f"{len(unmatched)} input contracts have no CUAD ground truth: {unmatched[:5]}")

    # Group output rows by contract: by id when it is an input id, else by normalized name.
    rows_by_doc, extra_rows = {}, 0
    for row in outputs:
        key = id_to_key.get(row.get("id")) if isinstance(row, dict) else None
        if key is None and isinstance(row, dict):
            name = row.get("name") or row.get("filename")
            key = _doc_key(name) if name else None
        if key not in docs:
            extra_rows += 1
            continue
        rows_by_doc.setdefault(key, []).append(row)

    counts = {c: {"tp": 0, "fp": 0, "fn": 0, "tasks": 0, "positive_tasks": 0, "negative_tasks": 0,
                  "negative_correct_empty": 0, "failed_tasks": 0} for c in cats}
    unknown_types, prefix_types = Counter(), Counter()
    malformed, missing_docs, failed_docs, duplicate_docs = 0, [], [], 0
    quote_lengths = []

    for key in sorted(docs):
        rows = rows_by_doc.get(key, [])
        duplicate_docs += len(rows) > 1
        valid = []
        for row in rows:
            clauses = row.get("clauses")
            if isinstance(clauses, dict):
                clauses = clauses.get("clauses")
            if isinstance(clauses, list):
                valid.append(clauses)
        # Missing contract, or no row with a clauses list: no predictions, flagged as failed.
        failed = not valid
        if not rows:
            missing_docs.append(docs[key])
        elif failed:
            failed_docs.append(docs[key])
        preds = {c: [] for c in cats}
        for clauses in valid:  # several rows for one contract: pool their quotes
            row_preds, info = predictions_from_clauses(clauses, cats)
            unknown_types.update(info["unknown_types"])
            prefix_types.update(info["prefix_types"])
            malformed += info["malformed_entries"]
            for c in cats:
                preds[c].extend(row_preds[c])
        for c in cats:
            quotes = list(dict.fromkeys(preds[c]))
            quote_lengths.extend(len(q) for q in quotes)
            answers = gold[key][c]
            tp, fp, fn = official_task_counts(answers, quotes, parties=(c == "parties"))
            k = counts[c]
            k["tp"] += tp
            k["fp"] += fp
            k["fn"] += fn
            k["tasks"] += 1
            if answers:
                k["positive_tasks"] += 1
            else:
                k["negative_tasks"] += 1
                # A failed or missing output is not a successful abstention.
                if not quotes and not failed:
                    k["negative_correct_empty"] += 1
            k["failed_tasks"] += failed

    per_metric = {}
    for c in cats:
        k = counts[c]
        precision, recall, f1 = _prf(k["tp"], k["fp"], k["fn"])
        per_metric[c] = {
            **k, "precision": precision, "recall": recall, "f1": f1,
            "negative_task_accuracy": (k["negative_correct_empty"] / k["negative_tasks"]
                                       if k["negative_tasks"] else None),
        }

    tp = sum(k["tp"] for k in counts.values())
    fp = sum(k["fp"] for k in counts.values())
    fn = sum(k["fn"] for k in counts.values())
    precision, recall, f1 = _prf(tp, fp, fn)
    scored = [m["f1"] for m in per_metric.values() if m["f1"] is not None]
    macro_f1 = sum(scored) / len(scored) if scored else None
    negative = sum(k["negative_tasks"] for k in counts.values())
    negative_ok = sum(k["negative_correct_empty"] for k in counts.values())

    result = {
        # Keys read by MOAR (mcts.py, ParetoFrontier.py) and evaluation/utils.py.
        "avg_f1": f1 if f1 is not None else 0.0,
        "avg_precision": precision,
        "avg_recall": recall if recall is not None else 0.0,
        "nan_fraction": 1 - len(scored) / len(cats),
        "avg_clause_length": sum(quote_lengths) / len(quote_lengths) if quote_lengths else 0.0,
        "per_metric": per_metric,
        # New fields.
        "evaluator_version": EVALUATOR_VERSION,
        "f1_micro": f1,
        "f1_macro": macro_f1,
        "precision": precision,
        "recall": recall,
        "tp": tp, "fp": fp, "fn": fn,
        "contracts": len(docs),
        "categories": cats,
        "tasks": sum(k["tasks"] for k in counts.values()),
        "positive_tasks": sum(k["positive_tasks"] for k in counts.values()),
        "gold_answers": tp + fn,
        "predicted_quotes": len(quote_lengths),
        "negative_tasks": negative,
        "negative_tasks_correct_empty": negative_ok,
        "negative_task_accuracy": negative_ok / negative if negative else None,
        "missing_contracts": len(missing_docs),
        "missing_contract_names": missing_docs,
        "failed_contracts": len(failed_docs),
        "failed_tasks": sum(k["failed_tasks"] for k in counts.values()),
        "duplicate_contract_rows": duplicate_docs,
        "extra_output_rows": extra_rows,
        "unknown_clause_types": dict(unknown_types.most_common()),
        "prefix_matched_clause_types": dict(prefix_types.most_common()),
        "malformed_entries": malformed,
    }
    result.update(_original_scores(method_name, results_file, ground_truth_file, original_json_file)
                  if categories is None else {})
    return result


def _original_scores(method_name, results_file, ground_truth_file, original_json_file):
    """The original evaluator's headline numbers, for comparison. Never raises."""
    try:
        old = evaluate_results_original(method_name, results_file, ground_truth_file, original_json_file)
        return {"original_avg_f1": float(old["avg_f1"]), "original_avg_precision": float(old["avg_precision"]),
                "original_avg_recall": float(old["avg_recall"])}
    except Exception as err:  # the original fails on some inputs, e.g. without original_json_file
        return {"original_avg_f1": None, "original_error": f"{type(err).__name__}: {err}"}


if __name__ == "__main__":
    import argparse
    import sys
    import os
    from docetl.utils import extract_output_from_json
    
    parser = argparse.ArgumentParser(description="Evaluate CUAD results from a single JSON file")
    parser.add_argument("results_file", help="Path to the results JSON file to evaluate")
    parser.add_argument("--ground_truth", "-gt", default="/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/experiments/reasoning/data/CUAD-master_clauses.csv", 
                       help="Path to the ground truth CSV file (default: experiments/reasoning/data/train/cuad_train.csv)")
    parser.add_argument("--method_name", "-m", default="evaluation", 
                       help="Name of the method being evaluated (default: evaluation)")
    parser.add_argument("--original_json", "-oj", 
                    help="Path to the original JSON file containing the list of files to evaluate (optional)")
    parser.add_argument("--categories", "-c", nargs="+",
                        help="Score only these categories (metric keys or CSV names)")

    args = parser.parse_args()
    

    # Check if results file exists
    if not os.path.exists(args.results_file):
        print(f"Error: Results file '{args.results_file}' not found.")
        sys.exit(1)
    
    # Check if ground truth file exists
    if not os.path.exists(args.ground_truth):
        print(f"Error: Ground truth file '{args.ground_truth}' not found.")
        print("Please provide the correct path using --ground_truth or -gt")
        sys.exit(1)
    
    try:
        print(f"Evaluating results from: {args.results_file}")
        print(f"Using ground truth from: {args.ground_truth}")
        print(f"Method name: {args.method_name}")
        print("-" * 50)
        
        # Evaluate the results
        results = evaluate_results(args.method_name, args.results_file, args.ground_truth, args.original_json,
                                   categories=args.categories)
        
        # Print results in a formatted way
        print(f"Evaluation Results for {args.method_name}:")
        print(f"Average Precision: {results['avg_precision']:.4f}")
        print(f"Average Recall: {results['avg_recall']:.4f}")
        print(f"Average F1 Score: {results['avg_f1']:.4f}")
        print(f"NaN Fraction: {results['nan_fraction']:.4f}")
        print(f"Average Clause Length: {results['avg_clause_length']:.1f} characters")
        print(f"Macro F1: {results['f1_macro']}")
        print(f"TP/FP/FN: {results['tp']}/{results['fp']}/{results['fn']}  "
              f"negative-task accuracy: {results['negative_task_accuracy']}")
        print(f"Missing/failed contracts: {results['missing_contracts']}/{results['failed_contracts']}")
        if results.get("original_avg_f1") is not None:
            print(f"Original evaluator F1: {results['original_avg_f1']:.4f}")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

