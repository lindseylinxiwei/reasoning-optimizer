"""Offline, exact-document candidate diagnostics plus the unchanged MOAR evaluator."""
import argparse
import ast
import csv
import json
import math
import re
from pathlib import Path

from adapters import ROOT, load_module
from pipeline import digest, save

ALIASES = {
    'Notice to Terminate Renewal': 'Notice Period To Terminate Renewal',
    'Right of First Refusal, Offer or Negotiation': 'Rofr/Rofo/Rofn',
    'Price Restriction': 'Price Restrictions',
    'Affiliate IP License-Licensor': 'Affiliate License-Licensor',
    'Affiliate IP License-Licensee': 'Affiliate License-Licensee',
}


def normalized(value):
    return re.sub(r'[^a-z0-9]', '', value.lower())


def filename(value):
    return normalized(Path(value).stem)


def similarity(a, b):
    def words(text):
        return set(re.sub(r'[^a-z0-9\s]', '', text.lower()).split())
    a, b = words(a), words(b)
    return len(a & b) / len(a | b) if a | b else 0.


def json_metrics(value):
    """Represent undefined MOAR metrics as JSON null, preserving numeric scores."""
    if isinstance(value, dict):
        return {key: json_metrics(item) for key, item in value.items()}
    return None if isinstance(value, float) and not math.isfinite(value) else value


def diagnostics(candidates, gold, decisions, threshold=.5):
    """Diagnostic span coverage, not a reimplementation of official CUAD scoring."""
    matched, retained, false_positive, true_positive, kept_false, kept_true = 0, 0, 0, 0, 0, 0
    for label, spans in gold.items():
        proposals = [c for c in candidates if c['clause_type'] == label]
        for span in spans:
            hits = [c for c in proposals if similarity(span, c['text']) >= threshold]
            matched += bool(hits)
            retained += any(decisions.get(c['candidate_id']) == 'KEEP' for c in hits)
        for candidate in proposals:
            correct = any(similarity(span, candidate['text']) >= threshold for span in spans)
            kept = decisions.get(candidate['candidate_id']) == 'KEEP'
            true_positive += correct
            false_positive += not correct
            kept_true += correct and kept
            kept_false += not correct and kept
    total = sum(map(len, gold.values()))
    return {'gold_spans': total, 'gold_covered_by_candidates': matched, 'gold_covered_after_jev': retained,
            'candidate_recall': matched / total if total else None,
            'verified_recall': retained / total if total else None,
            'candidate_precision': true_positive / len(candidates) if candidates else None,
            'verified_precision': kept_true / (kept_true + kept_false) if kept_true + kept_false else None,
            'correct_candidates_removed': true_positive - kept_true,
            'false_candidates_removed': false_positive - kept_false,
            'unresolved_candidates': sum(v == 'NEEDS_CONTEXT' for v in decisions.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--ground-truth', type=Path, default=ROOT / 'experiments/reasoning/data/CUAD-master_clauses.csv')
    args = parser.parse_args()
    records = json.loads((args.run / 'input.json').read_text())
    schema = json.loads((args.run / 'protocol.json').read_text())['schema']
    with args.ground_truth.open(newline='') as f:
        rows = list(csv.DictReader(f))
    indexed = {filename(row['Filename']): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError('Ambiguous normalized ground-truth filenames')
    columns = {normalized(k): k for k in rows[0] if k != 'Filename' and 'answer' not in k.lower()}
    report = {'matching': 'Exact normalized filename/category; diagnostic word-set Jaccard >=0.5, '
                          'without official CUAD Parties exception. No ground truth used for inference.',
              'documents': []}
    for row in records:
        source = indexed[filename(row['name'])]  # No nearest-name guessing.
        gold = {label: ast.literal_eval(source[columns[normalized(ALIASES.get(label, label))]] or '[]')
                for label in schema}
        if any(not isinstance(v, list) or any(not isinstance(s, str) for s in v) for v in gold.values()):
            raise ValueError('Expected lists of gold source strings')
        folder = args.run / digest(str(row['id']))[:16]
        candidates = json.loads((folder / 'extraction.json').read_text())['candidates']
        verified = folder / 'verification.json'
        decisions = json.loads(verified.read_text())['decisions'] if verified.exists() else {}
        scores = diagnostics(candidates, gold, decisions)
        if not verified.exists():
            for key in list(scores):
                if key not in ('gold_spans', 'gold_covered_by_candidates', 'candidate_recall', 'candidate_precision'):
                    del scores[key]
        report['documents'].append({'id': row['id'], **scores})
    evaluator = load_module('_cuad_moar_evaluation', ROOT / 'experiments/reasoning/evaluation/cuad.py')
    report['moar'] = {}
    for name in ('gliner', 'gliner_jev'):
        path = args.run / (name + '.json')
        if path.exists():
            if len(json.loads(path.read_text())) != len(records):
                raise ValueError('Refusing to score an incomplete run')
            report['moar'][name] = json_metrics(evaluator.evaluate_results(
                name, str(path), str(args.ground_truth), str(args.run / 'input.json')))
    save(args.run / 'evaluation.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
