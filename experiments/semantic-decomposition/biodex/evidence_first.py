"""Select source evidence before exhaustive label scanning and iterative ranking."""
import argparse
import json
import time
from pathlib import Path

from core import HERE, ROOT, digest, label_questions, load_module, save, validate_ids
from full_vocabulary import assemble, label_diagnostics
from iterative_evidence import copy_units, coverage, rank_labels, run_stage, select_evidence


def validate_evidence(text, result):
    """Only reuse evidence whose decisions and copied spans reproduce this source."""
    if result['source_sha256'] != digest(text):
        raise ValueError('Cached evidence belongs to a different source')
    copied = copy_units(text, result['units'], result['answers'])
    if any(result[key] != value for key, value in copied.items()):
        raise ValueError('Cached evidence does not reproduce its source spans')
    if result['source_characters'] != len(text):
        raise ValueError('Cached source length changed')
    return result


def scan_labels(evidence, candidates, decide, batch_size=256):
    """Same binary questions as the full-paper scan; only its state changes."""
    if not evidence.strip():
        return {'answers': {c['id']: {'choice': 'not_supported',
            'probabilities': {'supported': 0., 'not_supported': 1.}} for c in candidates},
            'supported_ids': [], 'probability_ranked_ids': [], 'empty_evidence': True}
    questions = list(label_questions(candidates).items())
    answers = {}
    for start in range(0, len(questions), batch_size):
        answers.update(decide('Medical paper:\n' + evidence, dict(questions[start:start + batch_size])))
        batch = start // batch_size + 1
        if batch == 1 or batch % 20 == 0 or start + batch_size >= len(questions):
            print(f'  label scan: {len(answers)}/{len(candidates)} decisions', flush=True)
    probability_ids, _ = assemble(candidates, answers)
    return {'answers': answers,
            'supported_ids': [c['id'] for c in candidates if answers[c['id']]['choice'] == 'supported'],
            'probability_ranked_ids': probability_ids, 'empty_evidence': False}


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'experiments/reasoning/data/train/biodex.json')
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--evidence-run', type=Path, help='Reuse exact evidence from an earlier iterative_evidence run')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    raw = args.input.read_text()
    data = json.loads(raw)
    if not 1 <= args.limit <= len(data):
        raise ValueError('Requested document count is unavailable')
    rows = data[:args.limit]
    if len({str(r['id']) for r in rows}) != len(rows):
        raise ValueError('Duplicate document IDs')
    vocabulary = rows[0]['possible_labels']
    if any(r['possible_labels'] != vocabulary for r in rows):
        raise ValueError('Inconsistent vocabulary')
    labels = [s.strip() for s in vocabulary.splitlines() if s.strip()]
    if len(labels) != len(set(labels)):
        raise ValueError('Duplicate official labels')
    candidates = [{'id': f'L{i:05d}', 'label': label} for i, label in enumerate(labels)]
    by_id = {c['id']: c['label'] for c in candidates}
    reused, evidence_hashes = {}, {}
    if args.evidence_run:
        for row in rows:
            key = str(row['id'])
            base = args.evidence_run / key / 'evidence'
            evidence = validate_evidence(row['fulltext_processed'], json.loads((base / 'result.json').read_text()))
            metrics = json.loads((base / 'metrics.json').read_text())
            reused[key] = (evidence, metrics)
            evidence_hashes[key] = digest(json.dumps({'result': evidence, 'metrics': metrics}, ensure_ascii=False))
    dependencies = [Path(__file__), HERE / 'iterative_evidence.py', HERE / 'full_vocabulary.py',
                    HERE / 'core.py', HERE.parent / 'company-extraction/pipeline.py']
    protocol = {'version': 1, 'source': str(args.input.resolve()), 'source_sha256': digest(raw),
                'documents': [str(r['id']) for r in rows],
                'selection': 'First N input documents in source order, fixed before calls',
                'implementation_sha256': {p.name: digest(p.read_text()) for p in dependencies},
                'evidence_run': str(args.evidence_run.resolve()) if args.evidence_run else None,
                'reused_evidence_sha256': evidence_hashes,
                'method': 'Evidence selection -> exhaustive short-context scan -> iterative selection',
                'model': 'jev-1.13.0', 'price_per_million_input_usd': .042,
                'labels_per_document': len(labels), 'scan_questions_per_request': 256,
                'scan_questions_sha256': digest(json.dumps(label_questions(candidates), ensure_ascii=False)),
                'scan_context': 'Copied evidence only; no full-paper fallback during scanning',
                'ranking_context': 'Copied evidence; MORE_CONTEXT can request the original paper',
                'ranking_group_size': 253, 'ranking_limit': 10,
                'accounting': 'Reused evidence excluded from new spend, included once in standalone totals',
                'ground_truth': 'Used only after all model decisions for evaluation'}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Refusing changed experiment resume')
    save(path, protocol)
    save(args.output / 'vocabulary.json', candidates)
    results, metrics_by_id, evidence_stats = [], {}, []
    for row in rows:
        key, text = str(row['id']), row['fulltext_processed']
        folder = args.output / key
        folder.mkdir(parents=True, exist_ok=True)
        if key in reused:
            evidence, em = reused[key]
            (folder / 'evidence').mkdir(exist_ok=True)
            save(folder / 'evidence/result.json', evidence)
            save(folder / 'evidence/metrics.json', em)
        else:
            evidence, em = run_stage(folder / 'evidence', lambda decide: select_evidence(text, decide))
        validate_evidence(text, evidence)
        print(f'{key}: evidence ready, scanning all {len(labels)} labels', flush=True)
        scan, sm = run_stage(folder / 'scan', lambda decide: scan_labels(evidence['evidence'], candidates, decide))
        supported = set(scan['supported_ids'])
        choices = [c for c in candidates if c['id'] in supported]
        ranking, rm = run_stage(folder / 'ranking', lambda decide: rank_labels(
            text, evidence['evidence'], choices, decide, allow_context=True))
        ids = validate_ids(ranking['ranked_ids'], choices, 10)
        # Evaluation is deliberately downstream of both scan and ranking.
        result = {'id': key, 'ground_truth_reactions': row['ground_truth_reactions'],
                  'ranked_ids': ids, 'ranked_reaction_labels': [by_id[i] for i in ids],
                  'supported_labels': len(choices), 'scan_diagnostics': label_diagnostics(
                      candidates, scan['answers'], row['ground_truth_reactions']),
                  'survivor_count': len(ranking['survivor_ids']),
                  'survivor_coverage': coverage(ranking['survivor_ids'], choices, row['ground_truth_reactions']),
                  'context_fallbacks': sum(len(r['requested_full_context']) for r in
                                           ranking['group_trace'] + ranking['final_trace']),
                  'empty_evidence': scan['empty_evidence']}
        save(folder / 'result.json', result)
        results.append(result)
        metrics_by_id[key] = {'evidence': em, 'scan': sm, 'ranking': rm}
        evidence_stats.append({'id': key, 'source_characters': len(text),
                               'retained_characters': evidence['retained_source_characters'],
                               'units': len(evidence['units']), 'reused': key in reused})
        save(args.output / 'results.json', results)
        save(args.output / 'metrics.json', metrics_by_id)
        print(f'{key}: {len(choices)} supported labels, {len(ids)} outputs, '
              f'new scan+ranking ${sm["known_api_cost_usd"]+rm["known_api_cost_usd"]:.6f}', flush=True)
    evaluator = load_module('evidence_first_evaluator', ROOT / 'experiments/reasoning/evaluation/biodex.py')
    summary = evaluator.evaluate_results('evidence_first', str(args.output / 'results.json'))
    summary['stages'] = {stage: {k: sum(m[stage][k] for m in metrics_by_id.values()) for k in
        ('known_api_cost_usd', 'unknown_cost_calls', 'api_calls', 'api_wall_seconds',
         'classification_wall_seconds', 'composed_seconds', 'retry_calls')}
        for stage in ('evidence', 'scan', 'ranking')}
    summary['standalone_cost_usd'] = sum(s['known_api_cost_usd'] for s in summary['stages'].values())
    summary['standalone_composed_seconds'] = sum(s['composed_seconds'] for s in summary['stages'].values())
    summary['new_known_cost_usd'] = sum(m[s]['known_api_cost_usd'] for key, m in metrics_by_id.items()
                                      for s in ('evidence', 'scan', 'ranking') if s != 'evidence' or key not in reused)
    summary['new_unknown_cost_calls'] = sum(m[s]['unknown_cost_calls'] for key, m in metrics_by_id.items()
                                          for s in ('evidence', 'scan', 'ranking') if s != 'evidence' or key not in reused)
    summary['supported_ground_truth_recall'] = sum(r['scan_diagnostics']['supported_ground_truth_recall']
                                                  for r in results) / len(results)
    summary['supported_oracle_rp5'] = sum(r['scan_diagnostics']['supported_oracle_rp5'] for r in results) / len(results)
    summary['process_wall_seconds'] = time.perf_counter() - started
    summary['evidence'] = evidence_stats
    save(args.output / 'summary.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('evidence', 'term_recall_scores')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
