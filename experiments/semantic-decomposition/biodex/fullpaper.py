"""Exhaustive full-paper Jev scan followed by iterative selection, with stage timing."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
from pathlib import Path

import numpy as np
from core import HERE, ROOT, JevRanker, digest, label_questions, load_module, save, validate_ids
from evidence_first import scan_labels
from full_vocabulary import label_diagnostics
from iterative_evidence import coverage, rank_labels, run_stage


def run_phase(rows, folder, stage, workers, operation):
    """Each document owns its client and ledger. Measure the whole phase's wall time."""
    started = time.perf_counter()
    results, metrics, errors = {}, {}, {}
    def process(row):
        key = str(row['id'])
        return run_stage(folder / key / stage, lambda decide: operation(row, decide))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = {pool.submit(process, row): str(row['id']) for row in rows}
        for task in as_completed(tasks):
            key = tasks[task]
            try:
                results[key], metrics[key] = task.result()
                print(f'{stage}: completed {len(results)}/{len(rows)} documents; {key}; '
                      f'${metrics[key]["known_api_cost_usd"]:.6f}', flush=True)
            except Exception as exc:
                errors[key] = {'error_type': type(exc).__name__, 'message': str(exc)}
                print(f'{stage}: {key} failed: {type(exc).__name__}', flush=True)
            save(folder / (stage + '_progress.json'), {'completed': list(results), 'errors': errors})
    phase = {'process_wall_seconds': time.perf_counter() - started,
             'completed_documents': len(results), 'errors': errors, 'workers': workers}
    save(folder / (stage + '_phase.json'), phase)
    if errors:
        raise RuntimeError(f'{stage}: {len(errors)} documents unresolved; see phase errors and request ledgers')
    return results, metrics, phase


def summarize_stage(metrics, phase):
    result = {key: sum(m[key] for m in metrics.values()) for key in
              ('known_api_cost_usd', 'unknown_cost_calls', 'api_calls', 'api_wall_seconds', 'retry_calls')}
    times = [m['composed_seconds'] for m in metrics.values()]
    result.update(phase_wall_seconds=phase['process_wall_seconds'],
                  sum_document_seconds=sum(times), mean_document_seconds=float(np.mean(times)),
                  median_document_seconds=float(np.median(times)), p95_document_seconds=float(np.percentile(times, 95)))
    return result


def preflight(rows, candidates, client):
    """Estimate the exact batching logic without making API calls."""
    questions = label_questions(candidates)
    sizes = [client._tokens(json.dumps({key: value}, ensure_ascii=False)) for key, value in questions.items()]
    plans = []
    for row in rows:
        state_tokens = client._tokens('Medical paper:\n' + row['fulltext_processed'])
        if state_tokens + max(sizes) > min(client.max_estimated_tokens, client.max_request_tokens):
            raise ValueError(f'Full paper {row["id"]} exceeds the existing conservative token guard')
        batches, estimate = 0, 0
        for start in range(0, len(sizes), 256):
            total, count = state_tokens, 0
            for size in sizes[start:start + 256]:
                if count and total + size > client.max_request_tokens:
                    estimate += total
                    batches += 1
                    total, count = state_tokens, 0
                total += size
                count += 1
            if count:
                estimate += total
                batches += 1
        plans.append({'id': str(row['id']), 'document_sha256': digest(row['fulltext_processed']),
                      'state_estimated_tokens': state_tokens, 'scan_calls': batches,
                      'estimated_scan_tokens': estimate, 'estimated_scan_cost_usd': estimate * .042 / 1e6})
    return plans


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'experiments/reasoning/data/test/biodex.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError('workers must be positive')
    started = time.perf_counter()
    raw = args.input.read_text()
    rows = json.loads(raw)
    if not rows or len({str(r['id']) for r in rows}) != len(rows):
        raise ValueError('Expected a nonempty dataset with unique IDs')
    vocabulary = rows[0]['possible_labels']
    if any(r['possible_labels'] != vocabulary for r in rows):
        raise ValueError('Vocabulary differs between documents')
    labels = [s.strip() for s in vocabulary.splitlines() if s.strip()]
    if len(labels) != len(set(labels)):
        raise ValueError('Duplicate official labels')
    candidates = [{'id': f'L{i:05d}', 'label': label} for i, label in enumerate(labels)]
    plans = preflight(rows, candidates, JevRanker(args.output / 'unused.jsonl').client)
    print(json.dumps({'documents': len(rows), 'labels': len(candidates), 'workers': args.workers,
                      'scan_calls': sum(p['scan_calls'] for p in plans),
                      'raw_estimated_scan_cost_usd': sum(p['estimated_scan_cost_usd'] for p in plans)}, indent=2), flush=True)
    if args.plan_only:
        return
    dependencies = [Path(__file__), HERE / 'core.py', HERE / 'evidence_first.py',
                    HERE / 'iterative_evidence.py', HERE / 'full_vocabulary.py',
                    HERE.parent / 'company-extraction/pipeline.py']
    protocol = {'version': 1, 'source': str(args.input.resolve()), 'source_sha256': digest(raw),
                'documents': plans, 'selection': 'Every document in the input file; no sampling',
                'method': 'Full-paper exhaustive scan -> iterative full-paper label selection',
                'implementation_sha256': {p.name: digest(p.read_text()) for p in dependencies},
                'model': 'jev-1.13.0', 'price_per_million_input_usd': .042,
                'vocabulary_sha256': digest(vocabulary), 'labels_per_document': len(candidates),
                'scan_questions_sha256': digest(json.dumps(label_questions(candidates), ensure_ascii=False)),
                'scan_questions_per_request': 256, 'ranking_group_size': 253, 'output_limit': 10,
                'full_paper_in_both_stages': True, 'evidence_filter': False, 'retrieval': False,
                'allow_context_request': False, 'workers': args.workers,
                'concurrency': 'Across documents; independent groups share requests, ranking rounds are sequential',
                'latency': 'Fresh phase wall time and document service times recorded separately',
                'retry_policy': 'Up to three attempts per request for transient failures; all charges logged',
                'evaluation': 'Unchanged MOAR evaluator; ground truth used only after predictions'}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Refusing changed protocol resume')
    save(path, protocol)
    save(args.output / 'vocabulary.json', candidates)
    scan, sm, sp = run_phase(rows, args.output, 'scan', args.workers,
        lambda row, decide: scan_labels(row['fulltext_processed'], candidates, decide))
    choice_map = {}
    for key, value in scan.items():
        supported = set(value['supported_ids'])
        choice_map[key] = [c for c in candidates if c['id'] in supported]
    ranking, rm, rp = run_phase(rows, args.output, 'ranking', args.workers,
        lambda row, decide: rank_labels(row['fulltext_processed'], row['fulltext_processed'],
                                       choice_map[str(row['id'])], decide, allow_context=False))
    by_id = {c['id']: c['label'] for c in candidates}
    results, probability_results = [], []
    for row in rows:
        key = str(row['id'])
        ids = validate_ids(ranking[key]['ranked_ids'], choice_map[key], 10)
        result = {'id': key, 'ground_truth_reactions': row['ground_truth_reactions'],
                  'ranked_ids': ids, 'ranked_reaction_labels': [by_id[i] for i in ids],
                  'supported_labels': len(choice_map[key]),
                  'scan_diagnostics': label_diagnostics(candidates, scan[key]['answers'], row['ground_truth_reactions']),
                  'survivor_count': len(ranking[key]['survivor_ids']),
                  'survivor_coverage': coverage(ranking[key]['survivor_ids'], choice_map[key], row['ground_truth_reactions']),
                  'status': 'success'}
        save(args.output / key / 'result.json', result)
        results.append(result)
        probability_results.append({'id': key, 'ground_truth_reactions': row['ground_truth_reactions'],
            'ranked_reaction_labels': [by_id[i] for i in scan[key]['probability_ranked_ids']]})
    save(args.output / 'results.json', results)
    save(args.output / 'probability_results.json', probability_results)
    evaluator = load_module('fullpaper_evaluator', ROOT / 'experiments/reasoning/evaluation/biodex.py')
    summary = evaluator.evaluate_results('iterative_jev_full_paper', str(args.output / 'results.json'))
    summary['probability_sort_baseline'] = evaluator.evaluate_results('same_scan_probability_sort', str(args.output / 'probability_results.json'))
    summary['stages'] = {'scan': summarize_stage(sm, sp), 'ranking': summarize_stage(rm, rp)}
    summary['known_api_cost_usd'] = sum(s['known_api_cost_usd'] for s in summary['stages'].values())
    summary['unknown_cost_calls'] = sum(s['unknown_cost_calls'] for s in summary['stages'].values())
    summary['process_wall_seconds'] = time.perf_counter() - started
    summary['sum_phase_wall_seconds'] = sp['process_wall_seconds'] + rp['process_wall_seconds']
    doc_times = [sm[str(r['id'])]['composed_seconds'] + rm[str(r['id'])]['composed_seconds'] for r in rows]
    summary['document_service_seconds'] = {'mean': float(np.mean(doc_times)), 'median': float(np.median(doc_times)),
                                         'p95': float(np.percentile(doc_times, 95)), 'sum': sum(doc_times)}
    summary['supported_ground_truth_recall'] = float(np.mean([r['scan_diagnostics']['supported_ground_truth_recall'] for r in results]))
    summary['supported_oracle_rp5'] = float(np.mean([r['scan_diagnostics']['supported_oracle_rp5'] for r in results]))
    summary['survivor_ground_truth_recall'] = float(np.mean([r['survivor_coverage']['recall'] for r in results]))
    summary['survivor_oracle_rp5'] = float(np.mean([r['survivor_coverage']['oracle_rp5'] for r in results]))
    summary['expected_scan_decisions'] = len(rows) * len(candidates)
    summary['errors'] = 0
    save(args.output / 'summary.json', summary)
    with (args.output / 'execution_segments.jsonl').open('a') as f:
        f.write(json.dumps({'process_wall_seconds': summary['process_wall_seconds'], 'completed': True}) + '\n')
    print(json.dumps({k:v for k,v in summary.items() if not isinstance(v, list) and k != 'probability_sort_baseline'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
