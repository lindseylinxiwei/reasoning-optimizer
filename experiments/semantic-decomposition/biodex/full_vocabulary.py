"""Exhaustive document × label decisions with durable per-request checkpoints."""
import argparse
import json
import time
from pathlib import Path

from core import (HERE, ROOT, JevRanker, accounting, digest, label_questions,
                  load_module, save, support_probability, validate_ids)

TRANSIENT_HTTP = {408, 429, 500, 502, 503, 504, 520, 522, 524}


def normalized(value):
    return value.strip().lower().replace("'", '').replace('^', '')


def assemble(candidates, answers, limit=10):
    """Stable official-vocabulary order resolves equal support probabilities."""
    if set(answers) != {c['id'] for c in candidates}:
        raise ValueError('Full answer coverage is required before producing predictions')
    scored = [(c, support_probability(answers[c['id']])) for c in candidates]
    scored.sort(key=lambda pair: -pair[1])
    selected = [c['id'] for c, _ in scored if answers[c['id']]['choice'] == 'supported'][:limit]
    validate_ids(selected, candidates, limit)
    return selected, scored


def request_batch(client, text, state, questions, folder, max_attempts=3,
                  answer_validator=support_probability):
    """Resume successful requests; retry only transient failures, preserving all charges."""
    folder.mkdir(parents=True, exist_ok=True)
    request_hash = digest(json.dumps({'state': state, 'questions': questions}, ensure_ascii=False))
    saved = folder / 'answers.json'
    if saved.exists():
        result = json.loads(saved.read_text())
        if result['request_sha256'] != request_hash:
            raise ValueError('Checkpoint request changed')
        return result['answers']
    for attempt in range(1, max_attempts + 1):
        ledger = folder / f'attempt-{attempt:02d}.jsonl'
        metadata = folder / f'attempt-{attempt:02d}.json'
        if ledger.exists() and ledger.stat().st_size:
            record = json.loads(ledger.read_text().splitlines()[-1])
            if digest(json.dumps({'state': record['request']['state'],
                                  'questions': record['request']['questions']}, ensure_ascii=False)) != request_hash:
                raise ValueError('Interrupted request changed')
            if record.get('status') == 'success':
                answers = record['response']['answers']
                save(saved, {'request_sha256': request_hash, 'answers': answers})
                return answers
            transient = record.get('http_status') in TRANSIENT_HTTP or record.get('error_type') in (
                'Timeout', 'ReadTimeout', 'ConnectTimeout', 'ConnectionError')
            if not transient:
                raise RuntimeError('Previous non-transient batch failure')
            continue
        client.ledger_path = ledger
        started = time.perf_counter()
        error = None
        try:
            response = client._request(text, state, questions, 1, attempt)
            for answer in response['answers'].values():
                answer_validator(answer)
            save(saved, {'request_sha256': request_hash, 'answers': response['answers']})
            return response['answers']
        except Exception as exc:
            error = type(exc).__name__
            record = (json.loads(ledger.read_text().splitlines()[-1])
                      if ledger.exists() and ledger.stat().st_size else {})
            transient = record.get('http_status') in TRANSIENT_HTTP or error in (
                'Timeout', 'ReadTimeout', 'ConnectTimeout', 'ConnectionError')
            if not transient or attempt == max_attempts:
                raise
            # This pause is included in attempt wall time and never hides a retry charge.
            time.sleep(2 ** attempt)
        finally:
            save(metadata, {'wall_seconds': time.perf_counter() - started, 'error_type': error})
    raise RuntimeError('Transient retry budget exhausted')


def call_metrics(folder):
    ledgers = sorted(folder.glob('batch-*/attempt-*.jsonl'))
    rows = [accounting(p) for p in ledgers]
    result = {key: sum(r[key] for r in rows) for key in
              ('known_api_cost_usd', 'unknown_cost_calls', 'api_calls', 'api_wall_seconds')}
    elapsed = 0.
    for path in ledgers:
        metadata = path.with_suffix('.json')
        elapsed += (json.loads(metadata.read_text())['wall_seconds'] if metadata.exists()
                    else accounting(path)['api_wall_seconds'])
    result['classification_wall_seconds'] = elapsed
    result['retry_calls'] = sum(int(p.stem.split('-')[-1]) > 1 for p in ledgers)
    return result


def label_diagnostics(candidates, answers, targets):
    """Ground truth is used only after all model decisions, for evaluation."""
    _, ranked = assemble(candidates, answers)
    by_label = {normalized(c['label']): c for c in candidates}
    supported = [(c, score) for c, score in ranked if answers[c['id']]['choice'] == 'supported']
    ranks = {c['id']: i + 1 for i, (c, _) in enumerate(supported)}
    details = []
    for target in sorted({normalized(t) for t in targets}):
        c = by_label.get(target)
        if c is None:
            details.append({'label': target, 'in_vocabulary': False})
            continue
        answer = answers[c['id']]
        details.append({'label': target, 'id': c['id'], 'in_vocabulary': True,
                        'choice': answer['choice'], 'support_probability': support_probability(answer),
                        'supported_rank': ranks.get(c['id'])})
    n = len(details)
    hits = sum(d.get('choice') == 'supported' for d in details)
    available = sum(d['in_vocabulary'] for d in details)
    cutoff = supported[4][1] if len(supported) >= 5 else None
    return {'ground_truth_decisions': details, 'supported_labels': len(supported),
            'vocabulary_recall': available / n if n else None,
            'supported_ground_truth_recall': hits / n if n else None,
            'vocabulary_oracle_rp5': min(5, available) / min(5, n) if n else None,
            'supported_oracle_rp5': min(5, hits) / min(5, n) if n else None,
            'top5_cutoff_probability': cutoff,
            'labels_tied_at_top5_cutoff': sum(score == cutoff for _, score in supported) if cutoff is not None else 0}


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'experiments/reasoning/data/train/biodex.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if args.limit < 1 or args.batch_size < 1 or args.max_attempts < 1:
        raise ValueError('Limits must be positive')
    wall_start = time.perf_counter()
    raw = args.input.read_text()
    data = json.loads(raw)
    selected = data[:args.limit]
    if len({str(r['id']) for r in selected}) != len(selected):
        raise ValueError('Duplicate document IDs')
    vocabulary = data[0]['possible_labels']
    if any(r['possible_labels'] != vocabulary for r in selected):
        raise ValueError('Inconsistent vocabulary')
    labels = [s.strip() for s in vocabulary.splitlines() if s.strip()]
    if len(labels) != len(set(labels)):
        raise ValueError('Duplicate official labels')
    candidates = [{'id': f'L{i:05d}', 'label': label} for i, label in enumerate(labels)]
    questions = label_questions(candidates)
    client = JevRanker(args.output / 'unused.jsonl').client
    groups_by_id, preparation, plans = {}, {}, []
    for row in selected:
        started = time.perf_counter()
        state = 'Medical paper:\n' + row['fulltext_processed']
        items = list(questions.items())
        groups = []
        for i in range(0, len(items), args.batch_size):
            groups.extend(client._pack(state, dict(items[i:i + args.batch_size])))
        keys = [k for group in groups for k in group]
        if keys != list(questions):
            raise ValueError('Batching lost or duplicated vocabulary entries')
        document_id = str(row['id'])
        groups_by_id[document_id] = groups
        preparation[document_id] = time.perf_counter() - started
        estimated = sum(client._tokens(state) + sum(client._tokens(json.dumps({k: v}, ensure_ascii=False))
                                                    for k, v in group.items()) for group in groups)
        plans.append({'id': document_id, 'document_sha256': digest(row['fulltext_processed']),
                      'labels': len(labels), 'batches': len(groups),
                      'estimated_input_tokens': estimated, 'estimated_api_usd': estimated * .042 / 1e6})
    protocol = {'version': 1, 'source': str(args.input.resolve()), 'source_sha256': digest(raw),
                'implementation_sha256': digest(Path(__file__).read_text()),
                'questions_sha256': digest(json.dumps(questions, ensure_ascii=False)),
                'document_selection': 'First N training documents in source order, fixed before calls',
                'documents': plans, 'model': 'jev-1.13.0', 'price_usd_per_million_input': .042,
                'max_questions_per_request': args.batch_size, 'max_attempts': args.max_attempts,
                'context_guards_cl100k': {'state_plus_question': 24000, 'whole_request': 48000},
                'ranking': 'Supported choices only, descending P(supported), ties by official vocabulary order',
                'output_limit': 10, 'no_retrieval': True, 'full_document': True,
                'retries': 'At most 3 total attempts by default for transient HTTP/network failures only; all logged'}
    print(json.dumps({'documents': plans, 'planned_calls': sum(p['batches'] for p in plans),
                      'estimated_api_usd': sum(p['estimated_api_usd'] for p in plans)}, indent=2), flush=True)
    if args.plan_only:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Refusing to resume a changed experiment')
    save(protocol_path, protocol)
    save(args.output / 'vocabulary.json', candidates)
    results, stages = [], []
    for row in selected:
        document_id = str(row['id'])
        folder = args.output / document_id
        folder.mkdir(exist_ok=True)
        if (folder / 'result.json').exists():
            results.append(json.loads((folder / 'result.json').read_text()))
            stages.append(json.loads((folder / 'metrics.json').read_text()))
            continue
        text = row['fulltext_processed']
        answers, failures = {}, []
        groups = groups_by_id[document_id]
        for index, group in enumerate(groups, 1):
            try:
                batch_answers = request_batch(client, text, 'Medical paper:\n' + text, group,
                    folder / f'batch-{index:04d}', args.max_attempts)
                if set(batch_answers) != set(group):
                    raise ValueError('Batch answer IDs mismatch')
                answers.update(batch_answers)
            except Exception as exc:
                failures.append({'batch': index, 'error_type': type(exc).__name__, 'labels': list(group)})
            if index == 1 or index % 10 == 0 or index == len(groups):
                print(f'{document_id}: {index}/{len(groups)} batches, {len(answers)}/{len(labels)} labels, '
                      f'{len(failures)} unresolved batches', flush=True)
        assembly_start = time.perf_counter()
        save(folder / 'all_answers.json', answers)
        result = {'id': document_id, 'ground_truth_reactions': row['ground_truth_reactions'],
                  'ranked_reaction_labels': [], 'ranked_ids': [], 'labels_evaluated': len(answers),
                  'failures': failures, 'status': 'error' if failures else 'success'}
        if not failures:
            ids, _ = assemble(candidates, answers)
            by_id = {c['id']: c['label'] for c in candidates}
            result.update(ranked_ids=ids, ranked_reaction_labels=[by_id[i] for i in ids],
                          diagnostics=label_diagnostics(candidates, answers, row['ground_truth_reactions']))
        metrics = {**call_metrics(folder), 'packing_wall_seconds': preparation[document_id],
                   'assembly_wall_seconds': time.perf_counter() - assembly_start}
        metrics['composed_wall_seconds'] = sum(metrics[k] for k in
            ('packing_wall_seconds', 'classification_wall_seconds', 'assembly_wall_seconds'))
        save(folder / 'metrics.json', metrics)
        save(folder / 'result.json', result)
        results.append(result)
        stages.append(metrics)
        save(args.output / 'results.json', results)
    evaluator = load_module('full_vocab_evaluator', ROOT / 'experiments/reasoning/evaluation/biodex.py')
    save(args.output / 'results.json', results)
    metrics = evaluator.evaluate_results('jev_full_vocabulary', str(args.output / 'results.json'))
    for key in stages[0]:
        metrics[key] = sum(s[key] for s in stages)
    metrics.update(errors=sum(r['status'] == 'error' for r in results),
                   labels_evaluated=sum(r['labels_evaluated'] for r in results),
                   expected_label_decisions=len(labels) * len(selected))
    save(args.output / 'summary.json', metrics)
    with (args.output / 'execution_segments.jsonl').open('a') as f:
        f.write(json.dumps({'wall_seconds': time.perf_counter() - wall_start,
                            'completed': True, 'note': 'Process wall time includes local preparation and output I/O'}) + '\n')
    print(json.dumps(metrics, indent=2), flush=True)
    if metrics['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
