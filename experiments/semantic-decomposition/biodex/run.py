"""Prepare shared candidates once; run embedding-only, Jev, and Gemini ranking."""
import argparse
import json
import time
from pathlib import Path

from core import (ROOT, EMBEDDING_MODEL, GEMINI_MODEL, RATES, TASK, Embeddings,
                  GeminiRanker, JevRanker, accounting, digest, load_module, passages,
                  retrieve, save, shared_evidence, validate_ids)


def prepare(args):
    import tiktoken
    raw = args.input.read_text()
    data = json.loads(raw)
    if not data or len({str(r['id']) for r in data}) != len(data):
        raise ValueError('Require nonempty input with unique IDs')
    vocab = data[0]['possible_labels']
    labels = [line.strip() for line in vocab.splitlines() if line.strip()]
    if len(labels) != len(set(labels)) or any(r['possible_labels'] != vocab for r in data):
        raise ValueError('Require a shared unique official vocabulary')
    if not 1 <= args.max_candidates <= len(labels):
        raise ValueError('Invalid maximum candidate budget')
    encoder = tiktoken.get_encoding('cl100k_base')
    if any(len(encoder.encode(label, disallowed_special=())) > 8191 for label in labels):
        raise ValueError('A label exceeds the embedding input budget')
    args.output.mkdir(parents=True, exist_ok=False)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    ledger = args.output / 'embedding_calls.jsonl'
    embed = Embeddings(args.cache, ledger)
    started = time.perf_counter()
    rows = []
    try:
        label_vectors = embed(labels)
        for record in data:
            t = time.perf_counter()
            text = record['fulltext_processed']
            chunks = passages(text, encoder, args.passage_tokens, args.overlap_tokens)
            if not chunks:
                raise ValueError(f"Empty paper: {record['id']}")
            candidates = retrieve(labels, label_vectors, chunks, embed([c['text'] for c in chunks]), args.max_candidates)
            evidence, mode = shared_evidence(text, chunks, candidates, encoder)
            rows.append({'id': str(record['id']), 'document_sha256': digest(text),
                'candidates': candidates, 'evidence': evidence, 'evidence_sha256': digest(evidence),
                'evidence_mode': mode, 'passages': chunks, 'retrieval_wall_seconds': time.perf_counter() - t})
            save(args.output / 'partial.json', rows)
            print(f"Prepared {len(rows)}/{len(data)}: {record['id']}", flush=True)
        # This artifact deliberately excludes ground truth.
        save(args.output / 'prepared.json', {'version': 1, 'source': str(args.input.resolve()),
            'source_sha256': digest(raw), 'embedding_model': EMBEDDING_MODEL,
            'passage_tokens': args.passage_tokens, 'overlap_tokens': args.overlap_tokens,
            'max_candidates': args.max_candidates, 'score': 'max passage cosine; vocabulary-order tie break',
            'rows': rows})
        (args.output / 'partial.json').unlink()
    finally:
        save(args.output / 'retrieval_metrics.json', {**accounting(ledger),
            'wall_seconds': time.perf_counter() - started, 'cache_hits': embed.cache_hits,
            'new_vectors': embed.new_vectors, 'rates': RATES,
            'note': 'Observed preparation cost/time; warm cache excludes historical embedding creation charges.'})
        embed.close()


def run(args):
    prepared_path = args.prepared / 'prepared.json'
    raw = prepared_path.read_text()
    prepared = json.loads(raw)
    if not 1 <= args.k <= prepared['max_candidates'] or not 1 <= args.output_limit <= args.k:
        raise ValueError('Require 1 <= output-limit <= k <= prepared candidate budget')
    source = Path(prepared['source']).read_text()
    if digest(source) != prepared['source_sha256']:
        raise ValueError('Source changed after preparation; refusing mismatched evaluation')
    originals = {str(r['id']): r for r in json.loads(source)}
    rows = prepared['rows']
    if len(rows) != len(originals) or {r['id'] for r in rows} != set(originals):
        raise ValueError('Prepared set differs from source documents')
    for row in rows:
        if digest(row['evidence']) != row['evidence_sha256'] or digest(originals[row['id']]['fulltext_processed']) != row['document_sha256']:
            raise ValueError('Evidence/source hash mismatch')
        validate_ids([c['id'] for c in row['candidates']], row['candidates'], prepared['max_candidates'])
    evaluator = load_module('biodex_evaluation', ROOT / 'experiments/reasoning/evaluation/biodex.py')
    args.output.mkdir(parents=True, exist_ok=False)
    shared = json.loads((args.prepared / 'retrieval_metrics.json').read_text())
    methods = ['embedding', 'jev', 'gemini'] if args.method == 'all' else [args.method]
    save(args.output / 'config.json', {'prepared_sha256': digest(raw), 'prepared_path': str(prepared_path.resolve()),
        'k': args.k, 'output_limit': args.output_limit, 'methods': methods, 'task': TASK,
        'models': {'embedding': EMBEDDING_MODEL, 'jev': 'jev-1.13.0', 'gemini': getattr(args, 'gemini_model', GEMINI_MODEL)},
        'pricing_assumptions': {**RATES, **({'gemini_input_per_million': 2., 'gemini_cached_per_million': .20, 'gemini_output_per_million': 12.} if getattr(args, 'gemini_model', GEMINI_MODEL) == 'gemini-3.1-pro-preview' else {})},
        'gemini_generation': {'thinking': getattr(args, 'thinking', 'minimal'), 'max_output_tokens': getattr(args, 'max_output_tokens', 2048), 'temperature': 0}, 'shared_retrieval': shared,
        'execution': 'Sequential; no retries or silent fallbacks. Local compute priced at zero.'})
    reports = {}
    for method in methods:
        folder = args.output / method
        folder.mkdir()
        ledger = folder / 'calls.jsonl'
        started = time.perf_counter()
        results = []
        ranker = None
        initialization_error = None
        try:
            if method == 'jev':
                ranker = JevRanker(ledger)
            elif method == 'gemini':
                ranker = GeminiRanker(ledger, model=args.gemini_model, thinking=args.thinking, max_output_tokens=args.max_output_tokens) if hasattr(args, 'gemini_model') else GeminiRanker(ledger)
        except Exception as exc:
            initialization_error = type(exc).__name__ + ': missing/invalid provider setup'
        for row in rows:
            t = time.perf_counter()
            candidates = row['candidates'][:args.k]
            result = {'id': row['id'], 'ranked_reaction_labels': [],
                'ground_truth_reactions': originals[row['id']]['ground_truth_reactions'],
                'evidence_sha256': row['evidence_sha256'],
                'candidate_ids': [c['id'] for c in candidates]}
            try:
                if initialization_error:
                    raise RuntimeError(initialization_error)
                ids, details = (ranker(row['evidence'], candidates, args.output_limit) if ranker else
                    ([c['id'] for c in candidates[:args.output_limit]], {}))
                validate_ids(ids, candidates, args.output_limit)
                by_id = {c['id']: c['label'] for c in candidates}
                result.update(status='success', ranked_ids=ids,
                    ranked_reaction_labels=[by_id[i] for i in ids], details=details)
            except Exception as exc:
                # Never copy arbitrary exception strings containing credentials/provider URLs.
                result.update(status='error', error_type=type(exc).__name__)
            result['wall_seconds'] = time.perf_counter() - t
            results.append(result)
            save(folder / 'results.json', results)
            print(f"{method} {row['id']}: {result['status']}", flush=True)
        metrics = {**accounting(ledger), 'wall_seconds': time.perf_counter() - started,
            'errors': sum(r['status'] == 'error' for r in results), 'documents': len(results),
            'initialization_error': initialization_error}
        # Shared work is charged once to each hypothetical standalone method, not re-executed.
        metrics['standalone_known_api_cost_usd'] = shared['known_api_cost_usd'] + metrics['known_api_cost_usd']
        metrics['standalone_unknown_cost_calls'] = shared['unknown_cost_calls'] + metrics['unknown_cost_calls']
        metrics['composed_wall_seconds'] = shared['wall_seconds'] + metrics['wall_seconds']
        metrics['evaluation'] = evaluator.evaluate_results(method, str(folder / 'results.json'))
        save(folder / 'metrics.json', metrics)
        reports[method] = metrics
    norm = lambda s: s.strip().lower().replace("'", '').replace('^', '')
    recalls, ceilings = [], []
    for row in rows:
        target = {norm(t) for t in originals[row['id']]['ground_truth_reactions']}
        if target:
            hits = len(target & {norm(c['label']) for c in row['candidates'][:args.k]})
            recalls.append(hits / len(target))
            ceilings.append(min(5, args.output_limit, hits) / min(5, len(target)))
    summary = {'candidate_recall': sum(recalls) / len(recalls) if recalls else None,
        'oracle_rp5': sum(ceilings) / len(ceilings) if ceilings else None,
        'labeled_documents': len(recalls), 'shared_retrieval': shared, 'methods': reports}
    save(args.output / 'summary.json', summary)
    lines = ['# Biodex shared-retrieval comparison', '',
        f"Candidates: {args.k}. Output limit: {args.output_limit}. Same cached candidates and evidence for every method.", '',
        f"Candidate label recall: {summary['candidate_recall']}. Candidate oracle RP@5: {summary['oracle_rp5']}.", '',
        '| Method | RP@5 | Errors | Ranking API USD | Ranking wall s | Shared + ranking USD | Shared + ranking s |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, m in reports.items():
        lines.append(f"| {name} | {m['evaluation']['avg_rp_at_5']:.4f} | {m['errors']} | {m['known_api_cost_usd']:.6f} | {m['wall_seconds']:.3f} | {m['standalone_known_api_cost_usd']:.6f} | {m['composed_wall_seconds']:.3f} |")
    lines += ['', f"Shared retrieval: ${shared['known_api_cost_usd']:.6f}, {shared['wall_seconds']:.3f}s; cache hits: {shared['cache_hits']}. Historical warm-cache charges are excluded.",
        'Combined times are composed from the shared preparation plus sequential ranking, not fresh end-to-end reruns.',
        'Costs are explicit token-price estimates; local compute is $0. Unknown-cost calls remain unknown (see metrics).',
        'Error outputs are empty and count as misses. Jev probabilities are a ranking heuristic, not calibrated cross-question probabilities.',
        'Embedding-only always returns the top labels; classifiers may abstain. The comparison measures the added selection/ranking stage.']
    (args.output / 'report.md').write_text('\n'.join(lines) + '\n')
    if any(m['errors'] for m in reports.values()):
        raise SystemExit(1)


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--cache', type=Path, default=Path(__file__).parent / 'cache/embeddings.sqlite')
    p.add_argument('--max-candidates', type=int, default=128)
    p.add_argument('--passage-tokens', type=int, default=256)
    p.add_argument('--overlap-tokens', type=int, default=64)
    p.set_defaults(func=prepare)
    p = sub.add_parser('run')
    p.add_argument('--prepared', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--method', choices=['all', 'embedding', 'jev', 'gemini'], default='all')
    p.add_argument('--gemini-model', choices=[GEMINI_MODEL, 'gemini-3.1-pro-preview'], default=GEMINI_MODEL)
    p.add_argument('--thinking', choices=['minimal', 'low', 'medium', 'high'], default='minimal')
    p.add_argument('--max-output-tokens', type=int, default=2048)
    p.add_argument('--k', required=True, type=int)
    p.add_argument('--output-limit', type=int, default=10)
    p.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
