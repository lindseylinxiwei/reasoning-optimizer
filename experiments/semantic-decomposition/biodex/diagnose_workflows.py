"""Explain completed Biodex workflow losses from saved traces; never calls APIs.

Ground truth is read only for this post-run evaluation. Counts are unique
(normalized label, document) pairs, not document accuracy or macro RP@5.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from core import HERE, digest, save
from hierarchy import validate_hierarchy


HIERARCHICAL = ('full_hierarchy', 'evidence_hierarchy')
CATEGORIES = ('candidate_hits', 'routing_none_loss', 'beam_loss',
              'not_in_vocabulary', 'request_failure', 'execution_failure',
              'no_selected_evidence', 'unexplained')


def normalize(text):
    return text.strip().lower().replace("'", '').replace('^', '')


def read(path):
    return json.loads(Path(path).read_text())


def vocabulary_paths(tree):
    paths = {}

    def visit(node_id, path):
        path = path + [node_id]
        node = tree['nodes'][node_id]
        if node['children']:
            for child in node['children']:
                visit(child, path)
        else:
            for index in node['label_indices']:
                paths[index] = path
    visit(tree['root'], [])
    return paths


def first_routing_loss(path, search, tree):
    """Follow a GT label's path, including already completed leaves in the beam."""
    current = 0
    rounds = search.get('details', {}).get('rounds', [])
    for number, round_ in enumerate(rounds, 1):
        node_id = path[current]
        retained = {entry['node'] for entry in round_['retained']}
        if tree['nodes'][node_id]['children']:
            if current + 1 >= len(path):
                return {'category': 'unexplained', 'reason': 'invalid_ground_truth_path'}
            node_id = path[current + 1]
            answer = round_['answers'].get(node_id)
            if answer is None:
                return {'category': 'unexplained', 'reason': 'missing_expected_branch_answer',
                        'node': node_id, 'round': number}
            details = {'node': node_id, 'round': number, 'depth': current + 1,
                       'answer_choice': answer.get('choice'),
                       'probabilities': answer.get('probabilities', {})}
            if answer.get('choice') == 'none':
                return {'category': 'routing_none_loss', **details}
            if answer.get('choice') not in ('supported', 'uncertain'):
                return {'category': 'unexplained', 'reason': 'unknown_branch_choice', **details}
            if node_id not in retained:
                return {'category': 'beam_loss', 'completed_leaf': False, **details}
            current += 1
        elif node_id not in retained:
            # A leaf retained earlier remains eligible but can be displaced by
            # newly expanded branches at a later global-beam pruning step.
            return {'category': 'beam_loss', 'node': node_id, 'round': number,
                    'depth': current, 'completed_leaf': True}
    return {'category': 'unexplained', 'reason': 'survived_trace_but_missing_from_candidates',
            'node': path[current]}


def failed_request(stage_dirs):
    for folder in stage_dirs:
        for ledger in folder.glob('*_calls.jsonl'):
            for line in ledger.read_text().splitlines():
                call = json.loads(line)
                if call.get('status') == 'error' or call.get('http_status', 200) >= 400:
                    return {'ledger': str(ledger), 'http_status': call.get('http_status'),
                            'error_type': call.get('error_type')}
    return None


def evidence_summary(run, rows, ids):
    documents = []
    counts = Counter()
    for doc_id in ids:
        row = rows[doc_id]
        folder = run / 'evidence' / doc_id
        path = folder / 'evidence.json'
        item = {'id': doc_id, 'source_characters': len(row['evidence']),
                'expected_passages': len(row['passages'])}
        if not path.exists():
            if not (folder / 'error.json').exists():
                raise ValueError(f'Incomplete evidence stage: {doc_id}')
            item.update(status='error', classified_passages_saved=0,
                        error=read(folder / 'error.json'), failed_request=failed_request([folder]))
        else:
            evidence = read(path)
            answers = evidence['answers']
            if set(answers) != {f'P{i:04d}' for i in range(len(row['passages']))}:
                raise ValueError(f'Incomplete saved passage decisions: {doc_id}')
            retained = evidence['retained_source_characters']
            if retained != sum(b - a for a, b in evidence['source_spans']):
                raise ValueError(f'Inconsistent retained-character count: {doc_id}')
            counts.update(answer['choice'] for answer in answers.values())
            item.update(status='success', classified_passages_saved=len(answers),
                        selected_passages=len(evidence['selected_passage_indices']),
                        retained_source_characters=retained,
                        retained_source_fraction=retained / len(row['evidence']))
        documents.append(item)
    successful = [d for d in documents if d['status'] == 'success']
    source_characters = sum(d['source_characters'] for d in successful)
    retained_characters = sum(d['retained_source_characters'] for d in successful)
    return {'documents': len(documents), 'successful_documents': len(successful),
            'error_documents': len(documents) - len(successful),
            'zero_selected_documents': sum(d['selected_passages'] == 0 for d in successful),
            'expected_passages': sum(d['expected_passages'] for d in documents),
            'classified_passages_saved': sum(d['classified_passages_saved'] for d in documents),
            'selected_passages': sum(d['selected_passages'] for d in successful),
            'passage_choice_counts': dict(counts),
            'source_characters_successful_documents': source_characters,
            'retained_source_characters': retained_characters,
            'weighted_retained_source_fraction': retained_characters / source_characters if source_characters else None,
            'mean_document_retained_source_fraction': sum(d['retained_source_fraction'] for d in successful) / len(successful) if successful else None,
            'per_document': documents}


def diagnose(run, prepared_dir, tree_path):
    run, prepared_dir, tree_path = map(Path, (run, prepared_dir, tree_path))
    if not (run / 'summary.json').exists():
        raise ValueError('Run is incomplete: summary.json is required')
    summary, protocol = read(run / 'summary.json'), read(run / 'protocol.json')
    if set(summary) != set(protocol['methods']):
        raise ValueError('Run summary is incomplete')
    raw_prepared = (prepared_dir / 'prepared.json').read_text()
    if digest(raw_prepared) != protocol['prepared_sha256']:
        raise ValueError('Prepared artifact differs from experiment protocol')
    prepared = json.loads(raw_prepared)
    source_raw = Path(prepared['source']).read_text()
    if digest(source_raw) != prepared['source_sha256']:
        raise ValueError('Original evaluation source hash mismatch')
    originals = {str(r['id']): r for r in json.loads(source_raw)}
    rows = {r['id']: r for r in prepared['rows']}
    labels = [line.strip() for line in next(iter(originals.values()))['possible_labels'].splitlines() if line.strip()]
    norm_to_index = {normalize(label): i for i, label in enumerate(labels)}
    if len(norm_to_index) != len(labels):
        raise ValueError('Ambiguous normalized official vocabulary; explicit alias handling required')
    tree_raw = tree_path.read_text()
    tree = json.loads(tree_raw)
    if digest(tree_raw) != read(run / 'setup.json')['tree_sha256']:
        raise ValueError('Hierarchy differs from the experiment')
    if tree['vocabulary_sha256'] != digest(json.dumps(labels, ensure_ascii=False)):
        raise ValueError('Hierarchy vocabulary mismatch')
    validate_hierarchy(tree, len(labels))
    paths = vocabulary_paths(tree)
    ids = protocol['document_ids']
    if len(ids) != len(set(ids)) or any(i not in originals or i not in rows for i in ids):
        raise ValueError('Invalid protocol document IDs')
    method_results = {}
    # Require every method's complete final artifact before diagnosing any method.
    for method in protocol['methods']:
        results = read(run / method / 'results.json')
        if len(results) != len(ids) or {r['id'] for r in results} != set(ids):
            raise ValueError(f'Incomplete results: {method}')
        if any(r['status'] not in ('success', 'error') for r in results):
            raise ValueError(f'Incomplete result statuses: {method}')
        method_results[method] = results
    report = {'version': 1, 'run': str(run.resolve()),
              'note': 'Post-run evaluation only. Counts are unique normalized ground-truth label/document pairs; routing causes describe saved decisions, not why the model made them.',
              'methods': {}}
    if any(method != 'full_hierarchy' for method in protocol['methods']):
        report['evidence'] = evidence_summary(run, rows, ids)
    for method in HIERARCHICAL:
        if method not in method_results:
            continue
        counts = Counter({name: 0 for name in CATEGORIES})
        final_losses = Counter()
        details = []
        for result in method_results[method]:
            doc_id = result['id']
            folder = run / method / doc_id
            targets = sorted({normalize(t) for t in originals[doc_id]['ground_truth_reactions']})
            candidates = {normalize(label) for label in result.get('candidate_labels', [])}
            top5 = {normalize(label) for label in result['ranked_reaction_labels'][:5]}
            search = read(folder / 'search.json') if (folder / 'search.json').exists() else None
            verification = read(folder / 'verification.json') if (folder / 'verification.json').exists() else None
            failure = failed_request([folder] + ([run / 'evidence' / doc_id] if method != 'full_hierarchy' else []))
            for target in targets:
                item = {'id': doc_id, 'ground_truth_label': target}
                if target in candidates:
                    item['category'] = 'candidate_hits'
                    item['in_top5'] = target in top5
                    if target not in top5:
                        answer = (verification or {}).get('details', {}).get('answers', {}).get(f'L{norm_to_index[target]:05d}', {})
                        reason = ('verification_failure' if result['status'] == 'error' else
                                  'classified_not_supported' if answer.get('choice') == 'not_supported' else
                                  'supported_below_top5' if answer.get('choice') == 'supported' else 'unexplained')
                        item['final_loss'] = reason
                        final_losses[reason] += 1
                elif target not in norm_to_index:
                    item['category'] = 'not_in_vocabulary'
                elif search and search.get('details', {}).get('reason') == 'no_selected_evidence':
                    item['category'] = 'no_selected_evidence'
                elif search is not None:
                    item.update(first_routing_loss(paths[norm_to_index[target]], search, tree))
                    if item.get('node'):
                        node = tree['nodes'][item['node']]
                        shown = node['representative_indices'] if node['children'] else node['label_indices']
                        item['ground_truth_label_shown_at_lost_node'] = norm_to_index[target] in shown
                        item['lost_node_descendant_labels'] = len(node['label_indices'])
                        item['lost_node_label_examples'] = [labels[i] for i in shown[:8]]
                elif failure:
                    item.update(category='request_failure', failure=failure)
                elif result['status'] == 'error':
                    item.update(category='execution_failure', error_type=result.get('error_type'))
                else:
                    item.update(category='unexplained', reason='missing_search_artifact')
                counts[item['category']] += 1
                details.append(item)
        hits = counts['candidate_hits']
        total = sum(counts.values())
        examples = []
        for item in sorted(details, key=lambda d: (d.get('depth', 999), d['id'], d['ground_truth_label'])):
            if item['category'] == 'routing_none_loss' and item['id'] not in {d['id'] for d in examples}:
                examples.append(item)
                if len(examples) == 2:
                    break
        report['methods'][method] = {
            'documents': len(method_results[method]), 'label_document_pairs': total,
            **dict(counts), 'candidate_recall_micro': hits / total if total else None,
            'candidate_hits_in_top5': hits - sum(final_losses.values()),
            'final_verification_or_ranking_loss': sum(final_losses.values()),
            'final_loss_breakdown': dict(final_losses),
            'routing_none_by_depth': dict(Counter(str(d['depth']) for d in details if d['category'] == 'routing_none_loss')),
            'beam_by_depth': dict(Counter(str(d['depth']) for d in details if d['category'] == 'beam_loss')),
            'routing_none_when_label_shown': sum(d.get('ground_truth_label_shown_at_lost_node', False) for d in details if d['category'] == 'routing_none_loss'),
            'illustrative_saved_trace_examples': examples,
            'per_label_document': details,
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=HERE / 'runs/workflows-train-v1')
    parser.add_argument('--prepared', type=Path, default=HERE / 'runs/train-prepared')
    parser.add_argument('--tree', type=Path, default=HERE / 'cache/label_hierarchy.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = diagnose(args.run, args.prepared, args.tree)
    output = args.output or args.run / 'hierarchy_diagnostics.json'
    save(output, report)
    concise = {name: {k: v for k, v in item.items() if k != 'per_label_document'}
               for name, item in report['methods'].items()}
    if 'evidence' in report:
        concise['evidence'] = {k: v for k, v in report['evidence'].items() if k != 'per_document'}
    print(json.dumps(concise, indent=2))
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
