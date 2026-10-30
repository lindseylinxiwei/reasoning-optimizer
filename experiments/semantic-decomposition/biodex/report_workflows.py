"""Generate a concise, artifact-grounded report; never makes model/API calls."""
import json
from pathlib import Path

import numpy as np

from core import HERE, ROOT, load_module


NAMES = {
    'evidence_retrieval': 'Evidence → dense retrieval → Jev',
    'full_hierarchy': 'Full paper → hierarchy → Jev',
    'evidence_hierarchy': 'Evidence → hierarchy → Jev',
}
BOOTSTRAP_SEED = 20260929
BOOTSTRAP_SAMPLES = 10000


def read(path):
    return json.loads(Path(path).read_text())


def paired_difference(current, baseline):
    """Pair documents by ID, use the unchanged MOAR metric, then resample pairs."""
    evaluator = load_module('workflow_report_evaluator',
                            ROOT / 'experiments/reasoning/evaluation/biodex.py')
    left = {str(r['id']): r for r in current}
    right = {str(r['id']): r for r in baseline}
    if len(left) != len(current) or len(right) != len(baseline) or set(left) != set(right):
        raise ValueError('Paired comparison requires unique, identical document IDs')
    differences = []
    for ident in sorted(left):
        a, b = left[ident], right[ident]
        normalize = lambda values: {v.strip().lower().replace("'", '').replace('^', '') for v in values}
        if normalize(a['ground_truth_reactions']) != normalize(b['ground_truth_reactions']):
            raise ValueError('Paired ground truth changed')
        if not a['ground_truth_reactions']:
            continue
        score = evaluator.rank_precision_at_k
        differences.append(score(a['ranked_reaction_labels'], a['ground_truth_reactions'], 5)
                           - score(b['ranked_reaction_labels'], b['ground_truth_reactions'], 5))
    values = np.asarray(differences, dtype=float)
    if not len(values):
        raise ValueError('No labeled pairs')
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = values[rng.integers(len(values), size=(BOOTSTRAP_SAMPLES, len(values)))].mean(axis=1)
    low, high = np.quantile(sampled, [.025, .975])
    return {'documents': len(values), 'difference': float(values.mean()),
            'ci_low': float(low), 'ci_high': float(high)}


def unique_ledger_accounting(run):
    """Count shared evidence once: sum actual ledger files, not method totals."""
    rows = []
    for path in sorted(run.rglob('*_calls.jsonl')):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return {'known': sum(row.get('cost_usd') or 0 for row in rows),
            'unknown': sum(row.get('cost_usd') is None for row in rows), 'calls': len(rows)}


def method_table(summary):
    lines = [
        '| Workflow | RP@5 | RP@10 | Candidate recall | Oracle RP@5 | Mean candidates | API USD | Composed s | Errors / unknown-cost calls |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for method, m in summary.items():
        lines.append(f"| {NAMES[method]} | {m['avg_rp_at_5']:.4f} | {m['avg_rp_at_10']:.4f} | "
                     f"{m['candidate_recall']:.4f} | {m['oracle_rp5']:.4f} | {m['mean_candidates']:.1f} | "
                     f"${m['known_api_cost_usd']:.6f} | {m['composed_wall_seconds']:.2f} | "
                     f"{m['errors']} / {m['unknown_cost_calls']} |")
    return lines


def stage_table(summary):
    lines = [
        '| Workflow | Evidence USD / s | Search USD / s | Verification USD / s |',
        '|---|---:|---:|---:|',
    ]
    for method, m in summary.items():
        parts = []
        for name in ('evidence', 'search', 'verification'):
            s = m['stages'].get(name)
            parts.append(f"${s['known_api_cost_usd']:.6f} / {s['wall_seconds']:.2f}" if s else '—')
        lines.append(f"| {NAMES[method]} | " + ' | '.join(parts) + ' |')
    return lines


def split_section(split, run, summary):
    baseline_dir = HERE / 'runs' / f'{split}-k64'
    baseline = read(baseline_dir / 'jev/metrics.json')
    prior = read(baseline_dir / 'summary.json')
    baseline_rows = read(baseline_dir / 'jev/results.json')
    b_eval = baseline['evaluation']
    count = len(read(run / 'protocol.json')['document_ids'])
    if len(baseline_rows) != count:
        raise ValueError('Expected complete matched split')
    title = f'Training comparison ({count} documents)' if split == 'train' else f'Test follow-up ({count} documents)'
    lines = [f'## {title}', '', *method_table(summary), '',
             'Costs and times above are the measured cached-stage execution; historical embeddings and one-time setup are excluded.', '',
             f"Prior full-paper dense retrieval + Jev at K=64 on this same split: **RP@5={b_eval['avg_rp_at_5']:.4f}**, "
             f"RP@10={b_eval['avg_rp_at_10']:.4f}; candidate recall={prior['candidate_recall']:.4f}, "
             f"oracle RP@5={prior['oracle_rp5']:.4f}.", '',
             '| Workflow vs prior Jev baseline | Paired ΔRP@5 | 95% bootstrap interval |',
             '|---|---:|---:|']
    comparisons = {}
    for method, m in summary.items():
        rows = read(run / method / 'results.json')
        comparison = comparisons[method] = paired_difference(rows, baseline_rows)
        if not np.isclose(b_eval['avg_rp_at_5'] + comparison['difference'], m['avg_rp_at_5']):
            raise ValueError('Saved evaluation differs from paired recomputation')
        lines.append(f"| {NAMES[method]} | {comparison['difference']:+.4f} | "
                     f"[{comparison['ci_low']:+.4f}, {comparison['ci_high']:+.4f}] |")
    lines.extend(['', 'Stage contributions:', '', *stage_table(summary), ''])
    prep = prior['shared_retrieval']
    lines.extend([
        f"The prior baseline's Jev ranking alone cost ${baseline['known_api_cost_usd']:.6f} / "
        f"{baseline['wall_seconds']:.2f}s. Its original shared preparation cost ${prep['known_api_cost_usd']:.6f} / "
        f"{prep['wall_seconds']:.2f}s. These are separate historical measurements, not a matched fresh-run timing comparison.", '',
    ])
    account = unique_ledger_accounting(run)
    lines.extend([
        f"Actual new API spend for this split: **${account['known']:.6f} known**, "
        f"{account['unknown']} unknown-cost calls, {account['calls']} total calls. Shared evidence is counted once here.", '',
        f"[Summary](runs/{run.name}/summary.json) · [Protocol](runs/{run.name}/protocol.json) · "
        f"[Setup](runs/{run.name}/setup.json)", '',
    ])
    for method in summary:
        lines.append(f"- [{NAMES[method]} outputs](runs/{run.name}/{method}/results.json)")
    lines.append('')
    return lines, comparisons, account


def routing_diagnostics(run, summary):
    path = run / 'hierarchy_diagnostics.json'
    if not path.exists():
        return []
    diagnostic = read(path)
    methods = diagnostic['methods']
    lines = ['## Why the hierarchy failed in this run', '',
             'The post-run trace audit locates where correct labels disappeared. Counts below are unique '
             '**ground-truth label/document pairs** (micro counts), not documents or the macro candidate recall above. '
             'Each pair has one pre-verification outcome.', '',
             '| Workflow | Total pairs | Reached candidates | Rejected as none | Pruned by beam | API failure | Absent from vocabulary |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for method, d in methods.items():
        lines.append(f"| {NAMES[method]} | {d['label_document_pairs']} | {d['candidate_hits']} | "
                     f"{d['routing_none_loss']} | {d['beam_loss']} | {d['request_failure']} | {d['not_in_vocabulary']} |")
    full = methods['full_hierarchy']
    evidence = diagnostic['evidence']
    lines.extend(['',
        f"For full-paper routing, {full['routing_none_by_depth'].get('1', 0)} pairs were already rejected at the first level. "
        f"For both hierarchy variants, **none of the rejected groups explicitly displayed the missed ground-truth label** "
        'among its examples. The full-paper candidate oracle equals its final RP@5, so final verification cannot repair this loss.', '',
    ])
    examples = full.get('illustrative_saved_trace_examples', [])
    example = next((e for e in examples if e['ground_truth_label'] == 'death'), None)
    if example:
        lines.extend([
            f"For example, paper {example['id']}'s label **death** belonged to a {example['lost_node_descendant_labels']:,}-label group "
            'represented by examples such as “Device psychogenic complication,” “Ainhum,” and “Military service.” '
            'The router chose none before ever seeing “death” as a candidate.', '',
        ])
    lines.extend([
        '**This cluster-description design failed:** a few representative labels did not communicate the contents of large, mixed groups. '
        'The traces support routing failure; they do not prove Jev cannot classify biomedical labels or that all label hierarchies fail.', '',
        f"Evidence selection retained **{evidence['weighted_retained_source_fraction']:.1%} of source characters** across "
        f"{evidence['successful_documents']} successful documents (pooled character ratio). It therefore removed relatively little text.", '',
    ])
    failed = [r for r in evidence.get('per_document', []) if r.get('failed_request')]
    if failed:
        failures = ', '.join(f"{r['id']} (HTTP {r['failed_request'].get('http_status', 'unknown')})" for r in failed)
        lines.extend([
            f"Shared evidence extraction failed for {failures}. This one request affects both evidence workflows, "
            'counts as a miss in each, and has one unknown charge in actual spend. No retry was made. '
            'Even a perfect result on that paper could not close the large gap between the selected retrieval workflow and either hierarchy workflow.', '',
        ])
    lines.extend([f'[Post-run routing audit](runs/{run.name}/hierarchy_diagnostics.json)', ''])
    return lines


def main():
    train = HERE / 'runs/workflows-train-v1'
    test = HERE / 'runs/workflows-test-v1'
    training = read(train / 'summary.json')
    if set(training) != set(NAMES):
        raise ValueError('Require all three completed training workflows')
    selected = min(training, key=lambda name: (-training[name]['avg_rp_at_5'],
                                              training[name]['known_api_cost_usd'],
                                              training[name]['composed_wall_seconds']))
    setup = read(train / 'setup.json')
    tree = read(HERE / 'cache/label_hierarchy.json')
    leaves = [node for node in tree['nodes'].values() if not node['children']]
    config = read(train / 'protocol.json')
    winner = training[selected]
    baseline_train = read(HERE / 'runs/train-k64/jev/metrics.json')['evaluation']['avg_rp_at_5']
    selection_path = HERE / 'runs/workflows-selection-v1.json'
    if selection_path.exists() and read(selection_path)['selected_method'] != selected:
        raise ValueError('Frozen selection disagrees with training rule')
    interpretation = (f"**None of the three new workflows improves training RP@5 over the prior retrieval + Jev baseline "
                      f"({baseline_train:.4f}).** " if winner['avg_rp_at_5'] <= baseline_train else '')
    lines = [
        '# Biodex: three Jev workflows', '',
        interpretation + f"Among the new workflows on the 40-document training split, **{NAMES[selected]}** has the highest RP@5 "
        f"({winner['avg_rp_at_5']:.4f}) and is selected by the prewritten rule: highest training RP@5, "
        'then lower API cost, then lower composed latency. This is a selection result, not an unbiased test estimate.', '',
        '[Frozen workflow selection](runs/workflows-selection-v1.json)', '',
        '## Methods', '',
        'All three methods use Jev 1.13.0, return up to 10 official labels, and use the unchanged MOAR evaluator. '
        'No document answer is generated as free text; code copies source evidence and official label names.', '',
        '1. **Evidence → dense retrieval → Jev:** Jev marks patient-relevant or uncertain passages; code copies them. '
        'The existing cached passage embeddings retrieve 64 labels. Jev verifies and ranks these labels.',
        '2. **Full paper → hierarchy → Jev:** Jev explores relevant label groups using the complete paper, then verifies labels in the reached leaves.',
        '3. **Evidence → hierarchy → Jev:** use the same copied evidence as method 1, then the same hierarchical search and verification as method 2.', '',
        'Evidence questions retain symptoms, diagnoses, medical history, laboratory/genetic findings, treatment failure and drug-use issues. '
        'Overlapping source spans are merged; an empty selection gives an empty prediction. No dense label filter is applied after hierarchical search.', '',
        f"The tree clusters all {setup['labels']:,} official labels using cached MOAR `text-embedding-3-small` vectors: "
        f"**{len(tree['nodes'])} nodes, {len(leaves)} leaves, at most {max(len(n['label_indices']) for n in leaves)} labels per leaf**. "
        f"It uses up to {config['hierarchy']['branching']} children, {config['hierarchy']['representatives']} representative labels per internal node, "
        f"and a global beam of {config['beam']}. This is an embedding-cluster search tree, not a medical ontology.", '',
        'Jev independently marks groups supported, uncertain or none. Non-none paths compete by the geometric mean of '
        '`P(supported) + 0.5 × P(uncertain)`; finished leaves also occupy beam slots. Final label questions are binary, '
        'and supported labels are ranked by P(supported). Both ranking scores are uncalibrated heuristics. '
        'Branch pruning can still miss labels, and the workflows use different candidate budgets.', '',
    ]
    section, train_ci, train_account = split_section('train', train, training)
    lines.extend(section)
    lines.extend(routing_diagnostics(train, training))
    all_accounts = [train_account]
    stats = {'seed': BOOTSTRAP_SEED, 'resamples': BOOTSTRAP_SAMPLES,
             'selected_method': selected, 'train': train_ci}
    test_summary = test / 'summary.json'
    if test_summary.exists():
        testing = read(test_summary)
        if set(testing) != {selected}:
            raise ValueError('Test report must contain only the training-selected workflow')
        section, test_ci, test_account = split_section('test', test, testing)
        lines.extend(section)
        all_accounts.append(test_account)
        stats['test'] = test_ci
        t = testing[selected]
        base_test = read(HERE / 'runs/test-k64/jev/metrics.json')['evaluation']['avg_rp_at_5']
        lines[2] += f" On the 100-document test follow-up it reaches **{t['avg_rp_at_5']:.4f}**, versus **{base_test:.4f}** for prior retrieval + Jev."
    else:
        lines.extend(['The training-selected workflow has no completed 100-document follow-up artifact yet.', ''])
    actual_known = sum(a['known'] for a in all_accounts)
    actual_unknown = sum(a['unknown'] for a in all_accounts)
    lines.extend([
        '## Interpretation and accounting', '',
        'RP@k is the number of correct labels in the first k outputs divided by min(k, number of ground-truth labels), '
        'averaged across labeled papers. Candidate recall measures ground-truth label coverage before verification; '
        'oracle RP@5 is the ceiling from that candidate set. Errors produce empty predictions and count as misses.', '',
        f"Paired percentile-bootstrap intervals use the same document IDs and ground truth, {BOOTSTRAP_SAMPLES:,} resamples, "
        f"seed {BOOTSTRAP_SEED}. They describe variation across these documents, not repeated model runs; training comparisons "
        'are subject to selection and the intervals are not adjusted for multiple comparisons. They do not establish equivalence. '
        'Earlier test results informed this follow-up, so even the separate test follow-up is exploratory.', '',
        'Shared evidence extraction ran once per split and was reused by both evidence workflows. Each standalone workflow total '
        'includes its evidence contribution; actual session spend counts those calls only once. '
        'Local computation is $0 by convention. Jev cost is recorded input usage × $0.042 per million tokens; these are estimates, not invoices. '
        'Unknown charges are not treated as free.', '',
        'These are warm-cache workflow measurements: historical label and passage embedding creation charges are excluded. '
        'Evidence + dense retrieval depends on cached document-passage embeddings; a fresh deployment must also create them. '
        'The hierarchy methods depend on cached label embeddings but need no document embedding API calls. '
        'The earlier training preparation was cold-cache, so its total cannot be used to claim a workflow speedup.', '',
        f"Training setup was recorded separately: **{setup['wall_seconds']:.2f}s**, ${setup['known_api_cost_usd']:.6f} API cost. "
        'It covers cache/tree loading or construction performed by the runner; any earlier tree prebuild is not included unless separately recorded. '
        'The same tree is reused for test. Setup is excluded from method tables.', '',
        'Stage times are sequential wall-clock contributions summed into composed totals. Shared-stage reuse means these are not '
        'three fresh end-to-end runs, and loading/reporting/setup overhead is excluded. Interruptions, if present, may leave a marked '
        'API-time lower bound; completed artifacts are resumed without automatic API retries.', '',
        f"Total new API spend in the completed experiments above: **${actual_known:.6f} known**, with **{actual_unknown} unknown-cost calls**.", '',
        '[Implementation](workflows.py) · [Tree builder](hierarchy.py) · [Previous experiments](biodex_jev.md) · '
        '[Paired statistics](biodex_jev_workflows_stats.json)', '',
    ])
    destination = HERE / 'biodex_jev_workflows.md'
    if len(lines) > 150:
        raise ValueError('Report exceeded requested length')
    destination.write_text('\n'.join(lines))
    (HERE / 'biodex_jev_workflows_stats.json').write_text(json.dumps(stats, indent=2) + '\n')
    print(destination)


if __name__ == '__main__':
    main()
