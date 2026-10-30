"""Evaluate the exhaustive scan and matched saved baselines without API calls."""
import json
from pathlib import Path

from core import HERE, ROOT, digest, load_module, save, support_probability
from full_vocabulary import assemble, normalized


def read(path):
    return json.loads(Path(path).read_text())


def main():
    run = HERE / 'runs/full-vocabulary-train5-v1'
    protocol, summary, results = (read(run / f'{name}.json') for name in ('protocol', 'summary', 'results'))
    ids = [d['id'] for d in protocol['documents']]
    if len(results) != len(ids) or {r['id'] for r in results} != set(ids) or summary['errors']:
        raise ValueError('Report requires complete successfully evaluated pilot documents')
    candidates = read(run / 'vocabulary.json')
    evaluate = load_module('full_vocab_report_eval', ROOT / 'experiments/reasoning/evaluation/biodex.py')
    score = evaluate.rank_precision_at_k
    prior_all = read(HERE / 'runs/train-k64/jev/results.json')
    prior = {r['id']: r for r in prior_all if r['id'] in ids}
    prepared = {r['id']: r for r in read(HERE / 'runs/train-prepared/prepared.json')['rows'] if r['id'] in ids}
    if set(prior) != set(ids) or set(prepared) != set(ids):
        raise ValueError('Unmatched historical baseline IDs')
    records, restricted = [], []
    for row in results:
        ident = row['id']
        target = {normalized(t) for t in row['ground_truth_reactions']}
        if target != {normalized(t) for t in prior[ident]['ground_truth_reactions']}:
            raise ValueError('Ground truth changed')
        answers = read(run / ident / 'all_answers.json')
        if set(answers) != {c['id'] for c in candidates}:
            raise ValueError('Not every official label was evaluated')
        old_ids = {c['id'] for c in prepared[ident]['candidates'][:64]}
        subset = [c for c in candidates if c['id'] in old_ids]
        selected, _ = assemble(subset, {c['id']: answers[c['id']] for c in subset})
        by_id = {c['id']: c['label'] for c in candidates}
        controlled = {'id': ident, 'ground_truth_reactions': row['ground_truth_reactions'],
                      'ranked_reaction_labels': [by_id[k] for k in selected]}
        restricted.append(controlled)
        d = row['diagnostics']
        old_labels = {normalized(c['label']) for c in subset}
        new_supported = [g for g in d['ground_truth_decisions']
                         if g.get('choice') == 'supported' and g['label'] not in old_labels]
        records.append({'id': ident, 'rp5': score(row['ranked_reaction_labels'], row['ground_truth_reactions'], 5),
                        'prior_rp5': score(prior[ident]['ranked_reaction_labels'], row['ground_truth_reactions'], 5),
                        'restricted_rp5': score(controlled['ranked_reaction_labels'], row['ground_truth_reactions'], 5),
                        'newly_reached_supported_gold': [g['label'] for g in new_supported],
                        'old_candidate_recall': len(target & old_labels) / len(target),
                        'old_candidate_oracle_rp5': min(5, len(target & old_labels)) / min(5, len(target)),
                        **d, **read(run / ident / 'metrics.json')})
    save(run / 'restricted-k64-results.json', restricted)
    all_methods = [('Prior retrieval + Jev, K=64 (historical)', [prior[i] for i in ids]),
                   ('Full vocabulary → Jev (new run)', results),
                   ('New answers restricted to prior K=64 (offline ablation)', restricted)]
    comparisons = {}
    for name, rows in all_methods:
        comparisons[name] = {f'rp{k}': sum(score(r['ranked_reaction_labels'], r['ground_truth_reactions'], k)
                                          for r in rows) / len(rows) for k in (5, 10)}
    means = {key: sum(r[key] for r in records) / len(records) for key in
             ('vocabulary_recall', 'supported_ground_truth_recall', 'vocabulary_oracle_rp5',
              'supported_oracle_rp5', 'old_candidate_recall', 'old_candidate_oracle_rp5')}
    novel_count = sum(len(r['newly_reached_supported_gold']) for r in records)
    segments = [json.loads(s) for s in (run / 'execution_segments.jsonl').read_text().splitlines()]
    wall = sum(s['wall_seconds'] for s in segments)
    comparison = {'methods': comparisons, 'mean_diagnostics': means, 'documents': records,
                  'newly_reached_supported_gold_pairs': novel_count, 'actual_recorded_process_wall_seconds': wall}
    save(run / 'comparison.json', comparison)
    previous, full, control = list(comparisons.values())
    outcome = 'improved' if full['rp5'] > previous['rp5'] else 'did not improve'
    lines = ['# Biodex: Jev over the full label vocabulary', '',
        f"On the first **{len(ids)} training papers**, removing retrieval **{outcome} RP@5**: "
        f"**{full['rp5']:.4f}**, versus **{previous['rp5']:.4f}** for the previous K=64 pipeline on these same papers. "
        f"All **{summary['labels_evaluated']:,} label/document decisions** completed. This is a small exploratory pilot, not a 100-document result.", '',
        '## Method', '',
        f"For each paper, Jev independently evaluates all **{len(candidates):,} official labels** against the **complete document**. "
        'There is no embedding retrieval, evidence filter, hierarchy, or label preselection. Each question asks whether one label is supported; '
        'many labels may be supported in the same request.', '',
        'The support question and model (`jev-1.13.0`) match the earlier Jev ranker. Requests contain up to 256 binary questions sharing one paper state, '
        'within conservative 24K state-plus-question and 48K whole-request token estimates. These are separate two-option questions, not a single winner among 256 labels. '
        'Each paper required 95 batches. Transient failures allow up to three total attempts, with every attempt logged.', '',
        'Code keeps supported labels, orders them by P(supported), and copies up to 10 official names. Equal probabilities are resolved by official-vocabulary order. '
        'No probabilities are multiplied or treated as a joint distribution. Ground truth is used only after inference.', '',
        'The first five papers in dataset order were fixed before calls. The protocol and every batch checkpoint are saved. '
        'The implementation reuses the existing question builder, transport, and MOAR evaluator.', '',
        '## Quality on matched papers', '',
        '| Method | MOAR RP@5 | MOAR RP@10 |', '|---|---:|---:|']
    for name, metrics in comparisons.items():
        lines.append(f"| {name} | {metrics['rp5']:.4f} | {metrics['rp10']:.4f} |")
    lines += ['', 'The offline ablation uses exactly the new full-scan answers, restricted to the original 64 labels. '
              'It uses vocabulary-order ties like the full scan and incurs no additional API cost. '
              'The historical baseline used smaller batches and retrieval-order ties; it is not a contemporaneous rerun.', '',
              '| Diagnostic (macro average across papers) | Value |', '|---|---:|',
              f"| Original K=64 candidate recall | {means['old_candidate_recall']:.4f} |",
              f"| Complete-vocabulary coverage of ground truth | {means['vocabulary_recall']:.4f} |",
              f"| Ground-truth recall after Jev support decisions | {means['supported_ground_truth_recall']:.4f} |",
              f"| Original candidate oracle RP@5 | {means['old_candidate_oracle_rp5']:.4f} |",
              f"| Complete-vocabulary oracle RP@5 | {means['vocabulary_oracle_rp5']:.4f} |",
              f"| Oracle RP@5 among Jev-supported labels | {means['supported_oracle_rp5']:.4f} |", '',
              f"Jev marked **{novel_count} ground-truth label/document pairs** supported that were absent from the old shortlist. "
              'This measures recovered support decisions, not necessarily recovered top-five outputs.', '',
              '| Paper | Prior RP@5 | Full-scan RP@5 | Supported labels | GT support recall | Supported oracle RP@5 | Labels tied at fifth score |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in records:
        lines.append(f"| {row['id']} | {row['prior_rp5']:.4f} | {row['rp5']:.4f} | {row['supported_labels']} | "
                     f"{row['supported_ground_truth_recall']:.4f} | {row['supported_oracle_rp5']:.4f} | {row['labels_tied_at_top5_cutoff']} |")
    lines += ['', 'RP@k is the number of matching ground-truth labels in the first k predictions divided by min(k, number of ground-truth labels), '
              'then averaged across papers. Support recall shows whether Jev recognizes the annotated labels; the supported-label oracle separates '
              'support decisions from final ranking. A large tie at the fifth score means deterministic tie-breaking can affect RP@5.', '',
              '## Cost and latency', '',
              f"New known API cost: **${summary['known_api_cost_usd']:.6f}**; **{summary['unknown_cost_calls']} unknown-cost calls**. "
              f"Requests: **{summary['api_calls']}**, including **{summary['retry_calls']} retries**. "
              f"Recorded process wall time: **{wall:.2f}s**. No embedding or Gemini API calls were made.", '',
              '| Stage | API cost USD | Wall time s |', '|---|---:|---:|',
              f"| Local request packing | $0 | {summary['packing_wall_seconds']:.2f} |",
              f"| Jev classification, including retry waits | ${summary['known_api_cost_usd']:.6f} | {summary['classification_wall_seconds']:.2f} |",
              f"| Local sorting, copying, and diagnostics | $0 | {summary['assembly_wall_seconds']:.2f} |", '',
              'Stage times exclude some startup, preflight estimation, and final reporting I/O; process wall time includes that work. '
              'Costs use provider-recorded input tokens at $0.042/M; local computation is counted as $0. Unknown failed-request charges are not treated as free. '
              '[Official Jev pricing and context limits](https://docs.typesafe.ai/models).', '',
              '## Per-label audit', '',
              '| Paper | Ground-truth label | P(supported) | Decision | Rank among supported labels |',
              '|---|---|---:|---|---:|']
    for row in records:
        for d in row['ground_truth_decisions']:
            label = d['label'].replace('|', '\\|')
            lines.append(f"| {row['id']} | {label} | {d.get('support_probability', '—')} | "
                         f"{d.get('choice', 'absent from vocabulary')} | {d.get('supported_rank') or '—'} |")
    min_supported = min(r['supported_labels'] for r in records)
    max_supported = max(r['supported_labels'] for r in records)
    lines += ['', '## Interpretation', '',
              f"Ground-truth support recall is **{means['supported_ground_truth_recall']:.1%}**, compared with "
              f"**{means['old_candidate_recall']:.1%}** candidate recall for the original shortlist. "
              f"The scan recovers {novel_count} annotated label/document pairs that the shortlist excluded. "
              'This directly tests the benefit of letting Jev see every label.', '',
              f"However, Jev supports **{min_supported}–{max_supported} labels per paper**. "
              f"Perfect ranking within that supported set could reach RP@5={means['supported_oracle_rp5']:.4f}, "
              f"while probability sorting reaches {full['rp5']:.4f}. "
              'High support recall alone does not establish high precision. Many supported labels are outside the annotations; '
              'source review would be needed to distinguish incorrect claims from valid but unannotated findings. '
              'The current support probabilities also do not reliably prioritize the benchmark labels, especially when many scores tie.', '',
              'This experiment removes retrieval exclusions for every official label. It can reveal three remaining limits: '
              'Jev rejecting an annotated label, Jev supporting many labels and ranking annotated ones too low, or an annotation not matching the supplied vocabulary. '
              'The resulting scores should not be generalized beyond these five papers, and exact annotation matching is not a source-reviewed assessment of every supported medical claim.', '',
              '[Protocol](runs/full-vocabulary-train5-v1/protocol.json) · [Predictions](runs/full-vocabulary-train5-v1/results.json) · '
              '[Metrics](runs/full-vocabulary-train5-v1/summary.json) · [Comparison data](runs/full-vocabulary-train5-v1/comparison.json) · '
              '[Runner](full_vocabulary.py)', '',
              'Each paper folder contains `all_answers.json` for all label probabilities, plus per-batch raw requests/responses, '
              'retry logs, cost, timing, and source hashes.', '']
    (HERE / 'biodex_jev_full_vocabulary.md').write_text('\n'.join(lines))
    print(HERE / 'biodex_jev_full_vocabulary.md')


if __name__ == '__main__':
    main()
