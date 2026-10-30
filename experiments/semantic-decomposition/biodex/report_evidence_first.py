"""Compare evidence-first pilot with the two saved iterative-ranking variants."""
import argparse
import json
from pathlib import Path

from core import HERE, ROOT, load_module, save
from full_vocabulary import normalized


def read(path):
    return json.loads(path.read_text())


def mean(values):
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=HERE / 'runs/evidence-first-train5-v1')
    parser.add_argument('--baseline', type=Path, default=HERE / 'runs/iterative-evidence-train5-v1')
    parser.add_argument('--output', type=Path, default=HERE / 'biodex_jev_evidence_first.md')
    args = parser.parse_args()
    run, baseline = args.run, args.baseline
    protocol, summary = read(run / 'protocol.json'), read(run / 'summary.json')
    old = read(baseline / 'summary.json')
    scan_path = Path(read(baseline / 'protocol.json')['scan'])
    old_scan = read(scan_path / 'summary.json')
    old_scan_rows = {r['id']: r for r in read(scan_path / 'results.json')}
    old_rows = {m: {r['id']: r for r in read(baseline / (m + '_results.json'))}
                for m in ('full_context', 'selected_evidence')}
    new_rows = {r['id']: r for r in read(run / 'results.json')}
    ids = protocol['documents']
    assert set(ids) == set(new_rows) == set(old_rows['full_context']) == set(old_scan_rows)
    evaluator = load_module('evidence_first_report_evaluator', ROOT / 'experiments/reasoning/evaluation/biodex.py')
    metrics = evaluator.rank_precision_at_k
    comparison = []
    for key in ids:
        new = new_rows[key]
        before = {normalized(d['label']): d for d in old_scan_rows[key]['diagnostics']['ground_truth_decisions']}
        after = {normalized(d['label']): d for d in new['scan_diagnostics']['ground_truth_decisions']}
        assert set(before) == set(after)
        lost = [label for label in before if before[label].get('choice') == 'supported'
                and after[label].get('choice') != 'supported']
        gained = [label for label in before if before[label].get('choice') != 'supported'
                  and after[label].get('choice') == 'supported']
        comparison.append({'id': key, 'newly_rejected_ground_truth': lost, 'newly_supported_ground_truth': gained,
            'previous_support_recall': old_scan_rows[key]['diagnostics']['supported_ground_truth_recall'],
            'new_support_recall': new['scan_diagnostics']['supported_ground_truth_recall'],
            'previous_rp5': metrics(old_rows['selected_evidence'][key]['ranked_reaction_labels'], new['ground_truth_reactions'], 5),
            'new_rp5': metrics(new['ranked_reaction_labels'], new['ground_truth_reactions'], 5)})
    save(run / 'comparison.json', comparison)
    cost_full = old_scan['known_api_cost_usd'] + old['stages']['full_context']['known_api_cost_usd']
    cost_late = old_scan['known_api_cost_usd'] + old['stages']['evidence']['known_api_cost_usd'] + old['stages']['selected_evidence']['known_api_cost_usd']
    time_full = old_scan['composed_wall_seconds'] + old['stages']['full_context']['composed_seconds']
    time_late = old_scan['composed_wall_seconds'] + old['stages']['evidence']['composed_seconds'] + old['stages']['selected_evidence']['composed_seconds']
    stages = summary['stages']
    reduction = 1 - summary['standalone_cost_usd'] / cost_full
    old_recall = mean([r['previous_support_recall'] for r in comparison])
    retained = sum(e['retained_characters'] for e in summary['evidence']) / sum(e['source_characters'] for e in summary['evidence'])
    lines = ['# Biodex: select evidence before scanning labels', '',
        f'On the same five-paper pilot, selecting evidence before the exhaustive label scan gives '
        f'**RP@5 = {summary["avg_rp_at_5"]:.4f}** and **RP@10 = {summary["avg_rp_at_10"]:.4f}**. '
        f'The composed standalone API cost is **${summary["standalone_cost_usd"]:.6f}**, '
        f'{abs(reduction):.1%} {"lower" if reduction >= 0 else "higher"} than scanning and ranking with full papers.', '',
        '## Method', '',
        '1. Jev classifies every numbered sentence, heading, or delimited table row as a patient finding, '
        'necessary context, background, or uncertain. Code copies findings, context, and uncertain units with associated headers from exact source offsets.',
        '2. Use only those copied excerpts as the state for all 24,313 independent supported/not-supported label questions. '
        'The scan prompts, vocabulary order, 256-question batching, and model are unchanged. No embeddings or label retrieval are used.',
        '3. Keep the supported labels and use the same iterative Jev ranking: choose the best remaining label, remove it, '
        'and repeat up to ten times. For pools above 253 labels, retain up to ten per group and rank the combined survivors. '
        'The ranking stage can request the original paper via MORE_CONTEXT; the scan cannot. Code copies official label names.', '',
        'This run reuses exactly the evidence from the earlier pilot. It therefore changes where that evidence enters the pipeline, '
        'without changing the evidence-selection decisions. All evidence source hashes and copied spans were checked. '
        'Ground truth is used only after model decisions.', '',
        '## Comparison on the same five papers', '',
        '| Method | RP@5 | RP@10 | Standalone API cost | Composed stage time |', '|---|---:|---:|---:|---:|',
        f'| Full-paper scan → full-paper iterative ranking | {old["methods"]["full_context"]["avg_rp_at_5"]:.4f} | '
        f'{old["methods"]["full_context"]["avg_rp_at_10"]:.4f} | ${cost_full:.6f} | {time_full:.2f}s |',
        f'| Full-paper scan → evidence selection → iterative ranking | {old["methods"]["selected_evidence"]["avg_rp_at_5"]:.4f} | '
        f'{old["methods"]["selected_evidence"]["avg_rp_at_10"]:.4f} | ${cost_late:.6f} | {time_late:.2f}s |',
        f'| **Evidence selection → short-text scan → iterative ranking** | **{summary["avg_rp_at_5"]:.4f}** | '
        f'**{summary["avg_rp_at_10"]:.4f}** | **${summary["standalone_cost_usd"]:.6f}** | **{summary["standalone_composed_seconds"]:.2f}s** |', '',
        'Costs include every required stage once. Timings combine separately measured stages, including historical reused work; '
        'they are not fresh end-to-end measurements. API service latency varies between runs. '
        'RP@k uses the unchanged MOAR evaluator: correct labels among the first k divided by min(k, number of ground-truth labels), averaged across documents.', '',
        '## Stage costs and latency', '',
        '| Stage | API cost | Stage wall time | API calls |', '|---|---:|---:|---:|']
    for key, title in [('evidence','Evidence selection + copying (reused)'),('scan','Scan all labels using short text'),('ranking','Iterative ranking, including any context fallback')]:
        s = stages[key]
        lines.append(f'| {title} | ${s["known_api_cost_usd"]:.6f} | {s["composed_seconds"]:.2f}s | {s["api_calls"]} |')
    lines += ['', f'**New API spend this run: ${summary["new_known_cost_usd"]:.6f}.** '
        f'New process wall time: **{summary["process_wall_seconds"]:.2f}s**. Evidence was reused and was not charged again, '
        'but its cost and original duration are included in the standalone comparison.', '',
        f'Unknown-cost new calls: {summary["new_unknown_cost_calls"]}. '
        f'New retries: {stages["scan"]["retry_calls"] + stages["ranking"]["retry_calls"]}. '
        f'Ranking full-context requests: {sum(r["context_fallbacks"] for r in new_rows.values())}. '
        'Costs use recorded Jev input tokens × $0.042 per million; these are usage estimates, not invoices. Local compute costs $0 by convention.', '',
        '## Does filtering remove useful label evidence?', '',
        f'The copied evidence retains **{retained:.1%}** of source characters. Macro ground-truth recall among scan-supported labels '
        f'changes from **{old_recall:.4f}** to **{summary["supported_ground_truth_recall"]:.4f}**. '
        f'The supported-label oracle RP@5 is **{summary["supported_oracle_rp5"]:.4f}**.', '',
        '| Paper | Old / new supported-label count | Old / new GT recall | Previous / new RP@5 | Newly rejected GT labels | Newly supported GT labels |',
        '|---|---:|---:|---:|---|---|']
    for row in comparison:
        key = row['id']
        lines.append(f'| {key} | {old_scan_rows[key]["diagnostics"]["supported_labels"]} / {new_rows[key]["supported_labels"]} | '
            f'{row["previous_support_recall"]:.1%} / {row["new_support_recall"]:.1%} | '
            f'{row["previous_rp5"]:.4f} / {row["new_rp5"]:.4f} | '
            + ('; '.join(row['newly_rejected_ground_truth']) or '—') + ' | '
            + ('; '.join(row['newly_supported_ground_truth']) or '—') + ' |')
    lines += ['', '“Newly rejected” means supported in the earlier full-paper scan but rejected in this short-text scan. '
        'These are changed model decisions, not proof that the corresponding source sentence was removed. '
        'Context and model variability can also change decisions. Labels rejected during scanning cannot be recovered by a ranking-only context fallback.', '',
        'For paper **33619451**, both ground-truth labels still passed the short-text scan, but neither survived the group shortlist. '
        'The previous ranking kept **Cardiac failure**; the new output includes **Cardiac failure congestive** instead. '
        'These are different official labels, so the unchanged exact-label metric gives no credit for the latter against the former. '
        'This paper accounts for the entire 0.10 drop in aggregate RP@5 and RP@10. The other four papers retain their previous scores. '
        'The changed candidate pool and a new ranking run can affect choices; this does not prove a loss of relevant source evidence.', '',
        '## Interpretation', '',
        '- This pilot saves only a small amount of API cost and does not show a latency improvement. '
        'Short-text scanning preserves the scan-supported ground-truth set, but downstream ranking is less accurate in this run.',
        '- This is a single run on five development papers. It does not establish accuracy or latency on the 100-document test set.',
        '- Shorter text reduces the document portion of repeated requests, but every request still contains label questions and instructions. '
        'A 34.6% reduction in source characters does not imply the same percentage reduction in API cost.',
        '- Changed supported-label pools also change ranking groups. Accuracy differences reflect the entire new workflow, not evidence selection in isolation.',
        '- All 33 offline tests passed before this run, including checks that the short-text scan uses the same label questions, '
        'covers all labels, sends only copied evidence, and rejects inconsistent cached source spans.', '',
        'Post-run audit verified all 121,565 scan decisions across 475 scan calls and 79 ranking calls. '
        'All 554 new calls succeeded with no retries. Scan states contain only copied evidence, every label is covered once, '
        'output labels are valid and unique, and the cost ledger matches the summary. [Audit](runs/evidence-first-train5-v1/audit.json).', '',
        '## Files', '',
        '- [Implementation](evidence_first.py) · [Tests](test_evidence_first.py).',
        '- [Protocol](runs/evidence-first-train5-v1/protocol.json) · [Summary](runs/evidence-first-train5-v1/summary.json).',
        '- [Predictions](runs/evidence-first-train5-v1/results.json) · [Changed ground-truth decisions](runs/evidence-first-train5-v1/comparison.json).',
        '- Per-paper folders retain copied evidence, all scan answers, ranking traces, exact request/response ledgers, costs, and timings.', '',
        '```bash',
        'python experiments/semantic-decomposition/biodex/evidence_first.py \\',
        '  --evidence-run experiments/semantic-decomposition/biodex/runs/iterative-evidence-train5-v1 \\',
        '  --output experiments/semantic-decomposition/biodex/runs/evidence-first-train5-v1',
        'python experiments/semantic-decomposition/biodex/report_evidence_first.py',
        '```', '']
    args.output.write_text('\n'.join(lines))
    print(args.output)


if __name__ == '__main__':
    main()
