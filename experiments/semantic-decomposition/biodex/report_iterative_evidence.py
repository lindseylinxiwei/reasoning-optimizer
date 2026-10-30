"""Rebuild the pilot report from saved decisions, ledgers, and unchanged MOAR metrics."""
import argparse
import json
from collections import Counter
from pathlib import Path

from core import HERE, ROOT, load_module


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=HERE / 'runs/iterative-evidence-train5-v1')
    parser.add_argument('--output', type=Path, default=HERE / 'biodex_jev_iterative_evidence.md')
    args = parser.parse_args()
    run = args.run
    protocol, summary = read(run / 'protocol.json'), read(run / 'summary.json')
    scan = Path(protocol['scan'])
    scan_summary = read(scan / 'summary.json')
    docs = protocol['documents']
    baseline = {r['id']: r for r in read(scan / 'results.json')}
    methods = {m: {r['id']: r for r in read(run / (m + '_results.json'))} for m in protocol['methods']}
    evaluator = load_module('report_fine_evaluator', ROOT / 'experiments/reasoning/evaluation/biodex.py')
    rp = evaluator.rank_precision_at_k
    evidence = {key: read(run / key / 'evidence/result.json') for key in docs}
    retained = sum(e['retained_source_characters'] for e in evidence.values())
    original = sum(e['source_characters'] for e in evidence.values())
    categories = Counter(a['choice'] for e in evidence.values() for a in e['answers'].values())
    full = summary['methods']['full_context']
    short = summary['methods']['selected_evidence']
    stages = summary['stages']
    baseline_rp = scan_summary['avg_rp_at_5']
    lines = [
        '# Biodex: finer evidence selection and iterative Jev ranking', '',
        f'On the same five-paper pilot, RP@5 is **{baseline_rp:.4f}** with the previous probability ranking, '
        f'**{full["avg_rp_at_5"]:.4f}** with iterative selection using the full paper, and '
        f'**{short["avg_rp_at_5"]:.4f}** with iterative selection using Jev-selected evidence.', '',
        '## Method', '',
        '1. Reuse the saved exhaustive Jev scan. Every one of the 24,313 labels was considered in that scan. '
        'Both new variants receive the identical supported-label pool for each paper, in official vocabulary order (190–641 labels). '
        'There is no embedding retrieval or new candidate generation in this pilot.',
        '2. Split every paper into numbered sentences, short headings, or delimited table rows using local deterministic rules. '
        'Classify every unit as a patient finding, necessary context, background, or uncertain. '
        'Each call shows a block of up to 16 target units, two neighboring units on each side, and associated headings.',
        '3. Code copies findings, necessary context, and uncertain units verbatim from saved source offsets. '
        'It also retains associated headings and detected table headers. The prompt explicitly preserves negation, drug relationships, '
        'qualifiers, treatment failure, and drug-use issues. These prompts encourage semantic preservation; they do not guarantee perfect evidence recall.',
        '4. Ask Jev to choose the best remaining supported label, remove that choice, and repeat up to ten times, stopping at NONE. '
        'For more than 253 candidates, select up to ten per group, combine survivors in original vocabulary order, and select the final ten. '
        'Independent group questions share a request; successive ranks depend on prior answers.',
        '5. The selected-evidence variant can choose MORE_CONTEXT. Code then supplies the full paper and keeps it for that group’s remaining rounds. '
        '253 labels plus NONE and MORE_CONTEXT fit the 255-option limit. Code assembles the final output by copying official label names.', '',
        '**Comparison:** the ranking algorithm, initial candidate pool, and candidate order are held fixed. '
        'The evidence variant adds filtering and an explicit full-context fallback; this evaluates that policy, not a strictly forced-short-context ablation. '
        'The full-paper variant has no MORE_CONTEXT option.', '',
        '## Accuracy', '',
        '| Method | RP@5 | RP@10 |', '|---|---:|---:|',
        f'| Previous full scan → probability sort | {scan_summary["avg_rp_at_5"]:.4f} | {scan_summary["avg_rp_at_10"]:.4f} |',
        f'| Same supported labels → iterative selection, full paper | {full["avg_rp_at_5"]:.4f} | {full["avg_rp_at_10"]:.4f} |',
        f'| Same supported labels → iterative selection, selected evidence | {short["avg_rp_at_5"]:.4f} | {short["avg_rp_at_10"]:.4f} |', '',
        'The unchanged MOAR evaluator computes RP@k as correct labels among the first k predictions divided by '
        'min(k, number of ground-truth labels), averaged over papers. It is not whole-document accuracy.', '',
        '| Paper | Supported labels | Previous RP@5 | Full-paper RP@5 | Selected-evidence RP@5 | Source characters retained | Full-context requests |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key in docs:
        gt = baseline[key]['ground_truth_reactions']
        vals = [rp(r['ranked_reaction_labels'], gt, 5) for r in
                (baseline[key], methods['full_context'][key], methods['selected_evidence'][key])]
        e = evidence[key]
        lines.append(f'| {key} | {protocol["candidate_counts"][key]} | {vals[0]:.4f} | {vals[1]:.4f} | '
                     f'{vals[2]:.4f} | {e["retained_source_characters"] / e["source_characters"]:.1%} | '
                     f'{methods["selected_evidence"][key]["context_fallbacks"]} |')
    lines += ['', f'Evidence selection classified **{sum(categories.values())} units**, retained **{retained / original:.1%} of source characters** '
              '(pooled across papers), and made ' + str(sum(r['context_fallbacks'] for r in methods['selected_evidence'].values())) +
              ' full-context requests during ranking. Decision counts: ' + ', '.join(f'{k}={v}' for k,v in sorted(categories.items())) + '.', '',
              '| Paper | Full-paper survivors / GT recall | Selected-evidence survivors / GT recall |', '|---|---:|---:|']
    for key in docs:
        a, b = methods['full_context'][key], methods['selected_evidence'][key]
        lines.append(f'| {key} | {a["survivor_count"]} / {a["survivor_coverage"]["recall"]:.1%} | '
                     f'{b["survivor_count"]} / {b["survivor_coverage"]["recall"]:.1%} |')
    lines += ['', 'Survivors are the labels entering the final ranking. For pools of at most 253, '
              'all initial supported labels enter directly. Ground truth is used only after model decisions.', '',
              '## Cost and latency', '',
              '| New stage, all five papers | API cost (USD) | Stage wall time (s) | API calls |', '|---|---:|---:|---:|']
    for stage, title in [('evidence', 'Sentence/row evidence classification + exact copying'),
                         ('full_context', 'Iterative ranking with full paper'),
                         ('selected_evidence', 'Iterative ranking with selected evidence, including fallback')]:
        s = stages[stage]
        lines.append(f'| {title} | ${s["known_api_cost_usd"]:.6f} | {s["composed_seconds"]:.2f} | {s["api_calls"]} |')
    e, f, s = stages['evidence'], stages['full_context'], stages['selected_evidence']
    lines += ['', '| Standalone variant after the saved scan | Incremental API cost | Incremental stage time |', '|---|---:|---:|',
              f'| Full-paper iterative ranking | ${f["known_api_cost_usd"]:.6f} | {f["composed_seconds"]:.2f}s |',
              f'| Evidence selection + iterative ranking | ${e["known_api_cost_usd"] + s["known_api_cost_usd"]:.6f} | '
              f'{e["composed_seconds"] + s["composed_seconds"]:.2f}s |', '',
              f'Actual new API spend across both variants: **${summary["new_known_cost_usd"]:.6f}**. '
              f'Process wall time: **{summary["process_wall_seconds"]:.2f}s**. '
              f'Retries: **{sum(x["retry_calls"] for x in stages.values())}**; '
              f'unknown-cost calls: **{sum(x["unknown_cost_calls"] for x in stages.values())}**.', '',
              f'The prior full-vocabulary scan cost **${scan_summary["known_api_cost_usd"]:.6f}** and '
              f'**{scan_summary["composed_wall_seconds"]:.2f}s** of composed stage time. It was reused, not rerun or charged again. '
              'Both methods still depend on that scan. Adding its historical cost and time gives the following composed estimates; '
              'these are not fresh end-to-end measurements:', '',
              '| Variant including historical scan | Composed API cost | Composed stage time |', '|---|---:|---:|']
    for title, cost, seconds in [
            ('Full-paper iterative ranking', f['known_api_cost_usd'], f['composed_seconds']),
            ('Evidence selection + iterative ranking', e['known_api_cost_usd'] + s['known_api_cost_usd'],
             e['composed_seconds'] + s['composed_seconds'])]:
        lines.append(f'| {title} | ${scan_summary["known_api_cost_usd"] + cost:.6f} | '
                     f'{scan_summary["composed_wall_seconds"] + seconds:.2f}s |')
    delta = short['avg_rp_at_5'] - full['avg_rp_at_5']
    lines += ['', 'Jev cost uses recorded input usage × $0.042 per million tokens for jev-1.13.0; '
              'these are usage-based estimates, not invoices. Local computation is counted as $0. '
              'Stage times include request waiting, local processing, and checkpoint I/O. API requests run sequentially, '
              'while independent questions may share a request. No Gemini or embedding API calls were made.', '',
              '## Interpretation and limits', '',
              f'- Iterative full-paper selection changes RP@5 by {full["avg_rp_at_5"] - baseline_rp:+.4f} versus the saved probability sort.',
              f'- Adding selected evidence and fallback changes RP@5 by {delta:+.4f} versus full-paper iterative selection. '
              'The filtering stage must be included when assessing cost or latency benefits.',
              '- In this run, evidence selection preserves RP@5 but does not pay for itself in cost or latency. Iterative ranking provides the observed gain.',
              '- This is a five-paper development pilot, with one run per variant. It does not establish statistical significance or generalization.',
              '- The earlier binary scan already rejected three ground-truth labels on one paper. Ranking cannot recover labels outside its frozen pool.',
              '- Group selection is a heuristic: it can discard correct labels before final ranking. The survivor recall table measures this loss.',
              '- Sentence boundaries and table detection are heuristic. Flattened source tables cannot reliably recover their original layout. '
              'Every non-whitespace source character is assigned to a unit, and copied spans are verified against the source.',
              '- We have not rescanned all 24,313 labels using the shorter evidence. This experiment therefore does not measure a cheaper exhaustive scan '
              'or independently measure evidence-selection recall against annotated source spans.', '',
              'Validation: all 29 offline Biodex tests passed, including six new tests for source copying, '
              'negation preservation, grouping, iterative removal, and context fallback. A post-run audit verified '
              'all 217 successful calls, all 964 evidence decisions, exact source spans, unique official output IDs, and ledger totals. '
              'The full-context fallback was tested offline but was not requested in the live pilot.', '',
              '## Saved artifacts', '',
              '- [Implementation](iterative_evidence.py) and [tests](test_iterative_evidence.py).',
              '- [Frozen protocol](runs/iterative-evidence-train5-v1/protocol.json) and [summary](runs/iterative-evidence-train5-v1/summary.json).',
              '- [Full-paper predictions](runs/iterative-evidence-train5-v1/full_context_results.json).',
              '- [Selected-evidence predictions](runs/iterative-evidence-train5-v1/selected_evidence_results.json).',
              '- Per-paper folders contain exact copied spans, every evidence decision, ranking rounds, context requests, '
              'raw API request/response ledgers, usage, costs, and timing.', '',
              'Reproduce (completed requests are checkpointed):', '', '```bash',
              'PYTHONDONTWRITEBYTECODE=1 TIKTOKEN_CACHE_DIR=/tmp/jev-tiktoken \\',
              '  /private/tmp/jev-comparison-venv/bin/python \\',
              '  experiments/semantic-decomposition/biodex/iterative_evidence.py \\',
              '  --output experiments/semantic-decomposition/biodex/runs/iterative-evidence-train5-v1',
              '```', '']
    args.output.write_text('\n'.join(lines))
    print(args.output)


if __name__ == '__main__':
    main()
