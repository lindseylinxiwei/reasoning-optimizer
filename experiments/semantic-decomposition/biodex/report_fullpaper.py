"""Report full-test accuracy, monetary cost, and measured batch/document latency."""
import argparse
import json
from pathlib import Path

from core import HERE


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=HERE / 'runs/fullpaper-test100-v1')
    parser.add_argument('--output', type=Path, default=HERE / 'biodex_jev_fullpaper_test100.md')
    args = parser.parse_args()
    run = args.run
    protocol, summary = read(run / 'protocol.json'), read(run / 'summary.json')
    rows = read(run / 'results.json')
    n = len(rows)
    assert n == summary['total_documents'] == len(protocol['documents'])
    stages = summary['stages']
    scan, ranking = stages['scan'], stages['ranking']
    latency = summary['document_service_seconds']
    probability = summary['probability_sort_baseline']
    ratio = scan['known_api_cost_usd'] / summary['known_api_cost_usd']
    doc_ms = {key: {name: read(run / key / name / 'metrics.json') for name in ('scan','ranking')}
              for key in (r['id'] for r in rows)}
    costs = [sum(doc_ms[r['id']][name]['known_api_cost_usd'] for name in ('scan','ranking')) for r in rows]
    lines = ['# Biodex: iterative Jev selection with full-paper context — 100 test documents', '',
        f'All **{n} test documents** completed. **RP@5 = {summary["avg_rp_at_5"]:.4f}**, '
        f'**RP@10 = {summary["avg_rp_at_10"]:.4f}**. Total known Jev API cost: '
        f'**${summary["known_api_cost_usd"]:.6f}** (${summary["known_api_cost_usd"]/n:.6f} per document). '
        f'Measured end-to-end process wall time: **{summary["process_wall_seconds"]:.2f}s '
        f'({summary["process_wall_seconds"]/60:.2f} min)** with up to **{protocol["workers"]} documents in flight**.', '',
        '## Method and evaluation', '',
        'For each complete paper, Jev independently evaluates every official label as supported or not supported. '
        'Up to 256 binary questions share one request and the same full-paper state: 24,313 labels and 95 scan calls per paper. '
        'All supported labels enter iterative comparative selection. Jev selects the best remaining label, code removes it, '
        'and selection repeats up to ten times or until NONE. Pools larger than 253 labels are split into groups; '
        'up to ten labels per group advance to the final selection. Code copies official label names into the ranked output.', '',
        'Both stages use the complete paper. There is no evidence filter, embedding retrieval, Gemini call, or free-text answer generation. '
        'The model (jev-1.13.0), prompts, scan batch size, vocabulary order, group size, and output limit match the five-paper full-context pilot. '
        'The execution change is parallel processing across eight documents; decisions within each document retain their dependencies.', '',
        'The source is the complete `experiments/reasoning/data/test/biodex.json` file, with IDs and hashes frozen before API calls. '
        'Ground truth enters only evaluation. The unchanged MOAR evaluator defines RP@k as correct labels among the first k '
        'divided by min(k, number of ground-truth labels), averaged across documents. It is not whole-document accuracy.', '',
        '| Evaluation on these same test documents | RP@5 | RP@10 |', '|---|---:|---:|',
        f'| Same exhaustive scan, probability sorting (offline baseline) | {probability["avg_rp_at_5"]:.4f} | {probability["avg_rp_at_10"]:.4f} |',
        f'| **Exhaustive scan + iterative full-paper selection** | **{summary["avg_rp_at_5"]:.4f}** | **{summary["avg_rp_at_10"]:.4f}** |', '',
        'The probability-sort baseline is computed from the same saved scan answers and makes no additional API calls. '
        'The earlier five-paper development result was RP@5=0.64 and RP@10=0.62; that pilot is not a matched accuracy comparison with these 100 papers.', '',
        '## Monetary cost and latency', '',
        '| Stage | Known API cost | Measured batch wall time | Mean service time/document | API calls including retries |',
        '|---|---:|---:|---:|---:|']
    for title, stage in [('Exhaustive label scan', scan), ('Iterative label selection', ranking)]:
        lines.append(f'| {title} | ${stage["known_api_cost_usd"]:.6f} | {stage["phase_wall_seconds"]:.2f}s | '
                     f'{stage["mean_document_seconds"]:.2f}s | {stage["api_calls"]} |')
    lines += [f'| **Total stages** | **${summary["known_api_cost_usd"]:.6f}** | **{summary["sum_phase_wall_seconds"]:.2f}s** | '
              f'**{latency["mean"]:.2f}s** | **{sum(s["api_calls"] for s in stages.values())}** |', '',
        f'End-to-end process wall time is **{summary["process_wall_seconds"]:.2f}s**; '
        f'the difference from summed phase wall time ({summary["process_wall_seconds"]-summary["sum_phase_wall_seconds"]:.2f}s) '
        'covers preflight, aggregation, evaluation, and output processing.', '',
        f'Per-document scan + ranking service time: mean **{latency["mean"]:.2f}s**, '
        f'median **{latency["median"]:.2f}s**, p95 **{latency["p95"]:.2f}s**. '
        'This adds each document’s two measured stages and excludes queueing between them. '
        'It is not batch time divided by 100, and it is not submission-to-result latency: '
        'the runner completes the scan phase for the batch before starting ranking.', '',
        f'The sum of all document stage times is **{latency["sum"]:.2f}s**; it exceeds wall time because documents overlap. '
        'This sum is not a measured single-worker run. Latency depends on concurrency and API conditions, '
        'so it should not be compared directly with the earlier single-worker pilot or MOAR runtimes measured differently.', '',
        f'The scan accounts for **{ratio:.1%}** of total cost. Cost uses recorded provider input tokens × '
        '$0.042 per million; it is a usage-based estimate rather than an invoice. Local compute is counted as $0. '
        'No prior pilot costs, optimizer search costs, or report-generation costs are included.', '',
        f'Unknown-cost calls: **{summary["unknown_cost_calls"]}**. '
        f'Retry calls: **{sum(s["retry_calls"] for s in stages.values())}**. '
        f'Unresolved document errors: **{summary["errors"]}**. '
        'All retry charges with reported usage are counted; calls lacking usage remain unknown rather than being assumed free.', '',
        '## Label coverage', '',
        f'- Exhaustive label decisions: **{summary["expected_scan_decisions"]:,}**.',
        f'- Macro ground-truth recall after binary support decisions: **{summary["supported_ground_truth_recall"]:.4f}**.',
        f'- Oracle RP@5 from scan-supported labels: **{summary["supported_oracle_rp5"]:.4f}**.',
        f'- Macro ground-truth recall after group selection: **{summary["survivor_ground_truth_recall"]:.4f}**.',
        f'- Oracle RP@5 among group survivors: **{summary["survivor_oracle_rp5"]:.4f}**.', '',
        'For candidate pools of at most 253, all supported labels enter final ranking directly. '
        'Larger pools can lose correct labels during group selection. Coverage and oracle metrics are evaluation-only diagnostics, '
        'not inputs to any Jev decision.', '',
        '## Per-document results', '',
        '| Document | RP@5 | RP@10 | Supported labels | Scan GT recall | API cost | Scan + ranking service time |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for i, row in enumerate(rows):
        seconds = sum(doc_ms[row['id']][name]['composed_seconds'] for name in ('scan','ranking'))
        lines.append(f'| {row["id"]} | {summary["rp_at_5_scores"][i]:.4f} | {summary["rp_at_10_scores"][i]:.4f} | '
                     f'{row["supported_labels"]} | {row["scan_diagnostics"]["supported_ground_truth_recall"]:.4f} | '
                     f'${costs[i]:.6f} | {seconds:.2f}s |')
    lines += ['', '## Scope and reproducibility', '',
        '- This is one execution per test paper, not a repeated-run variance estimate. Earlier experiments on this test set '
        'informed method development, so this is a full test-set evaluation rather than a claim of a completely untouched holdout.',
        '- Comparison with a paper’s MOAR frontier requires matching the document set, metric, cost accounting, and concurrency. '
        'This report does not infer a MOAR comparison from an unmatched plot.',
        '- All 35 offline tests passed before the run, including checks for per-document client isolation, checkpoint reuse, '
        'source copying, label coverage, and ranking behavior. Unresolved phase errors stop the runner rather than silently dropping papers.', '',
        'The post-run audit verified all 100 documents, all 2,431,300 scan decisions, all 10,936 successful API calls, '
        'complete original-paper contexts, valid unique output labels, recomputed evaluation metrics, and costs against provider usage. '
        'No retry or failed call occurred. The scan-question hash also matches the original pilot exactly.', '',
        'Artifacts:', '',
        '- [Protocol](runs/fullpaper-test100-v1/protocol.json) · [Summary](runs/fullpaper-test100-v1/summary.json) · [Audit](runs/fullpaper-test100-v1/audit.json).',
        '- [Predictions and ground-truth diagnostics](runs/fullpaper-test100-v1/results.json).',
        '- [Probability-sort predictions](runs/fullpaper-test100-v1/probability_results.json).',
        '- [Implementation](fullpaper.py) · [Tests](test_fullpaper.py).',
        '- Each document folder contains every scan answer, ranking trace, request/response ledger, API usage, cost, and timing.', '',
        '```bash',
        'python experiments/semantic-decomposition/biodex/fullpaper.py \\',
        '  --input experiments/reasoning/data/test/biodex.json \\',
        '  --output experiments/semantic-decomposition/biodex/runs/fullpaper-test100-v1 --workers 8',
        'python experiments/semantic-decomposition/biodex/report_fullpaper.py',
        '```', '',
        'Successful requests are checkpointed. Reusing an output folder resumes saved work; '
        'its new process wall time then measures a resumed execution, not a fresh run. '
        'Use a new output folder for a fresh end-to-end timing measurement.', '']
    comparison = run / 'moar_comparison.md'
    if comparison.exists():
        lines += ['', comparison.read_text()]
    args.output.write_text('\n'.join(lines))
    print(args.output)


if __name__ == '__main__':
    main()
