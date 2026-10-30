"""Summarize the two saved 11-document runs; performs no inference."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def read(path):
    return json.loads(path.read_text())


def main():
    source = ROOT / "experiments/semantic-decomposition/runs/20260928T183815Z"
    gt = {r["id"]: r for r in read(source / "evaluation/ground_truth.json")}
    spec = importlib.util.spec_from_file_location("moar", ROOT / "experiments/reasoning/evaluation/sustainability.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    similarity = module.calculate_name_similarity
    runs = [("GLiNER Small + Jev", "sustainability_11_gliner_compact"),
            ("GLiNER2.5 Base + Jev", "sustainability_11_gliner25_validated")]
    lines = ["# Company extraction — 11-document experiment", "",
        "Pipeline: full-document window scanning → organization candidates → Jev subject-company selection → exact source copying. Both runs use MPS, 320 word/punctuation-token windows, 64-token overlap (these are chunking units, not model subword tokens), extraction threshold 0.5, and the same 11 documents. Model checkpoints are pinned. Models are loaded once per run; documents and windows run sequentially. This is one run per method, not a repeated latency benchmark.", "",
        "## Stage costs and latency", "",
        "Local model loading, GLiNER extraction and deterministic processing cost $0 by the user's accounting convention. Jev cost is returned input tokens × $0.042/M, output free. No credits/free allowances are subtracted.", "",
        "| Method | Stage | Monetary cost (USD) | Latency, all 11 documents (s) |",
        "|---|---|---:|---:|"]
    report = {}
    for label, directory in runs:
        path = HERE / "runs" / directory
        rows, metrics = read(path / "results.json"), read(path / "metrics.json")
        inputs = {r['id']:r['text'] for r in read(path / 'input.json')}
        assert set(inputs)==set(gt) and len(rows)==11
        ledger = [json.loads(l) for l in (path / 'calls.jsonl').read_text().splitlines()] if (path / 'calls.jsonl').exists() else []
        assert abs(sum(x.get('cost_usd') or 0 for x in ledger)-metrics['selector_api_cost_usd'])<1e-10
        details = []
        for row in rows:
            i = row['id']; text = inputs[i]; target = gt[i]['GT company_name']
            for candidate in row.get('candidates', []):
                for span in candidate['mentions']:
                    assert text[span['start']:span['end']]==candidate['text']
            selected = row.get('company_name')
            if row['status']=='selected':
                span=row['selected_span']
                assert text[span['start']:span['end']]==selected
            score = similarity((selected or '').lower(), target.lower())
            coverage = any(similarity(c['text'].lower(), target.lower())>=0.5 for c in row.get('candidates', []))
            details.append({'id':i,'ground_truth':target,'prediction':selected,'status':row['status'],
                'error':row.get('error'),'candidate_count':len(row.get('candidates',[])),
                'window_count':row.get('window_count'),'invalid_spans':len(row.get('invalid_spans', [])),'name_similarity':score,
                'name_match_proxy':score>=0.5,'candidate_coverage_proxy':coverage,
                'extraction_seconds':row.get('extraction_wall_seconds'),
                'selection_seconds':row.get('selection_wall_seconds'),
                'copy_seconds':row.get('copying_wall_seconds'),
                'selection_cost_usd':row.get('stage_cost_usd',{}).get('selection')})
        report[label]={'directory':directory,'metrics':metrics,'documents':details,
            'name_matches_proxy':sum(x['name_match_proxy'] for x in details),
            'candidate_coverage_proxy':sum(x['candidate_coverage_proxy'] for x in details),
            'input_tokens':sum(x.get('response',{}).get('usage',{}).get('input_tokens',0) for x in ledger),
            'api_calls':len(ledger)}
        lines.append(f"| {label} | Model loading (separate) | $0 | {metrics['load_seconds']:.6f} |")
        for key, title in [('extraction','Candidate extraction + merging'),('selection','Jev selection'),('copying','Copying + decision validation')]:
            lines.append(f"| {label} | {title} | ${metrics['stage_cost_usd'][key]:.9f} | {metrics['stage_wall_seconds'][key]:.6f} |")
        lines.append(f"| {label} | Pipeline total, excluding load | ${metrics['selector_api_cost_usd']:.9f} | {metrics['pipeline_wall_seconds']:.6f} |")
    lines += ['', 'Extraction includes all windows, offset validation, and deduplication; selection includes payload construction, token-budget validation, HTTP and response processing. Copying includes deterministic ID lookup, span validation and slicing. Pipeline wall time also includes saving artifacts and bookkeeping, so it exceeds the sum of stages. First-inference warm-up is included; MPS is synchronized. Errors before a completed stage may have no stage timing; any such missing values are not zero-duration successful work.', '',
        '## Completion and provisional name matching', '',
        '| Method | Selected | Abstained | Errors | Candidate count range | Jev calls | Jev input tokens | Name-match proxy |',
        '|---|---:|---:|---:|---|---:|---:|---:|']
    for label,r in report.items():
        ds=r['documents'];counts=[x['candidate_count'] for x in ds]
        lines.append(f"| {label} | {sum(x['status']=='selected' for x in ds)} | {r['metrics']['abstentions']} | {r['metrics']['errors']} | {min(counts)}–{max(counts)} | {r['api_calls']} | {r['input_tokens']:,} | {r['name_matches_proxy']}/11 |")
    lines += ['', 'The name-match proxy compares each prediction against its own document’s existing GT company name using the MOAR name-similarity function at threshold ≥0.5. This is ID-aligned and differs from the original final-summary evaluator’s search over all GT names. Accepted source-span/alias annotations have not been reviewed, so neither this score nor the candidate-coverage proxy is a validated exact-span extraction accuracy. No model receives GT/company metadata.', '']
    for label,r in report.items():
        lines += ['## '+label, '', '| ID | GT company | Copied name | Candidates | Windows | Extract (s) | Select (s) | Copy (ms) | Jev cost (USD) | Status |', '|---|---|---|---:|---:|---:|---:|---:|---:|---|']
        for d in r['documents']:
            def fmt(x,scale=1): return '—' if x is None else f'{x*scale:.6f}'
            lines.append(f"| {d['id']} | {d['ground_truth']} | {d['prediction'] or '—'} | {d['candidate_count']} | {d['window_count']} | {fmt(d['extraction_seconds'])} | {fmt(d['selection_seconds'])} | {fmt(d['copy_seconds'],1000)} | {fmt(d['selection_cost_usd'])} | {d['status']} |")
        for d in r['documents']:
            if d['error']: lines.append(f"\nDocument {d['id']} error: {d['error']}")
        lines += ['', f"Invalid spans excluded after exact source-offset validation: {sum(d['invalid_spans'] for d in r['documents'])}. Invalid spans remain in the raw results for auditing; offsets are never repaired by guessing.", '']
        lines += ['', f"[Raw results](runs/{r['directory']}/results.json) · [Call ledger](runs/{r['directory']}/calls.jsonl) · [Run configuration](runs/{r['directory']}/config.json)", '']
    lines += ['## Scope', '', 'This experiment measures the decomposed extraction pipeline only. It does not yet include a direct Gemini company-name extraction baseline, so no cost/latency savings claim against generation follows from these runs. Earlier Gemini reduce costs belong to a different task. Local compute is treated as $0 by convention, not estimated hardware or electricity cost.', '']
    pilot = read(HERE / "runs/sustainability_11_gliner/metrics.json")
    base_pilot = read(HERE / 'runs/sustainability_11_gliner25_compact/metrics.json')
    total_cost = base_pilot['selector_api_cost_usd'] + pilot['selector_api_cost_usd'] + sum(r['metrics']['selector_api_cost_usd'] for r in report.values())
    lines += ['## Preliminary runs and total spend', '',
        'An initial GLiNER Small run included every candidate occurrence offset in the Jev state and estimated token length from serialized HTTP JSON. Five documents exceeded the conservative 24K estimate before any API request; six were selected. This was a request-format/preflight issue, not a 255-choice overflow or a provider-reported 32K overflow.', '',
        'The final runs above keep all source offsets locally, send the full unmodified document plus every candidate ID/name, and estimate actual state/question text rather than escaped transport bytes. Both final runs use this same format, with the 24K guard retained and no content/candidate pruning.', '',
        f"Preliminary run: Jev cost ${pilot['selector_api_cost_usd']:.9f}; pipeline wall time {pilot['pipeline_wall_seconds']:.6f}s; extraction {pilot['stage_wall_seconds']['extraction']:.6f}s, selection/preflight {pilot['stage_wall_seconds']['selection']:.6f}s, copying {pilot['stage_wall_seconds']['copying']:.6f}s; loading {pilot['load_seconds']:.6f}s separately. Local monetary costs are $0.", '',
        f"**Total Jev spend for this experiment session, including both preliminary runs: ${total_cost:.9f}.** Earlier synthetic smoke tests are excluded. The original preliminary artifacts are preserved under `runs/sustainability_11_gliner/`.", '']
    lines += [f"The preliminary GLiNER2.5 run cost ${base_pilot['selector_api_cost_usd']:.9f} across nine Jev calls, with pipeline wall time {base_pilot['pipeline_wall_seconds']:.6f}s and loading {base_pilot['load_seconds']:.6f}s. It stopped one document at an invalid extracted span and another at the candidate limit. Its extraction stage total omitted the failed extraction's time, so use its pipeline wall time for accounting. The final GLiNER2.5 run records and excludes invalid spans, keeping all valid candidates. The local diagnostic between runs cost $0; its latency was not instrumented and is outside the benchmark. Preliminary artifacts: `runs/sustainability_11_gliner25_compact/`.", '']
    report['session_accounting'] = {'preliminary_small_metrics': pilot, 'preliminary_base_metrics': base_pilot, 'total_api_cost_usd_including_preliminary': total_cost}
    (HERE/'extraction_experiment.md').write_text('\n'.join(lines))
    (HERE/'runs/extraction_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({name:{k:v for k,v in r.items() if k!='documents'} for name,r in report.items()},indent=2))


if __name__=='__main__':
    main()
