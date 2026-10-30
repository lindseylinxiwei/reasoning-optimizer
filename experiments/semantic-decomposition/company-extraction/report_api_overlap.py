"""Summarize saved sequential/overlap trials without model calls."""
import json,statistics,importlib.util
from pathlib import Path
h=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
root=h.parents[2]
spec=importlib.util.spec_from_file_location('evaluation',root/'experiments/reasoning/evaluation/sustainability.py');e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
gt={r['id']:r['GT company_name'] for r in read(root/'experiments/semantic-decomposition/runs/20260928T183815Z/evaluation/ground_truth.json')}
old={r['id']:r for r in read(h/'runs/sustainability_11_gliner_batch_16/results.json')}
def signature(r):return {(c['text'],m['start'],m['end']) for c in r['candidates'] for m in c['mentions']}
trials=[]
for i,mode in enumerate(['serial','overlap','overlap','serial'],1):
 p=h/f'runs/api_overlap_matched_{i}_{mode}';m=read(p/'metrics.json');rows=read(p/'results.json')
 assert len(rows)==11 and m['errors']==0
 ledger=[json.loads(x) for x in (p/'calls.jsonl').read_text().splitlines()]
 assert len(ledger)==11 and abs(sum(x['cost_usd'] for x in ledger)-m['selector_api_cost_usd'])<1e-12
 for r in rows:
  span=r['selected_span'];text=next(x['text'] for x in read(p/'input.json') if x['id']==r['id'])
  assert text[span['start']:span['end']]==r['company_name']
 matches=sum(e.calculate_name_similarity(r['company_name'].lower(),gt[r['id']].lower())>=.5 for r in rows)
 trials.append({'trial':i,'mode':mode,'directory':p.name,'metrics':m,'matches':matches,
 'changed_names':[r['id'] for r in rows if r['company_name']!=old[r['id']]['company_name']],
 'changed_candidates':[r['id'] for r in rows if signature(r)!=signature(old[r['id']])],
 'mean_selection_queue_seconds':statistics.mean(r['selection_queue_seconds'] for r in rows)})
means={mode:statistics.mean(t['metrics']['pipeline_wall_seconds'] for t in trials if t['mode']==mode) for mode in ['serial','overlap']}
lines=['## Overlapping extraction and Jev requests','',
'Controlled follow-up at GLiNER Small batch size 16 on MPS. Order: sequential, overlapped, overlapped, sequential. Each trial processes the same 11 documents with a freshly loaded model; loading is timed separately. No candidate/decision caching or retries. In overlap mode, one selection worker processes a document while the main thread extracts the next. At most one selection is active; extraction remains sequential. All trials use the same runner, prompt, model versions, windows and threshold.','',
'| Trial | Execution | Dataset wall time (s) | Extraction sum (s) | Selection sum (s) | Mean document latency (s) | API cost (USD) | Name-match proxy |',
'|---|---|---:|---:|---:|---:|---:|---:|']
for t in trials:
 m=t['metrics'];lines.append(f"| {t['trial']} | {t['mode']} | {m['pipeline_wall_seconds']:.3f} | {m['stage_wall_seconds']['extraction']:.3f} | {m['stage_wall_seconds']['selection']:.3f} | {m['mean_document_wall_seconds']:.3f} | ${m['selector_api_cost_usd']:.9f} | {t['matches']}/11 |")
speed=means['serial']/means['overlap'];reduction=100*(1-means['overlap']/means['serial'])
lines+=['',f"Mean dataset completion time: **{means['serial']:.3f}s sequential → {means['overlap']:.3f}s overlapped**, a **{speed:.2f}× speedup ({reduction:.1f}% reduction)**. Two trials per mode provide exploratory evidence, not statistical confidence. These are dataset completion times, excluding model load; per-document latency runs from extraction start to selection/copy completion and includes any selection queue wait. It excludes waiting before a document's extraction starts. Stage-duration sums overlap in time and must not be added to estimate dataset wall time.",'',
f"Changed final names versus the previous batch-16 run: {sum(len(t['changed_names']) for t in trials)} document-trials. Changed candidate-span sets: {sum(len(t['changed_candidates']) for t in trials)} document-trials. All 44 document executions completed successfully. Name-match scores retain the existing GT/alias limitations. Total additional API cost across these four trials: **${sum(t['metrics']['selector_api_cost_usd'] for t in trials):.9f}**. Local computation is $0 by convention.",'',
'Use `--extractor gliner --device mps --batch-size 16 --overlap-api` to enable this scheduling. The default remains sequential for reproducibility. This experiment does not rebenchmark Gemini with concurrency, so it establishes the scheduling improvement within our method rather than a fair concurrent-throughput comparison against Gemini.','',
'Raw artifacts include per-document start/completion timestamps, queue delay, stage timings, source spans, and complete API usage:','']
for t in trials:lines.append(f"- [Trial {t['trial']} ({t['mode']})](runs/{t['directory']}/metrics.json): [outputs](runs/{t['directory']}/results.json), [calls](runs/{t['directory']}/calls.jsonl), [configuration](runs/{t['directory']}/config.json).")
lines += ['', 'The only final-name change was document 4922: trial 3 returned `AB Volvo` instead of `Volvo Group`, reducing the automatic match score to 7/11. Its request payload was identical to the sequential trial (indeed all request payloads in these two trials matched). This is response variation under identical inputs, not evidence that batching changed candidates; the experiment cannot establish that scheduling caused the variation. The alias/subject-name distinction still requires review.', '', 'Mean document latency increased from approximately 2.39s sequentially to 2.78s under overlap, despite faster dataset completion. Selection-stage elapsed time also increased while extraction ran concurrently; these measurements alone do not isolate local contention from remote/network timing. Overlap improves throughput here, not individual-document response latency.', '']
pilot=read(h/'runs/api_overlap_trial_1_serial/metrics.json')
known_total=sum(t['metrics']['selector_api_cost_usd'] for t in trials)+pilot['selector_api_cost_usd']
lines += ['', '### Preliminary transport failure and session accounting', '',
    f"An earlier sequential trial had one TLS transport error on document 8667 and 10 successful selections. Its pipeline wall time was {pilot['pipeline_wall_seconds']:.3f}s and its known API charges were ${pilot['selector_api_cost_usd']:.9f}; one request has unknown billed usage. It is preserved at `runs/api_overlap_trial_1_serial/` and excluded from the matched successful-run timing table. Total known charges for this overlap experiment, including that preliminary run, are **${known_total:.9f}, plus any unreported charge for the failed request**. No automatic retries were used; the matched trials are fresh full runs.", '']
p=h/'gemini_extraction_comparison.md';s=p.read_text().split('\n## Overlapping extraction and Jev requests')[0];p.write_text(s.rstrip()+'\n\n'+'\n'.join(lines)+'\n')
(h/'runs/api_overlap_comparison.json').write_text(json.dumps({'means':means,'speedup':speed,'trials':trials},indent=2)+'\n')
print(json.dumps({'means':means,'speedup':speed,'reduction_percent':reduction,'trials':trials},indent=2))
