"""Append the saved batching measurements to the Gemini comparison report."""
import json,statistics,importlib.util
from pathlib import Path
h=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
b=read(h/'runs/window_batch_benchmark/metrics.json');trials=b['trials']
assert len(trials)==8
means={size:statistics.mean(t['wall_seconds'] for t in trials if t['batch_size']==size) for size in [1,8,16,32]}
best=min(means,key=means.get)
p=h/f'runs/sustainability_11_gliner_batch_{best}'
metrics=read(p/'metrics.json');rows=read(p/'results.json')
root=h.parents[2]
spec=importlib.util.spec_from_file_location('ev',root/'experiments/reasoning/evaluation/sustainability.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
gt={r['id']:r['GT company_name'] for r in read(root/'experiments/semantic-decomposition/runs/20260928T183815Z/evaluation/ground_truth.json')}
old={r['id']:r for r in read(h/'runs/sustainability_11_gliner_compact/results.json')}
matches=sum(m.calculate_name_similarity((r.get('company_name') or '').lower(),gt[r['id']].lower())>=.5 for r in rows)
changed=[{'id':r['id'],'before':old[r['id']].get('company_name'),'after':r.get('company_name')} for r in rows if r.get('company_name')!=old[r['id']].get('company_name')]
lines=['## Window-batching latency experiment','',
'Same Mac and MPS GPU, pinned GLiNER Small checkpoint, float precision unchanged, 11 documents / 559 windows, 320 word/punctuation units per window, overlap 64, threshold 0.5, CPU threads 4. Batches contain windows from one document at a time. No document truncation or candidate pruning. Two full-dataset trials per batch size, ordered 1, 8, 16, 32, 32, 16, 8, 1, with the model resident and one synthetic-window warm-up before all trials. First encounters of larger batch shapes may still include warm-up overhead. Timing includes candidate merging and source validation and synchronizes MPS before completion. Two trials are exploratory, not confidence intervals.','',
'| Windows per batch | Trial 1 (s) | Trial 2 (s) | Mean extraction (s) | Speedup vs batch 1 | Added / removed source spans across both trials |',
'|---|---:|---:|---:|---:|---:|']
for size,mean in means.items():
 ts=[t for t in trials if t['batch_size']==size]
 lines.append(f"| {size} | {ts[0]['wall_seconds']:.3f} | {ts[1]['wall_seconds']:.3f} | {mean:.3f} | {means[1]/mean:.2f}× | {sum(t['added_spans'] for t in ts)} / {sum(t['removed_spans'] for t in ts)} |")
lines+=['',f'Fastest tested mean: batch size **{best}**, **{means[1]/means[best]:.2f}×** extraction speedup. This is the fastest among the tested sizes, not a global optimum. Local benchmark API spend: **$0**. Candidate equality refers to names and exact source offsets; floating-point confidence scores need not be bit-identical.','',
'### Fresh end-to-end run at the fastest tested batch size','',
'| Stage | Wall time (s) | Monetary cost (USD) |','|---|---:|---:|',
f"| Model loading, separate | {metrics['load_seconds']:.3f} | $0 |"]
for key in ['extraction','selection','copying']:lines.append(f"| {key} | {metrics['stage_wall_seconds'][key]:.6f} | ${metrics['stage_cost_usd'][key]:.9f} |")
oldmetrics=read(h/'runs/sustainability_11_gliner_compact/metrics.json')
lines+=[f"| Pipeline total, excluding load | {metrics['pipeline_wall_seconds']:.3f} | ${metrics['selector_api_cost_usd']:.9f} |",'',
f"Fresh-run total speedup against the original {oldmetrics['pipeline_wall_seconds']:.3f}s run: **{oldmetrics['pipeline_wall_seconds']/metrics['pipeline_wall_seconds']:.2f}×**. This end-to-end comparison is one run each; the repeated extraction benchmark above is the stronger evidence for the batching improvement. Successful selections: {sum(r['status']=='selected' for r in rows)}/11. Name-match proxy: **{matches}/11** using the same existing GT and threshold; alias limitations still apply. Changed final names: {json.dumps(changed,ensure_ascii=False)}.",'',
'All original experiment outputs are preserved. The fresh run makes new Jev requests; it does not reuse historical decisions. Per-call usage and per-document stage latency are saved. The local benchmark contributes $0; the only additional API charge is the fresh run’s selection cost shown above.','',
f'[Batch benchmark records](runs/window_batch_benchmark/metrics.json) · [Fresh results](runs/{p.name}/results.json) · [Fresh call ledger](runs/{p.name}/calls.jsonl)','']
lines += ['', "### Potential ways to further improve latency\n\nThese are untested follow-up experiments; no additional speedup is assumed.\n\n- **Overlap extraction and API calls:** start GLiNER on the next document while Jev processes the previous one, with bounded request concurrency. This primarily improves total dataset completion time; it does not necessarily reduce one document's latency.\n- **Tune batching further:** test sizes near 16 and group similarly sized windows, potentially across documents, to reduce padding and partially filled batches. Preserve document IDs and source offsets and verify candidate equivalence.\n- **Reduce window overlap or adjust window length:** process fewer windows while staying within the extractor's input limit. Measure candidate recall, especially for names at window boundaries, before adopting the change.\n- **Use selective extraction with a full-document fallback:** scan the opening sections first and expand when evidence is insufficient. This may save substantial work but can miss the subject or its complete name, so evaluate the stopping rule separately.\n- **Benchmark alternative inference settings:** test supported lower-precision execution, optimized runtimes, or a different GPU. Measure actual speed and output changes; include any cloud GPU charges in the cost comparison.\n- **Keep the model loaded between jobs:** avoid repeated startup costs in a persistent process. This improves cold-start latency, but does not reduce the reported 26.02-second pipeline time, which already excludes loading.\n\nFor any concurrency change, compare against a similarly configured Gemini baseline and report both per-document latency and total dataset completion time. Recheck extraction quality and repeat timings before claiming a reliable improvement.\n"]
report=h/'gemini_extraction_comparison.md'
s=report.read_text().split('\n## Window-batching latency experiment')[0]
report.write_text(s.rstrip()+'\n\n'+'\n'.join(lines))
print(json.dumps({'best_batch_size':best,'means':means,'metrics':metrics,'matches':matches,'changed':changed},indent=2))
