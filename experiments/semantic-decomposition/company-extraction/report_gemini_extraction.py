import json,importlib.util
from pathlib import Path
here=Path(__file__).resolve().parent
root=here.parents[2]
read=lambda p:json.loads(p.read_text())
spec=importlib.util.spec_from_file_location('ev',root/'experiments/reasoning/evaluation/sustainability.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
gt={x['id']:x['GT company_name'] for x in read(root/'experiments/semantic-decomposition/runs/20260928T183815Z/evaluation/ground_truth.json')}
p=here/'runs/sustainability_11_gemini31_flash_lite';rows=read(p/'results.json');metrics=read(p/'metrics.json')
assert len(rows)==11 and metrics['errors']==0
for r in rows:
 r['ground_truth']=gt[r['id']];r['name_similarity']=m.calculate_name_similarity((r['company_name'] or '').lower(),gt[r['id']].lower());r['name_match_proxy']=r['name_similarity']>=.5
save=lambda p,d:p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
save(p/'evaluation.json',rows)
comparison=read(here/'runs/extraction_comparison.json')
lines=['# Direct Gemini vs. candidate extraction + Jev','',
'All methods receive the same 11 full documents. Gemini receives no extracted candidates or ground-truth metadata and directly returns a company name (or null) in JSON. Its prompt requests the complete subject-company name explicitly present in the source. Gemini uses temperature 0, minimal thinking, at most 256 output tokens, sequential standard requests, no retries or explicit caching. GLiNER costs $0 by convention. This is one run per method; latency is descriptive, not a statistically established speed difference.','',
'## Overall comparison','',
'| Method | Name-match proxy (all 11) | Successful outputs | API cost (USD) | Pipeline wall time, excluding model load (s) |',
'|---|---:|---:|---:|---:|']
for label,r in comparison.items():
 if 'documents' not in r:continue
 mm=r['metrics'];lines.append(f"| {label} | {r['name_matches_proxy']}/11 ({100*r['name_matches_proxy']/11:.1f}%) | {sum(d['status']=='selected' for d in r['documents'])}/11 | ${mm['selector_api_cost_usd']:.9f} | {mm['pipeline_wall_seconds']:.3f} |")
n=sum(r['name_match_proxy'] for r in rows)
lines.append(f"| Gemini 3.1 Flash-Lite, direct | {n}/11 ({100*n/11:.1f}%) | {sum(r['status']=='selected' for r in rows)}/11 | ${metrics['api_cost_usd']:.9f} | {metrics['pipeline_wall_seconds']:.3f} |")
lines+=['','The GLiNER2.5 row is the saved pre-batching run: Volvo Group failed at 402 candidates and is counted as incorrect. The new hierarchical selector has not been rerun here. Its lower total cost therefore does not represent equal completion. Small + Jev completed all 11 and is the primary complete-run comparison. GLiNER model loading was 4.727s (Small) and 3.618s (Base), separately.','',
'Accuracy uses the existing MOAR name-similarity function, threshold >=0.5, aligned by document ID. This is a provisional name-match proxy, not reviewed semantic correctness: GT aliases and suspected spelling errors remain unresolved (e.g., Amega, Deloite). Neither model sees GT. Exact source-substring checks are reported separately and do not establish that the correct company was selected.','',
'## Gemini per-document results','',
'| ID | GT | Gemini output | Match proxy | Exact substring | Cost (USD) | Latency (s) |','|---|---|---|---|---|---:|---:|']
for r in rows:lines.append(f"| {r['id']} | {r['ground_truth']} | {r['company_name']} | {r['name_match_proxy']} | {r['exact_source_substring']} | ${r['cost_usd']:.9f} | {r['wall_seconds']:.3f} |")
calls=[json.loads(x) for x in (p/'calls.jsonl').read_text().splitlines()]
usages=[r['response']['usageMetadata'] for r in calls]
lines+=['','## Cost and provenance','',f"Gemini totals: prompt tokens {sum(u.get('promptTokenCount',0) for u in usages):,}; cached prompt tokens {sum(u.get('cachedContentTokenCount',0) for u in usages):,}; response tokens {sum(u.get('candidatesTokenCount',0) for u in usages):,}; thinking tokens {sum(u.get('thoughtsTokenCount',0) for u in usages):,}. API/response-processing latency {metrics['api_wall_seconds']:.3f}s. Actual returned model versions: {sorted(set(r['response'].get('modelVersion','unknown') for r in calls))}.", '',
'Paid-tier equivalent cost uses $0.25/M uncached input, $0.025/M cached input, and $1.50/M output including thinking. This is usage-based list-price accounting, not a verified invoice; free tiers or credits are not subtracted. [Official pricing](https://ai.google.dev/gemini-api/docs/pricing).','',
'Gemini 2.5 Flash-Lite was attempted first because its published text price is lower. All 11 requests returned HTTP 404, explicitly saying the model was no longer available to new users. There were no successful generations or returned usage for those calls; billed cost is unreported, not asserted to be zero. Failed-attempt wall time was 3.460s and is excluded from the successful-run latency above. Those artifacts remain in `runs/sustainability_11_gemini25_flash_lite/`. The runner now stops on permanent HTTP request/model errors.','',
'[Gemini results](runs/sustainability_11_gemini31_flash_lite/results.json) · [Evaluation](runs/sustainability_11_gemini31_flash_lite/evaluation.json) · [Raw requests/responses and usage](runs/sustainability_11_gemini31_flash_lite/calls.jsonl) · [Configuration](runs/sustainability_11_gemini31_flash_lite/config.json) · [Earlier stage-level report](extraction_experiment.md)','']
(here/'gemini_extraction_comparison.md').write_text('\n'.join(lines))
print(json.dumps({'matches':n,'metrics':metrics,'predictions':[(r['id'],r['company_name'],r['name_match_proxy']) for r in rows]},indent=2))
