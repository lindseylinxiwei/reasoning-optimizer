"""Evaluate saved long-document experiments; never performs inference."""
import hashlib,importlib.util,json,statistics
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[2];P=H/'runs/long_document_evidence_v1'
def read(p):return json.loads(p.read_text())
def ledger(folder):
 p=folder/'calls.jsonl'
 return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
def accounting(calls):
 return {'calls':len(calls),'known_cost':sum(c.get('cost_usd') or 0 for c in calls),
 'unknown_cost_calls':sum(c.get('cost_usd') is None for c in calls),
 'latency_sum':sum(c.get('wall_seconds',0) for c in calls),
 'input_tokens':sum(c.get('response',{}).get('usage',{}).get('input_tokens',c.get('response',{}).get('usageMetadata',{}).get('promptTokenCount',0)) for c in calls)}
def esc(v):return str(v).replace('|','/').replace('\n',' ')

def main():
 spec=importlib.util.spec_from_file_location('moar',ROOT/'experiments/reasoning/evaluation/sustainability.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 gt={r['id']:r['GT company_name'] for r in read(ROOT/'experiments/reasoning/data/company_reports_gt.json')}
 def matched(name,i):return m.calculate_name_similarity((name or '').lower(),gt[i].lower())>=.5
 pilot=read(P/'pilot_input.json');test=read(P/'test_input.json');assert len(test)==100
 full={r['id']:r for r in read(P/'pilot_full/results.json')}
 retry=read(P/'pilot_full_retry/results.json') if (P/'pilot_full_retry/results.json').exists() else []
 for r in retry:full[r['id']]=r
 evidence={r['id']:r for r in read(P/'pilot_evidence/results.json')}
 test_e=read(P/'test_evidence/results.json');test_g=read(P/'test_gemini/results.json')
 assert len(test_e)==len(test_g)==100
 inputs={r['id']:r['text'] for r in test};gmap={r['id']:r for r in test_g}
 calls=ledger(P/'test_evidence');bysha={}
 for c in calls:bysha.setdefault(c['document_sha256'],[]).append(c)
 evals=[]
 for r in test_e:
  i=r['id'];t=inputs[i];g=gmap[i]
  for c in r.get('candidates',[]):
   for span in c['mentions']:assert t[span['start']:span['end']]==c['text']
  if r['status']=='selected':
   span=r['selected_span'];assert t[span['start']:span['end']]==r['company_name']
  lc=bysha.get(hashlib.sha256(t.encode()).hexdigest(),[])
  evals.append({'id':i,'ground_truth':gt[i],'tokens':r['estimated_document_tokens'],
    'candidate_count':len(r.get('candidates',[])),
    'candidate_coverage_proxy':any(matched(c['text'],i) for c in r.get('candidates',[])),
    'evidence_name':r.get('company_name'),'evidence_status':r['status'],'evidence_match':matched(r.get('company_name'),i),
    'gemini_name':g.get('company_name'),'gemini_status':g['status'],'gemini_match':matched(g.get('company_name'),i),
    'extraction_seconds':r.get('extraction_seconds'), 'selection_seconds':r.get('selection_copy_seconds'),
    'document_seconds':r['document_seconds'],'expanded':any(x['expansion_pass'] for x in (r.get('selection') or {}).get('rounds',[])),
    'jev_accounting':accounting(lc),'gemini_cost':g.get('cost_usd')})
 (P/'evaluation.json').write_text(json.dumps(evals,indent=2,ensure_ascii=False)+'\n')
 review=[]
 for r in test_e:
  i=r['id'];lc=bysha.get(hashlib.sha256(inputs[i].encode()).hexdigest(),[])
  selected=(r.get('selection') or {}).get('candidate_id')
  packets=[]
  for call in lc:
   for q in call['request']['questions'].values():
    if selected and selected!='NONE' and selected in q['criteria']:
     packets.append(json.loads(q['criteria'][selected]))
  review.append({'id':i,'ground_truth':gt[i],'selected_name':r.get('company_name'),
      'opening':inputs[i][:4000],'selected_candidate_evidence_by_round':packets,
      'evidence_sufficiency_label':None,'review_status':'pending; no automatic sufficiency claim'})
 (P/'evidence_review.json').write_text(json.dumps(review,indent=2,ensure_ascii=False)+'\n')
 # Mechanically verify evidence and budget compliance for every actual request.
 import tiktoken
 enc=tiktoken.get_encoding('cl100k_base');tokens=lambda s:len(enc.encode(s,disallowed_special=()))
 max_pair,max_request=0,0
 all_evidence_calls=ledger(P/'pilot_evidence')+calls
 all_inputs={r['id']:r['text'] for r in pilot+test};texts={hashlib.sha256(t.encode()).hexdigest():t for t in all_inputs.values()}
 for c in all_evidence_calls:
  req=c['request'];t=texts[c['document_sha256']];state=json.loads(req['state']);o=state['opening'];assert t[o['start']:o['end']]==o['text']
  sizes=[]
  for name,q in req['questions'].items():
   assert len(q['criteria'])<=255
   for key,value in q['criteria'].items():
    if key=='NONE':continue
    ev=json.loads(value)
    for span in ev['passages']:assert t[span['start']:span['end']]==span['text']
   sizes.append(tokens(json.dumps({name:q},ensure_ascii=False)))
  max_pair=max(max_pair,tokens(req['state'])+max(sizes));max_request=max(max_request,tokens(req['state'])+sum(sizes))
 assert max_pair<=24000 and max_request<=48000
 fa=accounting(ledger(P/'pilot_full')+ledger(P/'pilot_full_retry'));ea=accounting(ledger(P/'pilot_evidence'));ta=accounting(calls);ga=accounting(ledger(P/'test_gemini'))
 em=read(P/'test_evidence/metrics.json');gm=read(P/'test_gemini/metrics.json')
 lines=['# Long-document company extraction using candidate-focused evidence','',
 '## Scope and method','',
 'This experiment tests whether Jev can select a report’s subject company without receiving the full report. GLiNER Small scans the full original text on MPS using batch size 16, 320 word/punctuation-unit windows and overlap 64. No documents, candidates, or text windows are discarded based on length. Deterministic code collects a 4,000-character opening and up to two occurrence-centered passages per candidate (180 characters on each side). It starts with the first mention, then chooses the mention farthest from those already selected; overlapping intervals are merged. Names, occurrence counts, relative first position and original offsets accompany the verbatim passages.','',
 'Candidates are packed into choice questions by both token budget (10K estimated question tokens) and option count (254 candidates + NONE). Questions share the opening state and are packed into requests under conservative 24K state-plus-question and 48K total-token guards. Winners advance with their evidence; source offsets are copied after selection. If selection returns NONE, one expansion pass uses up to four dispersed mentions per candidate. There is no uncalibrated confidence cutoff, generated summary, GT-guided retrieval, or automatic API retry. Evidence expansion can add nothing for candidates with few mentions.','',
 'Evidence rules and source hashes were frozen before the held-out run. The pilot reuses exactly the same saved candidates for both selectors. The held-out evaluation uses the entire existing 100-document test split, disjoint from the 11 pilot documents, including annual, financial, half-year and sustainability reports. This evaluates standalone company extraction; no upstream sustainability filter is applied. Gemini 3.1 Flash-Lite receives each full document without candidates, uses minimal thinking and the same direct-extraction baseline prompt as earlier experiments. It runs after local evidence extraction finishes, to avoid overlapping benchmark workloads.','',
 '## Paired 11-document pilot','',
 '| Method | Name-match proxy | Selected | Known API cost (USD) | Calls incl. retry/failures | Sum of call latency (s) |',
 '|---|---:|---:|---:|---:|---:|']
 for label,rs,a in [('Full-document Jev',full,fa),('Evidence-based Jev',evidence,ea)]:
  lines.append(f"| {label} | {sum(matched(r.get('company_name'),i) for i,r in rs.items())}/11 | {sum(r['status']=='selected' for r in rs.values())}/11 | ${a['known_cost']:.9f} | {a['calls']} | {a['latency_sum']:.3f} |")
 lines+=['','The full-document pilot had a transport failure; its failed document was explicitly retried in a separate artifact directory. The table includes both attempts’ available charges and elapsed call time, and uses the retry result for matching. Unknown-cost calls are reported in session accounting. Pilot extraction was reused and is excluded from both selector measurements. One run per selector cannot establish a statistically reliable quality difference.','',
 '| ID | Existing GT | Full-document output | Evidence output |','|---|---|---|---|']
 for r in pilot:
  i=r['id'];lines.append(f"| {i} | {esc(gt[i])} | {esc(full[i].get('company_name'))} | {esc(evidence[i].get('company_name'))} |")
 lines+=['','## Held-out 100-document results','',
 '| Method | Name-match proxy | Selected | Abstentions | Errors | Known API cost (USD) | Pipeline time (s) |',
 '|---|---:|---:|---:|---:|---:|---:|']
 for label,rs,key,a,wall in [('GLiNER Small + evidence Jev',test_e,'evidence_match',ta,em['pipeline_wall_seconds']),('Gemini 3.1 Flash-Lite, full document',test_g,'gemini_match',ga,gm['pipeline_wall_seconds'])]:
  lines.append(f"| {label} | {sum(x[key] for x in evals)}/100 | {sum(r['status']=='selected' for r in rs)} | {sum(r['status']=='abstained' for r in rs)} | {sum(r['status']=='error' for r in rs)} | ${a['known_cost']:.9f} | {wall:.3f} |")
 lines+=['',f"Evidence-method candidate coverage proxy: **{sum(x['candidate_coverage_proxy'] for x in evals)}/100**. Final name-match proxy conditional on candidate coverage: **{sum(x['evidence_match'] and x['candidate_coverage_proxy'] for x in evals)}/{sum(x['candidate_coverage_proxy'] for x in evals)}**. Evidence expansion occurred in {sum(x['expanded'] for x in evals)} document results. Local extraction summed to {sum(x['extraction_seconds'] or 0 for x in evals):.3f}s, selection/copy to {sum(x['selection_seconds'] or 0 for x in evals):.3f}s; model loading was {em['load_seconds']:.3f}s separately. The held-out evidence runner is sequential, not the earlier overlap scheduler; these timings measure this implementation rather than an optimized serving comparison.",'',
 '### Results by original document length','',
 'Lengths below are cl100k_base estimates, not the provider tokenizer. A document above 32K estimated tokens illustrates the intended long-document regime; this is not a provider-confirmed full-context rejection. Even shorter documents may exceed context once questions are added.','',
 '| Estimated document tokens | Documents | Candidate coverage proxy | Evidence Jev match | Gemini match | Evidence API cost (USD) |','|---|---:|---:|---:|---:|---:|']
 for label,lo,hi in [('≤24K',0,24000),('24K–32K',24000,32000),('32K–64K',32000,64000),('64K–128K',64000,128000),('>128K',128000,float('inf'))]:
  rs=[x for x in evals if lo<x['tokens']<=hi];n=len(rs)
  lines.append(f"| {label} | {n} | {sum(x['candidate_coverage_proxy'] for x in rs)}/{n} | {sum(x['evidence_match'] for x in rs)}/{n} | {sum(x['gemini_match'] for x in rs)}/{n} | ${sum(x['jev_accounting']['known_cost'] for x in rs):.9f} |")
 lines+=['',f"Maximum observed evidence request sizes: **{max_pair:,}** estimated tokens for state + longest question and **{max_request:,}** for state + all questions. Every saved evidence passage was checked against its original source offsets, and every choice question had ≤255 options. The longest test document had **{max(x['tokens'] for x in evals):,}** estimated tokens.",'',
 '## Interpretation and limitations','',
 '- Name-match scores use the existing MOAR similarity function at threshold ≥0.5, aligned by document ID. They are automatic proxies; the follow-up source review, when available below, separately adjudicates company identity. Errors and abstentions count as unmatched.',
 '- Candidate coverage uses that same approximate matching rule. It is not strict entity recall. Evidence sufficiency has **not** been independently annotated: finding a candidate name in a passage does not prove that the passage establishes report subjecthood.',
 '- Eliminating full-document context overflow does not establish unchanged semantic accuracy. Selection can fail because GLiNER misses the name, deterministic retrieval omits the decisive passage, or Jev chooses incorrectly. These causes require source/evidence review.',
 '- Full-document GLiNER scanning still scales with document length; evidence reduction primarily bounds Jev input, not local extraction time. More candidates can require additional requests and tournament rounds.',
 '- One fixed test split and one run per method are not a generalization or statistical significance guarantee. Batch/round boundaries and API response variability may affect outcomes.',
 '- Monetary comparisons count local computation as $0 and use published paid-tier token prices (Jev $0.042/M input; Gemini $0.25/M input, $0.025/M cached input, $1.50/M output including thinking). These are token-equivalent costs, not verified invoices. Unknown usage is not assumed free.','',
 '## Session accounting and artifacts','']
 total=sum(a['known_cost'] for a in [fa,ea,ta,ga]);unknown=sum(a['unknown_cost_calls'] for a in [fa,ea,ta,ga])
 lines += [f"Total known API cost for this evidence experiment, including pilot, explicit pilot retry and both test methods: **${total:.9f}**, plus any unreported charges from **{unknown}** calls without usage. Previous experiments are excluded. Local processing is $0 by convention.",'',
 '[Frozen policy and source hashes](runs/long_document_evidence_v1/config.json) · [Per-document evaluation](runs/long_document_evidence_v1/evaluation.json) · [Evidence review packets (unannotated)](runs/long_document_evidence_v1/evidence_review.json) · [Evidence outputs](runs/long_document_evidence_v1/test_evidence/results.json) · [Jev request/response ledger](runs/long_document_evidence_v1/test_evidence/calls.jsonl) · [Gemini outputs](runs/long_document_evidence_v1/test_gemini/results.json) · [Gemini ledger](runs/long_document_evidence_v1/test_gemini/calls.jsonl)','',
 '## Per-document held-out outputs','',
 '| ID | Est. tokens | Candidates | Existing GT | Evidence output | Evidence match | Gemini output | Gemini match |','|---|---:|---:|---|---|---|---|---|']
 for x in evals:lines.append(f"| {x['id']} | {x['tokens']:,} | {x['candidate_count']} | {esc(x['ground_truth'])} | {esc(x['evidence_name'])} | {x['evidence_match']} | {esc(x['gemini_name'])} | {x['gemini_match']} |")
 long_rows=[x for x in evals if x['tokens']>32000]
 findings=['## Main findings', '',
  f"The evidence method completed all {len(test_e)} held-out documents, including {len(long_rows)} above 32K estimated tokens, without context errors or abstentions. The longest was {max(x['tokens'] for x in evals):,} estimated tokens. This supports the feasibility of bounded-context selection over long reports; it does not validate evidence sufficiency for every decision.", '',
  f"The name-match proxy was {sum(x['evidence_match'] for x in evals)}/100 versus {sum(x['gemini_match'] for x in evals)}/100 for Gemini. On the >32K subset it was {sum(x['evidence_match'] for x in long_rows)}/{len(long_rows)} versus {sum(x['gemini_match'] for x in long_rows)}/{len(long_rows)}. These small differences are not evidence of semantic equivalence or statistical superiority, particularly before alias review.", '',
  f"API cost fell by {100*(1-ta['known_cost']/ga['known_cost']):.1f}% (${ta['known_cost']:.6f} vs ${ga['known_cost']:.6f}), with local compute priced at $0. End-to-end execution was {em['pipeline_wall_seconds']/gm['pipeline_wall_seconds']:.2f}× slower ({em['pipeline_wall_seconds']:.1f}s vs {gm['pipeline_wall_seconds']:.1f}s). Local extraction remains the principal bottleneck.", '',
  f"Evidence Jev made {ta['calls']} requests and reported {ta['input_tokens']:,} input tokens across all batches/rounds; Gemini made {ga['calls']} requests and reported {ga['input_tokens']:,} prompt tokens. These token counts use different provider tokenizers and should not be interpreted as an exact text-compression ratio. The cost advantage also reflects Jev's lower token price. No held-out document invoked the NONE-triggered expansion, so the expansion path has offline test coverage but no empirical quality evidence from this run.", '']
 lines[2:2]=findings
 (H/'long_document_evidence_report.md').write_text('\n'.join(lines)+'\n')
 if (H/'runs/long_document_evidence_v1/semantic_review.json').exists():
  from report_semantic_review import main as render_semantic_review
  render_semantic_review()
 print(json.dumps({'pilot_full_matches':sum(matched(r.get('company_name'),i) for i,r in full.items()),'pilot_evidence_matches':sum(matched(r.get('company_name'),i) for i,r in evidence.items()),'test_evidence_matches':sum(x['evidence_match'] for x in evals),'test_gemini_matches':sum(x['gemini_match'] for x in evals),'long_documents':sum(x['tokens']>32000 for x in evals),'accounting':{'pilot_full':fa,'pilot_evidence':ea,'test_evidence':ta,'test_gemini':ga},'total_known_cost':total,'unknown_calls':unknown},indent=2))

if __name__=='__main__':main()
