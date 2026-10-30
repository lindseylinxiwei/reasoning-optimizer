"""Validate saved assistant adjudications and render the follow-up report; no API calls."""
import hashlib
import json
from pathlib import Path

H = Path(__file__).resolve().parent
P = H / 'runs/long_document_evidence_v1'
MARKER = '## Assistant-reviewed evaluation of all 100 documents'

def main():
    review = json.loads((P / 'semantic_review.json').read_text())
    rows = review['rows']
    docs = {r['id']: r['text'] for r in json.loads((P / 'test_input.json').read_text())}
    assert len(rows) == len({r['id'] for r in rows}) == 100
    assert {r['id'] for r in rows} == set(docs)
    for r in rows:
        text = docs[r['id']]
        assert hashlib.sha256(text.encode()).hexdigest() == r['source_sha256']
        for s in r['source_excerpts']:
            assert text[s['start']:s['end']] == s['text']
        for m in ('evidence_jev', 'gemini'):
            a = r[m]
            assert a['exact_source_substring'] == (a['name'] in text)
            assert a['whitespace_normalized_source_match'] == (' '.join(a['name'].split()) in ' '.join(text.split()))
    lines = [MARKER, '',
        'Follow-up review date: 2026-09-28. **All 100 documents and both outputs per document were reviewed against source excerpts**, including targeted searches for ambiguous names. The earlier 81/100 and 82/100 results are automatic string-similarity scores, not adjudicated semantic accuracy. Original predictions and GT are unchanged.', '',
        '**Rubric:** accept the report’s company, a source-supported brand/legal name, or an explicitly established reporting parent/group. Do not automatically equate arbitrary parents and subsidiaries. This scores entity identity, not completeness of the legal name or exact consolidation scope. Source-copy fidelity is measured separately.', '',
        '| Metric | GLiNER Small + evidence Jev | Gemini 3.1 Flash-Lite |',
        '|---|---:|---:|']
    metrics = [
        ('Confirmed correct company identity / all documents', lambda rs,m: f"{sum(r[m]['semantic_status']=='correct' for r in rs)}/{len(rs)}"),
        ('Ambiguous subject-versus-publisher cases', lambda rs,m: str(sum(r[m]['semantic_status']=='ambiguous' for r in rs))),
        ('Confirmed wrong-company outputs', lambda rs,m: str(sum(r[m]['semantic_status']=='incorrect' for r in rs))),
        ('Correct among 99 unambiguous documents', lambda rs,m: f"{sum(r[m]['semantic_status']=='correct' for r in rs)}/99"),
        ('Exact contiguous source-string occurrence', lambda rs,m: f"{sum(r[m]['exact_source_substring'] for r in rs)}/100"),
        ('Source occurrence after whitespace normalization', lambda rs,m: f"{sum(r[m]['whitespace_normalized_source_match'] for r in rs)}/100"),
        ('Automatic proxy negatives accepted after source review', lambda rs,m: str(sum(r[m]['semantic_status']=='correct' and not r[m]['original_proxy_match'] for r in rs))),
    ]
    for label, fn in metrics:
        lines.append(f"| {label} | {fn(rows,'evidence_jev')} | {fn(rows,'gemini')} |")
    long = [r for r in rows if r['estimated_document_tokens']>32000]
    lines += ['',f"On the **59 documents above 32K estimated tokens**, both methods have {sum(r['evidence_jev']['semantic_status']=='correct' for r in long)}/59 confirmed correct identities in this review. The ambiguous document is below 32K.", '',
        '**ID 6151 — Credit Suisse:** this is a thematic report on plastic pollution, produced by the Credit Suisse Center for Sustainability. Both methods return Credit Suisse, matching the publisher and existing GT. Under a strict subject-company task, NONE is defensible. We therefore retain one ambiguous case for each method: 99% confirmed correct over all 100, or 100% among the 99 unambiguous cases. If the task explicitly includes publisher extraction, this case can be accepted; that scope choice should be made before the next evaluation.', '',
        '**ID 7526 — FinDreams:** Gemini returns “FinDreams Battery Company Limited”; the source contains that full name across a line break. It is semantically supported and matches after whitespace normalization, but is not an exact contiguous copy. Jev’s returned “FinDreams Battery” occurs verbatim. String occurrence alone does not prove correct entity selection.', '',
        'Examples of automatic false negatives include Trex Company, Inc. versus Trex; Power Grid Corporation of India Limited versus PowerGrid; and Hilton Worldwide Holdings Inc. versus Hilton. The source also resolves scope-specific labels: Enel Russia PJSC is the subject of the Enel Russia report, while ATLAS Infrastructure is the reported organization whose governing body is GIP ATLAS Holdings Ltd.', '',
        '**Interpretation:** this review finds no confirmed identity-quality gap between the two methods on these 100 outputs. It strengthens the feasibility result for long-document extraction, but does not establish statistical equivalence or generalization. The original cost/latency result remains $0.180434 and 802.5s for the evidence method versus $1.496941 and 129.2s for Gemini. No new model calls were made for this review.', '',
        '**Limitations:** these are post-hoc, unblinded assistant judgments from source excerpts and targeted searches, not independent human-validated gold labels or exhaustive full-document readings. The review does not separately establish that every reduced evidence package was sufficient for Jev’s decision. A blinded human review and a frozen alias/scope rubric are appropriate before presenting these numbers as benchmark accuracy.', '',
        '[All 100 paired judgments, rationales, source hashes and exact source offsets](runs/long_document_evidence_v1/semantic_review.json). Offsets are zero-based, end-exclusive Python character indices into each `test_input.json` text. Regenerate this section with `python report_semantic_review.py` after editing adjudications.', '',
        '### Individual source-reviewed judgments', '',
        '| ID | Evidence Jev output | Gemini output | Judgment for both | Source-review rationale |',
        '|---|---|---|---|---|']
    def esc(v): return str(v).replace('|','\\|').replace('\n',' ')
    for r in rows:
        lines.append('| '+' | '.join(map(esc,[r['id'],r['evidence_jev']['name'],r['gemini']['name'],r['evidence_jev']['semantic_status'],r['rationale']]))+' |')
    section='\n'.join(lines)+'\n'
    (P/'semantic_review_summary.md').write_text(section)
    report=H/'long_document_evidence_report.md'
    old=report.read_text().split(MARKER)[0].rstrip()
    notice='> Follow-up: all 100 paired outputs now have a source-reviewed semantic evaluation; see [the reviewed results](#assistant-reviewed-evaluation-of-all-100-documents). The original automatic proxy results below are preserved.\n\n'
    if notice not in old:
        title, body=old.split('\n',1)
        old=title+'\n\n'+notice+body.lstrip('\n')
    report.write_text(old+'\n\n'+section)
    print(json.dumps({'documents_reviewed':len(rows),'source_excerpts_validated':sum(len(r['source_excerpts']) for r in rows),'confirmed_correct_each':99,'ambiguous_each':1,'long_documents':len(long)},indent=2))

if __name__=='__main__':
    main()
