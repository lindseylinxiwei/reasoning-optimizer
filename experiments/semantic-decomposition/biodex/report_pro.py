"""Append the stronger-model comparison without changing original experiment results."""
import json
from pathlib import Path
H=Path(__file__).resolve().parent
R=H/'runs'
marker='## Follow-up: Gemini 3.1 Pro with high thinking'

def main():
    original=json.loads((R/'test-k64/summary.json').read_text())
    follow=json.loads((R/'test-k64-pro-high/summary.json').read_text())
    pro=follow['methods']['gemini']
    base=json.loads((R/'test-k64/gemini/results.json').read_text())
    current=json.loads((R/'test-k64-pro-high/gemini/results.json').read_text())
    assert len(base)==len(current)==100
    for a,b in zip(base,current):
        assert a['id']==b['id'] and a['candidate_ids']==b['candidate_ids']
        assert a['evidence_sha256']==b['evidence_sha256']
    calls=[json.loads(s) for s in (R/'test-k64-pro-high/gemini/calls.jsonl').read_text().splitlines()]
    lines=[marker,'',
        'We ran `gemini-3.1-pro-preview` with **high thinking**, temperature 0, and a 32,768-token generation cap on the same 100 held-out papers. K remains 64, selected before this follow-up. Candidate IDs, paper evidence, task prompt and final output limit (10 labels) are unchanged. The output cap is higher than Flash-Lite’s 2,048 to allow reasoning; this tests the Pro/high configuration, not model identity alone.','',
        'Gemini 3.1 Pro is the current Pro model verified in the account’s model catalog. Newer Flash-family versions also exist; “Pro” is the selected flagship reasoning baseline, not a claim that it is best on every workload. [Model documentation](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview).','',
        '| Method | RP@5 | RP@10 | Errors | Ranking cost USD | Ranking time s | Including shared retrieval USD | Including shared retrieval s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,m in [(n,original['methods'][n]) for n in ['embedding','jev','gemini']]+[('Gemini 3.1 Pro / high',pro)]:
        label={'gemini':'Gemini 3.1 Flash-Lite / minimal','jev':'Jev'}.get(label,label)
        e=m['evaluation']
        lines.append(f"| {label} | {e['avg_rp_at_5']:.4f} | {e['avg_rp_at_10']:.4f} | {m['errors']} | ${m['known_api_cost_usd']:.6f} | {m['wall_seconds']:.2f} | ${m['standalone_known_api_cost_usd']:.6f} | {m['composed_wall_seconds']:.2f} |")
    score=pro['evaluation']['avg_rp_at_5'];oracle=follow['oracle_rp5']
    lines += ['',f"Pro/high RP@5 differs from Flash-Lite by **{score-original['methods']['gemini']['evaluation']['avg_rp_at_5']:+.4f}**, and from Jev by **{score-original['methods']['jev']['evaluation']['avg_rp_at_5']:+.4f}**. The shared candidate oracle remains **{oracle:.4f}**; a stronger model cannot recover missing labels. This is one run per configuration, not a significance test.",'',
        f"New follow-up API cost: **${pro['known_api_cost_usd']:.6f}**, plus **{pro['unknown_cost_calls']} unknown-cost calls**. Retrieval was reused, so its $0.012549 / 75.28s is included only in hypothetical standalone totals, not charged again to this run. Reported API latency is sequential wall time. Errors remain empty predictions and count as misses; no automatic retries or model fallback.",
        f"Reported token totals: input **{sum(c.get('response',{}).get('usageMetadata',{}).get('promptTokenCount',0) for c in calls):,}**, visible output **{sum(c.get('response',{}).get('usageMetadata',{}).get('candidatesTokenCount',0) for c in calls):,}**, thinking **{sum(c.get('response',{}).get('usageMetadata',{}).get('thoughtsTokenCount',0) for c in calls):,}**. Missing usage is not assumed zero for cost accounting.",
        'Pro prices used: $2/M input, $0.20/M cached input, $12/M output including thinking for prompts ≤200K tokens. All requests are below that threshold. [Official pricing](https://ai.google.dev/gemini-api/docs/pricing).','',
        '[Follow-up summary](runs/test-k64-pro-high/summary.json) · [Predictions](runs/test-k64-pro-high/gemini/results.json) · [API ledger](runs/test-k64-pro-high/gemini/calls.jsonl) · [Configuration](runs/test-k64-pro-high/config.json)']
    section='\n'.join(lines)+'\n'
    p=H/'biodex_jev.md'
    p.write_text(p.read_text().split(marker)[0].rstrip()+'\n\n'+section)
    print(section)

if __name__=='__main__':main()
