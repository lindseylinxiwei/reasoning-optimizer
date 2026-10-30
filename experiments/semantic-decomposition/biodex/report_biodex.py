import json
from pathlib import Path
H=Path(__file__).resolve().parent
R=H/'runs'
train={k:json.loads((R/f'train-k{k}/summary.json').read_text()) for k in [32,64,128]}
frozen=json.loads((R/'selected_k.json').read_text());k=frozen['k']
test=json.loads((R/f'test-k{k}/summary.json').read_text())
lines=['# Biodex: embedding retrieval + Jev','',
'We compare embedding-only ranking, retrieval + Jev, and retrieval + Gemini on the same candidate labels and document evidence. Each paper produces up to 10 unique official reaction labels. The primary metric is the existing MOAR RP@5.','',
'## Method','',
'We embed the official vocabulary and overlapping paper passages using OpenAI `text-embedding-3-small`, the embedding model used in MOAR. Passages contain 256 cl100k tokens with 64-token overlap. A label is scored by its maximum cosine similarity to any passage; the top K labels become candidates. This is our label-retrieval method, not an existing MOAR retrieval plan. There is no upstream LLM extraction.','',
'- **Embedding only:** return labels in retrieval order.\n- **Jev:** ask a supported/not-supported question for each candidate, sharing the paper state across questions. Keep supported labels and sort by P(supported); copy exact official names.\n- **Gemini 3.1 Flash-Lite:** select and rank supported candidate IDs, then copy exact official names.','',
'All 40 training papers and 100 test papers fit the shared full-document evidence budget; no paper was truncated. The saved passage offsets were validated against source text. Both classifiers may abstain; embedding-only returns its top labels. Jev probabilities across questions are an uncalibrated ranking heuristic. Gemini returns a ranked list directly, so this compares the two selection/ranking stages rather than identical model output parameterizations.','',
'## Training selection (40 documents)','',
'The rule was frozen before results: choose the highest mean training RP@5 across Jev and Gemini, breaking ties toward smaller K. Training and test IDs are disjoint. All K values reuse embeddings; each ranker uses identical candidate prefixes and frozen evidence.','',
'| K | Candidate recall | Candidate oracle RP@5 | Embedding RP@5 | Jev RP@5 | Gemini RP@5 | Jev ranking USD / s | Gemini ranking USD / s |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for n,s in train.items():
 m=s['methods'];j=m['jev'];g=m['gemini']
 lines.append(f"| {n} | {s['candidate_recall']:.4f} | {s['oracle_rp5']:.4f} | {m['embedding']['evaluation']['avg_rp_at_5']:.4f} | {j['evaluation']['avg_rp_at_5']:.4f} | {g['evaluation']['avg_rp_at_5']:.4f} | ${j['known_api_cost_usd']:.6f} / {j['wall_seconds']:.2f} | ${g['known_api_cost_usd']:.6f} / {g['wall_seconds']:.2f} |")
lines += ['',f'**Frozen test setting: K={k}.** No test-set tuning.','',
'Gemini had one HTTP 503 failure at K=128 (PMID 33573668); it counts as a miss and its charge is unknown. No retry was made. Even assigning that document a perfect score would only tie the selected K=64 mean; the smaller-K tie rule gives the same decision. All other training calls completed successfully.','', '## Held-out test (100 documents)','',
'| Method | MOAR RP@5 | MOAR RP@10 | Errors | Ranking API cost (USD) | Ranking wall time (s) | Shared retrieval + ranking USD | Shared retrieval + ranking s |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for name,m in test['methods'].items():
 e=m['evaluation']
 lines.append(f"| {name} | {e['avg_rp_at_5']:.4f} | {e['avg_rp_at_10']:.4f} | {m['errors']} | ${m['known_api_cost_usd']:.6f} | {m['wall_seconds']:.2f} | ${m['standalone_known_api_cost_usd']:.6f} | {m['composed_wall_seconds']:.2f} |")
shared=test['shared_retrieval']
lines += ['',f"Shared test retrieval cost: **${shared['known_api_cost_usd']:.6f}**; wall time: **{shared['wall_seconds']:.2f}s**. This warm-cache run reuses the official-vocabulary embeddings built during training.",
 f"Candidate recall: **{test['candidate_recall']:.4f}**. Candidate oracle RP@5: **{test['oracle_rp5']:.4f}**. These expose how much quality is lost before classification.",'',
'## Evaluation and interpretation','',
'RP@5 is the number of correct unique labels in the first five predictions divided by min(5, number of ground-truth labels), then averaged across labeled papers. It does not distinguish ordering within the top five. Candidate recall measures the fraction of GT labels retrieved per paper; oracle RP@5 assumes perfect selection from the candidate set. No source-review corrections or alias adjudication are applied.','']
e,j,g=[test['methods'][m] for m in ['embedding','jev','gemini']]
lines += [f"Jev changes RP@5 by {j['evaluation']['avg_rp_at_5']-e['evaluation']['avg_rp_at_5']:+.4f} versus embedding-only, and {j['evaluation']['avg_rp_at_5']-g['evaluation']['avg_rp_at_5']:+.4f} versus Gemini. These are observed results from one run, not a statistical equivalence claim.", '',
'## Accounting and artifacts','',
'Costs use recorded token usage and explicit experiment rates: embeddings $0.02/M input; Jev $0.042/M input; Gemini $0.25/M input, $0.025/M cached input, $1.50/M output including thinking. These are token-equivalent estimates, not verified invoices. Local compute is counted as $0. Shared + ranking times are composed from one shared preparation plus each sequential ranker, not fresh end-to-end reruns.','']
all_ledgers=list(R.glob('train-prepared/embedding_calls.jsonl'))+list(R.glob('test-prepared/embedding_calls.jsonl'))
all_ledgers += [p for n in [32,64,128] for p in (R/f'train-k{n}').glob('*/calls.jsonl')]
all_ledgers += list((R/f'test-k{k}').glob('*/calls.jsonl'))
calls=[json.loads(line) for p in all_ledgers for line in p.read_text().splitlines()]
lines += [f"Total known API cost for training selection plus test: **${sum(c.get('cost_usd') or 0 for c in calls):.6f}**, with **{sum(c.get('cost_usd') is None for c in calls)} calls with unknown cost**. Shared retrieval is counted once in this session total.",
 f"Training vocabulary/preparation setup: ${train[32]['shared_retrieval']['known_api_cost_usd']:.6f}, {train[32]['shared_retrieval']['wall_seconds']:.2f}s. This setup is excluded from warm-cache test totals.",'',
'[Selection protocol](runs/protocol.json) · [Frozen K](runs/selected_k.json) · [Implementation](README.md)',
 f'[Test summary](runs/test-k{k}/summary.json) · [Shared inputs](runs/test-prepared/prepared.json)', '',
'Per-document outputs, raw API responses, request usage and latency are saved under each run directory. Errors are retained as empty predictions and count as misses; unknown costs are not assumed free.']
j=test['methods']['jev'];g=test['methods']['gemini']
lines[2:2]=[f"At the training-selected K={k}, test RP@5 is **{j['evaluation']['avg_rp_at_5']:.4f} for Jev**, **{g['evaluation']['avg_rp_at_5']:.4f} for Gemini**, and **{test['methods']['embedding']['evaluation']['avg_rp_at_5']:.4f} for embedding-only**. Including shared test retrieval, Jev costs **${j['standalone_known_api_cost_usd']:.6f}** and takes **{j['composed_wall_seconds']:.2f}s**, versus **${g['standalone_known_api_cost_usd']:.6f}** and **{g['composed_wall_seconds']:.2f}s** for Gemini.", '', f"Jev uses {100*(1-j['standalone_known_api_cost_usd']/g['standalone_known_api_cost_usd']):.1f}% less API cost and {100*(1-j['composed_wall_seconds']/g['composed_wall_seconds']):.1f}% less composed wall time in this run. The quality difference is small and is not evidence of statistical superiority. **Retrieval is the main quality bottleneck:** candidate recall is only {100*test['candidate_recall']:.1f}%, and even perfect ranking of these candidates is capped at RP@5={test['oracle_rp5']:.3f}. This experiment does not establish high absolute extraction quality.", '']
(H/'biodex_jev.md').write_text('\n'.join(lines)+'\n')
print(H/'biodex_jev.md')

if (R/'test-k64-pro-high/summary.json').exists():
    from report_pro import main as append_pro
    append_pro()
