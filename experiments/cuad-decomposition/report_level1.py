"""Builds the Level 1 report as HTML (runs/level1/report/level1_report.html) for upload as a Google Doc."""
import base64
import html
import json
from pathlib import Path

from plot_level1 import code, frontier

HERE = Path(__file__).resolve().parent
L1 = HERE / "runs" / "level1"
scores = {r["plan"]: r for r in json.load(open(L1 / "scores.json"))}
check = json.load(open(L1 / "check.json"))["plans"]
configs = json.load(open(L1 / "configs.json"))
P = configs["proposals"]
front = {r["plan"] for r in frontier(list(scores.values()))}


def money(x):
    return f"${x:.3f}" if x < 1 else f"${x:.2f}"


def describe(plan):
    meta = configs["plans"][plan]["meta"]
    fam = configs["plans"][plan]["family"]
    if plan == "baseline":
        return "Baseline", "full contract, one map for all 41 categories"
    if fam == "view:head_tail":
        return "Head/tail view", f"first {meta['head_tokens']:,} + last {meta['tail_tokens']} tokens"
    if fam.startswith("view:topk"):
        method = "BM25" if meta["method"] == "fts" else "embedding"
        return "Top-k view", f"{method}, {meta['chunk_tokens']}-token chunks x {meta['k']}"
    if fam == "view:jev":
        return "Jev retrieval view", (f"8 targets, {meta['evidence_chars']:,} chars per target, "
                                      f"{meta['max_decisions']} decisions, {meta['policy'].replace('_', ' ')}")
    if fam == "task:presence_gate_jev":
        return "Presence gate", f"Jev checker, 36 checks, threshold 0.3 (instantiation {plan[-1]})"
    if fam == "task:presence_gate_llm":
        return "Presence gate", f"flash-lite checker, same 36 questions (instantiation {plan[-1]})"
    if fam == "task:isolating_subtasks":
        n = meta.get("groups") or 3
        how = f"{n} groups, LLM combine step"
        if plan in ("isolating_subtasks_1", "isolating_subtasks_2"):
            how += f" (directive default, instantiation {plan[-1]})"
        return "Isolating subtasks", how
    if fam == "task:isolating_structured":
        return "Isolating subtasks", f"{meta['groups']} groups, lists merged in code"
    if fam == "task:doc_chunking":
        ctx = "all earlier chunks as context (agent's config)" if plan.startswith("chunking_agent") else "previous chunk as context"
        return "Chunking", f"{meta['chunk_tokens'] // 1000}k-token chunks, {ctx}"
    if fam == "task:chunk_header_summary":
        return "Chunking + headers", f"{meta['chunk_tokens'] // 1000}k-token chunks, previous chunk + headers"
    return fam, ""


ORDER = (["baseline"] + [f"head_tail_{i}" for i in range(1, 6)] + [f"topk_bm25_{i}" for i in range(1, 6)]
         + [f"topk_emb_{i}" for i in range(1, 6)] + ["topk_emb_3_half_chunks"] + [f"jev_{i}" for i in range(1, 6)]
         + ["jev_3_adaptive", "gate_jev_1", "gate_jev_2", "gate_llm_1", "gate_llm_2", "isolating_subtasks_1",
            "isolating_subtasks_2", "isolating_subtasks_g6", "isolating_subtasks_g8", "isolating_subtasks_g12",
            "isolating_structured_1", "isolating_structured_g6", "isolating_structured_g8",
            "isolating_structured_g12"] + [f"chunking_{t}" for t in (2000, 4000, 8000, 16000)]
         + ["chunking_agent_10000"] + [f"chunk_header_{t}" for t in (2000, 4000, 8000, 16000)])
assert set(ORDER) == set(scores), set(scores) ^ set(ORDER)


def table():
    head = ["Code", "Technique", "Configuration", "Text kept", "Cost", "F1", "Macro F1", "Precision", "Recall",
            "Absent clauses left empty"]
    rows = ["<tr>" + "".join(f"<td><b>{h}</b></td>" for h in head) + "</tr>"]
    for plan in ORDER:
        r = scores[plan]
        tech, conf = describe(plan)
        kept = check[plan]["kept_share_median"]
        cells = [code(plan), tech, conf, f"{kept:.0%}" if kept is not None else "100%", money(r["cost"]),
                 f"{r['f1']:.3f}", f"{r['f1_macro']:.3f}", f"{r['precision']:.2f}", f"{r['recall']:.2f}",
                 f"{r['negative_task_accuracy']:.2f}"]
        cells = [html.escape(c) for c in cells]
        if plan in front:
            cells = [f"<b>{c}</b>" for c in cells]
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return "<table border='1' style='border-collapse:collapse'>" + "".join(rows) + "</table>"


def ul(items):
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


img = base64.b64encode((L1 / "report" / "frontier_level1_small.png").read_bytes()).decode()
s = scores
ht, tk, jv = P["head_tail"], P["topk"], P["jev"]["sweep"]
targets = "; ".join(f"{html.escape(t['name'])}" for t in jv["targets"])
total = sum(r["cost"] for r in scores.values())
est = sum(check[p]["estimated_usd"] for p in scores)
agent = sum(configs["agent_cost"].values())

doc = f"""<html><body>
<h1>Rewrite sweep on the MOAR CUAD workload: Level 1</h1>
<p>2026-11-03. Branch sustainability-decomposition of reasoning-optimizer. Follows the Sustainability data decomposition sweep.</p>

<h2>Summary</h2>
<p>On CUAD, splitting the task helps and trimming the text does not. Every data view scored below the full-contract
baseline (F1 0.535), at every level of every technique, because the 41 clause types sit anywhere in a contract: trimming
raised precision but lost more recall. The best view, Jev retrieval keeping about 70% of the text, reached 0.493 for
$0.23, slightly cheaper than the baseline's $0.28.</p>
<p>The presence gate and isolating subtasks raised F1 by 4 to 6 points. A presence gate with a Jev checker is the cheapest gain: 0.571 to 0.576 for
$0.32, about $0.04 more than the baseline. The same gate with a flash-lite checker reached 0.590 to 0.592 for $0.50, and
isolating subtasks with 3 groups reached 0.574 to 0.599 for $0.83 to $0.84. Splitting into 6, 8 or 12 groups, merging
the groups' lists in code, smaller chunks and chunk headers gave no further gain.</p>
<p>The Pareto frontier is H1, H2, H3, H5, J5, the baseline, GJ2, GL1 and I3a. Recall remains the limit (at most 0.56 in
any plan). Level 2 should combine the gate with isolating subtasks, chunking and the Jev view, and the leading plans
should be rerun to measure run-to-run noise.</p>

<h2>Setup</h2>
{ul([
 "<b>Pipeline:</b> the MOAR CUAD pipeline: one map reads the whole contract and returns, for each of 41 clause categories, the matching text spans. The output format was changed from one comma-joined string per category to one entry per category with a list of spans; the category definitions are unchanged.",
 "<b>Model:</b> every LLM step in every plan runs on gemini-3.5-flash-lite. Jev steps run on jev-1.13.0.",
 "<b>Data:</b> the 40 train contracts of the original MOAR CUAD runs (data/train/cuad.json): 1,640 contract-category tasks, 546 with a clause and 1,094 without, and 1,142 annotated quotes. The median contract is 34,461 characters (about 7,700 tokens); the longest is 266,213. The pipeline and the agent see only the contract text, file name and id.",
 "<b>Evaluation:</b> official CUAD matching on the returned quotes. A true quote is found if a predicted quote shares at least half of its words (Jaccard 0.5); each unmatched prediction is a false positive. F1 is pooled over all 1,640 tasks; macro F1 averages the 41 per-category F1 scores. Absent clauses left empty is the share of the 1,094 tasks without a clause where the plan returned nothing. These are fixed-output scores, not CUAD's confidence-swept AUPR.",
 "<b>Cost:</b> the API cost DocETL reports for each plan, from LiteLLM's price table for Gemini and $0.042 per million input tokens for Jev.",
 "<b>Runs:</b> each plan ran once on Modal, one container per plan, with the cache bypassed.",
])}

<h2>How the plans were chosen</h2>
<p>An agent (gemini-3.8-flash) received a label-free profile of the contracts (lengths, heading styles with real
example lines, paragraph statistics, excerpts) together with the pipeline's prompt. For the data views it used the same
proposal prompt as the Sustainability sweep, with limits changed for CUAD: each view keeps 15% to 70% of the median
contract (Sustainability: at most 30%), Jev gets 6 to 10 targets, and top-k chunks are 300 to 600 tokens. A first
proposal under the Sustainability limits kept at most 2,300 tokens of a contract and was replaced.</p>
<p>The task rewrites are MOAR's own directives, instantiated the way the MCTS search does it, plus the new presence gate
directive. Four comparisons were derived in code so that one setting changes at a time: the gate's Jev and LLM checkers
use the same questions; isolating subtasks were also run at exactly 6, 8 and 12 groups (the directive asks for 2 to 4);
the code-merge variants use the same groups and prompts as their LLM-combine counterparts; and chunking was run at a
fixed ladder of chunk sizes. The presence gate and the 3-group isolating plan were instantiated twice to see how much
the agent's choices vary.</p>
<p>Gleaning, clarify instructions, both document compression directives, chaining, summarization and the arbitrary
rewrite were left out. Before any paid run every plan was checked offline: DocETL accepted all 45, and the upper cost
estimate was ${est:.2f}. The agent proposals cost ${agent:.2f}.</p>

<h2>Methods</h2>
<h3>Head/tail view</h3>
<p>A code step keeps the first N tokens and the last M tokens of each contract. The agent set M to {ht['tail_tokens']}
tokens because governing law, assignment and third-party clauses often sit near the end, and chose N =
{', '.join(f'{x:,}' for x in ht['head_levels'])} tokens.</p>
<h3>Top-k view</h3>
<p>The contract is split into {tk['chunk_tokens']}-token chunks, each chunk is scored against one query, the k best are
kept and joined in document order. Every level was run with both BM25 keyword matching and gemini-embedding-001
embeddings, k = {', '.join(map(str, tk['k_levels']))}. The query lists the main clause topics:
"{html.escape(tk['query'])}". A control keeps the level-3 budget with chunks half the size.</p>
<h3>Jev retrieval view</h3>
<p>Each contract is split into paragraphs (the named heading styles failed the index checks on CUAD: numbered headings
match 20% of lines, and other styles appear in few contracts), pieces under 800 characters are merged, pieces over 2,400
are cut at sentence boundaries, and at most 8 children are grouped per node. Jev scores 4 sibling regions per request
against 8 targets, one per group of clause types: {targets}. The levels set the cap per target to
{', '.join(f'{x:,}' for x in jv['cap_levels'])} characters, with {', '.join(map(str, jv['decision_levels']))} navigation
decisions per target. J3-adapt keeps only regions scoring 0.3 or more instead of filling up to the cap.</p>
<h3>Presence gate</h3>
<p>A checker first decides, for each contract and each gated category, whether the contract contains that kind of clause.
The extraction prompt then shows only the categories judged present, and says the others are absent. The agent gated 36
categories and kept document name, parties, agreement date, effective date and governing law ungated. With the Jev
checker, Jev answers all 36 questions on 16,000-character chunks of the contract (a category is present if any chunk
scores 0.3 or more); with the LLM checker, one flash-lite call answers the same 36 questions over the whole contract.</p>
<h3>Isolating subtasks</h3>
<p>The categories are split into groups, and each group runs as its own prompt over the full contract in parallel. In
MOAR's directive each group returns plain text and a final LLM step rebuilds the clauses list from it; the code-merge
variants have each group return the clauses list directly and concatenate the lists in code. The agent grouped the
categories by topic, for example basic terms, restrictive covenants, IP and licensing, and liability.</p>
<h3>Chunking and chunk headers</h3>
<p>The contract is split into chunks, the extraction prompt runs on each chunk with the previous chunk as context, and a
reduce merges the chunk results. Chunk sizes are 2,000, 4,000, 8,000 and 16,000 tokens. C10k-agent is the agent's own
configuration: 10,000-token chunks with every earlier chunk as context. Chunk headers add a step that extracts each
chunk's section headings and a summary, which are passed along as context.</p>

<h2>Results</h2>
<p><b>Figure 1. Cost against F1 for all 45 plans.</b> The line is the Pareto frontier.</p>
<p>[Figure 1 goes here: runs/level1/report/frontier_level1.png]</p>
<p>Table 1 lists every plan. Bold rows are on the Pareto frontier. Codes match Figure 1: H is head/tail, B top-k with
BM25, E top-k with embeddings, J Jev, GJ and GL the gate with a Jev or LLM checker, I isolating subtasks with the LLM
combine step, IS with the code merge, C chunking and CH chunking with headers; the number is the level, group count or
chunk size. Text kept is the offline estimate for the median contract.</p>
{table()}

<h2>Findings</h2>
<h3>Trimming the contract lowered F1 at every level</h3>
<p>F1 rises with the share of text kept, and every view stays below the baseline. At about 70% kept, Jev scored 0.493,
head/tail 0.449, BM25 0.443 and embeddings 0.413. Most views raised precision (up to 0.82, against 0.66) and all of them
lowered recall (0.09 to 0.37, against 0.45). Unlike Sustainability, where the evidence sits on the first pages, CUAD clauses are spread
across the contract: the median annotated quote sits 38% of the way in.</p>
<h3>Among the views, Jev and BM25 did best for their cost</h3>
<p>Jev was the best view at the two largest levels. BM25 beat embeddings at every level and cost less. J3-adapt
collapsed to 0.161 because keeping only regions scoring 0.3 or more returned little text.</p>
<h3>The Jev presence gate is the cheapest improvement</h3>
<p>Both Jev-gate plans scored 0.571 and 0.576, against 0.535, for about $0.04 more than the baseline. Most of the gain is
recall (0.52 against 0.45): with fewer categories in the prompt, the model finds more clauses in the ones it is shown.
The LLM checker also raised precision (0.70 to 0.71, against 0.66 for the baseline and 0.63 to 0.65 for the
Jev gate) and scored 0.590 to 0.592, at $0.50.</p>
<h3>Three groups are enough for isolating subtasks</h3>
<p>Isolating subtasks with 3 groups scored 0.599 and 0.574 in two instantiations. 6, 8 and 12 groups scored 0.593,
0.594 and 0.592 at up to 2.8 times the cost. Merging the groups' lists in code did not beat the LLM combine step
(0.562, 0.596, 0.551 and 0.578 for 3, 6, 8 and 12 groups).</p>
<h3>Chunking helps slightly; chunk headers do not</h3>
<p>Chunking scored 0.541 to 0.580, best at 16,000 tokens (0.576) and with the agent's own configuration (0.580).
Smaller chunks did not help. Chunk headers scored 0.537 to 0.559 at about twice the cost of chunking.</p>
<h3>Recall is the remaining limit</h3>
<p>No plan found more than 56% of the annotated quotes, and the best task rewrites raised recall by 6 to 11 points over
the baseline.</p>

<h2>Limitations</h2>
{ul([
 "Each plan ran once. The two instantiations of the same rewrite differ by up to 2.5 F1 points (0.599 and 0.574), so gaps under about 3 points between task rewrites are within noise. Bootstrap intervals over contracts have not been computed yet.",
 "The 40 contracts are reused development data: the original MOAR runs and the earlier 8-type Jev experiments used them. They do not show held-out performance.",
 "One model, gemini-3.5-flash-lite, for every LLM step.",
 "Scores are fixed-output precision, recall and F1 with official CUAD matching, not CUAD's confidence-swept AUPR.",
 "Costs use LiteLLM's price table for Gemini and the documented Jev price, not billing records.",
])}

<h2>Problems found and fixed during the experiment</h2>
{ul([
 "The CUAD evaluator scored a perfect answer 0.92. It dropped 14 of 41 categories whose column names kept hyphens or slashes, dropped categories where every answer was wrong instead of scoring them 0, never counted a wrong span as a miss, and scored a correctly omitted clause with a similar clause's span. It now uses official CUAD matching and reproduces the earlier 8-type Jev results exactly (CUAD_EVAL.md).",
 "The pipeline asked for several spans as one comma-joined string. Under official matching that cost 21 F1 points even with perfect spans, so each category now returns a list of spans. The evaluator also maps singular and plural category names.",
 "The first data-view proposals followed Sustainability's limit of 30% of the median document and kept at most 2,300 tokens of a contract. The limits were changed for CUAD and the views re-proposed.",
 "The agent configured chunking so that every chunk also carries all earlier chunks, which makes cost grow with the square of the chunk count. The chunk-size ladder uses only the previous chunk as context; the agent's configuration ran as one extra plan.",
 "The arbitrary rewrite directive produced invalid pipeline edits twice, and its valid instance only changed the model, which the sweep fixes anyway. It was dropped.",
 f"The offline cost estimate was ${est:.2f}; the actual cost was ${total:.2f}. Jev's estimate assumes every target uses all its navigation decisions.",
])}

<h2>Cost</h2>
{ul([
 f"45 plans: ${total:.2f}.",
 f"Agent proposals: ${agent:.2f}, not counting two failed arbitrary-rewrite attempts.",
 "Smoke tests of the new output format and the presence gate: about $0.14.",
 "Modal compute: small; see the workspace dashboard.",
])}

<h2>Reproducing</h2>
<p>Code is in experiments/cuad-decomposition/ of the reasoning-optimizer repository. The data views are in
docetl/reasoning_optimizer/data_views.py, the presence gate in docetl/operations/presence_check.py and
docetl/reasoning_optimizer/directives/presence_gate.py, and the evaluator in experiments/reasoning/evaluation/cuad.py.</p>
{ul([
 "Propose the plans: python experiments/cuad-decomposition/sweep.py propose",
 "Check them offline: python experiments/cuad-decomposition/sweep.py check",
 "Run on Modal: MODAL_PROFILE=personal modal run experiments/cuad-decomposition/sweep.py --max-usd 50",
 "Score and plot: sweep.py score, then plot_level1.py (writes runs/level1/report/frontier_level1.png)",
])}
</body></html>"""
out = L1 / "report" / "level1_report.html"
out.write_text(doc)
print(out, len(doc))
