"""Per-model effective length on CUAD, from the first-layer sweep (no new model calls).

For each model: pooled recall (official CUAD matching) over contracts in length bins (cl100k tokens).
A bin passes when its recall is at least 85% of the model's recall on contracts up to 8K tokens
(a relative version of RULER's threshold). The effective length is the top of the longest run of
passing bins starting from the shortest; the train contracts reach 56K tokens, but only 8 exceed
16K, so values above 16K are not claimed.
Offline analysis with labels: it describes each model's capability on long inputs, not where the
answers are. Usage (repo root): python experiments/cuad-decomposition/measure_effective_length.py
"""
import json
import sys
from pathlib import Path

import tiktoken

sys.path.insert(0, "experiments/reasoning")
from evaluation.cuad import (CUAD_METRICS, _doc_key, load_cuad_gold, official_task_counts,  # noqa: E402
                             predictions_from_clauses)

SWEEP = Path("experiments/cuad-decomposition/runs/first_layer_sweep")
BINS = [(0, 8000), (8000, 16000), (16000, 10**9)]
PASS = 0.85


def main():
    enc = tiktoken.get_encoding("cl100k_base")
    docs = json.load(open("experiments/reasoning/data/train/cuad.json"))
    gold = load_cuad_gold("experiments/reasoning/data/CUAD-master_clauses.csv")
    length = {_doc_key(d["name"]): len(enc.encode(d["document"], disallowed_special=())) for d in docs}
    print("contracts per bin:", [sum(a <= v < b for v in length.values()) for a, b in BINS])
    table = {}
    for d in sorted(p for p in SWEEP.iterdir() if (p / f"{p.name}.json").exists()):
        rows = {}
        for r in json.load(open(d / f"{d.name}.json")):
            rows.setdefault(_doc_key(r.get("name") or ""), r)
        recall = []
        for a, b in BINS:
            tp = fn = 0
            for k, n in length.items():
                if a <= n < b:
                    preds, _ = predictions_from_clauses(rows.get(k, {}).get("clauses", []), CUAD_METRICS)
                    for m in CUAD_METRICS:
                        t, _, f = official_task_counts(gold[k][m], preds[m], parties=(m == "parties"))
                        tp, fn = tp + t, fn + f
            recall.append(tp / (tp + fn))
        eff = BINS[0][1]
        for (a, b), r in zip(BINS[1:2], recall[1:2]):
            if r >= PASS * recall[0]:
                eff = b
        model = "gemini-" + d.name.removeprefix("gemini_").removesuffix("_config").replace("_", "-")
        table[model] = eff
        print(f"{model:28s} recall " + " ".join(f"{r:.2f}" for r in recall)
              + f"  ratio(8-16K) {recall[1] / recall[0]:.2f}  ratio(>16K) {recall[2] / recall[0]:.2f}  -> {eff}")
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
