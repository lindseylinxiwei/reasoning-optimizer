"""Per-report correctness of the search plans' outputs on the 40-report sample.

Uses the same name matching as experiments/reasoning/evaluation/sustainability.py, but
restricted to the 40 sample reports so each output entry is attributed to one report.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reasoning"))
from evaluation.sustainability import calculate_name_similarity  # noqa: E402

from common import OUT, load  # noqa: E402

LOGS = Path.home() / "Downloads/search_logs/sustainability"
PLANS = {13: "gemini_2.5_flash_lite_config_13.json", 15: "gemini_2.5_flash_lite_config_15.json",
         17: "gemini_2.5_flash_lite_config_17.json", 20: "gemini_2.5_flash_lite_config_20.json",
         7: "gemini_2.5_flash_lite_config_7.json", 9: "gemini_2.5_flash_lite_config_9.json",
         18: "gemini_2.5_flash_lite_config_18.json"}

sample = [r for r in load() if r["in_sample"]]
result = {}
for plan, f in PLANS.items():
    out = json.load(open(LOGS / f))
    per = {r["id"]: {"name": False, "activity": False, "predicted": None} for r in sample}
    for group in out:
        act = group.get("economic_activity")
        for entry in group.get("economic_activity_summary", []):
            if isinstance(entry, str) or not entry.get("company_name", "").strip():
                continue
            name = entry["company_name"].strip().lower()
            best = max(sample, key=lambda r: calculate_name_similarity(name, r["GT company_name"].lower()))
            if calculate_name_similarity(name, best["GT company_name"].lower()) >= 0.5:
                p = per[best["id"]]
                p["name"] = True
                p["predicted"] = act
                p["activity"] = p["activity"] or act == best["GT economic_activity"]
    result[plan] = per
    n = len(sample)
    print(f"plan {plan:2d}: name found {sum(p['name'] for p in per.values())}/{n}, "
          f"activity correct {sum(p['activity'] for p in per.values())}/{n}")

ids = [r["id"] for r in sample]
always_wrong = [i for i in ids if not any(result[p][i]["activity"] for p in result)]
print("\nactivity wrong in every plan:", len(always_wrong))
gt = {r["id"]: r for r in sample}
for i in always_wrong:
    preds = {p: result[p][i]["predicted"] for p in result}
    print(f"  {i} {gt[i]['GT company_name']!r:32s} GT {gt[i]['GT economic_activity']:26s} predicted {preds}")

json.dump({str(k): {str(i): v for i, v in per.items()} for k, per in result.items()},
          open(OUT / "search_plan_per_report.json", "w"), indent=1)
