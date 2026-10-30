"""Create short-document Jev pipelines; run from the repository root."""
import argparse
import copy
import json
import sys
from pathlib import Path

import tiktoken
import yaml
from jinja2 import StrictUndefined, Environment

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from docetl.operations.utils.jev import build_questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--max-estimated-tokens", type=int, default=24000)
    parser.add_argument("--default-model", default="gemini/gemini-3.8-flash", help="Gemini model for non-Jev operators")
    args = parser.parse_args()
    if not args.default_model.startswith("gemini/"):
        parser.error("Non-Jev operators in this experiment require a gemini/ model")
    if not 1 <= args.max_estimated_tokens <= 24000:
        parser.error("Use a estimated token budget between 1 and 24000 for this conservative smoke test")
    root = Path(__file__).resolve().parents[2]
    folder = Path(__file__).resolve().parent
    config = yaml.safe_load((root / "experiments/reasoning/pipelines/sustainability.yaml").read_text())
    config["default_model"] = args.default_model
    source = root / f"experiments/reasoning/data/{args.split}/sustainability.json"
    records = json.loads(source.read_text())
    env = Environment(undefined=StrictUndefined)
    operations = config["operations"][:2]
    encoder = tiktoken.get_encoding("cl100k_base")
    retained, excluded = [], []
    for index, item in enumerate(records):
        sizes = []
        for op in operations:
            prompt = env.from_string(op["prompt"]).render(input=item)
            payload = {"model": "jev-1.13.0", "state": prompt, "questions": build_questions(op["output"]["schema"])}
            sizes.append(len(encoder.encode(json.dumps(payload, ensure_ascii=False), disallowed_special=())))
        if max(sizes) <= args.max_estimated_tokens:
            retained.append(item)
        else:
            excluded.append({"index": index, "id": item.get("id"), "estimated_tokens": max(sizes)})
    data_path = folder / f"sustainability_{args.split}_short.json"
    data_path.write_text(json.dumps(retained, ensure_ascii=False, indent=2))
    manifest = {"source": str(source), "policy": "cl100k_base estimate with headroom; not the Jev tokenizer", "max_estimated_tokens": args.max_estimated_tokens, "retained": len(retained), "excluded": excluded}
    (folder / f"selection_{args.split}.json").write_text(json.dumps(manifest, indent=2))
    config["datasets"]["company_reports"]["path"] = str(data_path)
    for op in operations:
        op["model"] = "jev"
    for classifiers_only in (False, True):
        variant = copy.deepcopy(config)
        suffix = "classifiers" if classifiers_only else "pipeline"
        if classifiers_only:
            variant["operations"] = variant["operations"][:2]
            variant["pipeline"]["steps"][0]["operations"] = variant["pipeline"]["steps"][0]["operations"][:2]
        variant["pipeline"]["output"]["path"] = str(folder / f"results_{args.split}_{suffix}.json")
        (folder / f"sustainability_{args.split}_{suffix}.yaml").write_text(yaml.safe_dump(variant, sort_keys=False))
    print(f"Retained {len(retained)}/{len(records)} documents; excluded {len(excluded)}. See selection_{args.split}.json")
    if not retained:
        raise SystemExit("No documents fit the conservative smoke-test budget")


if __name__ == "__main__":
    main()
