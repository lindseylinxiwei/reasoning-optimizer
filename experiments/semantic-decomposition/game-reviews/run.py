"""Run the GameReviews decomposition with Jev choices and deterministic code."""
import argparse
import importlib.util
import json
import time
from collections import Counter
from pathlib import Path

from pipeline import (SENTIMENT_POLICY, SUMMARY_POLICY, digest, parse_reviews, save,
                      select_reviews, sentiment_questions, summarize)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SHARED = HERE.parent / 'company-extraction/pipeline.py'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Choices:
    def __init__(self, folder, document):
        self.folder, self.document = Path(folder), document
        self.folder.mkdir(parents=True, exist_ok=True)
        self.client = load_module('_game_reviews_shared', SHARED).JevSelector(self.folder / 'calls.jsonl')

    def __call__(self, state, questions):
        answers, entries = {}, list(questions.items())
        for start in range(0, len(entries), 64):
            for group in self.client._pack(state, dict(entries[start:start + 64])):
                identity = {'model': 'jev-1.13.0', 'state': state, 'questions': group}
                key = digest(json.dumps(identity, sort_keys=True, ensure_ascii=False))
                path = self.folder / (key + '.json')
                if path.exists():
                    result = json.loads(path.read_text())
                else:
                    result = self.client._request(self.document, state, group, 1, start // 64 + 1)
                    save(path, result)
                if set(result['answers']) != set(group) or any(
                    a.get('choice') not in group[k]['criteria'] for k, a in result['answers'].items()):
                    raise ValueError('Jev returned missing or invalid choices')
                answers.update(result['answers'])
        return answers


def metrics(folder):
    rows = [(p.parent.name, json.loads(line)) for p in Path(folder).glob('*/requests/*/calls.jsonl')
            for line in p.read_text().split('\n') if line]
    return {'api_calls': len(rows), 'calls_by_stage': dict(Counter(stage for stage, _ in rows)),
            'known_api_cost_usd': sum(r['cost_usd'] for _, r in rows if r.get('cost_usd') is not None),
            'unknown_cost_calls': sum(r.get('cost_usd') is None for _, r in rows),
            'input_tokens': sum(r.get('response', {}).get('usage', {}).get('input_tokens', 0) for _, r in rows),
            'api_wall_seconds': sum(r['wall_seconds'] for _, r in rows),
            'api_price_assumption': 'Existing client: USD 0.042 per million returned input tokens.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'experiments/reasoning/data/train/game_reviews.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=40)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if min(args.limit, args.count) < 1:
        parser.error('Limit and count must be positive')
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    records = json.loads(args.input.read_text())[:args.limit]
    if not records or len({r['app_id'] for r in records}) != len(records):
        raise ValueError('Expected unique game IDs')
    parsed = [parse_reviews(r['concatenated_reviews']) for r in records]
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {'input_sha256': digest(json.dumps(records, sort_keys=True)), 'count': args.count,
                'model': 'jev-1.13.0', 'max_questions_per_batch': 64,
                'code_sha256': {str(p.relative_to(ROOT)): digest(p.read_text())
                               for p in (HERE / 'pipeline.py', HERE / 'run.py', SHARED)}}
    path = args.output / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Existing run has a different protocol; use a new output directory')
    save(path, protocol)
    save(args.output / 'input.json', records)
    save(args.output / 'parsing.json', {'games': len(records), 'reviews': sum(map(len, parsed)),
         'per_game': [{'app_id': r['app_id'], 'reviews': len(rs)} for r, rs in zip(records, parsed)]})
    if args.dry_run:
        print(f'Parsed {len(records)} games and {sum(map(len, parsed))} complete review records; no API calls.')
        return
    outputs, started = [], time.perf_counter()
    try:
        for row, reviews in zip(records, parsed):
            folder = args.output / str(row['app_id'])
            folder.mkdir(exist_ok=True)
            state = SENTIMENT_POLICY + '\nGame: ' + row['app_name']
            decisions = Choices(folder / 'requests/sentiment', row['concatenated_reviews'])(state, sentiment_questions(reviews))
            save(folder / 'sentiment.json', decisions)
            selected = select_reviews(reviews, decisions, args.count)
            result, trace = summarize(selected, Choices(folder / 'requests/summary', row['concatenated_reviews']),
                                      SUMMARY_POLICY + '\nGame: ' + row['app_name'])
            save(folder / 'summaries.json', trace)
            save(folder / 'selection.json', {'pool_counts': dict(Counter(a['choice'] for a in decisions.values())),
                 'selected_ids': {label: [r['review_id'] for r in group] for label, group in selected.items()}})
            outputs.append({**row, 'positive_reviews': result['POSITIVE'], 'negative_reviews': result['NEGATIVE']})
            save(args.output / 'results.json', outputs)
            print(f"{row['app_name']}: classified {len(reviews)}, selected {len(result['POSITIVE'])} positive / "
                  f"{len(result['NEGATIVE'])} negative ({len(outputs)}/{len(records)} games)", flush=True)
    finally:
        save(args.output / 'metrics.json', {**metrics(args.output), 'games_completed': len(outputs),
             'elapsed_seconds_this_run': time.perf_counter() - started})


if __name__ == '__main__':
    main()
