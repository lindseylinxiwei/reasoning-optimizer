"""Compare complete outputs with the saved baseline; audit task properties separately."""
import argparse
import json
import statistics
from collections import Counter
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from pipeline import digest, evenly_spaced, parse_reviews, save, seconds
from run import ROOT, load_module


def temporal_error(selected, start, end):
    """Mean distance to uniform full-game time targets, normalized by game duration."""
    if not selected:
        return None
    times = sorted(seconds(r['timestamp']) for r in selected)
    targets = [start + (end - start) * i / (len(times) - 1) for i in range(len(times))] if len(times) > 1 else [(start + end) / 2]
    return statistics.mean(abs(t - target) for t, target in zip(times, targets)) / (end - start) if end > start else 0.


def audit_outputs(records, outputs, ground, limit):
    indexed = {r['app_id']: r for r in outputs}
    if len(indexed) != len(outputs) or set(indexed) != {r['app_id'] for r in records}:
        raise ValueError('Results must contain each input game exactly once')
    details, totals, errors = [], Counter(), []
    for row in records:
        result = indexed[row['app_id']]
        if result['concatenated_reviews'] != row['concatenated_reviews']:
            raise ValueError('Results do not match the original source input')
        reviews = parse_reviews(row['concatenated_reviews'])
        by_id = {r['review_id']: r for r in reviews}
        start, end = min(seconds(r['timestamp']) for r in reviews), max(seconds(r['timestamp']) for r in reviews)
        counts, lists, seen = Counter(), {}, set()
        for field, positive in (('positive_reviews', True), ('negative_reviews', False)):
            selected = result[field]
            if not isinstance(selected, list):
                raise ValueError('Expected review lists')
            counts['returned'] += len(selected)
            counts['lists_with_exact_count'] += len(selected) == limit
            valid = []
            for position, selection in enumerate(selected):
                key = selection.get('review_id')
                duplicate = key in seen
                counts['duplicate_selections'] += duplicate
                seen.add(key)
                if key not in by_id:
                    counts['invalid_ids'] += 1
                    continue
                source = by_id[key]
                valid.append(source)
                counts['valid_ids'] += 1
                counts['correct_timestamps'] += selection.get('timestamp') == source['timestamp']
                counts['sentiment_agrees_with_vader'] += ground[row['app_id']][key] == positive
                counts['distinct_correct_requested_slots'] += (ground[row['app_id']][key] == positive
                                                               and not duplicate and position < limit)
                summary = selection.get('review_summary', '')
                counts['nonempty_summaries'] += bool(summary.strip())
                counts['verbatim_summaries'] += bool(summary) and summary in source['review_text']
            times = [seconds(r['timestamp']) for r in valid]
            ordered = times == sorted(times)
            counts['chronological_lists'] += ordered
            error = temporal_error(valid, start, end)
            if error is not None:
                errors.append(error)
            lists[field] = {'returned': len(selected), 'valid_ids': len(valid),
                            'chronological_by_source_timestamp': ordered, 'normalized_time_target_error': error,
                            'time_range_coverage': (max(times) - min(times)) / (end - start) if times and end > start else None}
        totals.update(counts)
        details.append({'app_id': row['app_id'], 'app_name': row['app_name'], **dict(counts), 'lists': lists})
    return {'counts': dict(totals), 'games_with_10_per_class': sum(
                d['lists_with_exact_count'] == 2 for d in details),
            'vader_agreement_per_requested_slot': totals['distinct_correct_requested_slots'] / (len(records) * 2 * limit),
            'vader_agreement_per_returned_review': totals['sentiment_agrees_with_vader'] / totals['returned'],
            'mean_normalized_time_target_error': statistics.mean(errors) if errors else None,
            'documents': details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=ROOT / 'experiments/reasoning/outputs/game_reviews_baseline.json')
    args = parser.parse_args()
    # Require the real lexicon before loading the legacy evaluator; suppress only its
    # redundant download attempt. Its sentiment and scoring functions remain unchanged.
    analyzer = SentimentIntensityAnalyzer()
    evaluator_path = ROOT / 'experiments/reasoning/evaluation/game_reviews.py'
    with patch('nltk.download', return_value=True):
        legacy = load_module('_game_reviews_legacy', evaluator_path)
    legacy.extract_reviews_from_input = lru_cache(maxsize=None)(legacy.extract_reviews_from_input)
    records = json.loads((args.run / 'input.json').read_text())
    output = json.loads((args.run / 'results.json').read_text())
    limit = json.loads((args.run / 'protocol.json').read_text())['count']
    baseline_rows = json.loads(args.baseline.read_text())
    baseline_index = {r['app_id']: r for r in baseline_rows}
    if len(baseline_index) != len(baseline_rows):
        raise ValueError('Duplicate baseline game IDs')
    baseline = [baseline_index[row['app_id']] for row in records]
    save(args.run / 'baseline_matched.json', baseline)

    ground, disagreements, classification, shortages = {}, [], Counter(), []
    parser_differences = 0
    for row in records:
        reviews = parse_reviews(row['concatenated_reviews'])
        parsed = legacy.extract_reviews_from_input(row['concatenated_reviews'])
        by_id = {r['review_id']: r for r in parsed}
        if len(by_id) != len(parsed) or set(by_id) != {r['review_id'] for r in reviews}:
            raise ValueError('Legacy parser does not recover the same review IDs')
        ground[row['app_id']] = {r['review_id']: r['is_positive'] for r in parsed}
        decisions = json.loads((args.run / str(row['app_id']) / 'sentiment.json').read_text())
        pools = Counter()
        for review in reviews:
            key = review['review_id']
            parser_differences += ' '.join(review['review_text'].split()) != ' '.join(by_id[key]['review_text'].split())
            choice = decisions['R' + key]['choice']
            expected = 'POSITIVE' if by_id[key]['is_positive'] else 'NEGATIVE'
            classification['reviews'] += 1
            classification[choice] += 1
            classification['agrees_with_vader'] += choice == expected
            classification[f'{choice}_vs_{expected}'] += 1
            pools[choice] += 1
            if choice != expected:
                disagreements.append({'app_id': row['app_id'], 'app_name': row['app_name'], **review,
                                      'jev': choice, 'vader': expected})
        if pools['POSITIVE'] < limit or pools['NEGATIVE'] < limit:
            shortages.append({'app_id': row['app_id'], 'app_name': row['app_name'], 'jev_pool_counts': dict(pools),
                              'vader_pool_counts': dict(Counter('POSITIVE' if r['is_positive'] else 'NEGATIVE' for r in parsed))})
        print(f"Evaluated source labels: {row['app_name']}", flush=True)

    audits = {name: audit_outputs(records, rows, ground, limit)
              for name, rows in [('jev_decomposition', output), ('saved_baseline', baseline)]}
    # Lower error is better. Check against the same classified pools, not an oracle label set.
    for row, result in zip(records, output):
        assert result['app_id'] == row['app_id']
        reviews = parse_reviews(row['concatenated_reviews'])
        decisions = json.loads((args.run / str(row['app_id']) / 'sentiment.json').read_text())
        start, end = min(seconds(r['timestamp']) for r in reviews), max(seconds(r['timestamp']) for r in reviews)
        for label, field in (('POSITIVE', 'positive_reviews'), ('NEGATIVE', 'negative_reviews')):
            pool = [r for r in reviews if decisions['R' + r['review_id']]['choice'] == label]
            expected = evenly_spaced(pool, start, end, limit)
            assert [r['review_id'] for r in expected] == [r['review_id'] for r in result[field]]
    report = {
        'protocol': {'games': len(records), 'split': 'train', 'evaluator_sha256': digest(evaluator_path.read_text()),
                     'evaluation_code_sha256': digest(Path(__file__).read_text()),
                     'vader_lexicon_sha256': digest(json.dumps(analyzer.lexicon, sort_keys=True)),
                     'nltk_version': nltk.__version__, 'baseline': str(args.baseline),
                     'baseline_matching': 'Exact app_id and identical concatenated_reviews; all input games required'},
        'limitations': ['VADER labels are automatic proxies, not human sentiment annotations.',
                        'The legacy temporal metric concatenates positive and negative lists before scoring.',
                        'Legacy scoring does not measure even time coverage, exact list sizes, or summary quality.',
                        'Jev summaries are selected source passages, not abstractive summaries.',
                        'No new baseline inference was run; its original cost is not established here.'],
        'classification': dict(classification), 'parser_text_differences': parser_differences,
        'shortages': shortages, 'audit': audits,
        'legacy_moar': {name: legacy.evaluate_results(name, str(path)) for name, path in
                        [('jev_decomposition', args.run / 'results.json'), ('saved_baseline', args.run / 'baseline_matched.json')]},
        'temporal_selection_matches_dynamic_program': True,
    }
    save(args.run / 'evaluation.json', report)
    with (args.run / 'sentiment_disagreements.jsonl').open('w') as f:
        for row in disagreements:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps({'legacy_moar': report['legacy_moar'], 'classification': report['classification'],
                      'shortage_games': len(shortages)}, indent=2))


if __name__ == '__main__':
    main()
