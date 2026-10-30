"""Verify exhaustive request coverage, source context, output IDs, and billed usage."""
import argparse
import json
import math
from pathlib import Path

from core import HERE, digest, label_questions, load_module, ROOT, save
from full_vocabulary import support_probability


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=HERE / 'runs/fullpaper-test100-v1')
    args = parser.parse_args()
    run = args.run
    protocol, summary = read(run/'protocol.json'), read(run/'summary.json')
    raw = Path(protocol['source']).read_text()
    assert digest(raw) == protocol['source_sha256']
    source = {str(r['id']):r for r in json.loads(raw)}
    results = {r['id']:r for r in read(run/'results.json')}
    assert set(source) == set(results) == {p['id'] for p in protocol['documents']}
    candidates = read(run/'vocabulary.json')
    expected = label_questions(candidates)
    labels = {c['id']:c['label'] for c in candidates}
    calls = {'scan':0, 'ranking':0}
    costs = {'scan':0., 'ranking':0.}
    input_tokens = {'scan':0, 'ranking':0}
    unknown = {'scan':0, 'ranking':0}
    errors, retries, count = [], 0, 0
    for index, (key, row) in enumerate(source.items(),1):
        scan_keys = []
        seen_scan_answers = {}
        for stage in calls:
            for path in sorted((run/key/stage).glob('batch-*/attempt-*.jsonl')):
                for line in path.read_text().splitlines():
                    record = json.loads(line)
                    calls[stage] += 1
                    retries += int(path.stem.split('-')[-1]) > 1
                    costs[stage] += record.get('cost_usd') or 0
                    unknown[stage] += record.get('cost_usd') is None
                    if record.get('status') != 'success':
                        errors.append({'document':key,'stage':stage,'status':record.get('http_status'),
                                       'error_type':record.get('error_type'),'path':str(path)})
                        continue
                    assert record['request']['model'] == 'jev-1.13.0'
                    prefix = 'Medical paper:\n' if stage == 'scan' else 'Medical paper evidence:\n'
                    assert record['request']['state'] == prefix + row['fulltext_processed']
                    questions = record['request']['questions']
                    answers = record['response']['answers']
                    assert set(questions) == set(answers)
                    for k,q in questions.items():
                        assert answers[k]['choice'] in q['criteria']
                        assert len(q['criteria']) <= 255
                        assert 'MORE_CONTEXT' not in q['criteria']
                    tokens = record['response']['usage']['input_tokens']
                    input_tokens[stage] += tokens
                    assert math.isclose(record['cost_usd'], tokens * .042 / 1e6)
                    if stage == 'scan':
                        assert len(questions) <= 256
                        assert all(q == expected[k] for k,q in questions.items())
                        assert all(len(q['criteria']) == 2 for q in questions.values())
                        scan_keys.extend(questions)
                        seen_scan_answers.update(answers)
        assert scan_keys == list(expected)
        count += len(scan_keys)
        scan = read(run/key/'scan/result.json')
        assert scan['answers'] == seen_scan_answers
        assert scan['supported_ids'] == [k for k in expected if seen_scan_answers[k]['choice']=='supported']
        for answer in seen_scan_answers.values():
            support_probability(answer)
        output = results[key]
        ids = output['ranked_ids']
        assert len(ids) == len(set(ids)) <= 10
        assert set(ids) <= set(scan['supported_ids'])
        assert output['ranked_reaction_labels'] == [labels[i] for i in ids]
        assert output['ground_truth_reactions'] == row['ground_truth_reactions']
        assert output['status'] == 'success'
        ranking = read(run/key/'ranking/result.json')
        assert ranking['ranked_ids'] == ids
        assert not any(r['requested_full_context'] for r in ranking['group_trace']+ranking['final_trace'])
        if index%20 == 0:
            print(f'Audited {index}/{len(source)} documents',flush=True)
    for stage in calls:
        s=summary['stages'][stage]
        assert s['api_calls'] == calls[stage]
        assert math.isclose(s['known_api_cost_usd'], costs[stage])
        assert s['unknown_cost_calls'] == unknown[stage]
    assert math.isclose(summary['known_api_cost_usd'], sum(costs.values()))
    assert count == summary['expected_scan_decisions']
    evaluator=load_module('audit_fullpaper_eval', ROOT/'experiments/reasoning/evaluation/biodex.py')
    score=evaluator.evaluate_results('audit',str(run/'results.json'))
    for k in ['avg_rp_at_5','avg_rp_at_10','rp_at_5_scores','rp_at_10_scores','total_documents']:
        assert score[k] == summary[k]
    audit={'documents':len(source),'scan_decisions':count,'api_calls':calls,'input_tokens':input_tokens,
           'known_api_cost_usd':costs,'unknown_cost_calls':unknown,'failed_attempts':errors,'retry_calls':retries,
           'all_scan_questions_match_pilot':True,'all_contexts_are_complete_source_papers':True,
           'each_label_scanned_once_successfully':True,'ranking_uses_unique_supported_official_labels':True,
           'costs_match_provider_usage':True,'metrics_recomputed':True}
    save(run/'audit.json',audit)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
