"""Preserve original evidence, then top up using only its saved scored regions.

Pure source-only function: no labels, provider clients, model calls, or imports
from the evaluation code. Scores are used at the finest observed granularity.
"""
import math
import hashlib


def merge(intervals):
    out = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def subtract(interval, excluded):
    a, b = interval
    result = []
    for c, d in merge(excluded):
        if d <= a:
            continue
        if c >= b:
            break
        if c > a:
            result.append([a, c])
        a = max(a, d)
        if a >= b:
            break
    if a < b:
        result.append([a, b])
    return result


def candidates_from_trace(packet, source_length):
    requests = {}
    for event in packet['trace']:
        if event.get('stage') == 'request':
            if event['call'] in requests:
                raise ValueError('Duplicate request call')
            requests[event['call']] = event
    candidates = {}
    for event in packet['trace']:
        if event.get('stage') != 'response':
            continue
        request = requests[event['call']]
        regions = request['context']['regions']
        if set(regions) != set(event['scores']):
            raise ValueError('Score/region mismatch')
        for label, region in regions.items():
            a, b = region['start'], region['end']
            score = event['scores'][label]
            if not (type(a) is int and type(b) is int and 0 <= a < b <= source_length):
                raise ValueError('Invalid candidate interval')
            if type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid cached score')
            key = (a, b)
            if key in candidates and candidates[key]['score'] != score:
                raise ValueError('Conflicting scores for identical source interval')
            candidates.setdefault(key, {'start':a, 'end':b, 'score':score,
                'node_id':region['node_id'], 'call':event['call'],
                'request_sha256':request['request_sha256'], 'region_label':label})
    # Use scored descendants wherever available; parent scores only cover any
    # residual area lacking a more specific score. Never rank a coarse ancestor
    # above already-scored low-scoring children simply because its score is high.
    candidates = list(candidates.values())
    refined = []
    for candidate in candidates:
        a, b = candidate['start'], candidate['end']
        descendants = []
        for other in candidates:
            c, d = other['start'], other['end']
            if a <= c and d <= b and (a, b) != (c, d):
                descendants.append([c, d])
            elif max(a, c) < min(b, d) and not (c <= a and b <= d):
                raise ValueError('Crossing scored regions are not a hierarchy')
        for c, d in subtract([a, b], descendants):
            refined.append({**candidate, 'score_region_start':a, 'score_region_end':b,
                            'start':c, 'end':d})
    return sorted(refined, key=lambda x: (-x['score'], x['start'], x['end'], x['node_id']))


def preserve_then_fill(text, original):
    cap = original['config']['evidence_chars']
    if type(cap) is not int or cap <= 0:
        raise ValueError('Invalid evidence cap')
    if hashlib.sha256(text.encode()).hexdigest() != original['source_sha256']:
        raise ValueError('Source hash mismatch')
    selected = []
    for part in original['evidence']:
        a, b = part['start'], part['end']
        if not (0 <= a < b <= len(text)) or part['text'] != text[a:b]:
            raise ValueError('Original evidence differs from source')
        selected.append([a,b])
    selected = merge(selected)
    original_chars = sum(b-a for a,b in selected)
    if original_chars != original['evidence_chars'] or original_chars > cap:
        raise ValueError('Original evidence accounting mismatch')
    preserved = [x[:] for x in selected]
    candidates = candidates_from_trace(original, len(text))
    target = min(cap, len(text))
    remaining = target-original_chars
    added = []
    for candidate in candidates:
        if not remaining:
            break
        for a, b in subtract([candidate['start'],candidate['end']], selected):
            if not remaining:
                break
            end = a + min(b-a, remaining)
            added.append({**candidate, 'admitted_start':a, 'admitted_end':end,
                          'clipped':end < b})
            selected = merge(selected+[[a,end]])
            remaining -= end-a
    evidence = [{'id':f'E{i+1}', 'start':a, 'end':b, 'text':text[a:b]}
                for i,(a,b) in enumerate(selected)]
    assert all(not subtract(part,selected) for part in preserved)
    return {'schema_version':1, 'method':'preserve_original_then_fill',
        'source_sha256':original['source_sha256'], 'task':original['task'],
        'cap_chars':cap, 'original_threshold':original['config']['threshold'],
        'original_decision_limit':original['config']['max_calls'],
        'original_controller_calls':original['controller_calls'],
        'original_evidence':original['evidence'], 'original_evidence_chars':original_chars,
        'evidence':evidence, 'evidence_chars':target-remaining,
        'added_chars':target-remaining-original_chars, 'additions':added,
        'target_filled':remaining==0, 'shortfall_chars':remaining,
        'stop_reason':('scored_candidates_exhausted' if remaining else
                       'short_source_exhausted' if len(text)<cap else 'evidence_cap_reached'),
        'candidate_count':len(candidates), 'new_provider_calls':0,
        'candidate_policy':'finest cached scored regions from original run; descending score; source-order ties',
        'retrieval_partial':target-remaining<len(text)}
