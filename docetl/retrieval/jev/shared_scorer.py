"""Single-flight source-node scoring shared across a fixed query bundle.

No credential access or HTTP here. Injected backend owns raw calls/accounting.
Caches are confined to one document, variant and retrieval realization.
"""
from __future__ import annotations
from concurrent.futures import Future
import hashlib
import json
import math
import threading


def _target(task):
    if isinstance(task,dict):
        name=task['name']; description=task.get('description',task.get('instruction'))
    else:
        name=task.name; description=getattr(task,'description',getattr(task,'instruction',None))
    if not isinstance(name,str) or not name or not isinstance(description,str) or not description:
        raise ValueError('Every target needs its original name and definition')
    return {'name':name,'description':description}


class SharedScorer:
    supports_concurrent_calls=True
    def __init__(self,tasks,backend,*,model='jev-1.13.0',max_request_bytes=28000,question_profile='extract-presence'):
        if question_profile not in {'extract-presence','evidence-support'}:
            raise ValueError('Question profile outside generic search')
        self.question_profile=question_profile
        self.targets=[_target(t) for t in tasks]
        if not self.targets or len({t['name'] for t in self.targets})!=len(self.targets):
            raise ValueError('A fixed unique query bundle is required')
        self.names={t['name']:f'T{i}' for i,t in enumerate(self.targets)}
        self.backend,self.model,self.max_request_bytes=backend,model,max_request_bytes
        self.lock=threading.RLock();self.futures={}
        self.counters={'request_computations':0,'cache_hits':0,'singleflight_waits':0,
                       'question_count':0,'request_bytes':0}

    def payload(self,single):
        regions=list(single['questions'])
        if not 1<=len(regions)*len(self.targets)<=32:
            raise ValueError('Shared regions times targets exceeds32 questions; page before scoring')
        state=dict(single['state'])
        state['task']={'type':'extraction','targets':{f'T{i}':v for i,v in enumerate(self.targets)},
            'objective':'Each target description defines the meaning of its name. Identify all original '
                        'source spans matching it, including conditions and exceptions. '
                        'Keep distinct matching occurrences even when their meaning is similar.'}
        state['target_labels_definition']='Tn identifies the corresponding extraction target in task.targets.'
        if self.question_profile=='evidence-support':
            state['evidence_definition']='Supporting evidence includes text that answers or qualifies a target, including exceptions, conditions and negative answers. Topic-only mentions and table-of-contents entries are insufficient.'
        state.pop('selected_evidence',None)
        payload={'model':self.model,'state':state,'questions':{
            f'{region}_T{i}':{'type':'noul','instructions':
                f'Does {region} contain text to extract for T{i}?' if self.question_profile=='extract-presence'
                else f'Does {region} provide supporting evidence for T{i}?'}
            for region in regions for i in range(len(self.targets))}}
        wire=json.dumps(payload,ensure_ascii=False,allow_nan=False)
        if len(wire.encode())>self.max_request_bytes:
            raise ValueError('Shared request exceeds capacity; original source must be paged without loss')
        return payload,wire

    @staticmethod
    def _parse(payload,raw):
        if (not isinstance(raw,dict) or raw.get('model')!=payload['model']
                or not isinstance(raw.get('answers'),dict)
                or set(raw['answers'])!=set(payload['questions'])):
            raise ValueError('Missing or mismatched shared response')
        scores={}
        for name,answer in raw['answers'].items():
            value=answer.get('noul') if isinstance(answer,dict) else None
            if (not isinstance(answer,dict) or answer.get('type')!='noul'
                    or type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1):
                raise ValueError('Invalid independent Noul score')
            scores[name]=float(value)
        return scores

    def score(self,payload,context):
        name=context['task']['name']
        if name not in self.names:
            raise ValueError('Task outside fixed shared query bundle')
        shared,wire=self.payload(payload)
        key=hashlib.sha256(wire.encode()).hexdigest()
        with self.lock:
            first=key not in self.futures
            if first:
                future=Future();self.futures[key]=future
                self.counters['request_computations']+=1
                self.counters['question_count']+=len(shared['questions'])
                self.counters['request_bytes']+=len(wire.encode())
            else:
                future=self.futures[key]
                self.counters['cache_hits']+=1
                if not future.done():self.counters['singleflight_waits']+=1
        if first:
            try:
                raw=self.backend.decide(shared)
                future.set_result(self._parse(shared,raw))
            except BaseException as exc:
                future.set_exception(exc)
                raise
        scores=future.result()
        label=self.names[name]
        return {region:scores[f'{region}_{label}'] for region in payload['questions']}

    def stats(self):
        with self.lock:
            result=dict(self.counters)
            # Live guarded backends expose admitted calls after durable reserve.
            # Mock .calls counters are fixture backend calls, never paid results.
            count=getattr(self.backend,'admitted_calls',getattr(self.backend,'calls',None))
            if count is None and str(getattr(self.backend,'kind','')).startswith('offline_'):
                records=getattr(self.backend,'payloads',None)
                count=len(records) if isinstance(records,list) else None
            result['network_calls']=count if type(count)is int else None
            result['counter_kind']='guarded_admitted_attempts' if hasattr(self.backend,'admitted_calls') else 'offline_fixture_calls_or_unavailable'
            return result
