import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from shared_scorer import SharedScorer

class Backend:
    def __init__(self):self.calls=0;self.payloads=[];self.lock=threading.Lock()
    def decide(self,p):
        with self.lock:self.calls+=1;self.payloads.append(p)
        time.sleep(.01)
        return {'model':p['model'],'answers':{key:{'type':'noul','noul':.8 if key.endswith('_T0') else .2}
                for key in p['questions']}}

class SharedTests(unittest.TestCase):
    def request(self):
        return {'model':'jev-1.13.0','state':{'task':{},'source_text':'⟦REGION|BEGIN C0⟧\nEntire region.\n⟦REGION|END C0⟧'},
                'questions':{'C0':{'type':'noul','instructions':'Does C0 contain text to extract?'}}}
    def test_concurrent_targets_share_exactly_one_real_backend_attempt(self):
        backend=Backend();tasks=[{'name':f't{i}','description':f'definition{i}'} for i in range(8)]
        scorer=SharedScorer(tasks,backend)
        def call(i):return scorer.score(self.request(),{'task':tasks[i]})
        with ThreadPoolExecutor(max_workers=8) as pool:values=list(pool.map(call,range(8)))
        self.assertEqual(backend.calls,1)
        self.assertEqual(values,[{'C0':.8}]+[{'C0':.2}]*7)
        self.assertEqual(scorer.stats()['question_count'],8)
        self.assertEqual(scorer.stats()['cache_hits'],7)
        self.assertEqual(backend.payloads[0]['state']['source_text'],self.request()['state']['source_text'])
    def test_32question_limit_is_not_silent_target_discard(self):
        tasks=[{'name':f't{i}','description':'d'} for i in range(8)]
        scorer=SharedScorer(tasks,Backend());req=self.request()
        req['questions']={f'C{i}':{'type':'noul'} for i in range(5)}
        with self.assertRaises(ValueError):scorer.payload(req)
    def test_failures_remain_cached_no_retry(self):
        class Failed(Backend):
            def decide(self,p):self.calls+=1;raise RuntimeError('synthetic failure')
        backend=Failed();tasks=[{'name':'t','description':'d'}];scorer=SharedScorer(tasks,backend)
        for _ in range(2):
            with self.assertRaises(RuntimeError):scorer.score(self.request(),{'task':tasks[0]})
        self.assertEqual(backend.calls,1)
    def test_independent_variants_do_not_share_provider_predictions(self):
        backend=Backend();tasks=[{'name':'t','description':'d'}]
        for _ in range(2):SharedScorer(tasks,backend).score(self.request(),{'task':tasks[0]})
        self.assertEqual(backend.calls,2)
    def test_refused_dispatch_is_not_counted_as_network_admission(self):
        class Refused:
            kind='live_jev_guarded'
            admitted_calls=0
            def decide(self,p):raise RuntimeError('budget refused before dispatch')
        scorer=SharedScorer([{'name':'t','description':'d'}],Refused())
        with self.assertRaises(RuntimeError):scorer.score(self.request(),{'task':{'name':'t'}})
        self.assertEqual(scorer.stats()['request_computations'],1)
        self.assertEqual(scorer.stats()['network_calls'],0)
    def test_task_outside_bundle_refused(self):
        scorer=SharedScorer([{'name':'t','description':'d'}],Backend())
        with self.assertRaises(ValueError):scorer.score(self.request(),{'task':{'name':'other'}})

if __name__=='__main__':unittest.main()
