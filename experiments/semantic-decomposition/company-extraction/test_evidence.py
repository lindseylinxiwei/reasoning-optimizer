import unittest
from evidence_selector import evidence_for,EvidenceSelector
from test_jev_batches import ByteEncoder

class EvidenceTests(unittest.TestCase):
    def test_spread_and_verbatim_offsets(self):
        text='A'+('x'*1000)+'A'+('x'*1000)+'A'
        candidate={'id':'C1','text':'A','mentions':[{'start':i,'end':i+1,'text':'A'} for i in [0,1001,2002]]}
        a=evidence_for(text,candidate,2);b=evidence_for(text,candidate,4)
        self.assertEqual(len(a['passages']),2)
        self.assertEqual(a['passages'][-1]['end'],len(text))
        self.assertEqual(len(b['passages']),3)
        for p in b['passages']:self.assertEqual(text[p['start']:p['end']],p['text'])

    def test_none_expands_once_and_returns_original_id(self):
        selector=EvidenceSelector.__new__(EvidenceSelector)
        selector.encoder=ByteEncoder();selector.max_estimated_tokens=24000;selector.max_request_tokens=48000
        requests=[]
        def request(text,state,questions,round_number,request_number):
            requests.append(questions)
            return {'answers':{k:{'type':'choice','choice':'NONE' if round_number==1 else 'C1'} for k in questions},'usage':{'input_tokens':10},'cost_usd':.1}
        selector._request=request
        result=selector('Apple', [{'id':'C1','text':'Apple','mentions':[{'start':0,'end':5,'text':'Apple'}]}])
        self.assertEqual(result['candidate_id'],'C1');self.assertEqual(len(requests),2)
        self.assertEqual(result['usage']['input_tokens'],20);self.assertAlmostEqual(result['cost_usd'],.2)

    def test_token_budget_splits_below_option_limit(self):
        selector=EvidenceSelector.__new__(EvidenceSelector);selector.encoder=ByteEncoder();selector.max_estimated_tokens=24000;selector.max_request_tokens=48000
        text='Apple '*300
        candidates=[{'id':f'C{i}','text':'Apple','mentions':[{'start':i*6,'end':i*6+5,'text':'Apple'}]} for i in range(100)]
        qs=selector._questions(text,candidates,2,'opening')
        self.assertGreater(len(qs),1)
        self.assertEqual(sum(len(q['criteria'])-1 for q in qs.values()),100)
        for group in selector._pack('opening',qs):
            self.assertLessEqual(selector._tokens('opening')+sum(selector._tokens(__import__('json').dumps({k:v},ensure_ascii=False)) for k,v in group.items()),48000)
