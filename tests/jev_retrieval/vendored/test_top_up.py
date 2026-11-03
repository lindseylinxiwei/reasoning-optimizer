import unittest, hashlib, copy, random
from top_up import preserve_then_fill, candidates_from_trace

def fixture(regions, selected, cap, text='abcdefghij'):
    trace=[]
    for i,(a,b,score) in enumerate(regions):
        trace += [{'stage':'request','call':i,'request_sha256':str(i),
                   'context':{'regions':{'C0':{'start':a,'end':b,'node_id':f'n{i}'}}}},
                  {'stage':'response','call':i,'scores':{'C0':score}}]
    return text, {'source_sha256':hashlib.sha256(text.encode()).hexdigest(),
      'config':{'evidence_chars':cap,'threshold':.3,'max_calls':128},'controller_calls':len(regions),
      'task':{'name':'synthetic'},'evidence':[{'id':f'E{i}','start':a,'end':b,'text':text[a:b]} for i,(a,b) in enumerate(selected)],
      'evidence_chars':sum(b-a for a,b in selected),'trace':trace}

class TopUpTests(unittest.TestCase):
    def test_preserves_late_evidence_before_early_low_score(self):
        t,p=fixture([(0,5,.2),(5,8,.1),(8,10,.9)],[(8,10)],5)
        r=preserve_then_fill(t,p)
        self.assertEqual([(x['start'],x['end']) for x in r['evidence']],[(0,3),(8,10)])
    def test_parent_cannot_override_child_score(self):
        t,p=fixture([(0,10,.99),(0,5,.1),(5,10,.2)],[],3)
        r=preserve_then_fill(t,p);self.assertEqual(r['evidence'][0]['start'],5)
    def test_partial_child_retains_parent_residual(self):
        t,p=fixture([(0,10,.9),(0,5,.1)],[],4)
        self.assertEqual(preserve_then_fill(t,p)['evidence'][0]['text'],'fghi')
    def test_overlap_original_is_not_double_counted(self):
        t,p=fixture([(0,10,.1)],[(3,6)],7)
        r=preserve_then_fill(t,p);self.assertEqual(r['evidence_chars'],7);self.assertEqual(r['evidence'][0]['text'],'abcdefg')
    def test_short_source(self):
        t,p=fixture([(0,10,0)],[],20)
        r=preserve_then_fill(t,p);self.assertTrue(r['target_filled']);self.assertEqual(r['stop_reason'],'short_source_exhausted')
    def test_no_score_no_padding(self):
        t,p=fixture([],[(0,2)],5)
        r=preserve_then_fill(t,p);self.assertEqual(r['shortfall_chars'],3)
    def test_full_original_unchanged(self):
        t,p=fixture([(0,10,.8)],[(7,10)],3)
        r=preserve_then_fill(t,p);self.assertEqual(r['added_chars'],0)
    def test_invalid_source_and_conflicting_scores_fail(self):
        t,p=fixture([(0,5,.1),(0,5,.2)],[],5)
        with self.assertRaises(ValueError):preserve_then_fill(t,p)
        t,p=fixture([],[(0,2)],5)
        with self.assertRaises(ValueError):preserve_then_fill('z'+t[1:],p)
    def test_random_superset_cap_and_source(self):
        rng=random.Random(20261030)
        for _ in range(300):
            cap=rng.randint(1,10);a=rng.randrange(10);b=min(10,a+rng.randint(1,cap))
            t,p=fixture([(i,i+1,rng.random()) for i in range(10)],[(a,b)],cap)
            before=copy.deepcopy(p);r=preserve_then_fill(t,p)
            chars={i for x in r['evidence'] for i in range(x['start'],x['end'])}
            self.assertTrue(set(range(a,b))<=chars);self.assertEqual(len(chars),cap)
            self.assertEqual(p,before)

if __name__=='__main__':unittest.main()
