"""Paired 11-document ablation, then a frozen 100-document held-out run."""
import hashlib,json,os,time
from pathlib import Path
from dotenv import load_dotenv
from pipeline import LocalExtractor,JevSelector,extract_candidates,select_and_copy
from evidence_selector import EvidenceSelector,POLICY

H=Path(__file__).resolve().parent
ROOT=H.parents[2]
OUT=H/'runs/long_document_evidence_v1'
def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

def main():
    load_dotenv(ROOT/'.env');os.environ.setdefault('HF_HOME','/private/tmp/gliner-smoke-hf');os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    OUT.mkdir(exist_ok=False)
    pilot=json.loads((H/'runs/sustainability_11_gliner_compact/input.json').read_text())
    cached={r['id']:r for r in json.loads((H/'runs/sustainability_11_gliner_compact/results.json').read_text())}
    raw=json.loads((ROOT/'experiments/reasoning/data/test/sustainability.json').read_text())
    test=[{'id':r['id'],'text':r['tot_text_raw']} for r in raw]
    assert not {r['id'] for r in test}&{r['id'] for r in pilot}
    save(OUT/'pilot_input.json',pilot);save(OUT/'test_input.json',test)
    save(OUT/'config.json',{'policy':POLICY,'pilot_documents':len(pilot),'test_documents':len(test),
      'test_selection':'Entire existing held-out test split, no length or result filtering',
      'policy_frozen_before_test':True,'source_hashes':{f:hashlib.sha256((H/f).read_bytes()).hexdigest() for f in ['pipeline.py','evidence_selector.py']},
      'extraction':{'model':'urchade/gliner_small-v2.1','device':'mps','batch_size':16,'window':320,'overlap':64,'threshold':.5},
      'local_cost_usd':0,'retry_policy':'No automatic retries; errors retained','pilot_extraction':'Reuse identical saved candidates, exclude historical extraction latency from selector ablation',
      'context_estimates':'cl100k_base, not provider tokenizer; 24K per question+state, 48K total request guard',
      'test_input_sha256':hashlib.sha256((OUT/'test_input.json').read_bytes()).hexdigest()})
    for mode,cls in [('pilot_full',JevSelector),('pilot_evidence',EvidenceSelector)]:
        folder=OUT/mode;folder.mkdir();selector=cls(folder/'calls.jsonl');results=[];started=time.perf_counter()
        for r in pilot:
            row={'id':r['id'],'candidate_count':len(cached[r['id']]['candidates'])};t=time.perf_counter()
            try:row.update(select_and_copy(r['text'],cached[r['id']]['candidates'],selector))
            except Exception as exc:row.update(status='error',error=str(exc))
            row['selection_copy_seconds']=time.perf_counter()-t;results.append(row);save(folder/'results.json',results)
            print(mode,r['id'],row['status'],flush=True)
        save(folder/'metrics.json',{'wall_seconds':time.perf_counter()-started})
    folder=OUT/'test_evidence';folder.mkdir();selector=EvidenceSelector(folder/'calls.jsonl')
    import torch
    torch.set_num_threads(4);t=time.perf_counter();extractor=LocalExtractor('gliner','mps',.5)
    load_seconds=time.perf_counter()-t;results=[];started=time.perf_counter()
    for index,r in enumerate(test):
        row={'id':r['id'],'text_chars':len(r['text']),
             'estimated_document_tokens':selector._tokens(r['text'])}
        document_started=time.perf_counter()
        t=time.perf_counter()
        try:
            extraction=extract_candidates(r['text'],extractor,batch_size=16)
            row.update(extraction)
            row['extraction_seconds']=time.perf_counter()-t
            with (folder/'candidates.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            t=time.perf_counter()
            try:row.update(select_and_copy(r['text'],row['candidates'],selector))
            finally:row['selection_copy_seconds']=time.perf_counter()-t
        except Exception as exc:row.update(status='error',error=str(exc))
        row['document_seconds']=time.perf_counter()-document_started
        results.append(row);save(folder/'results.json',results)
        save(folder/'metrics.json',{'load_seconds':load_seconds,'pipeline_wall_seconds':time.perf_counter()-started,'completed_documents':len(results),'expected_documents':len(test)})
        print(f"test {index+1}/{len(test)} id={r['id']} status={row['status']} tokens={row['estimated_document_tokens']} candidates={len(row.get('candidates',[]))} seconds={row['document_seconds']:.2f}",flush=True)

if __name__=='__main__':main()
