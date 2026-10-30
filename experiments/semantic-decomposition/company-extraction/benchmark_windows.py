"""Two counter-ordered full-dataset trials per batch size; no API calls."""
import json,os,time,statistics
from pathlib import Path
from pipeline import LocalExtractor,extract_candidates
h=Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME','/private/tmp/gliner-smoke-hf')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ['HF_HUB_OFFLINE']='1'
import torch
torch.set_num_threads(4)
p=h/'runs/window_batch_benchmark';p.mkdir(exist_ok=False)
records=json.loads((h/'runs/sustainability_11_gliner_compact/input.json').read_text())
old={r['id']:r for r in json.loads((h/'runs/sustainability_11_gliner_compact/results.json').read_text())}
def signature(r):return {(c['text'],m['start'],m['end']) for c in r['candidates'] for m in c['mentions']}
t=time.perf_counter();extractor=LocalExtractor('gliner','mps',.5);load=time.perf_counter()-t
extractor('Volvo Group published its annual report.')
trials=[]
for trial,b in enumerate([1,8,16,32,32,16,8,1],1):
 rows=[];start=time.perf_counter()
 for r in records:
  t=time.perf_counter();out=extract_candidates(r['text'],extractor,batch_size=b);elapsed=time.perf_counter()-t
  out.update(id=r['id'],wall_seconds=elapsed)
  before,after=signature(old[r['id']]),signature(out)
  out.update(added_spans=len(after-before),removed_spans=len(before-after),
             previous_selected_name_retained=any(c['text']==old[r['id']]['company_name'] for c in out['candidates']))
  rows.append(out)
 elapsed=time.perf_counter()-start
 (p/f'trial_{trial}_batch_{b}.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
 info={'trial':trial,'batch_size':b,'wall_seconds':elapsed,'added_spans':sum(r['added_spans'] for r in rows),'removed_spans':sum(r['removed_spans'] for r in rows),'previous_selected_names_retained':sum(r['previous_selected_name_retained'] for r in rows)}
 trials.append(info);(p/'metrics.json').write_text(json.dumps({'load_seconds':load,'trials':trials,'device':'mps','cost_usd':0,'warmup':'one synthetic window before trials','order':'1,8,16,32,32,16,8,1'},indent=2))
 print(json.dumps(info),flush=True)
print('DONE',flush=True)
