import subprocess,os
from pathlib import Path
h=Path(__file__).resolve().parent
env=dict(os.environ,HF_HUB_OFFLINE='1',TIKTOKEN_CACHE_DIR='/tmp/jev-tiktoken')
for i,mode in enumerate(['serial','overlap','overlap','serial'],1):
 cmd=['/private/tmp/gliner-smoke-venv/bin/python',str(h/'run_extraction.py'),'--input',str(h/'runs/sustainability_11_gliner_compact/input.json'),'--text-field','text','--output-dir',str(h/f'runs/api_overlap_matched_{i}_{mode}'),'--extractor','gliner','--device','mps','--batch-size','16']
 if mode=='overlap':cmd.append('--overlap-api')
 with open(f'/tmp/api-overlap-trial-{i}.log','w') as log:
  subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print(f'Trial {i} {mode} complete',flush=True)
