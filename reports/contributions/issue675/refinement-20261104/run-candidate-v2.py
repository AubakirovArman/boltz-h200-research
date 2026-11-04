from pathlib import Path
import os,subprocess,json,hashlib
root=Path.cwd(); folder=root/'reports/contributions/issue675/refinement-20261104'
env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']='GPU-6b984734-8ed6-11dd-3b9a-0b6c36ba5e11'; env['PYTHONPATH']=str(root/'contributions/boltz/src')
commands=[('candidate-v2',root/'research/contributions/covalent_guidance_pilot.py',[]),('candidate-v2-panel',root/'research/contributions/covalent_refinement_validation.py',['--label','candidate-v2','--cases','original','ace_cys','ace_cy3_aly','linear_aly_control','interchain_control'])]
for label,script,extra in commands:
 with (folder/(label+'.log')).open('w') as log:
  p=subprocess.Popen([str(root/'.venv/bin/python'),str(script),'--output',str(folder/label),*extra],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  (folder/(label+'-job.json')).write_text(json.dumps({'pid':p.pid,'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest()},indent=2)+'\n')
  code=p.wait()
  print(label,code,flush=True)
  if code: raise SystemExit(code)
