from pathlib import Path
import os,subprocess,json
root=Path.cwd(); folder=root/'reports/contributions/issue675/refinement-20261104'
python=str(root/'.venv/bin/python'); driver=str(root/'research/contributions/covalent_refinement_validation.py')
commands=[('final-holdout','contributions/boltz',[python,driver,'--output',str(folder/'final-holdout'),'--label','final-holdout','--cases','original','ace_cys','ace_cy3_aly','--seeds','2026']),('native-cli-final','contributions/boltz',[python,'-m','boltz.main','predict',str(folder/'cli-final-input'),'--model','boltz2','--cache',str(root/'cache'),'--out_dir',str(folder/'native-cli-final'),'--use_potentials','--recycling_steps','3','--sampling_steps','200','--diffusion_samples','1','--max_parallel_samples','1','--seed','0','--num_workers','0','--preprocessing-threads','1']),('upstream-control-repeat','upstream',[python,driver,'--output',str(folder/'upstream-control-repeat'),'--label','upstream-repeat','--cases','linear_aly_control','--seeds','3'])]
for label,source,cmd in commands:
 env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']='GPU-6b984734-8ed6-11dd-3b9a-0b6c36ba5e11'; env['PYTHONPATH']=str(root/source/'src')
 with (folder/(label+'.log')).open('w') as log:
  p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  (folder/(label+'-job.json')).write_text(json.dumps({'pid':p.pid,'cmd':cmd},indent=2)+'\n')
  status=p.wait()
  (folder/(label+'-exit.json')).write_text(json.dumps({'exit_code':status},indent=2)+'\n')
  print(label,status,flush=True)
  if status: raise SystemExit(status)
