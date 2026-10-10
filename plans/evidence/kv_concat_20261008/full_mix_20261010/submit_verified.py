import json,os,re,subprocess,time
from pathlib import Path
r=Path('/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/mix_full_20261010')
assert (r/'release.json').is_file() and not (r/'failed.json').exists()
assert not (r/'submitted.json').exists()
def job(j):return subprocess.check_output(['scontrol','show','job','-o',str(j)],text=True)
old=job(4798)
for expected in ['UserId=anhdh35(', 'BatchFlag=0', 'Command=/bin/bash', 'NodeList=worker-3', 'JobState=RUNNING']:
 assert expected in old, (expected,old)
env=os.environ.copy();env.update(MIX_SOURCE=str(r/'source'),MIX_RELEASE=str(r/'release.json'),MIX_OUTPUT_PARENT=str(r.parent))
cmd=['sbatch','--parsable','--dependency=afterany:4798','--nodelist=worker-3','--output='+str(r/'slurm-%j.log'),str(r/'source/scripts/submit_full_mix.sbatch')]
subprocess.run([*cmd[:1],'--test-only',*cmd[1:]],env=env,check=True)
new=int(subprocess.check_output(cmd,env=env,text=True).strip().split(';')[0])
state=job(new)
record=dict(job=new,interactive=4798,submitted_unix=time.time(),command=cmd,job_description=state)
(r/'submitted.json').write_text(json.dumps(record,indent=2))
assert 'afterany:4798' in state and 'UserId=anhdh35(' in state and 'gres/gpu=2' in state
assert 'JobState=PENDING' in state
# Submission exists and is bound to this allocation before releasing it.
subprocess.run(['scancel','4798'],check=True)
record['interactive_cancel_requested_unix']=time.time()
(r/'submitted.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
