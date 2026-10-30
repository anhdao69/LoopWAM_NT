from pathlib import Path
import hashlib,json,subprocess,time
r=Path('/mnt/data/vmo-ai-task/anhdh35/robotwin_eval_runtime_20261030');c=Path('/mnt/data/vmo-ai-task/anhdh35/LoopWAM_robotwin_queue')
p=Path('/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robotwin_full_v0_20261028')
q=p/'posttrain_20261030';q.mkdir(exist_ok=True);(q/'queue').mkdir(exist_ok=True)
paths=[]
for folder in ['src/fastwam','experiments/robotwin/loopwam_policy','experiments/robotwin/chronoloop_policy']:
 paths+=list((c/folder).rglob('*.py'))+list((c/folder).rglob('*.yml'))
paths+=list((c/'scripts').glob('*robotwin*.py'))+[c/'scripts/supervise_robotwin_queue.sh',c/'third_party/RoboTwin/envs/_base_task.py',c/'third_party/RoboTwin/script/eval_policy.py']
cfg=dict(job_id=5489,parent_dir=str(p/'release_v2/production'),queue_dir=str(q/'queue'),eval_dir=str(q/'eval_epoch05'),continuation_dir=str(q/'epochs06_10'),
 preflight=str(r/'preflight_passed.json'),train_env=str(r/'train_env.sh'),eval_env=str(r/'eval_env.sh'),robotwin_root=str(r/'RoboTwin'),
 shared_dir='/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robotwin50_20261025/shared/robotwin50',
 init_artifact='/mnt/data/vmo-ai-task/anhdh35/FastWAM/checkpoints/LoopWAM/wan21_compact_donors.pt',eval_per_gpu=2,
 source_hashes={str(x.relative_to(c)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(set(paths))},created=time.time(),
 source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=c,text=True).strip())
config=q/'config.json'
if config.exists():raise ValueError('Refuse to overwrite an armed queue configuration')
config.write_text(json.dumps(cfg,indent=2))
cmd=['srun','--jobid=5489','--overlap','--ntasks=1','--cpus-per-task=1','--gres=none','--job-name=test_training_queue',
 'bash',str(c/'scripts/supervise_robotwin_queue.sh'),str(r/'train_env.sh'),str(c),str(config)]
(q/'launch.json').write_text(json.dumps(dict(command=cmd,created=time.time()),indent=2))
with (q/'controller.log').open('w') as f:proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=proc.pid,config=str(config),preflight_ready=Path(cfg['preflight']).exists())))
