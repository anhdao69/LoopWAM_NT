#!/usr/bin/env python3
"""Persistent independent rollout replicas sharing GPUs, with dynamic episode assignment."""
from __future__ import annotations
import argparse,hashlib,json,multiprocessing as mp,os,queue,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')

def write(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(p)

def make_jobs(suites,seeds,tasks,episodes):
 suites=list(suites);tasks=list(tasks)
 if (not suites or len(set(suites))!=len(suites) or any(s not in SUITES for s in suites)
     or not seeds or any(type(s)!=int or s<0 for s in seeds)
     or not tasks or len(set(tasks))!=len(tasks) or any(type(t)!=int or not 0<=t<10 for t in tasks)
     or type(episodes)!=int or not 1<=episodes<=50):raise ValueError('Invalid evaluation grid')
 # Long episodes start first, reducing final stragglers. Per-episode seeds are independent of assignment.
 return [dict(id=f'r{ri}_{s}_t{t}_e{e}',round=ri,base_seed=seed,suite=s,task=t,episode=e,
              seed=seed+t*100000+e*1000)
         for s in sorted(suites,key=lambda x:0 if x=='libero_10' else 1)
         for ri,seed in enumerate(seeds) for t in tasks for e in range(episodes)]

def verify_results(jobs,rows,checkpoint_hash,mode):
 expected={j['id']:j for j in jobs}
 if len(rows)!=len(expected) or {r['id'] for r in rows}!=set(expected):raise ValueError('Missing/duplicate episodes')
 for r in rows:
  if any(r.get(k)!=v for k,v in expected[r['id']].items()):raise ValueError('Episode identity or seed mismatch')
  if r.get('checkpoint_sha256')!=checkpoint_hash or r.get('mode')!=mode:raise ValueError('Checkpoint/protocol mismatch')
  p=Path(r.get('video',''))
  if not p.is_file() or p.stat().st_size==0:raise ValueError('Missing video')
 rounds={}
 for ri in sorted({j['round'] for j in jobs}):
  rr=[r for r in rows if r['round']==ri];per_suite={}
  for s in sorted({r['suite'] for r in rr}):
   rs=[r for r in rr if r['suite']==s];per_suite[s]=dict(episodes=len(rs),successes=sum(bool(r['success']) for r in rs),success_rate=sum(bool(r['success']) for r in rs)/len(rs))
  rounds[str(ri)]=dict(seed=rr[0]['base_seed'],episodes=len(rr),successes=sum(bool(r['success']) for r in rr),success_rate=sum(bool(r['success']) for r in rr)/len(rr),per_suite=per_suite)
 return dict(total_episodes=len(rows),successes=sum(bool(r['success']) for r in rows),
             success_rate=sum(bool(r['success']) for r in rows)/len(rows),rounds=rounds,
             episodes=sorted(rows,key=lambda r:r['id']),checkpoint_sha256=checkpoint_hash,mode=mode)

def worker(index,gpu,args,jobs,results,checkpoint_hash,data,stats):
 # Spawn starts a fresh interpreter. CUDA is initialized only after restricting visibility.
 os.environ['CUDA_VISIBLE_DEVICES']=gpu
 os.environ['OMP_NUM_THREADS']='1';os.environ['LP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
 try:
  import random,numpy as np,torch
  torch.set_num_threads(1);torch.cuda.set_device(0)
  from scripts.evaluate_loopwam_libero import ObservationAdapter,initial_states,run_episode
  from fastwam.models.wan22.loopwam import create_loopwam
  from libero.libero import benchmark
  from experiments.libero.libero_utils import get_libero_env,save_rollout_video
  model=create_loopwam(checkpoint_path=args.checkpoint,vae_path=args.vae_path,device='cuda:0',model_dtype=torch.float32).eval()
  adapter=ObservationAdapter(stats,data['text_cache_dir'],data['text_cache_files'])
  suites={s:benchmark.get_benchmark_dict()[s]() for s in args.suites}
  out=Path(args.output_dir);env=None;current=None;count=0
  results.put(dict(event='ready',worker=index,gpu=gpu))
  try:
   while True:
    job=jobs.get()
    if job is None:break
    key=(job['suite'],job['task'],job['base_seed'])
    if key!=current:
     if env is not None:env.close()
     task=suites[job['suite']].get_task(job['task']);context,mask=adapter.text(task.language)
     states=initial_states(task);env,description=get_libero_env(task,256,job['base_seed']);current=key
    seed=job['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);env.seed(seed)
    trace=[];digest=hashlib.sha256();started=time.perf_counter()
    def predict(obs,sampler_seed):
     image,proprio=adapter.observation(obs)
     with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
      action=model.infer_action(input_image=image,proprio=proprio,context=context,context_mask=mask,
           action_horizon=32,num_inference_steps=10,text_cfg_scale=1.0,seed=sampler_seed)['action']
     action=adapter.libero_actions(action)
     if args.trace_actions:
      trace.append(action.copy());digest.update(action.tobytes())
     return action
    row,frames=run_episode(env,states[job['episode']],predict,seed=seed,max_steps=args.max_steps,wait_steps=30,replan_steps=10)
    row.update(job,worker=index,checkpoint_sha256=checkpoint_hash,mode='smoke' if args.smoke else 'final_rollout',
         initial_state_index=job['episode'],task_id=job['task'],episode_index=job['episode'],task_description=description,
         duration_seconds=time.perf_counter()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
    row['video']=save_rollout_video(out/'videos',frames,job['id'],row['success'],description)
    if args.trace_actions:
     dest=out/'traces'/f"{job['id']}.npz";dest.parent.mkdir(exist_ok=True)
     np.savez_compressed(dest,actions=np.stack(trace) if trace else np.empty((0,32,7)))
     row['action_sha256']=digest.hexdigest();row['trace']=str(dest)
    write(out/'episodes'/f"{job['id']}.json",row);results.put(dict(event='episode',row=row));count+=1
  finally:
   if env is not None:env.close()
  results.put(dict(event='done',worker=index,episodes=count))
 except BaseException:
  results.put(dict(event='error',worker=index,error=traceback.format_exc()));raise

def run(args):
 import torch
 from scripts.evaluate_loopwam_libero import validate_checkpoint,checkpoint_policy_spec,sha256_file
 out=Path(args.output_dir).resolve();args.output_dir=str(out)
 if out.exists():raise ValueError('Fresh evaluation directory required')
 out.mkdir(parents=True);(out/'videos').mkdir()
 checkpoint=Path(args.checkpoint).resolve();args.checkpoint=str(checkpoint)
 stats_path=Path(args.stats).resolve();data=json.loads((stats_path.parent/'data_manifest.json').read_text());stats=json.loads(stats_path.read_text())
 payload=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
 for suite in args.suites:validate_checkpoint(payload,data,sha256_file(stats_path),smoke=args.smoke,suite=suite)
 version,loops=checkpoint_policy_spec(payload);action_loops=payload.get('action_loops',loops);step=payload['step'];del payload
 m=json.loads((checkpoint.parent/'manifest.json').read_text())
 assert sha256_file(args.vae_path)==m['asset_sha256']['vae']
 if not args.smoke:assert json.loads((checkpoint.parent/'timing.json').read_text())['status']=='complete'
 checkpoint_hash=sha256_file(checkpoint)
 visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
 if not visible or '' in visible or len(set(visible))!=len(visible):raise ValueError('Explicit allocated GPUs required')
 if args.workers_per_gpu<1 or args.workers_per_gpu>8:raise ValueError('Use 1..8 workers per GPU')
 grid=make_jobs(args.suites,args.seeds,args.tasks,args.episodes_per_task)
 manifest=dict(checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_hash,checkpoint_step=step,version=version,
   video_loops=loops,action_loops=action_loops,arguments=vars(args),gpus=visible,workers_per_gpu=args.workers_per_gpu,
   worker_count=len(visible)*args.workers_per_gpu,mode='smoke' if args.smoke else 'final_rollout',
   normalization_sha256=sha256_file(stats_path),seed_formula='base_seed + task_id*100000 + episode_index*1000 + replan_index',
   protocol=dict(max_policy_steps=args.max_steps,settling_steps=30,replan_steps=10,denoising_steps=10,cfg=1,action_chunk=32),
   source_script_sha256=sha256_file(__file__),slurm_job_id=os.getenv('SLURM_JOB_ID'))
 write(out/'manifest.json',manifest)
 ctx=mp.get_context('spawn');q=ctx.Queue();results=ctx.Queue();children=[];started=time.perf_counter();ready=set();rows=[];done=set();first_ready=None;all_ready=None
 for job in grid:q.put(job)
 for _ in range(manifest['worker_count']):q.put(None)
 try:
  for i in range(manifest['worker_count']):
   child=ctx.Process(target=worker,args=(i,visible[i%len(visible)],args,q,results,checkpoint_hash,data,stats));child.start();children.append(child)
  while len(done)<len(children):
   try:message=results.get(timeout=5)
   except queue.Empty:
    if any(p.exitcode not in (None,0) for p in children):raise RuntimeError('Evaluation worker exited with error')
    if all(p.exitcode is not None for p in children):raise RuntimeError('Workers exited without all completion records')
    continue
   event=message['event']
   if event=='error':raise RuntimeError(message['error'])
   if event=='ready':
    ready.add(message['worker']);first_ready=first_ready or time.perf_counter()
    if len(ready)==len(children):all_ready=time.perf_counter()
   elif event=='episode':rows.append(message['row'])
   elif event=='done':done.add(message['worker'])
   write(out/'progress.json',dict(ready=len(ready),workers=len(children),episodes=len(rows),planned=len(grid),successes=sum(bool(r['success']) for r in rows),elapsed_seconds=time.perf_counter()-started))
  for p in children:
   p.join(timeout=30)
   if p.exitcode!=0:raise RuntimeError('Worker teardown failed')
  summary=verify_results(grid,rows,checkpoint_hash,manifest['mode'])
  summary.update(checkpoint_step=step,version=version,video_loops=loops,action_loops=action_loops,
      elapsed_seconds=time.perf_counter()-started,model_startup_seconds=(all_ready or time.perf_counter())-started,
      episode_phase_seconds=time.perf_counter()-(first_ready or started),workers_per_gpu=args.workers_per_gpu,
      max_worker_gpu_gib=max(r['peak_gpu_gib'] for r in rows))
  write(out/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ('episodes','rounds')}),flush=True)
 finally:
  for p in children:
   if p.is_alive():p.terminate()
  for p in children:p.join(timeout=30)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--checkpoint',required=True);p.add_argument('--stats',required=True);p.add_argument('--vae-path',required=True)
 p.add_argument('--output-dir',required=True);p.add_argument('--workers-per-gpu',type=int,default=1)
 p.add_argument('--suites',nargs='+',default=list(SUITES));p.add_argument('--seeds',nargs='+',type=int,default=[42,42,42])
 p.add_argument('--tasks',nargs='+',type=int,default=list(range(10)));p.add_argument('--episodes-per-task',type=int,default=10)
 p.add_argument('--max-steps',type=int,default=700);p.add_argument('--smoke',action='store_true');p.add_argument('--trace-actions',action='store_true')
 a=p.parse_args()
 if a.max_steps<1 or (not a.smoke and a.max_steps!=700):p.error('Final evaluation requires 700 policy steps')
 run(a)
if __name__=='__main__':main()
