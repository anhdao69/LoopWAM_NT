#!/usr/bin/env python3
"""Export final native policy weights, verify tensors, and publish one private HF repo."""
import argparse,datetime,hashlib,json,os,shutil,time
from pathlib import Path
import torch
from huggingface_hub import HfApi

def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--spec',required=True);a.add_argument('--output',required=True);a.add_argument('--token-file',required=True);a.add_argument('--repo',default='anhdao69/LoopWAM_NT');args=a.parse_args()
 root=Path(args.root);out=Path(args.output);stage=out/'repository';stage.mkdir(parents=True,exist_ok=True)
 tokenfile=Path(args.token_file);api=HfApi(token=tokenfile.read_text().strip());torch.set_num_threads(2)
 def status(stage_name,**kw):
  x=dict(stage=stage_name,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw);write(out/'status.json',x);print(json.dumps(x),flush=True)
 try:
  assert api.whoami()['name']=='anhdao69'
  api.create_repo(args.repo,repo_type='model',private=True,exist_ok=True)
  assert api.model_info(args.repo).private,'Destination must remain private'
  inventory=[]
  for spec in read(args.spec):
   train=root/spec['training_path'];evaluation=root/spec['evaluation_path'];dst=stage/spec['repo_path'];dst.mkdir(parents=True,exist_ok=True)
   timing=read(train/'timing.json');manifest=read(train/'manifest.json');summary=read(evaluation/'summary.json')
   assert timing['status']=='complete' and timing['completed_updates']==manifest['planned_updates']
   source=train/'latest.pt';target=dst/'policy.pt';infofile=dst/'export.json'
   if infofile.exists() and target.exists():
    info=read(infofile);assert sha(target)==info['export_sha256'];inventory.append(info);continue
   status('verifying_source',model=spec['id'],bytes=source.stat().st_size)
   source_sha=sha(source);assert source_sha==summary['checkpoint_sha256'],f"Source differs from evaluated checkpoint: {spec['id']}"
   payload=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
   assert payload['format_version']=='loopwam-s-v1' and payload['step']==timing['completed_updates']
   policy={k:v for k,v in payload.items() if k!='optimizer'}
   # Keep the full training contract for the native evaluator, but export no Adam moments.
   tmp=target.with_suffix('.pt.tmp');torch.save(policy,tmp);tmp.replace(target)
   restored=torch.load(target,map_location='cpu',mmap=True,weights_only=False)
   from fastwam.models.wan22.loopwam import _validate_checkpoint_depth
   _validate_checkpoint_depth(restored)
   for group in ('mot','proprio_encoder'):
    assert restored[group].keys()==payload[group].keys()
    for k,v in payload[group].items():assert torch.equal(v,restored[group][k]),f'Tensor mismatch: {group}/{k}'
   assert restored['step']==payload['step'] and restored['architecture']==payload['architecture']
   export_sha=sha(target)
   info=dict(**spec,source_sha256=source_sha,source_bytes=source.stat().st_size,export_sha256=export_sha,
       export_bytes=target.stat().st_size,format='loopwam-s-v1',optimizer_included=False,step=payload['step'],
       version=payload['version'],video_loops=payload.get('video_loops',payload['inference_loops']),
       action_loops=payload.get('action_loops',payload['inference_loops']),parameters=manifest['policy_parameters'],
       success_rate=summary['success_rate'],successes=summary['successes'],episodes=summary['total_episodes'],
       training_seconds=timing['elapsed_training_seconds'],tensor_equality_verified=True,
       source_revision=manifest.get('git_revision'),normalization_sha256=manifest['data']['normalization_sha256'])
   shutil.copy2(train/'data/dataset_stats.json',dst/'dataset_stats.json')
   assert sha(dst/'dataset_stats.json')==info['normalization_sha256']
   for src,name in ((train/'manifest.json','training_manifest.json'),(train/'data/data_manifest.json','data_manifest.json'),(train/'timing.json','training_timing.json'),(evaluation/'summary.json','evaluation_summary.json')):
    shutil.copy2(src,dst/name)
   write(infofile,info);inventory.append(info)
   del restored,policy,payload
   status('export_verified',model=spec['id'],bytes=info['export_bytes'],sha256=export_sha)
  write(stage/'checkpoint_index.json',dict(repo=args.repo,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checkpoints=inventory))
  card=['---','language: en','tags:','- robotics','- libero','- loopwam','- world-action-model','---','','# LoopWAM final checkpoints','','Final trained policy weights from the LoopWAM experiments. This repository contains **11 checkpoints**, separated by training data scope. Both original and repeated 4/4 runs are retained. All production checkpoints are from completed ten-epoch runs with global batch 128 and training seed 42.','','## Checkpoint index','','| Directory | Model | Video/action loops | Parameters | Final evaluation |','|---|---|---|---:|---|']
  for i in inventory:card.append(f"| [{i['repo_path']}]({i['repo_path']}) | {i['version']} | {i['video_loops']}/{i['action_loops']} | {i['parameters']:,} | {i['successes']}/{i['episodes']} ({i['success_rate']:.0%}) |")
  card += ['','`libero-long/`: matched 344-training / 44-validation demonstration split, 7,250 updates. Dense-S12/S30 execute their independent blocks once; their 1/1 metadata does not mean the dense networks have equal depth. S12 has 12 independent block pairs; S30 has 30. Loop models use three prelude, six shared core and three coda pairs, with 3 + 6K + 3 effective block applications per expert.','','`libero-all-suites/`: one v0 trained on all 1,712 locally available Spatial/Object/Goal/Long demonstrations, 21,700 updates. Its 388/400 result aggregates all four suites, so it is a different training/evaluation scope from the Long-only rows. The same four-suite checkpoint also scored 720/2,000 (36%) on LIBERO-Pro.','','All Long rows use 100 final-checkpoint rollouts at evaluation seed 42. Standard-suite evaluations use 700 policy steps, 30 settling steps, 32-action predictions, replanning every ten actions, ten denoising steps and CFG 1. The original 4/4 checkpoint additionally scored 96%, 88%, 91% across evaluation seeds 42/43/44. These are one training seed and cannot establish across-training-seed significance.','','## Files and integrity','','Each directory contains `policy.pt`, `dataset_stats.json`, `data_manifest.json`, `training_manifest.json`, `training_timing.json`, `evaluation_summary.json`, and `export.json`. `policy.pt` is the native `loopwam-s-v1` checkpoint, containing FP32 model tensors, architecture/depth metadata and the training contract. Adam optimizer states are omitted; use it for inference or fresh-optimizer fine-tuning, not exact optimizer-state resume. All exported model tensors were checked for exact equality with the evaluated final checkpoint. `export.json` records both the original checkpoint SHA-256 and the exported file SHA-256; these hashes differ because optimizer removal changes serialization.','','The frozen VAE and text encoder/embeddings are external inference dependencies. The native loader does not require the donor initialization artifact when `checkpoint_path` is provided. Normalization files are specific to each checkpoint and must remain matched. Videos, dataset videos and original optimizer checkpoints remain on the training server.','','## Loading','','Use the compatible [LoopWAM code](https://github.com/anhdao69/LoopWAM_NT/tree/LoopWAM_NT), at revision `8a29ffdce1537409e2b7e975e838d990ebbe068a` or a compatible later revision. This is a custom policy, not a Transformers AutoModel checkpoint. Install that repository’s inference dependencies first. Authenticate with a Hugging Face account authorized to access this private repository.','','```python','import torch','from huggingface_hub import hf_hub_download','from fastwam.models.wan22.loopwam import create_loopwam','','repo = "anhdao69/LoopWAM_NT"','variant = "libero-long/v0-video4-action4-repeat"','checkpoint = hf_hub_download(repo, f"{variant}/policy.pt")','stats = hf_hub_download(repo, f"{variant}/dataset_stats.json")','vae = hf_hub_download("Wan-AI/Wan2.1-T2V-1.3B", "Wan2.1_VAE.pth")','model = create_loopwam(checkpoint_path=checkpoint, vae_path=vae,','                       model_dtype=torch.float32, device="cuda").eval()','```','','Depth/version are reconstructed from checkpoint metadata. Use the repository’s observation adapter, matched normalization, text embeddings with their padding masks, and BF16 autocast during prediction. See `scripts/evaluate_loopwam_libero.py` for the complete simulator pipeline. Evaluation summaries retain the original checkpoint hashes; use the tensor equality and source/export mapping in `export.json` when auditing exports. The evaluator computes the exported file’s own hash for a new evaluation.','','## Provenance and rights','','These research checkpoints derive from pretrained Wan and the FastWAM/LoopWAM implementation. Consult the upstream model and code licenses before redistribution or use; this model card does not grant additional upstream rights. The repository is private.','','For detailed comparison caveats, runtime accounting and per-task results, see the [consolidated report](https://github.com/anhdao69/LoopWAM_NT/blob/LoopWAM_NT/plans/performance/LoopWAM_Consolidated_Results_20261008.md). That report’s earlier snapshot labels the repeat pending; the repeat is now complete at 91/100, as recorded here.']
  (stage/'README.md').write_text('\n'.join(card)+'\n')
  files=[p for p in stage.rglob('*') if p.is_file() and '.cache' not in p.parts]
  expected={str(p.relative_to(stage)):dict(size=p.stat().st_size,sha256=sha(p)) for p in files}
  status('uploading',files=len(files),bytes=sum(x['size'] for x in expected.values()))
  api.upload_large_folder(repo_id=args.repo,repo_type='model',folder_path=stage,private=True,num_workers=4,print_report=True,print_report_every=60)
  status('verifying_hub')
  remote={f.rfilename:f for f in api.model_info(args.repo,files_metadata=True).siblings}
  for name,meta in expected.items():
   assert name in remote and remote[name].size==meta['size'],f'Missing/wrong-size Hub file: {name}'
   lfs=remote[name].lfs
   if lfs is not None:
    remote_sha=lfs.sha256 if hasattr(lfs,'sha256') else lfs['sha256']
    assert remote_sha==meta['sha256'],f'Hub checksum mismatch: {name}'
  result=dict(repo=args.repo,private=api.model_info(args.repo).private,revision=api.model_info(args.repo).sha,
      files=len(expected),checkpoints=len(inventory),bytes=sum(x['size'] for x in expected.values()),
      verified_files=expected,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
  write(out/'upload_verification.json',result);status('complete',repo=args.repo,revision=result['revision'],checkpoints=len(inventory),bytes=result['bytes'])
 finally:
  tokenfile.unlink(missing_ok=True)
if __name__=='__main__':main()
