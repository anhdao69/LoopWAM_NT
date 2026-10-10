import pathlib,json,hashlib,subprocess
R=pathlib.Path(__file__).resolve().parent
repos={'Light-WAM':R.parent,'LoopWAM':pathlib.Path('/storage/anhdh35/LoopWAM_NT-eval'),'Fast-WAM':pathlib.Path('/storage/anhdh35/FastWAM-eval')}
evidence={}
for name,p in repos.items():
 evidence[name]={}
 for folder in ['scripts','experiments/libero','src/fastwam/models/wan22']:
  for f in sorted((p/folder).glob('*.py')):
   evidence[name][str(f.relative_to(p))]=hashlib.sha256(f.read_bytes()).hexdigest()
 for f in [p/'setup_logs/environment.lock.txt',p/'scripts/libero_long_requirements.lock.txt']:
  if f.exists():evidence[name][str(f.relative_to(p))]={'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'contents':f.read_text()}
(R/'source_environment_evidence.json').write_text(json.dumps(evidence,indent=2))
provenance={'Python':'3.10','torch':'2.7.1+cu128','torchvision':'0.22.1+cu128','numpy':'1.26.4','mujoco':'3.3.2','robosuite':'1.4.0','envs':['lightwam-libero-eval','loopwam-libero-eval','fastwam-libero-eval'],'conda_root':'/storage/anhdh35/miniconda3','LoopWAM_original_release_revision':'006be1b959e8b188350bfa9db3944dfd0a8a1755','LoopWAM_v1a4_release_revision':'90453499bb825be2502702f20022895e1dfcda30','FastWAM_release_revision':'8eaceeb24c3cc92ff2a9c9a9d266a4941b836705','FastWAM_weight':'libero_uncond_2cam224.pt','FastWAM_checkpoint_sha256':'1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579','Wan22_VAE_sha256':'20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36','FastWAM_training_metadata':'Not independently recovered from original checkpoint; no assumed training seed/demo count.'}
(R/'release_environment_provenance.json').write_text(json.dumps(provenance,indent=2))
a=json.loads((R/'audit_findings.json').read_text())
a['sections']['Native model inference and fairness'].append('Original Fast-WAM is explicitly the pinned historical official code snapshot 8afd20f with action shift 5. Current main 7faa711 defaults to shift 1. The source configuration does not prove the released checkpoint historical training shift, and the checkpoint loader does not restore scheduler settings. Its historical training seed and demonstration count are not independently verified.')
a['sections']['Reproduction and artifacts'][0]=a['sections']['Reproduction and artifacts'][0].replace('scripts/evaluate_matched_lightwam.py and the campaign plan','scripts/run_matched_campaign.sh, scripts/evaluate_matched_lightwam.py and the campaign plan')
(R/'audit_findings.json').write_text(json.dumps(a,indent=2))
