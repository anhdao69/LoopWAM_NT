import json,hashlib,time
from pathlib import Path
import torch
from fastwam.models.wan22.loopwam_robot import create_loopwam_robot, payload_variant
B=Path('/media/vmo-perception/disk_2/anhdh35/loopwam_robotwin10_v0_4x4_20261105');torch.set_num_threads(2)
vae=Path('/media/vmo-perception/disk_2/anhdh35/loopwam_robotwin10_v0_4x4_20261104/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth')
c=json.loads((B/'checkpoint/config.json').read_text())
model,p=create_loopwam_robot(str(vae),checkpoint_path=str(B/'checkpoint/policy.pt'),device='cpu',checkpoint_blocks=False,variant='v0-video4-action4')
s=p['training_state'];contract=s['contract'];assert contract==c['training_contract']
assert p['step']==s['update']==45445 and s['next_epoch']==5 and s['next_batch']==0 and s['windows_seen']==5816755
assert payload_variant(p)=='v0-video4-action4' and model.mot.version=='v0' and model.mot.loops==model.mot.action_loops==4
assert len(model.video_expert.blocks)==len(model.action_expert.blocks)==12
params=sum(x.numel() for x in model.policy_parameters());assert params==584567886
assert all(torch.isfinite(x).all().item() for x in model.policy_parameters())
stats=json.loads((B/'checkpoint/dataset_stats.json').read_text());assert hashlib.sha256(json.dumps(stats,sort_keys=True).encode()).hexdigest()==contract['normalization_sha256']
assert hashlib.sha256(vae.read_bytes()).hexdigest()==contract['vae_sha256']
r=dict(strict_load=True,finite_weights=True,parameters=params,variant='v0-video4-action4',step=45445,epoch=5,physical_blocks_per_expert=12,video_loops=4,action_loops=4,normalization_matches=True,vae_matches=True,tasks=contract['tasks'],trained_contract=contract,validated_unix=time.time())
(B/'cpu_validation.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
