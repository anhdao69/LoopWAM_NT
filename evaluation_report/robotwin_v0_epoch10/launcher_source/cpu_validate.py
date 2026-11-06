import json,hashlib,time
from pathlib import Path
import torch
from fastwam.models.wan22.loopwam_robot import create_loopwam_robot
B=Path('/data14t/anhdh35/loopwam_robotwin_v0_epoch10_20261106')
VAE=Path('/storage/anhdh35/Light-WAM/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth')
torch.set_num_threads(2)
config=json.loads((B/'checkpoint/config.json').read_text());stats=json.loads((B/'checkpoint/dataset_stats.json').read_text())
model,p=create_loopwam_robot(str(VAE),checkpoint_path=str(B/'checkpoint/policy.pt'),device='cpu',checkpoint_blocks=False)
state=p['training_state'];contract=state['contract']
assert p['step']==config['optimizer_update']==state['update']==474620
assert state['next_epoch']==10 and state['next_batch']==0 and state['windows_seen']==60751030
assert contract==config['training_contract']
assert hashlib.sha256(json.dumps(stats,sort_keys=True).encode()).hexdigest()==contract['normalization_sha256']
assert hashlib.sha256(VAE.read_bytes()).hexdigest()==contract['vae_sha256']
parameters=sum(x.numel() for x in model.policy_parameters());assert parameters==584567886
weights=list(p['mot'].values())+list(p['proprio_encoder'].values())
assert all(torch.isfinite(x).all().item() for x in weights)
assert all(x.dtype==torch.float32 for x in weights if torch.is_floating_point(x))
old=json.loads(Path('/data14t/anhdh35/loopwam_robotwin_v0_20261030/cpu_validation.json').read_text())['trained_contract']
keys=['model','partition','video_loops','action_loops','memory_tokens','history_frame','init_sha256','vae_sha256','normalization_sha256','data_manifest_sha256','seed','global_batch']
assert all(contract[k]==old[k] for k in keys)
report=dict(strict_load=True,finite_weights=True,parameters=parameters,step=p['step'],epoch=10,normalization_matches=True,vae_matches=True,epoch5_common_contract_matches=True,trained_contract=contract,validated_unix=time.time())
(B/'cpu_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
