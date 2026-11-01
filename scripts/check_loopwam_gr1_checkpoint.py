"""GR1 v0 strict reload, loss-wrapper equivalence and cached-observation inference gate (one GPU)."""
import argparse, json
from pathlib import Path

import numpy as np
import torch

from fastwam.datasets.loopwam_latent_cache import LoopWAMLatentCache
from fastwam.datasets.robocasa_gr1 import GR1Windows, LATENT_SHAPE, OBS_TOKENS, ROBOT_DIM, VIDEO_HW, gr1_spec, latent_provenance_gr1
from fastwam.models.wan22.loop_mot import LoopMoT
from fastwam.models.wan22.loopwam_gr1 import PARAMETERS, RobotTrainForward, create_loopwam_gr1
from fastwam.models.wan22.loopwam_init import sha256_file
from train_loopwam_gr1 import slot_noise

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True); p.add_argument('--shared', required=True); p.add_argument('--latent-dir', required=True)
p.add_argument('--output', required=True); p.add_argument('--vae-path', default='checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth')
a = p.parse_args()
torch.set_num_threads(2); torch.cuda.set_device(0)
model, payload = create_loopwam_gr1(a.vae_path, checkpoint_path=a.checkpoint, device='cuda:0', checkpoint_blocks=False)
assert type(model.mot) is LoopMoT and sum(x.numel() for x in model.policy_parameters()) == PARAMETERS
assert all(x.dtype == torch.float32 for x in model.policy_parameters())
has_opt = 'optimizer' in payload
if has_opt:
    assert payload['optimizer']['state']
    assert all(v.dtype == torch.float32 for s in payload['optimizer']['state'].values() for k, v in s.items() if k in ('exp_avg', 'exp_avg_sq'))
assert not any('memory' in key for key in payload['mot'])
for family, state in [('mot', model.mot.state_dict()), ('proprio_encoder', model.proprio_encoder.state_dict())]:
    assert state.keys() == payload[family].keys()
    assert all(torch.equal(v.detach().cpu(), payload[family][k]) for k, v in state.items())
contract = payload['training_state']['contract']
assert contract['vae_sha256'] == sha256_file(a.vae_path)
model.mot.structured_attention = True; model.mot.structured_attention_observation_tokens = OBS_TOKENS
shared = Path(a.shared)
w = GR1Windows(gr1_spec(stats_path=shared / 'data/dataset_stats_29d.json'), cache_dir=shared / 'data', text_threads=4)
cache = LoopWAMLatentCache(Path(a.latent_dir), len(w), latent_provenance_gr1(w, contract['vae_sha256']), latent_shape=LATENT_SHAPE)
idx = int(w.episodes[0][2]) + 10
f = w.fields(idx); sample = {k: v.unsqueeze(0).cuda() for k, v in f.items() if isinstance(v, torch.Tensor)}
sample['latents'] = torch.from_numpy(np.array(cache._latents[idx:idx + 1]).view(np.int16)).view(torch.bfloat16).float().cuda()
sample['video'] = torch.zeros((), device='cuda').expand(1, 3, 9, *VIDEO_HW)
noise = slot_noise(model, [42], torch.device('cuda:0')); model.eval()
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    noisy = model.prepare_training_batch(sample, noise_video=noise['video'], noise_action=noise['action'],
                                         timestep_video=noise['t_video'], timestep_action=noise['t_action'])
    original, _ = model.reduce_flow_losses(model.forward_exits(noisy), noisy)
    wrapped, _, _ = RobotTrainForward(model)([dict(sample=sample, noise=noise)])
    assert torch.equal(original, wrapped) and torch.isfinite(wrapped)
    image = model.vae.model.decode(sample['latents'][:, :, :1].to(torch.bfloat16), model.vae.scale)[:, :, 0].clamp(-1, 1)
    model.vae.model.clear_cache()
    kw = dict(input_image=image, proprio=sample['proprio'][0, 0], context=sample['context'], context_mask=sample['context_mask'],
              num_inference_steps=10, seed=42)
    first = model.infer_action(**kw)['action']; second = model.infer_action(**kw)['action']
    assert first.shape == (32, ROBOT_DIM) and torch.isfinite(first).all() and torch.equal(first, second)
    mse = float(((first.float() - sample['action'][0].float()) ** 2).mean())
result = dict(passed=True, step=payload['step'], parameters=PARAMETERS, model='original LoopWAM v0 / LoopMoT (GR1 29-D)',
              policy_dtype='float32', optimizer_present=has_opt, optimizer_moments='float32' if has_opt else None,
              model_weights_restored_exact=True, loss_wrapper_exact=True, loss=float(wrapped), inference_shape=list(first.shape),
              inference_finite=True, repeated_seed_exact=True, window=idx, normalized_action_mse_vs_ground_truth=mse)
Path(a.output).write_text(json.dumps(result, indent=2)); print(json.dumps(result))
