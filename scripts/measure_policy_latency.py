#!/usr/bin/env python3
"""Synchronized batch-one FastWAM/LoopWAM action-query latency on one H100."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from omegaconf import OmegaConf
from scripts.evaluate_loopwam_libero import ObservationAdapter, initial_states
from fastwam.models.wan22.loopwam import create_loopwam
from fastwam.models.wan22.loopwam_init import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--loop-checkpoint', required=True)
    p.add_argument('--fast-checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--samples', type=int, default=50)
    p.add_argument('--warmup', type=int, default=5)
    args = p.parse_args()
    if args.samples < 20 or args.warmup < 3:
        p.error('Require at least 20 measured queries and three warmups')
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    train = Path(args.loop_checkpoint).parent
    stats = json.loads((train / 'data/dataset_stats.json').read_text())
    data = json.loads((train / 'data/data_manifest.json').read_text())
    adapter = ObservationAdapter(stats, data['text_cache_dir'], data['text_cache_files'])
    from libero.libero import benchmark
    from experiments.libero.libero_utils import get_libero_env
    suite = benchmark.get_benchmark_dict()['libero_10']()
    task = suite.get_task(0)
    env, _ = get_libero_env(task, 256, 42)
    try:
        env.reset()
        obs = env.set_init_state(initial_states(task)[0])
        for _ in range(30):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    finally:
        env.close()
    context, mask = adapter.text(task.language)
    result = dict(hardware=torch.cuda.get_device_name(0), torch_version=torch.__version__,
        batch_size=1, camera_size=[224, 224], cameras=2, action_horizon=32,
        diffusion_steps=10, warmup_queries=args.warmup, measured_queries=args.samples,
        precision='FP32 weights, BF16 autocast compute', text='Cached embeddings with actual padding mask',
        timing='Synchronized wall time; includes observation preprocessing, transfers, online VAE, visual prefill and action denoising; excludes simulator, text encoder and model loading',
        models={})
    for name, checkpoint in [('fastwam', args.fast_checkpoint), ('loopwam_v0', args.loop_checkpoint)]:
        if name == 'fastwam':
            from fastwam.runtime import create_fastwam
            config = OmegaConf.to_container(OmegaConf.load('configs/model/fastwam.yaml'), resolve=False)
            config.pop('_target_')
            config['proprio_dim'] = 8
            for expert in ['video_dit_config', 'action_dit_config']:
                config[expert]['action_dim'] = 7
                config[expert]['use_gradient_checkpointing'] = False
            config.update(skip_dit_load_from_pretrain=True, load_text_encoder=False,
                          device='cuda:0', model_dtype=torch.float32)
            model = create_fastwam(**config).eval()
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
            model.mot.load_state_dict(payload['mot'], strict=True)
            model.proprio_encoder.load_state_dict(payload['proprio_encoder'], strict=True)
            del payload
        else:
            model = create_loopwam(checkpoint_path=checkpoint,
                vae_path='checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth',
                version='v0', loops=4, device='cuda:0', model_dtype=torch.float32).eval()
        times = []
        torch.cuda.reset_peak_memory_stats()
        for i in range(args.warmup + args.samples):
            torch.cuda.synchronize()
            started = time.perf_counter()
            image, proprio = adapter.observation(obs)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                action = model.infer_action(prompt=None, input_image=image, proprio=proprio,
                    context=context, context_mask=mask, action_horizon=32,
                    num_inference_steps=10, text_cfg_scale=1.0, seed=42+i)['action']
                cpu_action = action.detach().float().cpu()
            torch.cuda.synchronize()
            elapsed = 1000 * (time.perf_counter() - started)
            if cpu_action.shape != (32, 7) or not torch.isfinite(cpu_action).all():
                raise ValueError('Invalid predicted action')
            if i >= args.warmup:
                times.append(elapsed)
        result['models'][name] = dict(checkpoint=checkpoint, checkpoint_sha256=sha256_file(checkpoint),
            mean_ms=float(np.mean(times)), p50_ms=float(np.percentile(times, 50)),
            p95_ms=float(np.percentile(times, 95)), min_ms=min(times), max_ms=max(times),
            samples_ms=times, peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9)
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(json.dumps({name: result['models'][name]}), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
