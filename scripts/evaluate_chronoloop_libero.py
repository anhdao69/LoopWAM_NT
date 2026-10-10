#!/usr/bin/env python3
"""ChronoLoop LIBERO rollouts (plan 6.1), task-sharded by torchrun, on the matched harness.

torchrun --standalone --nproc_per_node=4 scripts/evaluate_chronoloop_libero.py \
  --checkpoint HF_DIR/epoch_10/policy.pt --suite libero_10 --episodes-per-task 50 \
  --seed 42 --noise fixed --output-dir OUT

Protocol: 700 policy steps, 30 settling steps, replan every 10 actions, 10 denoising steps,
CFG 1, the parent's observation/action adapters and normalizer (shared/data/dataset_stats.json).
Memory state is reset at episode start; settling steps never touch it; it advances once per
replanning query. CL-FRAME keeps the observation of query k-3 (first query when k < 3).
Noise: fixed = the same noise vector at every query of an episode (primary); fresh = new per query.
CL-TF (eval-only, CL-0 checkpoint) is not implemented here.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from evaluate_loopwam_libero import (LIBERO_SUITES, ObservationAdapter, initial_states, run_episode,  # noqa: E402
                                     sha256_file, shard_tasks, write_json)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config', help='config.json next to the HF checkpoint (default: beside --checkpoint)')
    p.add_argument('--suite', choices=LIBERO_SUITES, default='libero_10')
    p.add_argument('--vae-path', default='checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth')
    p.add_argument('--data-dir', default=os.environ.get('CHRONO_SHARED', '') + '/data',
                   help='directory with dataset_stats.json and data_manifest.json of the training data')
    p.add_argument('--text-cache-dir', default='data/text_embeds_cache/libero')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--tasks', default=','.join(map(str, range(10))))
    p.add_argument('--episodes-per-task', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--noise', choices=['fixed', 'fresh'], default='fixed')
    p.add_argument('--max-steps', type=int, default=700)
    p.add_argument('--wait-steps', type=int, default=30)
    p.add_argument('--replan-steps', type=int, default=10)
    p.add_argument('--no-video', action='store_true')
    args = p.parse_args()
    rank, world, local = (int(os.environ.get(k, d)) for k, d in (('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)))
    tasks = [int(x) for x in args.tasks.split(',')]
    assigned = shard_tasks(tasks, rank, world)
    ckpt = Path(args.checkpoint).resolve()
    payload = torch.load(ckpt, map_location='cpu', weights_only=False, mmap=True)
    from fastwam.models.wan22.chronoloop import ChronoConfig, create_chronoloop
    cfg = ChronoConfig(**payload['chrono'])
    del payload
    data_dir = Path(args.data_dir)
    stats = json.loads((data_dir / 'dataset_stats.json').read_text())
    data = json.loads((data_dir / 'data_manifest.json').read_text())
    torch.cuda.set_device(local)
    model, _ = create_chronoloop(cfg, args.vae_path, checkpoint_path=str(ckpt), device=f'cuda:{local}')
    model.eval()
    adapter = ObservationAdapter(stats, args.text_cache_dir, data['text_cache_files'])
    from libero.libero import benchmark
    from experiments.libero.libero_utils import get_libero_env, save_rollout_video
    suite = benchmark.get_benchmark_dict()[args.suite]()
    out = Path(args.output_dir); (out / 'episodes').mkdir(parents=True, exist_ok=True)
    ckpt_hash = sha256_file(ckpt) if rank == 0 else None
    if rank == 0:
        write_json(out / 'manifest.json', dict(checkpoint=str(ckpt), checkpoint_sha256=ckpt_hash, chrono=cfg.__dict__,
            suite=args.suite, noise=args.noise, seed=args.seed, episodes_per_task=args.episodes_per_task,
            protocol=dict(max_policy_steps=args.max_steps, settling_steps=args.wait_steps, replan_steps=args.replan_steps,
                          inference_steps=10, cfg=1.0, action_chunk=32),
            seed_formula='seed + task_id*100000 + episode_index*1000 (+ replan index when noise=fresh)'))
    episodes = []
    for task_id in assigned:
        task = suite.get_task(task_id)
        context, mask = adapter.text(task.language)
        states = initial_states(task)
        env, description = get_libero_env(task, 256, args.seed)
        try:
            for ep in range(args.episodes_per_task):
                seed = args.seed + task_id * 100000 + ep * 1000
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); env.seed(seed)
                memo = dict(state=None, images=[], latency=[])

                def predict(obs, sampler_seed):
                    image, proprio = adapter.observation(obs)
                    memo['images'].append(image)
                    k = len(memo['images']) - 1
                    history = memo['images'][max(0, k - 3)] if cfg.history_frame else None
                    t0 = time.perf_counter()
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                        result = model.chrono_infer_action(image, proprio, context, mask, state=memo['state'],
                            history_image=history, num_inference_steps=10,
                            seed=seed if args.noise == 'fixed' else sampler_seed)
                    torch.cuda.synchronize(); memo['latency'].append(time.perf_counter() - t0)
                    memo['state'] = result['state']
                    return adapter.libero_actions(result['action'])

                started = time.perf_counter()
                result, frames = run_episode(env, states[ep % len(states)], predict, seed=seed,
                    max_steps=args.max_steps, wait_steps=args.wait_steps, replan_steps=args.replan_steps)
                result.update(suite=args.suite, task_id=task_id, episode_index=ep, initial_state_index=ep % len(states),
                              noise=args.noise, rank=rank, duration_seconds=time.perf_counter() - started,
                              query_latency_ms_p50=float(np.median(memo['latency']) * 1e3))
                if not args.no_video:
                    result['video'] = save_rollout_video(out / 'videos', frames, f'task{task_id}_trial{ep}',
                                                         result['success'], description)
                write_json(out / 'episodes' / f'task{task_id}_episode{ep}.json', result)
                episodes.append(result)
                print(json.dumps(dict(event='episode_complete', task=task_id, episode=ep, success=result['success'])), flush=True)
        finally:
            env.close()
    write_json(out / f'rank{rank}.json', dict(rank=rank, episodes=episodes))
    if world > 1:
        import torch.distributed as dist
        dist.init_process_group('gloo'); dist.barrier()
    if rank == 0:
        rows = []
        for r in range(world):
            rows += json.loads((out / f'rank{r}.json').read_text())['episodes']
        succ = sum(r['success'] for r in rows)
        write_json(out / 'summary.json', dict(suite=args.suite, noise=args.noise, seed=args.seed, checkpoint_sha256=ckpt_hash,
            chrono=cfg.__dict__, episodes=len(rows), successes=succ, success_rate=succ / max(1, len(rows)),
            per_task={str(t): sum(r['success'] for r in rows if r['task_id'] == t) for t in tasks}))
        print(json.dumps(dict(event='done', successes=succ, episodes=len(rows))), flush=True)


if __name__ == '__main__':
    main()
