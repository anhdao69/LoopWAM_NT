#!/usr/bin/env python3
"""Evaluate a completed four-suite v0 on isolated official PRO/Plus installs."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.evaluate_loopwam_libero import ObservationAdapter, validate_checkpoint, run_episode, sha256_file, write_json

SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
HORIZONS = dict(zip(SUITES, (220, 280, 300, 520)))
DIMENSIONS = ('object', 'swap', 'lan', 'task', 'env')


def task_specs(kind):
    return [(base, base if kind == 'plus' else base + '_' + dimension)
            for base in SUITES for dimension in (('plus',) if kind == 'plus' else DIMENSIONS)]


def language_from_bddl(task):
    from libero.libero import get_libero_path
    from libero.libero.envs import bddl_utils
    path = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    if '_view_' in path.name:
        path = path.with_name(path.name.split('_view_')[0] + '.bddl')
    return bddl_utils.get_problem_info(str(path))['language_instruction']


def make_tasks(kind):
    from libero.libero import benchmark
    # Official initial-state files are trusted legacy NumPy pickle payloads.
    # Scope compatibility to the benchmark module rather than global torch.
    benchmark.torch = SimpleNamespace(load=lambda path: torch.load(path, weights_only=False))
    tasks = []
    for base, name in task_specs(kind):
        suite = benchmark.get_benchmark_dict()[name]()
        for index in range(suite.n_tasks):
            task = suite.get_task(index)
            tasks.append((base, name, index, task, suite))
    random.Random(42).shuffle(tasks)
    return tasks


def task_categories(kind):
    if kind != 'plus':
        return {}
    from libero.libero import get_libero_path
    data = json.loads((Path(get_libero_path('benchmark_root')) / 'benchmark/task_classification.json').read_text())
    return {(suite, row['name']): row for suite, rows in data.items() for row in rows}


def prepare_text(tasks, cache, existing):
    from fastwam.datasets.loopwam_long import DEFAULT_PROMPT
    from fastwam.models.wan22.helpers.loader import _resolve_configs, _load_registered_model
    from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
    cache.mkdir(parents=True, exist_ok=True)
    prompts = sorted({DEFAULT_PROMPT.format(task=language_from_bddl(t[3])) for t in tasks})
    pending = []
    for prompt in prompts:
        name = hashlib.sha256(prompt.encode()).hexdigest() + '.t5_len128.wan22ti2v5b.pt'
        if not (cache / name).exists():
            if (existing / name).exists():
                (cache / name).symlink_to((existing / name).resolve())
            else:
                pending.append((prompt, name))
    if pending:
        _, text_config, _, tokenizer_config = _resolve_configs(
            model_id='Wan-AI/Wan2.2-TI2V-5B', tokenizer_model_id='Wan-AI/Wan2.1-T2V-1.3B',
            redirect_common_files=False)
        text_config.download_if_necessary()
        tokenizer_config.download_if_necessary()
        encoder = _load_registered_model(text_config.path, 'wan_video_text_encoder',
                                         torch_dtype=torch.bfloat16, device='cuda:0').eval()
        tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=128, clean='whitespace')
        with torch.inference_mode():
            for start in range(0, len(pending), 8):
                batch = pending[start:start+8]
                ids, mask = tokenizer([x[0] for x in batch], return_mask=True, add_special_tokens=True)
                ids, mask = ids.cuda(), mask.cuda().bool()
                context = encoder(ids, mask)
                for i, (_, name) in enumerate(batch):
                    torch.save(dict(context=context[i].cpu().bfloat16().contiguous(),
                                    mask=mask[i].cpu().contiguous()), cache / name)
                print(json.dumps(dict(event='text_progress', encoded=min(start+8, len(pending)), total=len(pending))), flush=True)
    write_json(cache / 'manifest.json', dict(prompts=len(prompts), generated=len(pending),
        files={p.name: sha256_file(p) for p in cache.glob('*.pt')}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', choices=['pro', 'plus'], required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--text-cache', required=True)
    p.add_argument('--prepare-text', action='store_true')
    p.add_argument('--preflight', action='store_true')
    a = p.parse_args()
    rank, world, local = (int(os.getenv(k, default)) for k, default in
                          [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')])
    torch.cuda.set_device(local)
    torch.set_num_threads(4)
    out, checkpoint, cache = Path(a.output), Path(a.checkpoint), Path(a.text_cache)
    data_dir = checkpoint.parent / 'data'
    data = json.loads((data_dir / 'data_manifest.json').read_text())
    stats = data_dir / 'dataset_stats.json'
    payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    contract = validate_checkpoint(payload, data, sha256_file(stats))
    if (payload['version'] != 'v0' or data.get('dataset_scope') != 'full_libero'
            or contract.get('epochs') != 10):
        raise ValueError('Requires the completed ten-epoch four-suite v0 checkpoint')
    step = payload['step']
    del payload
    tasks = make_tasks(a.benchmark)
    if a.prepare_text:
        if world != 1:
            raise ValueError('Prepare text with one GPU process')
        prepare_text(tasks, cache, Path(data['text_cache_dir']))
        return
    out.mkdir(parents=True, exist_ok=True)
    result_file = out / f'rank{rank}.jsonl'
    if result_file.exists():
        raise ValueError('Fresh output required; refusing to mix repeated episodes')
    checkpoint_hash = sha256_file(checkpoint)
    episodes_per_task = 1 if a.benchmark == 'plus' else 10
    from fastwam.models.wan22.loopwam import create_loopwam
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from experiments.libero.libero_utils import save_rollout_video
    adapter = ObservationAdapter(json.loads(stats.read_text()), cache)
    model = create_loopwam(checkpoint_path=str(checkpoint),
        vae_path='checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth',
        version='v0', loops=4, device=f'cuda:{local}', model_dtype=torch.float32).eval()
    assigned = tasks[rank::world]
    categories = task_categories(a.benchmark)
    if a.preflight:
        selected = []
        groups = set()
        for spec in assigned:
            category = (spec[1].rsplit('_', 1)[-1] if a.benchmark == 'pro'
                        else categories[(spec[0], spec[3].name)]['category'])
            if category not in groups:
                selected.append(spec)
                groups.add(category)
        assigned = selected
    if rank == 0:
        write_json(out / 'manifest.json', dict(benchmark=a.benchmark, mode='preflight' if a.preflight else 'full',
            checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash, checkpoint_step=step,
            version='v0', loops=4, world_size=world, tasks=len(tasks),
            episodes_per_task=episodes_per_task, planned_episodes=len(tasks)*episodes_per_task,
            horizons=HORIZONS, settling_steps=30, replan_steps=10, diffusion_steps=10,
            normalization_sha256=sha256_file(stats), text_manifest_sha256=sha256_file(cache/'manifest.json'),
            libero_module=str(Path(get_libero_path('benchmark_root')).resolve()),
            script_sha256=sha256_file(__file__), slurm_job_id=os.getenv('SLURM_JOB_ID')))
    completed = successes = 0
    started = time.time()
    for base, name, index, task, suite in assigned:
        description = language_from_bddl(task)
        context, mask = adapter.text(description)
        path = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(path), camera_heights=256, camera_widths=256)
        try:
            states = suite.get_task_init_states(index)
            count = 1 if a.preflight else episodes_per_task
            if len(states) < count:
                raise ValueError(f'Insufficient distinct initial states: {name}/{index}')
            def predict(obs, seed):
                image, proprio = adapter.observation(obs)
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    action = model.infer_action(input_image=image, proprio=proprio, context=context,
                        context_mask=mask, action_horizon=32, num_inference_steps=10,
                        text_cfg_scale=1.0, seed=seed)['action']
                return adapter.libero_actions(action)
            for episode in range(count):
                seed = 42 + task_specs(a.benchmark).index((base,name))*10000000 + index*1000 + episode*100
                random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed); env.seed(seed)
                then = time.perf_counter()
                row, frames = run_episode(env, states[episode], predict, seed=seed,
                    max_steps=20 if a.preflight else HORIZONS[base], wait_steps=30, replan_steps=10)
                videos = out / 'videos'; videos.mkdir(exist_ok=True)
                video = save_rollout_video(videos, frames, f'{name}_{index}_{episode}', row['success'],
                    re.sub(r'[^a-zA-Z0-9 ]', '_', description))
                row.update(suite=name, base_suite=base, task_id=index, task_name=task.name,
                    episode_index=episode, initial_state_index=episode, description=description,
                    rank=rank, checkpoint_sha256=checkpoint_hash, video=str(video),
                    duration_seconds=time.perf_counter()-then, mode='preflight' if a.preflight else 'full')
                row['category'] = (name.rsplit('_', 1)[-1] if a.benchmark == 'pro'
                                   else categories[(base, task.name)]['category'])
                with result_file.open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
                completed += 1; successes += row['success']
                write_json(out/f'progress_rank{rank}.json', dict(completed=completed, successes=successes,
                    planned=sum(1 if a.preflight else episodes_per_task for _ in assigned),
                    elapsed_seconds=time.time()-started, updated_unix=time.time()))
                print(json.dumps(dict(event='episode_complete', **row)), flush=True)
        finally:
            env.close()
    write_json(out/f'done_rank{rank}.json', dict(completed=completed, successes=successes,
        checkpoint_sha256=checkpoint_hash, elapsed_seconds=time.time()-started))


if __name__ == '__main__':
    main()
