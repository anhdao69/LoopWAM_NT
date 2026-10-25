#!/usr/bin/env python3
"""RoboTwin 2.0 evaluation of ChronoLoop robot checkpoints with the official harness.

python scripts/evaluate_chronoloop_robotwin.py --checkpoint RUN/checkpoints/epoch_05.pt --output-dir OUT \
    --gpus 0,1 --per-gpu 5 [--tasks hanging_mug,...] [--configs demo_clean,demo_randomized] [--episodes 50] \
    [--stop-unix T]

One RoboTwin `script/eval_policy.py` process per (task, setting), run from the RoboTwin root with the policy
`policy/chronoloop_policy` (experiments/robotwin/chronoloop_policy). Each process walks seeds from
100000 * (1 + seed); a seed counts only if the expert planner solves it, and the policy then gets the first
`--episodes` such seeds, with an instruction drawn from the task's seen or unseen templates (seeded per episode in
our harness copy, so every policy sees the same scene and instruction at episode k).
All jobs can run at once (`--per-gpu`), so partial results stay balanced across tasks; one shared T5 server
(t5_server.py) encodes the instructions. Jobs run at lower CPU priority (`--nice`) so a training run on the same
node keeps its speed. `--split-from REF --parts N` runs each job as N processes: part i starts at the accepted
seed of REF's episode 50*i/N (REF = an earlier evaluation dir of the same tasks) with the episode index offset to
match, so the parts together are exactly the sequential job. `--seeds-from REF` instead evaluates exactly the
scene seeds REF accepted (one process per task and setting), so models evaluated against the same REF are scored on
identical scenes even where the expert check would accept a slightly different seed set. At `--stop-unix`, or when OUT/STOP exists, the runner
stops every job; OUT/summary.json keeps each finished episode
(seed, success), so a stopped job still counts its first k episodes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('hanging_mug', 'blocks_ranking_size', 'scan_object', 'place_can_basket', 'handover_mic')
EPISODE = re.compile(r'Success rate: (?:\x1b\[\d+m)?(\d+)/(\d+).*?current seed: (?:\x1b\[\d+m)?(\d+)')


def episodes(log):
    """[(seed, success)] for every finished episode in a RoboTwin eval log."""
    out, prev = [], 0
    for k, n, seed in EPISODE.findall(log.read_text(errors='replace')) if log.exists() else []:
        out.append((int(seed), int(k) - prev))
        prev = int(k)
    return out


def job_dirs(out, task, config):
    """A job's directories in episode order: TASK__CONFIG, then TASK__CONFIG__fromKK for later parts."""
    parts = sorted(out.glob(f'{task}__{config}__from*'), key=lambda d: int(d.name.rsplit('from', 1)[1]))
    return [out / f'{task}__{config}'] + parts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--robotwin-root', default='/groups/yshang/an221229/code/RoboTwin_eval/RoboTwin')
    p.add_argument('--overlay', default='/groups/yshang/an221229/envs/robotwin_overlay',
                   help='sapien, mplib, curobo, ... installed with --target next to the training environment')
    p.add_argument('--numpy-overlay', default='/groups/yshang/an221229/envs/robotwin_overlay_np1',
                   help='numpy 1.26.4 ahead of the environment: mplib 0.2.1 segfaults under numpy 2 (fcl.Box)')
    p.add_argument('--gcc-lib', default='/apps/gcc/gcc-11.2.0/lib64', help="libstdc++ of the compiler CuRobo was built with")
    p.add_argument('--data-dir', default='/groups/yshang/an221229/checkpoints/ChronoLoop/robot/robotwin5hard/data')
    p.add_argument('--dataset', default='robotwin5hard')
    p.add_argument('--tasks', default=','.join(TASKS))
    p.add_argument('--configs', default='demo_clean,demo_randomized')
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--instruction-type', default='unseen', choices=['seen', 'unseen'])
    p.add_argument('--replan-steps', type=int, default=24)
    p.add_argument('--gpus', default='0')
    p.add_argument('--per-gpu', type=int, default=1, help='processes per entry of --gpus (a GPU may be listed twice)')
    p.add_argument('--t5-gpu', default=None, help='GPU of the shared T5 server (11 GB); default: the first of --gpus')
    p.add_argument('--nice', type=int, default=10)
    p.add_argument('--stop-unix', type=float, default=None, help='stop all jobs at this time, keeping finished episodes')
    p.add_argument('--split-from', default=None, help='evaluation dir whose accepted seeds define the split points')
    p.add_argument('--parts', type=int, default=1)
    p.add_argument('--seeds-from', default=None, help='complete evaluation dir whose accepted seeds are evaluated exactly')
    a = p.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    ckpt = Path(a.checkpoint).resolve()
    if a.seeds_from and a.split_from:
        raise SystemExit('--seeds-from and --split-from are exclusive')
    seed_lists = {}                                       # (task, config) -> file with the reference's seeds
    if a.seeds_from:
        (out / 'seed_lists').mkdir(exist_ok=True)
        for c in a.configs.split(','):
            for t in a.tasks.split(','):
                seeds = [s for d in job_dirs(Path(a.seeds_from), t, c) for s, _ in episodes(d / 'eval.log')]
                if len(seeds) != a.episodes or len(set(seeds)) != len(seeds):
                    raise SystemExit(f'{a.seeds_from}: {t}/{c} has {len(seeds)} seeds, need {a.episodes} distinct')
                seed_lists[t, c] = out / 'seed_lists' / f'{t}__{c}.json'
                seed_lists[t, c].write_text(json.dumps(seeds))
    jobs = []                                             # (task, config, first episode, episodes, first seed)
    for c in a.configs.split(','):
        for t in a.tasks.split(','):
            ref = [s for s, _ in episodes(Path(a.split_from) / f'{t}__{c}' / 'eval.log')] if a.split_from else []
            starts = [round(i * a.episodes / a.parts) for i in range(a.parts)] if a.split_from else [0]
            starts = [b for b in starts if b == 0 or b < len(ref)]          # a split needs the reference seed
            for b, e in zip(starts, starts[1:] + [a.episodes]):
                jobs.append((t, c, b, e - b, ref[b] if b else None))
    jobs.sort(key=lambda j: j[2])                         # first parts first
    gpus = a.gpus.split(',')
    slots = [g for _ in range(a.per_gpu) for g in gpus]
    env = dict(os.environ)
    env['PYTHONPATH'] = f'{a.numpy_overlay}:{a.overlay}:{a.robotwin_root}:' + env.get('PYTHONPATH', '')
    env['PATH'] = f'{a.overlay}/bin:' + env['PATH']
    env['LD_LIBRARY_PATH'] = f'{a.gcc_lib}:' + env.get('LD_LIBRARY_PATH', '')
    env['PYTHONUNBUFFERED'] = '1'                         # per-episode progress in eval.log
    git = lambda *cmd: subprocess.run(['git', *cmd], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (out / 'manifest.json').write_text(json.dumps(dict(
        checkpoint=str(ckpt), robotwin_root=a.robotwin_root, dataset=a.dataset, data_dir=a.data_dir, tasks=a.tasks,
        configs=a.configs, episodes=a.episodes, seed=a.seed, first_seed=100000 * (1 + a.seed),
        split_from=a.split_from, seeds_from=a.seeds_from, parts=[dict(task=t, config=c, first_episode=b, episodes=n, first_seed=s)
                                        for t, c, b, n, s in jobs],
        instruction_type=a.instruction_type, instruction_seeded_per_episode=True, replan_steps=a.replan_steps,
        num_inference_steps=10, action_chunk=32, noise='fixed per episode', expert_check=True,
        git_revision=git('rev-parse', 'HEAD'), git_dirty=bool(git('status', '--porcelain', '--', 'experiments', 'scripts')),
        started=time.time()), indent=2))

    sock = f'/tmp/chronoloop_t5_{os.getpid()}.sock'
    server = subprocess.Popen([sys.executable, str(ROOT / 'experiments/robotwin/chronoloop_policy/t5_server.py'),
                               '--address', sock], env={**env, 'CUDA_VISIBLE_DEVICES': a.t5_gpu or gpus[0]},
                              stdout=(out / 't5_server.log').open('w'), stderr=subprocess.STDOUT)
    while 'ready' not in (out / 't5_server.log').read_text(errors='replace'):
        if server.poll() is not None:
            raise SystemExit(f'T5 server exited with {server.returncode}; see {out / "t5_server.log"}')
        time.sleep(5)

    running, results = {}, {}

    def finish(slot, stopped=False):
        proc, (task, config, first, count, _), jdir, t0 = running.pop(slot)
        eps = episodes(jdir / 'eval.log')
        key = f'{task}/{config}/{first}'
        results[key] = dict(task=task, config=config, first_episode=first, episodes=len(eps),
                            successes=sum(s for _, s in eps), seeds=[s for s, _ in eps], success=[x for _, x in eps],
                            exit=proc.returncode, stopped=stopped,
                            # a pinned job that walked its whole seed list is complete even if it had to skip a seed
                            # whose scene could not be built (see skipped; the report compares matched scenes)
                            complete=proc.returncode == 0 and not stopped
                                     and (len(eps) == count or (task, config) in seed_lists),
                            skipped=count - len(eps) if (task, config) in seed_lists and not stopped else 0,
                            hours=round((time.time() - t0) / 3600, 3))
        print(json.dumps(dict(event='end', **{k: v for k, v in results[key].items() if k not in ('seeds', 'success')})),
              flush=True)

    try:
        while jobs or running:
            if (out / 'STOP').exists() or (a.stop_unix and time.time() > a.stop_unix):
                for slot, (proc, *_rest) in list(running.items()):
                    proc.terminate()
                    try:
                        proc.wait(60)
                    except subprocess.TimeoutExpired:
                        proc.kill(); proc.wait()
                    finish(slot, stopped=True)
                if jobs:
                    print(json.dumps(dict(event='not_started', jobs=[j[:3] for j in jobs])), flush=True)
                break
            for slot in [s for s in range(len(slots)) if s not in running]:
                if not jobs:
                    break
                job = jobs.pop(0)
                task, config, first, count, first_seed = job
                jdir = out / (f'{task}__{config}' + (f'__from{first:02d}' if first else '')); jdir.mkdir(exist_ok=True)
                cmd = [sys.executable, 'script/eval_policy.py', '--config', 'policy/chronoloop_policy/deploy_policy.yml',
                       '--overrides', '--task_name', task, '--task_config', config, '--ckpt_setting', str(ckpt),
                       '--seed', str(a.seed), '--eval_num_episodes', str(count), '--eval_output_dir', str(jdir),
                       '--instruction_type', a.instruction_type, '--dataset', a.dataset, '--data_dir', a.data_dir,
                       '--replan_steps', str(a.replan_steps), '--t5_server', sock, '--episode_offset', str(first),
                       '--instruction_pool', str(a.episodes)]
                if first_seed is not None:
                    cmd += ['--st_seed', str(first_seed)]
                if (task, config) in seed_lists:
                    cmd += ['--seed_list', str(seed_lists[task, config])]
                proc = subprocess.Popen(cmd, cwd=a.robotwin_root, env={**env, 'CUDA_VISIBLE_DEVICES': slots[slot]},
                                        stdout=(jdir / 'eval.log').open('w'), stderr=subprocess.STDOUT,
                                        preexec_fn=lambda: os.nice(a.nice))
                running[slot] = (proc, job, jdir, time.time())
                print(json.dumps(dict(event='start', task=task, config=config, first_episode=first, episodes=count,
                                      first_seed=first_seed, gpu=slots[slot], pid=proc.pid)), flush=True)
            time.sleep(20)
            for slot, (proc, *_rest) in list(running.items()):
                if proc.poll() is not None:
                    finish(slot)
            partial = {f'{t}/{c}/{b}': dict(zip(('episodes', 'successes'), (len(e), sum(s for _, s in e))))
                       for _, (t, c, b, *_x), j, _ in running.values() for e in [episodes(j / 'eval.log')]}
            (out / 'summary.json').write_text(json.dumps(dict(finished=results, running=partial, updated=time.time()),
                                                         indent=2))
    finally:
        server.terminate()
    (out / 'summary.json').write_text(json.dumps(dict(finished=results, running={}, updated=time.time()), indent=2))
    print(json.dumps(dict(event='done', jobs=len(results))), flush=True)


if __name__ == '__main__':
    main()
