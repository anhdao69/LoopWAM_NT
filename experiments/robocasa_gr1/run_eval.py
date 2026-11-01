#!/usr/bin/env python3
"""RoboCasa-GR1 evaluation driver: LoopWAM policy servers on GPUs + parallel seeded sim clients (resumable).

python experiments/robocasa_gr1/run_eval.py --checkpoint CK --stats S --out OUT --steps 8,12,16,24 --episodes 10 --seed 0
Each (n_action_steps, task) cell writes OUT/n{steps}/{task}.json via sim_client.py; existing cells are skipped.
Servers serve any n_action_steps (sent per request). Thread caps follow the THAW-VLA GR1 runner
(SERVER_THREADS=8, SIM_THREADS=2, osmesa, NUMBA_DISABLE_JIT=1, MUJOCO_EGL_DEVICE_ID unset).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / 'src'))
from fastwam.datasets.robocasa_gr1 import GR1_ENV_NAMES  # noqa: E402

EVAL_ROOT = Path(os.getenv('GR1_EVAL_ROOT', '/mnt/data/vmo-ai-task/anhdh35/gr1_eval'))
FASTWAM_PY = os.getenv('FASTWAM_PY', '/mnt/data/vmo-ai-task/anhdh35/FastWAM/.venv/bin/python')
ROBOCASA_PY = str(EVAL_ROOT / 'envs/robocasa/bin/python')
STARVLA = str(EVAL_ROOT / 'third_party/starVLA')
OSMESA_LIB = os.getenv('OSMESA_LIB', '/mnt/data/vmo-ai-task/anhdh35/FastWAM/third_party/system_libs/usr/lib/x86_64-linux-gnu')


def port_free(port):
    with socket.socket() as s:
        try:
            s.bind(('0.0.0.0', port)); return True
        except OSError:
            return False


def port_open(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(('127.0.0.1', port)) == 0


def sim_env(threads):
    env = dict(os.environ, MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa', NUMBA_DISABLE_JIT='1', PYTHONPATH=STARVLA,
               OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
               LD_LIBRARY_PATH=f'{OSMESA_LIB}:{os.getenv("LD_LIBRARY_PATH", "")}', CUDA_VISIBLE_DEVICES='')
    env.pop('MUJOCO_EGL_DEVICE_ID', None)
    return env


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True); p.add_argument('--stats', required=True); p.add_argument('--out', required=True)
    p.add_argument('--steps', default='8,12,16,24'); p.add_argument('--episodes', type=int, default=10); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--tasks', default=None, help='comma list of env names (default: all 24)')
    p.add_argument('--gpus', default='0,1,2,3'); p.add_argument('--servers-per-gpu', type=int, default=1)
    p.add_argument('--base-port', type=int, default=6500); p.add_argument('--parallel', type=int, default=24)
    p.add_argument('--n-envs', type=int, default=1); p.add_argument('--max-episode-steps', type=int, default=720)
    p.add_argument('--server-threads', type=int, default=8); p.add_argument('--sim-threads', type=int, default=2)
    p.add_argument('--video', action='store_true'); p.add_argument('--expected-step', type=int, default=None)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); (out / 'logs').mkdir(exist_ok=True)
    tasks = a.tasks.split(',') if a.tasks else list(GR1_ENV_NAMES)
    steps = [int(s) for s in a.steps.split(',')]
    cells = [(s, t) for s in steps for t in tasks if not (out / f'n{s}' / f'{t.split("/")[-1]}.json').exists()]
    (out / 'eval_config.json').write_text(json.dumps(dict(vars(a), tasks=tasks, pending=len(cells)), indent=2))
    if not cells:
        print('nothing to do'); return summarize(out, steps, tasks, a)
    gpus = a.gpus.split(','); ports = [a.base_port + i for i in range(len(gpus) * a.servers_per_gpu)]
    busy = [q for q in ports if not port_free(q)]
    if busy:
        raise SystemExit(f'ports in use: {busy}')
    servers = []
    for i, port in enumerate(ports):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[i % len(gpus)], SERVER_THREADS=str(a.server_threads),
                   OMP_NUM_THREADS=str(a.server_threads), PYTHONPATH=f'{EVAL_ROOT}/pydeps_fastwam:{STARVLA}:{CODE}/src:{CODE}')
        cmd = [FASTWAM_PY, str(CODE / 'experiments/robocasa_gr1/loopwam_policy_server.py'), '--checkpoint', a.checkpoint,
               '--stats', a.stats, '--port', str(port)] + (['--expected-step', str(a.expected_step)] if a.expected_step else [])
        servers.append(subprocess.Popen(cmd, env=env, stdout=open(out / 'logs' / f'server_{port}.log', 'a'), stderr=subprocess.STDOUT))
    try:
        deadline = time.time() + 900
        while not all(port_open(q) for q in ports):
            if any(s.poll() is not None for s in servers):
                raise SystemExit('a policy server exited during startup; see logs/server_*.log')
            if time.time() > deadline:
                raise SystemExit('policy servers did not open their ports in 15 min')
            time.sleep(5)
        print(json.dumps(dict(servers=ports, cells=len(cells))), flush=True)
        running, k, started = [], 0, time.time()
        while cells or running:
            while cells and len(running) < a.parallel:
                s, t = cells.pop(0); name = t.split('/')[-1]; port = ports[k % len(ports)]; k += 1
                res = out / f'n{s}' / f'{name}.json'; res.parent.mkdir(exist_ok=True)
                cmd = [ROBOCASA_PY, str(CODE / 'experiments/robocasa_gr1/sim_client.py'), '--env-name', t, '--port', str(port),
                       '--n-episodes', str(a.episodes), '--n-envs', str(a.n_envs), '--n-action-steps', str(s),
                       '--max-episode-steps', str(a.max_episode_steps), '--seed', str(a.seed), '--out', str(res)]
                if a.video:
                    cmd += ['--video-dir', str(out / 'videos' / f'n{s}' / name)]
                log = open(out / 'logs' / f'n{s}_{name}.log', 'a')
                running.append((subprocess.Popen(cmd, env=sim_env(a.sim_threads), stdout=log, stderr=subprocess.STDOUT, cwd=STARVLA), s, t))
            time.sleep(10)
            for item in list(running):
                code = item[0].poll()
                if code is not None:
                    running.remove(item)
                    print(json.dumps(dict(steps=item[1], task=item[2].split('/')[-1], exit=code,
                                          done=k - len(running), elapsed_min=round((time.time() - started) / 60, 1))), flush=True)
            if any(s.poll() is not None for s in servers):
                raise SystemExit('a policy server died; see logs/server_*.log')
    finally:
        for s in servers:
            s.terminate()
        for s in servers:
            try:
                s.wait(60)
            except subprocess.TimeoutExpired:
                s.kill()
    summarize(out, steps, tasks, a)


def summarize(out, steps, tasks, a):
    table = {}
    for s in steps:
        rows = {}
        for t in tasks:
            f = out / f'n{s}' / f'{t.split("/")[-1]}.json'
            if f.exists():
                r = json.loads(f.read_text()); rows[t.split('/')[-1]] = dict(success_rate=r['success_rate'], successes=r['successes_first_n'],
                                                                         episodes=r['n_episodes'])
        done = len(rows) == len(tasks)
        table[s] = dict(complete=done, tasks=len(rows), mean_success=(sum(r['success_rate'] for r in rows.values()) / len(rows)) if rows else None,
                        per_task=rows)
    (out / 'summary.json').write_text(json.dumps(dict(seed=a.seed, episodes=a.episodes, checkpoint=a.checkpoint, by_n_action_steps=table), indent=2))
    print(json.dumps({s: (v['mean_success'], v['tasks']) for s, v in table.items()}), flush=True)


if __name__ == '__main__':
    main()
