#!/usr/bin/env python3
"""RoboTwin-10 node agent: one per interactive allocation (a Slurm step started by dispatcher.py).

Stages the latent cache to node-local disk, then keeps the trainers listed in control/agent_<job>.json running
on their assigned GPUs: starts or resumes each run, asks runs that lost their assignment to stop (stop file ->
trainer saves latest.pt and exits), restarts crashed runs (bounded), and runs publish tasks. Status is written
to control/agent_<job>.status.json every loop. No torch import (runs before staging, cheap restarts).
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path

WORK = Path('/groups/yshang/an221229/checkpoints/LoopWAM_NT/robotwin10_work')
RUNS = Path('/groups/yshang/an221229/checkpoints/LoopWAM_NT/robotwin10')
CONTROL = WORK / 'control'
LOCAL = Path('/tmp/loopwam_rt10/shared')
TOTAL_UPDATES = 45445


def now():
    return time.time()


def stamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json(path, payload):
    path = Path(path); tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps(payload, indent=2)); tmp.replace(path)


def job_end_unix(job):
    out = subprocess.run(['squeue', '-h', '-j', str(job), '-o', '%e'], stdout=subprocess.PIPE, text=True).stdout.strip()
    return datetime.fromisoformat(out).timestamp() if out and out[0].isdigit() else None


class Agent:
    def __init__(self, job, code):
        self.job, self.code, self.host = str(job), code, socket.gethostname()
        self.end = job_end_unix(job)
        self.procs = {}       # variant -> dict(proc, gpus, started, log, stop_requested)
        self.tasks = {}       # task name -> dict(proc, started, log)
        self.failures = {}    # variant -> consecutive crash count
        self.failed_at = {}   # variant -> time of the last crash
        self.crashes = {}     # variant -> total crashes on this agent
        self.events = []
        self.ready = False
        self.log_path = CONTROL / f'agent_{self.job}.log'

    def log(self, msg):
        line = f'[{stamp()}] {msg}'
        print(line, flush=True)
        self.events = (self.events + [line])[-50:]

    # ------------------------------------------------------------------ staging
    def stage(self):
        # Retries until the master cache exists and the local copy verifies (e.g. while the first build is running).
        while True:
            proc = subprocess.Popen(['bash', str(WORK / 'stage_cache.sh')], stdout=open(CONTROL / f'stage_{self.job}.log', 'a'),
                                    stderr=subprocess.STDOUT)
            while proc.poll() is None:
                self.write_status('staging'); time.sleep(15)
            if proc.returncode == 0:
                break
            self.log(f'cache staging not possible yet (rc={proc.returncode}); retry in 60 s')
            if self.end and now() > self.end - 2400:
                raise SystemExit('allocation ending before the cache could be staged')
            for _ in range(4):
                self.write_status('waiting_cache'); time.sleep(15)
        self.ready = True
        self.log(f'cache ready on {self.host}')

    # ------------------------------------------------------------------ runs
    def lock_path(self, variant):
        return RUNS / variant / 'lock.json'

    def lock_holder(self, variant):
        lock = read_json(self.lock_path(variant))
        if not lock:
            return None
        if lock.get('job') == self.job and lock.get('host') == self.host:
            return None if variant not in self.procs else lock
        if now() - lock.get('heartbeat', 0) > 300:
            return None  # stale: the holder stopped refreshing it
        return lock

    def start_run(self, variant, gpus, microbatch):
        run = RUNS / variant
        (run / 'train').mkdir(parents=True, exist_ok=True); (run / 'logs').mkdir(exist_ok=True)
        holder = self.lock_holder(variant)
        if holder:
            return  # still running elsewhere (e.g. stopping on the previous allocation); retry next loop
        if (run / 'train' / 'COMPLETE').exists():
            return
        stop = run / f'STOP_{self.job}'
        stop.unlink(missing_ok=True)
        resume = (run / 'train' / 'latest.pt').exists()
        if not resume and any((run / 'train' / n).exists() for n in ('manifest.json', 'metrics.jsonl')):
            moved = run / f'train.quarantine.{int(now())}'
            (run / 'train').rename(moved); (run / 'train').mkdir()
            self.log(f'{variant}: records without latest.pt quarantined to {moved.name}')
        seg = len(list((run / 'logs').glob('segment_*.log'))) + 1
        log = run / 'logs' / f'segment_{seg:03d}_job{self.job}_gpu{"".join(map(str, gpus))}.log'
        cmd = ['torchrun', '--standalone', f'--nproc_per_node={len(gpus)}', 'scripts/train_loopwam_robotwin10.py',
               '--variant', variant, '--output-dir', str(run / 'train'), '--shared-dir', str(WORK / 'shared'),
               '--latent-dir', str(LOCAL / 'latents'), '--init-artifact', 'checkpoints/LoopWAM/wan21_compact_donors.pt',
               '--microbatch', str(microbatch), '--compile', '--stop-file', str(stop), '--log-every', '50',
               '--checkpoint-every', '1000', '--checkpoint-minutes', '30']
        if self.end:
            cmd += ['--stop-at-unix', str(self.end), '--stop-margin', '900']
        if resume:
            cmd += ['--resume']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)), OMP_NUM_THREADS='2')
        shell = f'source {WORK}/env.sh && export RT10_CODE={self.code} && cd {self.code} && exec ' + ' '.join(cmd)
        with open(log, 'w') as f:
            f.write(f'# {stamp()} job {self.job} host {self.host} gpus {gpus} microbatch {microbatch} resume {resume}\n# {" ".join(cmd)}\n')
            f.flush()
            proc = subprocess.Popen(['bash', '-c', shell], stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        self.procs[variant] = dict(proc=proc, gpus=list(gpus), microbatch=microbatch, started=now(), log=str(log), stop_requested=False)
        write_json(self.lock_path(variant), dict(job=self.job, host=self.host, gpus=list(gpus), pid=proc.pid, started=now(),
                                                heartbeat=now(), log=str(log)))
        self.log(f'{variant}: started on gpus {gpus} mb {microbatch} resume={resume} pid {proc.pid}')

    def request_stop(self, variant, why):
        info = self.procs[variant]
        if not info['stop_requested']:
            (RUNS / variant / f'STOP_{self.job}').touch()
            info['stop_requested'] = now()
            self.log(f'{variant}: stop requested ({why})')

    def reap(self):
        for variant, info in list(self.procs.items()):
            rc = info['proc'].poll()
            if rc is None:
                lock = read_json(self.lock_path(variant)) or {}
                if lock.get('job') == self.job:
                    lock['heartbeat'] = now(); write_json(self.lock_path(variant), lock)
                # A stop that did not complete within 20 minutes (hung collective): kill the process group.
                if info['stop_requested'] and now() - info['stop_requested'] > 1200 and not info.get('killed'):
                    os.killpg(info['proc'].pid, signal.SIGKILL); info['killed'] = True
                    self.log(f'{variant}: killed after stop timeout')
                # No completed update for 30 minutes (hang): kill; the reaper counts it as a crash and it restarts.
                # timing.json is rewritten every update; its final write (complete/stopped) has no heartbeat field.
                tpath = RUNS / variant / 'train' / 'timing.json'
                hb = tpath.stat().st_mtime if tpath.exists() else 0
                if now() - max(hb, info['started']) > 1800 and not info.get('killed'):
                    os.killpg(info['proc'].pid, signal.SIGKILL); info['killed'] = True
                    self.log(f'{variant}: killed after 30 min without an update (stall)')
                continue
            tpath = RUNS / variant / 'train' / 'timing.json'
            timing = read_json(tpath, {})
            # Only a status written by this segment counts (a stale 'stopped' from an earlier segment does not).
            fresh = tpath.exists() and tpath.stat().st_mtime > info['started']
            status = timing.get('status') if fresh else None
            del self.procs[variant]
            (RUNS / variant / f'STOP_{self.job}').unlink(missing_ok=True)
            lock = read_json(self.lock_path(variant)) or {}
            if lock.get('job') == self.job:
                self.lock_path(variant).unlink(missing_ok=True)
            if status in ('complete', 'stopped') and (rc == 0 or info.get('killed')):
                self.failures[variant] = 0
                self.log(f'{variant}: exited {status} at update {timing.get("update")}')
            else:
                self.failures[variant] = self.failures.get(variant, 0) + 1
                self.crashes[variant] = self.crashes.get(variant, 0) + 1; self.failed_at[variant] = now()
                self.log(f'{variant}: CRASHED rc={rc} status={status} update={timing.get("update")} '
                         f'(failure {self.failures[variant]}) log {info["log"]}')
        # After three consecutive crashes a run rests 30 minutes, then may try again (at most 10 crashes per agent).
        for variant, n in self.failures.items():
            if n >= 3 and now() - self.failed_at.get(variant, 0) > 1800 and self.crashes.get(variant, 0) < 10:
                self.failures[variant] = 0
                self.log(f'{variant}: crash cooldown over; may restart')

    def run_task(self, name, argv):
        if name in self.tasks:
            return
        log = CONTROL / f'task_{name}_{self.job}.log'
        shell = f'source {WORK}/env.sh && export RT10_CODE={self.code} && cd {self.code} && exec ' + ' '.join(argv)
        f = open(log, 'a'); f.write(f'# {stamp()} {argv}\n'); f.flush()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        self.tasks[name] = dict(proc=subprocess.Popen(['bash', '-c', shell], stdout=f, stderr=subprocess.STDOUT, env=env,
                                                      start_new_session=True), started=now(), log=str(log))
        self.log(f'task {name} started')

    def reap_tasks(self):
        for name, info in list(self.tasks.items()):
            rc = info['proc'].poll()
            if rc is not None:
                del self.tasks[name]
                self.log(f'task {name} exited rc={rc}')
                write_json(CONTROL / 'tasks' / f'{name}.done.json', dict(rc=rc, job=self.job, finished=now(), log=info['log']))

    # ------------------------------------------------------------------ loop
    def write_status(self, state):
        runs = {}
        for v, info in self.procs.items():
            t = read_json(RUNS / v / 'train' / 'timing.json', {})
            runs[v] = dict(gpus=info['gpus'], microbatch=info['microbatch'], pid=info['proc'].pid, started=info['started'],
                           stop_requested=bool(info['stop_requested']), update=t.get('update'),
                           last_update_seconds=t.get('last_update_seconds'), log=info['log'])
        write_json(CONTROL / f'agent_{self.job}.status.json', dict(
            job=self.job, host=self.host, state=state, ready=self.ready, heartbeat=now(), end_unix=self.end, runs=runs,
            tasks={k: dict(started=v['started'], log=v['log']) for k, v in self.tasks.items()}, failures=self.failures,
            events=self.events[-20:]))

    def step(self):
        ctl = read_json(CONTROL / f'agent_{self.job}.json', {})
        assign = ctl.get('assign', {})            # variant -> dict(gpus=[...], microbatch=n)
        self.reap(); self.reap_tasks()
        for variant, info in list(self.procs.items()):
            want = assign.get(variant)
            if want is None or sorted(want['gpus']) != sorted(info['gpus']) or want['microbatch'] != info['microbatch']:
                self.request_stop(variant, 'reassigned')
        busy = {g for info in self.procs.values() for g in info['gpus']}
        closing = self.end and now() > self.end - 1800
        for variant, want in assign.items():
            if closing or variant in self.procs or self.failures.get(variant, 0) >= 3:
                continue
            if busy & set(want['gpus']):
                continue  # GPUs still held by a stopping run
            self.start_run(variant, want['gpus'], want['microbatch'])
            if variant in self.procs:
                busy |= set(want['gpus'])
        for name, argv in ctl.get('tasks', {}).items():
            if name not in self.tasks and not (CONTROL / 'tasks' / f'{name}.done.json').exists():
                self.run_task(name, argv)
        self.write_status('running')

    def main(self):
        (CONTROL / 'tasks').mkdir(parents=True, exist_ok=True)
        self.log(f'agent start job {self.job} host {self.host} end {self.end} code {self.code}')
        self.stage()
        while True:
            self.step()
            if self.end and now() > self.end - 240:
                self.log('allocation ending; agent exits')
                break
            if (CONTROL / f'agent_{self.job}.EXIT').exists() and not self.procs:
                self.log('exit requested'); break
            time.sleep(15)
        self.write_status('exited')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--job', required=True)
    p.add_argument('--code', required=True)
    a = p.parse_args()
    Agent(a.job, a.code).main()
