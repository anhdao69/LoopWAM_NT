#!/usr/bin/env python3
"""RoboTwin-10 dispatcher: runs on the login node (evuser2), survives the Claude session.

Every minute: read the user's interactive pool jobs from squeue; keep one node agent (node_agent.py, a Slurm step
started with srun --overlap) alive in each RUNNING pool job; when the set of ready allocations or incomplete runs
changes, re-plan which run trains on which GPUs and write control/agent_<job>.json; queue a publish task (HF upload)
for every completed run and the index update once all seven are published. Status: control/STATUS.txt.

Plan (config gpus_per_run=4, the user's choice): every run trains on all four GPUs of one allocation. Runs are
never preempted while their allocation is usable; a free allocation takes the waiting run with the most remaining
work; a run on an allocation that is about to end moves to a free allocation if there is one, else finishes its
segment there. (Without gpus_per_run: water-fill 1/2/4 GPUs per run and bin-pack.)
"""
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

WORK = Path('/groups/yshang/an221229/checkpoints/LoopWAM_NT/robotwin10_work')
RUNS = Path('/groups/yshang/an221229/checkpoints/LoopWAM_NT/robotwin10')
CONTROL = WORK / 'control'
PY = '/groups/yshang/an221229/envs/FastWAM/bin/python'
TOTAL_UPDATES = 45445
GPUS_PER_JOB = 4


def now():
    return time.time()


def stamp(t=None):
    return datetime.fromtimestamp(t or now()).strftime('%m-%d %H:%M')


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json(path, payload):
    path = Path(path); tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps(payload, indent=2)); tmp.replace(path)


def log(msg):
    line = f'[{stamp()}] {msg}'
    print(line, flush=True)
    with open(CONTROL / 'dispatcher_events.log', 'a') as f:
        f.write(line + '\n')


def squeue(jobs):
    res = subprocess.run(['squeue', '-h', '-u', os.environ.get('USER', 'an221229'), '-o', '%i|%T|%N|%e|%S'],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    if res.returncode:
        raise RuntimeError('squeue failed')
    info = {}
    for line in res.stdout.splitlines():
        job, state, node, end, start = line.split('|')
        if job not in jobs:
            continue
        def ts(x):
            try:
                return datetime.fromisoformat(x).timestamp()
            except ValueError:
                return None
        info[job] = dict(state=state, node=node, end=ts(end), start=ts(start))
    return info


def pid_alive(pid, needle):
    try:
        return needle in Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='ignore')
    except OSError:
        return False


def water_fill(work, gpus):
    """GPU count per run in {1,2,4}: one GPU each by remaining work, then double the slowest-finishing run."""
    order = sorted(work, key=lambda v: -work[v])
    g = {}
    for v in order:
        if gpus >= 1:
            g[v] = 1; gpus -= 1
    while True:
        cands = [v for v in g if g[v] < 4 and gpus >= g[v]]
        if not cands:
            return g
        v = max(cands, key=lambda v: (work[v] / g[v], work[v]))
        gpus -= g[v]; g[v] *= 2


def pack(g, allocs, current, sticky=True):
    """Place sizes onto allocations (GPU ids). Runs whose size is unchanged keep their exact GPUs; the rest are
    placed largest first, preferring their current allocation, then best fit. Falls back to a fresh packing if
    kept runs fragment the allocations."""
    free = {j: set(range(GPUS_PER_JOB)) for j in allocs}
    placement = {}
    if sticky:
        for v in sorted(g):
            cur = current.get(v)
            if cur and cur['job'] in free and len(cur['gpus']) == g[v] and set(cur['gpus']) <= free[cur['job']]:
                placement[v] = (cur['job'], sorted(cur['gpus'])); free[cur['job']] -= set(cur['gpus'])
    for v in sorted((v for v in g if v not in placement), key=lambda v: (-g[v], v)):
        cur = current.get(v)
        fits = [j for j in free if len(free[j]) >= g[v]]
        if not fits:
            return pack(g, allocs, current, sticky=False) if sticky else placement
        prefer = [j for j in fits if cur and j == cur['job']]
        j = prefer[0] if prefer else min(fits, key=lambda j: (len(free[j]), j))   # best fit
        pairs = [[a, a + 1] for a in (0, 2) if {a, a + 1} <= free[j]]
        chosen = pairs[0] if g[v] == 2 and pairs else sorted(free[j])[:g[v]]
        placement[v] = (j, chosen); free[j] -= set(chosen)
    return placement


def whole_allocation_plan(ready, incomplete, work, current, previous, jobs):
    placement, used = {}, set()
    def keep(v, j):
        if v in incomplete and v not in placement and j in ready and j not in used:
            placement[v] = (j, list(range(GPUS_PER_JOB))); used.add(j)
    for v, c in current.items():
        if not c['stopping'] and len(c['gpus']) == GPUS_PER_JOB:
            keep(v, c['job'])
    for v, (j, _) in previous.items():
        keep(v, j)
    free = sorted((j for j in ready if j not in used), key=lambda j: -(jobs[j]['end'] or 0))
    waiting = sorted((v for v in incomplete if v not in placement), key=lambda v: -work[v])
    for j, v in zip(free, waiting):
        placement[v] = (j, list(range(GPUS_PER_JOB)))
    return placement


class Dispatcher:
    def __init__(self):
        self.signature = None
        self.plan = {}         # variant -> (job, gpus)

    def config(self):
        return read_json(CONTROL / 'config.json')

    def run_state(self, cfg):
        runs = {}
        for v in cfg['variants']:
            d = RUNS / v
            t = read_json(d / 'train' / 'timing.json', {})
            runs[v] = dict(complete=(d / 'train' / 'COMPLETE').exists(), published=(d / 'PUBLISHED').exists(),
                           update=int(t.get('update') or 0), status=t.get('status'), heartbeat=t.get('heartbeat_unix'),
                           last_update_seconds=t.get('last_update_seconds'))
        return runs

    def ensure_agents(self, jobs, cfg):
        agents = {}
        for job, info in jobs.items():
            status = read_json(CONTROL / f'agent_{job}.status.json', {})
            launch = read_json(CONTROL / f'agent_{job}.launch.json', {})
            fresh = status and now() - status.get('heartbeat', 0) < 180 and status.get('state') != 'exited'
            client = launch.get('pid') and pid_alive(launch['pid'], 'node_agent.py')
            if info['state'] == 'RUNNING' and not fresh and not client:
                if now() - launch.get('time', 0) < 120:
                    continue
                if info['end'] and info['end'] - now() < 2400:
                    continue    # too late to start anything on this allocation
                cmd = ['srun', f'--jobid={job}', '--overlap', '--gres=gpu:4', '--cpus-per-task=16', '--mem=0',
                       '--cpu-bind=none', '--job-name=rt10_agent', PY, '-u', str(WORK / 'node_agent.py'), '--job', job,
                       '--code', cfg['code']]
                out = open(CONTROL / f'agent_{job}.srun.log', 'a')
                out.write(f'# {stamp()} {" ".join(cmd)}\n'); out.flush()
                proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        start_new_session=True)
                write_json(CONTROL / f'agent_{job}.launch.json', dict(pid=proc.pid, time=now(), node=info['node']))
                log(f'launched agent in job {job} on {info["node"]} (srun pid {proc.pid})')
            agents[job] = dict(status=status, fresh=bool(fresh), ready=bool(fresh and status.get('ready')))
        return agents

    def step(self):
        cfg = self.config()
        pool = [str(j) for j in cfg['pool']]
        jobs = squeue(pool)
        agents = self.ensure_agents(jobs, cfg)
        runs = self.run_state(cfg)
        current = {}
        for job, a in agents.items():
            for v, r in (a['status'].get('runs') or {}).items():
                current[v] = dict(job=job, gpus=r['gpus'], microbatch=r['microbatch'], stopping=r.get('stop_requested'))
        min_left = cfg.get('min_remaining_minutes', 45) * 60
        ready = sorted(j for j, a in agents.items() if a['ready'] and jobs.get(j, {}).get('state') == 'RUNNING'
                       and (jobs[j]['end'] or 0) - now() > min_left and not (CONTROL / f'agent_{j}.EXIT').exists())
        failures = {v: n for a in agents.values() for v, n in (a['status'].get('failures') or {}).items() if n >= 3}
        incomplete = sorted(v for v, r in runs.items() if not r['complete'] and v not in failures)
        signature = (tuple(ready), tuple(incomplete), (CONTROL / 'REPLAN').exists())
        if signature != self.signature:
            (CONTROL / 'REPLAN').unlink(missing_ok=True)
            work = {v: (TOTAL_UPDATES - runs[v]['update']) * cfg['variants'][v]['gpu_sec_per_update'] for v in incomplete}
            if cfg.get('gpus_per_run') == GPUS_PER_JOB:
                placement = whole_allocation_plan(ready, incomplete, work, current, self.plan, jobs)
            else:
                g = water_fill(work, GPUS_PER_JOB * len(ready))
                placement = pack(g, ready, {v: c for v, c in current.items() if not c['stopping']})
            # Runs left without GPUs keep training where they are until their allocation's own stop.
            for v, c in current.items():
                if v in incomplete and v not in placement and c['job'] not in ready and not c['stopping']:
                    placement[v] = (c['job'], sorted(c['gpus']))
            self.plan = placement; self.signature = signature
            desc = ', '.join(f'{v}@{j}:{"".join(map(str, gp))}' for v, (j, gp) in sorted(placement.items()))
            log(f'plan for ready={list(ready)} incomplete={len(incomplete)}: {desc or "nothing"}')
        # Control files (assignments plus tasks).
        tasks_by_job = {j: {} for j in agents}
        unpublished = [v for v, r in runs.items() if r['complete'] and not r['published']]
        task_job = next((j for j in sorted(agents, key=lambda j: -(jobs.get(j, {}).get('end') or 0)) if agents[j]['ready']), None)
        running_tasks = {t: j for j, a in agents.items() if a['fresh'] for t in (a['status'].get('tasks') or {})}
        wanted = {}
        for v in unpublished:
            attempt = len(list((CONTROL / 'tasks').glob(f'publish_{v}_try*.done.json')))
            if attempt < 3:
                wanted[f'publish_{v}_try{attempt + 1}'] = ['python', 'scripts/publish_robotwin10.py', '--variant', v]
        if all(r['published'] for r in runs.values()) and not (RUNS / 'INDEX_PUBLISHED').exists():
            attempt = len(list((CONTROL / 'tasks').glob('index_try*.done.json')))
            if attempt < 3:
                wanted[f'index_try{attempt + 1}'] = ['python', 'scripts/publish_robotwin10.py', '--index']
        for name, argv in wanted.items():
            j = running_tasks.get(name, task_job)
            if j:
                tasks_by_job.setdefault(j, {})[name] = argv
        for job in agents:
            assign = {v: dict(gpus=gp, microbatch=cfg['variants'][v]['microbatch'][str(len(gp))])
                      for v, (j, gp) in self.plan.items() if j == job and not runs[v]['complete']}
            payload = dict(assign=assign, tasks=tasks_by_job.get(job, {}))
            if read_json(CONTROL / f'agent_{job}.json') != payload:
                write_json(CONTROL / f'agent_{job}.json', payload)
        self.report(jobs, agents, runs, cfg, failures)

    def report(self, jobs, agents, runs, cfg, failures):
        lines = [f'RoboTwin-10 dispatcher {stamp()}', '', 'Allocations:']
        for job in [str(j) for j in cfg['pool']]:
            j = jobs.get(job)
            if not j:
                lines.append(f'  {job}: gone'); continue
            a = agents.get(job, {}); st = a.get('status') or {}
            lines.append(f'  {job}: {j["state"]:8s} {j["node"]:8s} start {stamp(j["start"]) if j["start"] else "?"} '
                         f'end {stamp(j["end"]) if j["end"] else "?"} agent={st.get("state", "-")}{" ready" if a.get("ready") else ""}')
        lines += ['', 'Runs:']
        for v, r in runs.items():
            where = next((f'{j}:{"".join(map(str, s["gpus"]))}' for j, a in agents.items()
                          for vv, s in (a['status'].get('runs') or {}).items() if vv == v), '-')
            lines.append(f'  {v:20s} {r["update"]:6d}/{TOTAL_UPDATES} {100 * r["update"] / TOTAL_UPDATES:5.1f}% '
                         f'{"COMPLETE" if r["complete"] else (r["status"] or "-"):9s} on {where:12s} '
                         f'{"PUBLISHED" if r["published"] else ""}{" FAILED" if v in failures else ""} '
                         f's/upd {r["last_update_seconds"] or 0:.2f}')
        (CONTROL / 'STATUS.txt').write_text('\n'.join(lines) + '\n')
        write_json(CONTROL / 'dispatcher_status.json', dict(time=now(), plan={v: list(p) for v, p in self.plan.items()},
                                                             jobs=jobs, runs=runs, failures=failures))

    def main(self):
        log(f'dispatcher start pid {os.getpid()} on {os.uname().nodename}')
        write_json(CONTROL / 'dispatcher.pid.json', dict(pid=os.getpid(), host=os.uname().nodename, time=now()))
        while not (CONTROL / 'DISPATCHER_STOP').exists():
            try:
                self.step()
            except Exception as e:  # keep going; the next loop re-reads everything
                log(f'error: {type(e).__name__}: {e}')
            if (RUNS / 'INDEX_PUBLISHED').exists():
                log('all runs trained and published; dispatcher exits'); break
            time.sleep(60)


if __name__ == '__main__':
    Dispatcher().main()
