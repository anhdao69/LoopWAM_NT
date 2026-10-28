#!/usr/bin/env python3
"""Elastic campaign dispatcher; runs on login node evuser2 (nohup setsid; log logs/dispatcher.log).

User request 2026-10-28: actively watch all of the user's running and queued interactive jobs and start the remaining
elastic runs on GPUs that are empty (especially job 1080498). Job 990547 is excluded by the user.

Every poll: a GPU is a candidate only when it has no compute process and <1 GiB in use for IDLE_POLLS consecutive
polls, is not used by an elastic lane, and its job has no other managed `srun --jobid` step (e.g. the stepwise
variant on 1080498 keeps that job off-limits until its step ends). Pairs of candidate GPUs get a lane
(lane.sh, world 2) while unowned incomplete runs remain. Near each job's end, its lanes are stopped gracefully
(recovery saved) and its /dev/shm cache removed; once every run is complete all caches are removed and this exits.
Python 3.6 compatible (system python on the login node).
"""
import json, os, re, subprocess, sys, time

WORK = "/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_work"
RUNS = "/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic"
QUEUE = ["E0", "E2", "E1", "SP11", "F0-PB"]
EXCLUDE = {"990547"}
POLL, IDLE_POLLS = 300, 3
MIN_LEFT_START, STOP_LEFT = 3600, 1500
MAX_LAUNCH_RECORDS = 30
USER = os.environ["USER"]
DRY = "--dry-run" in sys.argv

def log(msg):
    print("[%s] %s" % (time.strftime("%Y-%m-%dT%H:%M:%S"), msg), flush=True)

def sh(args, timeout=150):
    try:
        r = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
                           timeout=timeout, stdin=subprocess.DEVNULL)
        return r.returncode, r.stdout
    except subprocess.TimeoutExpired:
        return 124, ""

def running_jobs():
    code, out = sh(["squeue", "-h", "-u", USER, "-t", "RUNNING", "-o", "%i|%e|%N|%j|%b"])
    jobs = {}
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 5: continue
        jid, end, node, name, gres = parts
        if jid in EXCLUDE or name != "bash" or "gpu" not in gres: continue
        try: left = time.mktime(time.strptime(end, "%Y-%m-%dT%H:%M:%S")) - time.time()
        except ValueError: continue
        jobs[jid] = dict(node=node, left=left)
    return jobs

def job_state(jid):
    return sh(["squeue", "-h", "-j", jid, "-o", "%T"], 60)[1].strip()

def mtime_age(path):
    try: return time.time() - os.stat(path).st_mtime
    except OSError: return None

def status(run):
    try:
        s = json.load(open(os.path.join(RUNS, run, "status.json")))
        return s.get("status"), s.get("update")
    except (OSError, ValueError):
        return None, None

def complete(run):
    st, up = status(run)
    return st == "complete" and up == 21700

def owned(run, states):
    """True when a lane anywhere currently owns the run (live claim or training output in the last 15 min)."""
    claim = os.path.join(RUNS, run, "claim.d")
    if os.path.isdir(claim):
        try: job = open(os.path.join(claim, "job")).read().strip()
        except OSError: job = ""
        beat = mtime_age(os.path.join(claim, "heartbeat"))
        if beat is None:
            age = mtime_age(claim)
            if age is not None and age < 120: return True
        elif job and states.get(job) == "RUNNING" and beat < 600:
            return True
    for name in ("train.log", "metrics.jsonl"):
        age = mtime_age(os.path.join(RUNS, run, name))
        if age is not None and age < 900: return True
    return False

def launch_records(run):
    try: return sum(1 for _ in open(os.path.join(RUNS, run, "launches.jsonl")))
    except OSError: return 0

def login_steps():
    """srun clients on this login node: {job: {"lanes": {lane: gpus}, "foreign": [cmd...]}}."""
    code, out = sh(["ps", "-u", USER, "-o", "args="], 30)
    steps = {}
    for args in out.splitlines():
        if not args.startswith("srun ") or "--jobid=" not in args: continue
        jid = re.search(r"--jobid=(\d+)", args).group(1)
        entry = steps.setdefault(jid, dict(lanes={}, foreign=[]))
        m = re.search(r"lane\.sh (\S+) (\S+)", args)
        if m: entry["lanes"][m.group(1)] = m.group(2)
        elif "nvidia-smi" not in args and "stop_lanes.sh" not in args and "test -d /dev/shm" not in args:
            entry["foreign"].append(args)
    return steps

def srun(jid, command, cpus=1, timeout=150):
    return sh(["srun", "--jobid=%s" % jid, "--overlap", "--nodes=1", "--ntasks=1", "--cpus-per-task=%d" % cpus,
               "--gres=gpu:4", "--mem=0"] + command, timeout)

def probe(jid):
    """{gpu index: idle} from inside the job, or None if the probe failed."""
    code, out = srun(jid, ["bash", "-c", "nvidia-smi --query-gpu=index,pci.bus_id,memory.used,utilization.gpu "
                                         "--format=csv,noheader,nounits; echo APPS; nvidia-smi "
                                         "--query-compute-apps=gpu_bus_id --format=csv,noheader"])
    if code != 0 or "APPS" not in out: return None
    gpus, apps = out.split("APPS")
    busy = {line.strip().lower() for line in apps.splitlines() if line.strip()}
    result = {}
    for line in gpus.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4 or not parts[0].isdigit(): continue
        index, bus, memory, util = int(parts[0]), parts[1].lower(), float(parts[2]), float(parts[3])
        result[index] = memory < 1024 and util == 0 and bus not in busy
    return result if len(result) == 4 else None

def pairs(free):
    free = sorted(free); chosen = []
    for a, b in ((0, 1), (2, 3)):
        if a in free and b in free: chosen.append((a, b)); free.remove(a); free.remove(b)
    while len(free) >= 2: chosen.append((free.pop(0), free.pop(0)))
    return chosen

def main():
    log("dispatcher pid %d on %s; excluded jobs %s; dry_run=%s" % (os.getpid(), os.uname()[1], sorted(EXCLUDE), DRY))
    streak, children, stopped = {}, [], set()
    while True:
        children = [p for p in children if p.poll() is None]
        jobs = running_jobs()
        states = {j: "RUNNING" for j in jobs}
        for run in QUEUE:  # owner jobs outside the running set (e.g. ended) need their real state
            try: owner = open(os.path.join(RUNS, run, "claim.d", "job")).read().strip()
            except OSError: owner = ""
            if owner and owner not in states: states[owner] = job_state(owner)
        if all(complete(r) for r in QUEUE):
            log("all five runs complete; removing node-local caches")
            for jid in jobs:
                if srun(jid, ["test", "-d", "/dev/shm/elastic_latents_%s" % jid])[0] == 0 and not DRY:
                    log("cleanup job %s: %s" % (jid, srun(jid, ["bash", WORK + "/stop_lanes.sh", jid], timeout=1200)[1].strip()))
            log("dispatcher done"); return
        steps = login_steps()
        failed = [r for r in QUEUE if not complete(r) and launch_records(r) >= MAX_LAUNCH_RECORDS]
        pending = [r for r in QUEUE if not complete(r) and not owned(r, states) and r not in failed]
        lanes = sum(len(s["lanes"]) for s in steps.values())
        active = sum(1 for r in QUEUE if not complete(r) and owned(r, states))
        idle_lanes = max(0, lanes - active)
        need = max(0, len(pending) - idle_lanes)
        summary = ", ".join("%s:%s" % (r, "done" if complete(r) else ("owned" if owned(r, states) else
                            ("FAILED" if r in failed else "pending"))) for r in QUEUE)
        log("runs [%s]; lanes %d (idle %d); need %d; jobs %s" % (summary, lanes, idle_lanes, need,
            {j: "%s %.1fh" % (v["node"], v["left"] / 3600) for j, v in jobs.items()}))
        for jid, info in sorted(jobs.items()):
            entry = steps.get(jid, dict(lanes={}, foreign=[]))
            if info["left"] < STOP_LEFT:
                if jid not in stopped and not DRY:
                    log("job %s ends in %.0f min: graceful stop of its lanes and cache cleanup" % (jid, info["left"] / 60))
                    children.append(subprocess.Popen(["srun", "--jobid=%s" % jid, "--overlap", "--nodes=1", "--ntasks=1",
                        "--cpus-per-task=1", "--gres=gpu:4", "--mem=0", "bash", WORK + "/stop_lanes.sh", jid],
                        stdout=open(WORK + "/logs/stop_%s.log" % jid, "a"), stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, start_new_session=True))
                    stopped.add(jid)
                continue
            if entry["foreign"]:
                streak.pop(jid, None)
                log("job %s has another managed step; not touching it (%s)" % (jid, entry["foreign"][0][:120]))
                continue
            used = set()
            for gpus in entry["lanes"].values(): used.update(int(g) for g in gpus.split(","))
            if len(used) == 4 or need == 0:
                continue
            idle = probe(jid)
            if idle is None:
                log("job %s: GPU probe failed" % jid); continue
            counts = streak.setdefault(jid, {})
            for g, is_idle in idle.items():
                counts[g] = counts.get(g, 0) + 1 if (is_idle and g not in used) else 0
            free = [g for g, c in counts.items() if c >= IDLE_POLLS]
            log("job %s (%s): idle streaks %s; elastic GPUs %s" % (jid, info["node"], counts, sorted(used)))
            if info["left"] < MIN_LEFT_START: continue
            for a, b in pairs(free):
                if need == 0: break
                lane = "j%sg%d%d" % (jid, a, b)
                command = ["srun", "--jobid=%s" % jid, "--overlap", "--nodes=1", "--ntasks=1", "--cpus-per-task=8",
                           "--gres=gpu:4", "--mem=0", "bash", WORK + "/lane.sh", lane, "%d,%d" % (a, b)]
                log("launching lane %s on job %s (%s) GPUs %d,%d" % (lane, jid, info["node"], a, b))
                if not DRY:
                    children.append(subprocess.Popen(command, stdout=open(WORK + "/logs/lane_%s.log" % lane, "a"),
                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True))
                counts[a] = counts[b] = 0
                need -= 1
        if DRY: return
        time.sleep(POLL)

if __name__ == "__main__":
    main()
