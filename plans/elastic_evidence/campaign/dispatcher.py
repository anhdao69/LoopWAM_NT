#!/usr/bin/env python3
"""Elastic campaign dispatcher (v3: training + evaluation); runs on login node evuser2 (nohup setsid; log logs/dispatcher.log).

User requests 2026-10-28: actively watch all of the user's running and queued interactive jobs and start the remaining
elastic runs on GPUs that are empty (especially job 1080498; job 990547 is excluded by the user); then also evaluate the
decisive conditions (plan section 12.2: E0/E1/E2 x [1,1], [1]x10, [4]x10) with the matched protocol, finishing
training and inference as fast as possible.

Every poll: a GPU is a candidate only when it has no compute process and <1 GiB in use for IDLE_POLLS consecutive
polls (one poll for GPUs that our own lanes just released), is not used by an elastic lane, and its job has no other
managed `srun --jobid` step. Pairs of candidate GPUs go first to training lanes (lane.sh, world 2) while unowned
incomplete runs remain, then to evaluation lanes (eval_lane.sh) while unclaimed evaluation groups of finished runs
remain; evaluation is CPU-bound, so its workers per GPU follow the job's CPUs not used by training. If a run whose
evaluation is requested (E0/E2/E1) waits for GPUs for PREEMPT_POLLS polls, one evaluation lane is stopped to make
room (its in-flight groups are re-run elsewhere). v3 (2026-10-29): while unclaimed evaluation groups remain and
no evaluation lane runs anywhere, a pair goes to training only for a run whose evaluation is requested (F0-PB, which is
not evaluated, then waits for such a pair instead of displacing the last evaluation lane). Near each job's end its
lanes are stopped gracefully and its /dev/shm cache removed. Once training is complete the caches are removed; once evaluation is complete this exits.
Python 3.6 compatible (system python on the login node).
"""
import glob, json, os, re, subprocess, sys, time

WORK = "/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_work"
RUNS = "/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic"
EVAL_OUT = "/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_eval"
QUEUE = ["E0", "E2", "E1", "SP11", "F0-PB"]
EVAL_RUNS = ["E0", "E2", "E1"]
EVAL_SCHEDULES = ["flat_k1_n2", "flat_k1_n10", "flat_k4_n10"]
EVAL_GROUPS = 4 * 10 * 3          # suites x tasks x seeds per condition (50 episodes each)
EXCLUDE = {"990547"}
POLL, IDLE_POLLS, PREEMPT_POLLS = 120, 6, 10
MIN_LEFT_START, EVAL_MIN_LEFT, STOP_LEFT = 3600, 3600, 1500
TRAIN_CPUS, MAX_WPG = 3, 8
MAX_LAUNCH_RECORDS = 30
USER = os.environ["USER"]
DRY = "--dry-run" in sys.argv
JOB_GPUS = {}   # job id -> GPUs per node (from squeue), so steps request exactly the job's GPUs
KNOWN = ("nvidia-smi", "stop_lanes.sh", "stop_eval_lane.sh", "clean_cache.sh", "test -d /dev/shm")

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
    code, out = sh(["squeue", "-h", "-u", USER, "-t", "RUNNING", "-o", "%i|%e|%N|%j|%b|%C"])
    jobs = {}
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 6: continue
        jid, end, node, name, gres, cpus = parts
        if jid in EXCLUDE or name != "bash" or "gpu" not in gres: continue
        try: left = time.mktime(time.strptime(end, "%Y-%m-%dT%H:%M:%S")) - time.time()
        except ValueError: continue
        count = gres.rsplit(":", 1)[-1]
        jobs[jid] = dict(node=node, left=left, cpus=int(cpus) if cpus.isdigit() else 16,
                         gpus=int(count) if count.isdigit() else 4)
        JOB_GPUS[jid] = jobs[jid]["gpus"]
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

def eval_ready(run):
    return complete(run) and os.path.exists(os.path.join(RUNS, run, "epoch_10.pt"))

def eval_dirs(run, schedule):
    return glob.glob(os.path.join(EVAL_OUT, "final_rollout", "elastic__%s__%s__*" % (run, schedule)))

def eval_done(run, schedule):
    return any(os.path.exists(os.path.join(d, "summary.json")) for d in eval_dirs(run, schedule))

def live_token(token):
    try: return time.time() - json.load(open(os.path.join(EVAL_OUT, "runners", token + ".json")))["time"] < 600
    except (OSError, ValueError, KeyError): return False

def eval_unclaimed():
    """Groups of ready, unfinished conditions that are neither scored nor claimed by a live runner."""
    total = 0
    for run in EVAL_RUNS:
        if not eval_ready(run): continue
        for schedule in EVAL_SCHEDULES:
            if eval_done(run, schedule): continue
            dirs = eval_dirs(run, schedule)
            if not dirs: total += EVAL_GROUPS; continue
            d = dirs[0]
            done = len(glob.glob(os.path.join(d, "groups", "*.json")))
            claimed = 0
            for c in glob.glob(os.path.join(d, "claims", "*.claim")):
                try: token = open(c).read().strip()
                except OSError: continue
                if token and live_token(token): claimed += 1
            total += max(0, EVAL_GROUPS - done - claimed)
    return total

def login_steps():
    """srun clients on this login node: {job: {"lanes": {lane: gpus}, "evals": {lane: (gpus, wpg)}, "foreign": [cmd]}}."""
    code, out = sh(["ps", "-u", USER, "-o", "args="], 30)
    steps = {}
    for args in out.splitlines():
        if not args.startswith("srun ") or "--jobid=" not in args: continue
        jid = re.search(r"--jobid=(\d+)", args).group(1)
        entry = steps.setdefault(jid, dict(lanes={}, evals={}, foreign=[]))
        e = re.search(r"eval_lane\.sh (\S+) (\S+) (\d+)", args)
        m = re.search(r"/lane\.sh (\S+) (\S+)", args)
        if e: entry["evals"][e.group(1)] = (e.group(2), int(e.group(3)))
        elif m: entry["lanes"][m.group(1)] = m.group(2)
        elif not any(k in args for k in KNOWN):
            entry["foreign"].append(args)
    return steps

def srun_cmd(jid, cpus, command):
    return ["srun", "--jobid=%s" % jid, "--overlap", "--nodes=1", "--ntasks=1", "--cpus-per-task=%d" % cpus,
            "--gres=gpu:%d" % JOB_GPUS.get(jid, 4), "--mem=0"] + command

def srun(jid, command, cpus=1, timeout=150):
    return sh(srun_cmd(jid, cpus, command), timeout)

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
    return result if len(result) == JOB_GPUS.get(jid, 4) else None

def pairs(free):
    free = sorted(free); chosen = []
    for a, b in ((0, 1), (2, 3)):
        if a in free and b in free: chosen.append((a, b)); free.remove(a); free.remove(b)
    while len(free) >= 2: chosen.append((free.pop(0), free.pop(0)))
    return chosen

def spawn(command, logname):
    log("  $ " + " ".join(command[-4:]))
    if DRY: return None
    return subprocess.Popen(command, stdout=open(os.path.join(WORK, "logs", logname), "a"), stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)

def main():
    log("dispatcher v3.1 pid %d on %s; excluded jobs %s; dry_run=%s" % (os.getpid(), os.uname()[1], sorted(EXCLUDE), DRY))
    streak, children, stopped, prev_used = {}, [], set(), {}
    starving, caches_cleaned, eval_launches = 0, False, []
    while True:
        children = [p for p in children if p.poll() is None]
        jobs = running_jobs()
        states = {j: "RUNNING" for j in jobs}
        for run in QUEUE:  # owner jobs outside the running set (e.g. ended) need their real state
            try: owner = open(os.path.join(RUNS, run, "claim.d", "job")).read().strip()
            except OSError: owner = ""
            if owner and owner not in states: states[owner] = job_state(owner)
        training_done = all(complete(r) for r in QUEUE)
        eval_all_done = all(eval_done(r, s) for r in EVAL_RUNS for s in EVAL_SCHEDULES)
        steps = login_steps()
        if training_done and not caches_cleaned and not DRY:
            for jid in jobs:
                if not steps.get(jid, {}).get("lanes") and srun(jid, ["test", "-d", "/dev/shm/elastic_latents_%s" % jid])[0] == 0:
                    log("training complete; cache cleanup on job %s: %s" % (jid, srun(jid, ["bash", WORK + "/clean_cache.sh", jid],
                        timeout=300)[1].strip()))
            caches_cleaned = True
        if training_done and eval_all_done:
            log("all five runs trained and all %d evaluation conditions complete; dispatcher done" %
                (len(EVAL_RUNS) * len(EVAL_SCHEDULES)))
            return
        failed = [r for r in QUEUE if not complete(r) and launch_records(r) >= MAX_LAUNCH_RECORDS]
        pending = [r for r in QUEUE if not complete(r) and not owned(r, states) and r not in failed]
        lanes = sum(len(s["lanes"]) for s in steps.values())
        active = sum(1 for r in QUEUE if not complete(r) and owned(r, states))
        idle_lanes = max(0, lanes - active)
        need = max(0, len(pending) - idle_lanes)
        critical = [r for r in pending if r in EVAL_RUNS]
        need_critical = max(0, len(critical) - idle_lanes)
        unclaimed = eval_unclaimed()
        evals = sum(len(s["evals"]) for s in steps.values())
        evals_live = evals
        summary = ", ".join("%s:%s" % (r, "done" if complete(r) else ("owned" if owned(r, states) else
                            ("FAILED" if r in failed else "pending"))) for r in QUEUE)
        evsum = ", ".join("%s/%s:%s" % (r, s.replace("flat_", ""), "done" if eval_done(r, s) else
                          ("ready" if eval_ready(r) else "-")) for r in EVAL_RUNS for s in EVAL_SCHEDULES)
        log("runs [%s]; lanes %d (idle %d); need %d; eval [%s]; eval lanes %d, unclaimed groups %d; jobs %s" % (
            summary, lanes, idle_lanes, need, evsum, evals, unclaimed,
            {j: "%s %.1fh %dcpu" % (v["node"], v["left"] / 3600, v["cpus"]) for j, v in jobs.items()}))
        launched_training = False
        for jid, info in sorted(jobs.items(), key=lambda kv: -kv[1]["left"]):
            entry = steps.get(jid, dict(lanes={}, evals={}, foreign=[]))
            if info["left"] < STOP_LEFT:
                if jid not in stopped and not DRY:
                    log("job %s ends in %.0f min: graceful stop of its lanes and cache cleanup" % (jid, info["left"] / 60))
                    children.append(spawn(srun_cmd(jid, 1, ["bash", WORK + "/stop_lanes.sh", jid]), "stop_%s.log" % jid))
                    stopped.add(jid)
                continue
            if entry["foreign"]:
                streak.pop(jid, None)
                log("job %s has another managed step; not touching it (%s)" % (jid, entry["foreign"][0][:120]))
                continue
            used = set()
            for gpus in entry["lanes"].values(): used.update(int(g) for g in gpus.split(","))
            for gpus, _ in entry["evals"].values(): used.update(int(g) for g in gpus.split(","))
            counts = streak.setdefault(jid, {})
            for g in prev_used.get(jid, set()) - used:   # released by our own lanes: no need to wait long
                counts[g] = max(counts.get(g, 0), IDLE_POLLS - 1)
            prev_used[jid] = used
            eval_launches = [t for t in eval_launches if time.time() - t < 3600]
            want_eval = unclaimed > 0 and info["left"] >= EVAL_MIN_LEFT and len(eval_launches) < 8
            if unclaimed > 0 and len(eval_launches) >= 8:
                log("8 eval lanes launched within the last hour: holding further eval launches (check eval logs)")
            if len(used) >= info["gpus"] or (need == 0 and not want_eval):
                continue
            idle = probe(jid)
            if idle is None:
                log("job %s: GPU probe failed" % jid); continue
            for g, is_idle in idle.items():
                counts[g] = counts.get(g, 0) + 1 if (is_idle and g not in used) else 0
            free = [g for g, c in counts.items() if c >= IDLE_POLLS]
            log("job %s (%s): idle streaks %s; elastic GPUs %s" % (jid, info["node"], counts, sorted(used)))
            chosen = pairs(free)
            training_lanes = len(entry["lanes"])
            while chosen and need > 0 and info["left"] >= MIN_LEFT_START and (
                    need_critical > 0 or unclaimed == 0 or evals_live > 0):
                a, b = chosen.pop(0)
                lane = "j%sg%d%d" % (jid, a, b)
                log("launching training lane %s on job %s (%s) GPUs %d,%d" % (lane, jid, info["node"], a, b))
                children.append(spawn(srun_cmd(jid, 8, ["bash", WORK + "/lane.sh", lane, "%d,%d" % (a, b)]),
                                      "lane_%s.log" % lane))
                counts[a] = counts[b] = 0; need -= 1; need_critical = max(0, need_critical - 1)
                training_lanes += 1; launched_training = True
            if chosen and want_eval:
                target = info["cpus"] - TRAIN_CPUS * training_lanes
                existing = sum(len(g.split(",")) * w for g, w in entry["evals"].values())
                wpg = max(1, min(MAX_WPG, (target - existing) // (2 * len(chosen))))
                for a, b in chosen:
                    lane = "e%sg%d%d" % (jid, a, b)
                    log("launching eval lane %s on job %s (%s) GPUs %d,%d, %d workers/GPU" % (lane, jid, info["node"], a, b, wpg))
                    children.append(spawn(srun_cmd(jid, info["cpus"], ["bash", WORK + "/eval_lane.sh", lane,
                                                                        "%d,%d" % (a, b), str(wpg)]), "eval_%s.log" % lane))
                    counts[a] = counts[b] = 0; eval_launches.append(time.time()); evals_live += 1
        # A run whose evaluation is requested must not wait behind evaluation lanes for long.
        starving = starving + 1 if (critical and need > 0 and not launched_training) else 0
        if starving >= PREEMPT_POLLS:
            victims = [(info["left"], jid, lane, gpus) for jid, info in jobs.items() if info["left"] >= MIN_LEFT_START + 1800
                       for lane, (gpus, _) in steps.get(jid, dict(evals={})).get("evals", {}).items()]
            if victims:
                left, jid, lane, gpus = max(victims)
                log("training %s has waited %d polls: stopping eval lane %s on job %s to free GPUs %s" % (
                    critical, starving, lane, jid, gpus))
                if not DRY:
                    log(srun(jid, ["bash", WORK + "/stop_eval_lane.sh", jid, lane], timeout=300)[1].strip())
                    for g in gpus.split(","): streak.setdefault(jid, {})[int(g)] = IDLE_POLLS - 1
            starving = 0
        if DRY: return
        time.sleep(POLL)

if __name__ == "__main__":
    main()
