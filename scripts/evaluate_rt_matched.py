#!/usr/bin/env python3
"""Resumable Residual Transport LIBERO evaluation matching the remote campaign protocol.

Protocol (evaluation_report/audit_setup.py and campaign manifests):
- four suites x ten tasks x official initial states 0..49 x evaluation seeds 42/43/44;
- one task/seed group = one fresh environment (env.seed(base seed)) reused for its 50 episodes in order;
  the global Python/NumPy/Torch seed is set once per group; every action query uses the base seed;
- 30 settling actions [0,0,0,0,0,0,-1] whose done flag is ignored (Light-WAM loop), then policy control
  capped at 400 (Spatial/Object/Goal) or 700 (Long) steps; 32-action chunks, ten executed per query;
- success is the environment done flag; FP32 model with BF16 autocast; observation adapter, gripper rule
  and transport model come unchanged from scripts/evaluate_loopwam_libero.py and TransportInference.

Every task/seed group is written atomically, so a restarted run resumes from completed groups only.
"""
from __future__ import annotations
import argparse, hashlib, json, math, multiprocessing as mp, os, queue, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MAX_STEPS = {"libero_spatial": 400, "libero_object": 400, "libero_goal": 400, "libero_10": 700}
SEEDS = (42, 43, 44)
SETTLING = 30
REPLAN = 10
DUMMY = [0, 0, 0, 0, 0, 0, -1]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False)); tmp.replace(path)


def heartbeat_fresh(root, token, limit=600):
    path = Path(root) / "runners" / f"{token}.json"
    try:
        return time.time() - json.loads(path.read_text())["time"] < limit
    except (OSError, ValueError, KeyError):
        return False


def claim(path, token, root):
    """Atomically own a task/seed group; take over only claims whose runner stopped heartbeating."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, token.encode()); os.close(fd); return True
        except FileExistsError:
            try:
                owner = path.read_text().strip()
            except OSError:
                owner = ""
            if owner == token:
                return True
            if owner and heartbeat_fresh(root, owner):
                return False
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    return False


def condition_key(run, schedule, sha, prefix="residual_trans"):
    return f"{prefix}__{run}__{schedule}__{sha}"


def groups_for(suites, seeds, tasks):
    # Long first to reduce stragglers; ordering never changes per-group seeding.
    order = sorted(suites, key=lambda s: 0 if s == "libero_10" else 1 + SUITES.index(s))
    return [dict(id=f"{s}_seed{seed}_task{t}", suite=s, seed=seed, task=t)
            for s in order for t in tasks for seed in seeds]


def run_group(model, adapter, schedule_steps, group, episodes, backend, video_dir):
    import random
    import numpy as np, torch
    from libero.libero import benchmark
    from experiments.libero.libero_utils import get_libero_env, get_libero_image, save_rollout_video
    from scripts.evaluate_loopwam_libero import initial_states
    from fastwam.utils.pytorch_utils import set_global_seed
    suite_name, seed, task_id = group["suite"], group["seed"], group["task"]
    set_global_seed(seed, get_worker_init_fn=False)          # once per task/seed group
    task = benchmark.get_benchmark_dict()[suite_name]().get_task(task_id)
    context, mask = adapter.text(task.language)
    states = initial_states(task)
    if len(states) < episodes:
        raise ValueError("Fewer official initial states than requested episodes")
    env, description = get_libero_env(task, 256, seed)      # env.seed(base seed) once
    max_steps = MAX_STEPS[suite_name]
    rows = []
    try:
        for episode in range(episodes):
            started = time.perf_counter(); query_seconds = 0.0; queries = 0
            env.reset()
            obs = env.set_init_state(states[episode])
            frames = []
            for _ in range(SETTLING):                     # Light-WAM: done ignored while settling
                obs, _, done, _ = env.step(DUMMY)
            pending = []; steps = 0; done = False
            while steps < max_steps:
                if not pending:
                    q0 = time.perf_counter()
                    image, proprio = adapter.observation(obs)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        action = model.infer_action(input_image=image, proprio=proprio, context=context,
                            context_mask=mask, action_horizon=32, num_inference_steps=schedule_steps,
                            text_cfg_scale=1.0, seed=seed, compile_action_infer=(backend == "cuda-graph"))["action"]
                    chunk = adapter.libero_actions(action)
                    query_seconds += time.perf_counter() - q0; queries += 1
                    pending = chunk[:REPLAN].tolist()
                frames.append(get_libero_image(obs))
                obs, _, done, _ = env.step(pending.pop(0))
                steps += 1
                if done:
                    break
            rollout = time.perf_counter() - started
            success = bool(done)
            video = save_rollout_video(video_dir, frames, f"{group['id']}_trial{episode}", success, description) if video_dir else None
            rows.append(dict(id=group["id"], suite=suite_name, seed=seed, task=task_id, episode=episode,
                             initial_state_index=episode, success=success, policy_steps=steps, queries=queries,
                             query_seconds=query_seconds, rollout_seconds=rollout,
                             total_seconds=time.perf_counter() - started, video=video))
    finally:
        env.close()
    return rows, description


def worker(index, gpu, args, conditions, work, results):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["MUJOCO_EGL_DEVICE_ID"] = gpu
    for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "SLURM_PROCID"):
        os.environ.pop(k, None)
    try:
        import torch
        torch.set_num_threads(args.threads); torch.cuda.set_device(0)
        from scripts.evaluate_loopwam_libero import ObservationAdapter
        from fastwam.models.wan22.transport_inference import TransportInference
        from fastwam.models.wan22.residual_transport import SCHEDULES
        loaded = None; model = None; adapter = None
        results.put(dict(event="ready", worker=index, gpu=gpu))
        while True:
            item = work.get()
            if item is None:
                break
            cond = conditions[item["condition"]]
            if loaded != cond["checkpoint"]:
                model = None; torch.cuda.empty_cache()
                model = TransportInference(cond["checkpoint"], args.vae_path, device="cuda:0",
                                           schedule=cond["schedule"]).eval()
                data = json.loads(Path(cond["data_manifest"]).read_text())
                adapter = ObservationAdapter(json.loads(Path(cond["stats"]).read_text()),
                                             args.text_cache_dir, data["text_cache_files"] if args.verify_text_hash else None)
                loaded = cond["checkpoint"]
            model.schedule = cond["schedule"]
            out = Path(cond["output"]); group = item["group"]
            if (out / "groups" / f"{group['id']}.json").exists() or not claim(
                    out / "claims" / f"{group['id']}.claim", args.token, args.output_root):
                results.put(dict(event="skipped", condition=item["condition"], group=group["id"], worker=index))
                continue
            video_dir = out / "videos" / group["id"] if args.videos else None
            if video_dir: video_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            rows, description = run_group(model, adapter, len(SCHEDULES[cond["schedule"]]), group,
                                          args.episodes, cond["backend"], video_dir)
            record = dict(condition=item["condition"], group=group, task_description=description,
                          checkpoint_sha256=cond["sha256"], schedule=cond["schedule"], backend=cond["backend"],
                          worker=index, gpu=gpu, elapsed_seconds=time.perf_counter() - t0, episodes=rows,
                          successes=sum(r["success"] for r in rows), completed_at=time.time())
            write_json(out / "groups" / f"{group['id']}.json", record)
            (out / "claims" / f"{group['id']}.claim").unlink(missing_ok=True)
            results.put(dict(event="group", condition=item["condition"], group=group["id"],
                             successes=record["successes"], episodes=len(rows), worker=index,
                             seconds=record["elapsed_seconds"]))
        results.put(dict(event="done", worker=index))
    except BaseException:
        results.put(dict(event="error", worker=index, error=traceback.format_exc()))
        raise


def summarize(cond, groups, episodes):
    out = Path(cond["output"]); rows = []
    for g in groups:
        p = out / "groups" / f"{g['id']}.json"
        if not p.exists():
            return None
        rec = json.loads(p.read_text())
        if rec["checkpoint_sha256"] != cond["sha256"] or rec["schedule"] != cond["schedule"]:
            raise ValueError(f"Identity mismatch in {p}")
        if [r["episode"] for r in rec["episodes"]] != list(range(episodes)):
            raise ValueError(f"Incomplete group {p}")
        rows += rec["episodes"]
    per_suite = {}
    for s in SUITES:
        rs = [r for r in rows if r["suite"] == s]
        if not rs: continue
        seeds = {str(seed): dict(successes=sum(r["success"] for r in rs if r["seed"] == seed),
                                 episodes=sum(1 for r in rs if r["seed"] == seed)) for seed in sorted({r["seed"] for r in rs})}
        for v in seeds.values(): v["success_rate"] = v["successes"] / v["episodes"]
        per_suite[s] = dict(successes=sum(r["success"] for r in rs), episodes=len(rs),
                            success_rate=sum(r["success"] for r in rs) / len(rs), seeds=seeds)
    summary = dict(run=cond["run"], schedule=cond["schedule"], checkpoint_sha256=cond["sha256"],
                   backend=cond["backend"], episodes=len(rows), successes=sum(r["success"] for r in rows),
                   success_rate=sum(r["success"] for r in rows) / len(rows), per_suite=per_suite,
                   suite_mean_success_rate=sum(v["success_rate"] for v in per_suite.values()) / len(per_suite))
    write_json(out / "summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--conditions", required=True, help="Comma list RUN:SCHEDULE[:backend], e.g. RT-A:S4,RT-A2:S7")
    p.add_argument("--weights-root", default="/groups/yshang/an221229/checkpoints/LoopWAM_NT")
    p.add_argument("--epoch-file", default="epoch_10.pt")
    p.add_argument("--expected-sha", default="", help="Comma list RUN=sha256 checked before evaluation")
    p.add_argument("--output-root", required=True)
    p.add_argument("--vae-path", required=True)
    p.add_argument("--text-cache-dir", required=True)
    p.add_argument("--verify-text-hash", action="store_true")
    p.add_argument("--backend", default="cuda-graph", choices=("cuda-graph", "eager"))
    p.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    p.add_argument("--workers-per-gpu", type=int, default=3)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--suites", default=",".join(SUITES))
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    p.add_argument("--tasks", default=",".join(map(str, range(10))))
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--no-videos", dest="videos", action="store_false")
    p.add_argument("--tag", default="final_rollout")
    p.add_argument("--key-prefix", default="residual_trans")
    p.add_argument("--interleave", action="store_true",
                   help="Queue groups group-major across the schedules of one checkpoint (spreads heavy and light "
                        "schedules over workers; never changes per-group seeding)")
    p.add_argument("--smoke", action="store_true",
                   help="Unscored harness check of a non-final checkpoint; requires a smoke* tag")
    args = p.parse_args()
    if args.smoke and not args.tag.startswith("smoke"):
        raise SystemExit("--smoke requires a tag starting with 'smoke'")
    args.token = f"{os.uname().nodename}-{os.environ.get('SLURM_JOB_ID', 'nojob')}-{os.getpid()}-{int(time.time())}"
    def beat():
        write_json(Path(args.output_root) / "runners" / f"{args.token}.json", dict(time=time.time(), token=args.token))
    beat()
    import torch
    from scripts.evaluate_loopwam_libero import validate_transport_checkpoint
    from fastwam.models.wan22.residual_transport import SCHEDULES
    suites = args.suites.split(","); seeds = [int(x) for x in args.seeds.split(",")]
    tasks = [int(x) for x in args.tasks.split(",")]
    if any(s not in SUITES for s in suites) or not 1 <= args.episodes <= 50:
        raise SystemExit("invalid grid")
    groups = groups_for(suites, seeds, tasks)
    expected = dict(x.split("=") for x in args.expected_sha.split(",") if x)
    vae_hash = sha256_file(args.vae_path)
    root = Path(args.output_root); root.mkdir(parents=True, exist_ok=True)
    conditions = {}; sha_cache = {}
    for spec in args.conditions.split(","):
        parts = spec.split(":"); run, schedule = parts[0], parts[1]
        backend = parts[2] if len(parts) > 2 else args.backend
        if schedule not in SCHEDULES:
            raise SystemExit(f"unknown schedule {schedule}")
        folder = Path(args.weights_root) / run
        ckpt = folder / args.epoch_file
        if ckpt not in sha_cache:
            sha_cache[ckpt] = sha256_file(ckpt)
        sha = sha_cache[ckpt]
        if run in expected and expected[run] != sha:
            raise SystemExit(f"{run} checkpoint sha {sha} != expected {expected[run]}")
        config = json.loads((folder / "config.json").read_text())
        if config["asset_sha256"]["vae"] != vae_hash:
            raise SystemExit("VAE differs from the one used in training")
        stats = folder / "dataset_stats.json"; data_manifest = folder / "data" / "data_manifest.json"
        payload = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
        if args.smoke:
            contract = payload["training_state"]["contract"]
        else:
            contract = validate_transport_checkpoint(payload, json.loads(data_manifest.read_text()), sha256_file(stats))
        if config["config"].get("campaign") == "elastic-v1":
            if payload.get("transport_config") != config["config"]:
                raise SystemExit(f"{run}: checkpoint transport config differs from config.json")
            # Full-depth deployment ([4]xN, plan sections 7 and 8: the "full3" of first-full runs) is evaluated as a
            # retention probe; it is allowed only at a step count N the run was trained with, so that the step
            # condition d = 1/N it receives is one it was trained on.
            trained = config["config"]["schedules"]
            full_depth = set(SCHEDULES[schedule]) == {4} and any(len(SCHEDULES[t]) == len(SCHEDULES[schedule]) for t in trained)
            if schedule not in trained and not full_depth:
                raise SystemExit(f"{run} was not trained on schedule {schedule}")
        epoch = payload["training_state"]["epoch"]; del payload
        key = condition_key(run, schedule, sha, args.key_prefix)
        out = root / args.tag / key
        cond = dict(run=run, schedule=schedule, depths=SCHEDULES[schedule], checkpoint=str(ckpt), sha256=sha,
                    stats=str(stats), data_manifest=str(data_manifest), backend=backend, output=str(out),
                    epoch=epoch, transport_config=config["config"])
        manifest = dict(cond, protocol=dict(suites=suites, seeds=seeds, tasks=tasks, episodes=args.episodes,
            max_policy_steps=MAX_STEPS, settling_steps=SETTLING, settling_action=DUMMY, settling_done="ignored",
            replan_steps=REPLAN, action_horizon=32, seed_rule="global seed once per task/seed group; env.seed(base) at "
            "construction, env reused for 50 sequential initial states; every query uses the base seed",
            success="environment done flag", precision="FP32 model, BF16 autocast", camera=256, model_image="2x224 horizontal",
            videos=args.videos), vae_sha256=vae_hash, contract_world=contract.get("world"),
            runner_sha256=sha256_file(__file__), adapter_source_sha256=sha256_file(ROOT / "scripts/evaluate_loopwam_libero.py"),
            text_cache_dir=args.text_cache_dir, host=os.uname().nodename, slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            mujoco_gl=os.environ.get("MUJOCO_GL"), torch=torch.__version__,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        mpath = out / "manifest.json"
        if mpath.exists():
            old = json.loads(mpath.read_text())
            strip = lambda d: {k: v for k, v in d.items() if k != "videos"}  # videos never affect outcomes
            for k in ("sha256", "schedule", "backend", "protocol", "vae_sha256"):
                if (strip(old[k]) if k == "protocol" else old[k]) != (strip(manifest[k]) if k == "protocol" else manifest[k]):
                    raise SystemExit(f"Resume refused: {k} differs for {key}")
        else:
            write_json(mpath, manifest)
        conditions[key] = cond
    if args.interleave:
        runs = list(dict.fromkeys(c["checkpoint"] for c in conditions.values()))
        order = [(k, g) for r in runs for g in groups for k in conditions if conditions[k]["checkpoint"] == r]
    else:
        order = [(k, g) for k in conditions for g in groups]
    todo = [dict(condition=k, group=g) for k, g in order
            if not (Path(conditions[k]["output"]) / "groups" / f"{g['id']}.json").exists()]
    print(json.dumps(dict(event="plan", conditions=list(conditions), pending_groups=len(todo),
                          total_groups=len(groups) * len(conditions))), flush=True)
    if todo:
        gpus = args.gpus.split(","); nworkers = len(gpus) * args.workers_per_gpu
        ctx = mp.get_context("spawn"); work = ctx.Queue(); results = ctx.Queue()
        for item in todo: work.put(item)
        for _ in range(nworkers): work.put(None)
        procs = [ctx.Process(target=worker, args=(i, gpus[i % len(gpus)], args, conditions, work, results), daemon=False)
                 for i in range(nworkers)]
        for pr in procs: pr.start()
        finished = 0; done_workers = 0; t0 = time.time(); eps = 0; errors = 0
        last_beat = time.time()
        while done_workers + errors < nworkers:
            if time.time() - last_beat > 60:
                beat(); last_beat = time.time()
            try:
                msg = results.get(timeout=60)
            except queue.Empty:
                continue
            if msg["event"] == "skipped":
                finished += 1; continue
            if msg["event"] == "group":
                finished += 1; eps += msg["episodes"]
                print(json.dumps(dict(msg, finished=finished, pending=len(todo) - finished,
                                      eps_per_hour=round(eps / (time.time() - t0) * 3600))), flush=True)
            elif msg["event"] == "error":
                errors += 1; print(msg["error"], file=sys.stderr, flush=True)
            elif msg["event"] == "done":
                done_workers += 1
            else:
                print(json.dumps(msg), flush=True)
        for pr in procs: pr.join()
    for k, cond in conditions.items():
        s = summarize(cond, groups, args.episodes)
        print(json.dumps(dict(event="summary", condition=k, complete=s is not None,
                              success_rate=s and s["success_rate"], episodes=s and s["episodes"])), flush=True)


if __name__ == "__main__":
    main()
