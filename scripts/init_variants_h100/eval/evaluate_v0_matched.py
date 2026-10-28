#!/usr/bin/env python3
"""Resumable matched LIBERO evaluation of one LoopWAM v0 checkpoint (campaign protocol lightwam_20261009_matched).

Adapted from the residual_trans H100 runner scripts/evaluate_rt_matched.py, which follows
evaluation_report/audit_setup.py and reproduced the remote campaign (RT-A S2 parity):
- four suites x ten tasks x official initial states 0..49 x evaluation seeds 42/43/44 = 6,000 episodes;
- one task/seed group = one environment (env.seed(base seed) at construction) reused for its 50 initial states in
  order; the global Python/NumPy/Torch seed is set once per group; every action query uses the base seed;
- 30 settling actions [0,0,0,0,0,0,-1] whose done flag is ignored, then policy control capped at 400
  (Spatial/Object/Goal) or 700 (Long) steps; 32-action chunks, ten executed per query; success = env done flag;
- native LoopWAM (create_loopwam strict loader): FP32 weights, BF16 autocast, BF16 Wan2.1 VAE, ten action denoising
  steps, CFG 1, action shift 1; observation adapter and normalization from scripts/evaluate_loopwam_libero.py;
  gripper sign(-(2*g-1)) on the denormalized gripper.
CUDA graph: one captured replay of the unchanged native video-prefill + action-denoising computation (the VAE
encode stays eager). Production uses it only after --gate shows bit-exact actions against native eager
infer_action on real simulator observations. Videos are optional and never affect outcomes.
"""
from __future__ import annotations
import argparse, hashlib, json, multiprocessing as mp, os, queue, sys, time, traceback
from pathlib import Path

ROOT = Path(os.environ["LOOPWAM_ROOT"])
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MAX_STEPS = {"libero_spatial": 400, "libero_object": 400, "libero_goal": 400, "libero_10": 700}
SEEDS = (42, 43, 44)
SETTLING = 30
REPLAN = 10
STEPS = 10
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
    try:
        return time.time() - json.loads((Path(root) / "runners" / f"{token}.json").read_text())["time"] < limit
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


def groups_for(suites, seeds, tasks):
    # Long first to reduce stragglers; ordering never changes per-group seeding.
    order = sorted(suites, key=lambda s: 0 if s == "libero_10" else 1 + SUITES.index(s))
    return [dict(id=f"{s}_seed{seed}_task{t}", suite=s, seed=seed, task=t)
            for s in order for t in tasks for seed in seeds]


class Policy:
    """Native LoopWAM action inference with an optional CUDA-graph replay of its prefill + denoising core."""

    def __init__(self, checkpoint, vae_path):
        import torch
        from fastwam.models.wan22.loopwam import create_loopwam
        self.torch = torch
        self.model = create_loopwam(checkpoint_path=checkpoint, vae_path=vae_path, device="cuda:0",
                                    model_dtype=torch.float32).eval()
        # No autograd state: also keeps autocast from caching weight casts outside the captured graph.
        self.model.requires_grad_(False)
        self.fuse = bool(getattr(self.model.video_expert, "fuse_vae_embedding_in_latents", False))
        self.graph = None

    def native(self, image, proprio, context, mask, seed):
        return self.model.infer_action(input_image=image, proprio=proprio, context=context, context_mask=mask,
                                       action_horizon=32, num_inference_steps=STEPS, text_cfg_scale=1.0,
                                       seed=seed)["action"]

    def _inputs(self, image, proprio, context, mask, seed):
        # Same tensors, dtypes and RNG stream as FastWAM.infer_action.
        torch, m = self.torch, self.model
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn((1, 32, m.action_expert.action_dim), generator=generator, device="cpu",
                            dtype=torch.float32).to(device=m.device, dtype=m.torch_dtype)
        image = image.unsqueeze(0) if image.ndim == 3 else image
        latents = m._encode_input_image_latents_tensor(input_image=image.to(device=m.device, dtype=m.torch_dtype))
        proprio = proprio.unsqueeze(0).to(device=m.device, dtype=m.torch_dtype)
        context = (context.unsqueeze(0) if context.ndim == 2 else context).to(device=m.device, dtype=m.torch_dtype)
        mask = (mask.unsqueeze(0) if mask.ndim == 1 else mask).to(device=m.device, dtype=torch.bool)
        return [latents, proprio, context, mask, noise]

    def _core(self, latents, proprio, context, mask, noise):
        # FastWAM.infer_action after noise/VAE: proprio token, video prefill, ten action denoising steps.
        torch, m = self.torch, self.model
        context, mask = m._append_proprio_to_context(context=context, context_mask=mask, proprio=proprio)
        timestep_video = torch.zeros((latents.shape[0],), dtype=latents.dtype, device=m.device)
        (tokens, _t, t_mod, video_context, video_mask, freqs, _f, _h, _w, per_frame) = m.video_expert.prepare(
            x=latents, timestep=timestep_video, context=context, context_mask=mask, action=None,
            fuse_vae_embedding_in_latents=self.fuse)
        n = int(tokens.shape[1])
        attention = m._build_mot_attention_mask(video_seq_len=n, action_seq_len=noise.shape[1],
                                                video_tokens_per_frame=per_frame, device=tokens.device)
        cache_k, cache_v = m.mot.prefill_video_cache_tensor(
            video_tokens=tokens, video_freqs=freqs, video_t_mod=t_mod, video_context=video_context,
            video_context_mask=video_mask, video_attention_mask=attention[:n, :n])
        timesteps, deltas = m.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=STEPS, device=m.device, dtype=noise.dtype, shift_override=None)
        x = noise
        for step_t, delta in zip(timesteps, deltas):
            pred = m._denoise_action_with_video_cache(
                latents_action=x, timestep_action=step_t.unsqueeze(0).to(dtype=x.dtype, device=m.device),
                context=context, context_mask=mask, video_cache_k=cache_k, video_cache_v=cache_v,
                action_attention_mask=attention[n:, :])
            x = m.infer_action_scheduler.step(pred, delta, x)
        return x

    def graphed(self, image, proprio, context, mask, seed):
        torch = self.torch
        args = self._inputs(image, proprio, context, mask, seed)
        if self.graph is None:
            self.static = [a.clone() for a in args]
            stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                for _ in range(3):
                    self._core(*self.static)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                self.out = self._core(*self.static)
        for dst, src in zip(self.static, args):
            if dst.shape != src.shape or dst.dtype != src.dtype or dst.device != src.device:
                raise ValueError("CUDA graph input signature changed")
            dst.copy_(src)
        self.graph.replay()
        return self.out[0].detach().to(device="cpu", dtype=torch.float32)

    def __call__(self, image, proprio, context, mask, seed, backend):
        torch = self.torch
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if backend == "eager":
                return self.native(image, proprio, context, mask, seed)
            return self.graphed(image, proprio, context, mask, seed)


def libero_actions(adapter, action):
    import numpy as np, torch
    action = action.detach().to(device="cpu", dtype=torch.float32)
    if action.shape != (32, 7) or not torch.isfinite(action).all():
        raise ValueError("Expected finite 32x7 predicted actions")
    action = adapter.dataset.denormalize_action(action).numpy().copy()
    action[:, -1] = np.sign(-(2 * action[:, -1] - 1))      # protocol gripper rule; exact zero stays zero
    return action


def run_group(policy, adapter, group, episodes, backend, video_dir):
    import torch
    from libero.libero import benchmark
    from experiments.libero.libero_utils import get_libero_env, get_libero_image, save_rollout_video
    from evaluate_loopwam_libero import initial_states
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
                    image, proprio = adapter.observation(obs)
                    q0 = time.perf_counter()
                    action = policy(image, proprio, context, mask, seed, backend)
                    query_seconds += time.perf_counter() - q0; queries += 1
                    pending = libero_actions(adapter, action)[:REPLAN].tolist()
                if video_dir:
                    frames.append(get_libero_image(obs))
                obs, _, done, _ = env.step(pending.pop(0))
                steps += 1
                if done:
                    break
            success = bool(done)
            video = save_rollout_video(video_dir, frames, f"{group['id']}_trial{episode}", success,
                                       description) if video_dir else None
            rows.append(dict(id=group["id"], suite=suite_name, seed=seed, task=task_id, episode=episode,
                             initial_state_index=episode, success=success, policy_steps=steps, queries=queries,
                             query_seconds=query_seconds, total_seconds=time.perf_counter() - started, video=video))
    finally:
        env.close()
    return rows, description


def make_adapter(cond):
    from evaluate_loopwam_libero import ObservationAdapter
    data = json.loads(Path(cond["data_manifest"]).read_text())
    return ObservationAdapter(json.loads(Path(cond["stats"]).read_text()), cond["text_cache_dir"],
                              data["text_cache_files"])


def worker(index, gpu, args, cond, work, results):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["MUJOCO_EGL_DEVICE_ID"] = gpu
    for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "SLURM_PROCID", "LOCAL_WORLD_SIZE"):
        os.environ.pop(k, None)
    try:
        import torch
        torch.set_num_threads(args.threads); torch.cuda.set_device(0)
        policy = Policy(cond["checkpoint"], cond["vae_path"])
        adapter = make_adapter(cond)
        out = Path(cond["output"])
        results.put(dict(event="ready", worker=index, gpu=gpu))
        while True:
            group = work.get()
            if group is None:
                break
            if (out / "groups" / f"{group['id']}.json").exists() or not claim(
                    out / "claims" / f"{group['id']}.claim", args.token, out):
                results.put(dict(event="skipped", group=group["id"], worker=index))
                continue
            video_dir = out / "videos" / group["id"] if args.videos else None
            if video_dir: video_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            rows, description = run_group(policy, adapter, group, args.episodes, cond["backend"], video_dir)
            record = dict(group=group, task_description=description, checkpoint_sha256=cond["sha256"],
                          backend=cond["backend"], worker=index, gpu=gpu, token=args.token,
                          elapsed_seconds=time.perf_counter() - t0, episodes=rows,
                          successes=sum(r["success"] for r in rows), completed_at=time.time())
            write_json(out / "groups" / f"{group['id']}.json", record)
            (out / "claims" / f"{group['id']}.claim").unlink(missing_ok=True)
            results.put(dict(event="group", group=group["id"], successes=record["successes"], episodes=len(rows),
                             worker=index, seconds=round(record["elapsed_seconds"], 1),
                             policy_steps=sum(r["policy_steps"] for r in rows)))
        results.put(dict(event="done", worker=index))
    except BaseException:
        results.put(dict(event="error", worker=index, error=traceback.format_exc()))
        raise


def summarize(cond, groups, episodes):
    out = Path(cond["output"]); rows = []; records = []
    for g in groups:
        p = out / "groups" / f"{g['id']}.json"
        if not p.exists():
            return None
        rec = json.loads(p.read_text())
        if rec["checkpoint_sha256"] != cond["sha256"] or rec["group"] != g:
            raise ValueError(f"Identity mismatch in {p}")
        if ([r["episode"] for r in rec["episodes"]] != list(range(episodes))
                or any(r["initial_state_index"] != r["episode"] or r["seed"] != g["seed"] for r in rec["episodes"])):
            raise ValueError(f"Incomplete group {p}")
        rows += rec["episodes"]; records.append(rec)
    keys = {(r["suite"], r["seed"], r["task"], r["episode"]) for r in rows}
    if len(keys) != len(rows):
        raise ValueError("Duplicate episode keys")
    rate = lambda rs: sum(r["success"] for r in rs) / len(rs)
    per_suite = {}
    for s in SUITES:
        rs = [r for r in rows if r["suite"] == s]
        if not rs: continue
        seeds = {str(seed): dict(successes=sum(r["success"] for r in rs if r["seed"] == seed),
                                 episodes=sum(1 for r in rs if r["seed"] == seed),
                                 success_rate=rate([r for r in rs if r["seed"] == seed]))
                 for seed in sorted({r["seed"] for r in rs})}
        per_suite[s] = dict(successes=sum(r["success"] for r in rs), episodes=len(rs), success_rate=rate(rs),
                            seeds=seeds)
    per_seed = {str(seed): dict(successes=sum(r["success"] for r in rows if r["seed"] == seed),
                                episodes=sum(1 for r in rows if r["seed"] == seed),
                                success_rate=rate([r for r in rows if r["seed"] == seed]))
                for seed in sorted({r["seed"] for r in rows})}
    summary = dict(run=cond["run"], checkpoint=cond["checkpoint"], checkpoint_sha256=cond["sha256"],
                   checkpoint_step=cond["step"], backend=cond["backend"], episodes=len(rows),
                   successes=sum(r["success"] for r in rows), success_rate=rate(rows), per_suite=per_suite,
                   per_seed=per_seed, suite_mean_success_rate=sum(v["success_rate"] for v in per_suite.values()) / len(per_suite),
                   total_policy_steps=sum(r["policy_steps"] for r in rows), total_queries=sum(r["queries"] for r in rows),
                   total_query_seconds=sum(r["query_seconds"] for r in rows),
                   group_seconds=sum(rec["elapsed_seconds"] for rec in records),
                   first_group_completed_at=min(rec["completed_at"] for rec in records),
                   last_group_completed_at=max(rec["completed_at"] for rec in records))
    write_json(out / "summary.json", summary)
    return summary


def build_condition(args):
    import torch
    from evaluate_loopwam_libero import validate_checkpoint
    ckpt = Path(args.checkpoint).resolve(); train = ckpt.parent
    sha = sha256_file(ckpt)
    if args.expected_sha and sha != args.expected_sha:
        raise SystemExit(f"checkpoint sha {sha} != expected {args.expected_sha}")
    manifest = json.loads((train / "manifest.json").read_text())
    vae_hash = sha256_file(args.vae_path)
    if manifest["asset_sha256"]["vae"] != vae_hash:
        raise SystemExit("VAE differs from the one used in training")
    stats = train / "data" / "dataset_stats.json"; data_manifest = train / "data" / "data_manifest.json"
    payload = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    contract = validate_checkpoint(payload, json.loads(data_manifest.read_text()), sha256_file(stats),
                                   suite="libero_10")
    if (payload["version"] != "v0" or payload["video_loops"] != 4 or payload["action_loops"] != 4
            or payload.get("action_kv_mode", "aligned") != "aligned"):
        raise SystemExit("Expected a v0 aligned 4/4 checkpoint")
    step = payload["step"]; del payload
    return dict(run=args.run, checkpoint=str(ckpt), sha256=sha, step=step, stats=str(stats),
                data_manifest=str(data_manifest), vae_path=args.vae_path, vae_sha256=vae_hash,
                text_cache_dir=args.text_cache_dir, backend=args.backend, output=args.output_dir,
                contract_world=contract.get("world"), initialization_mode=manifest.get("initialization_mode"),
                init_sha256=manifest["asset_sha256"].get("initialization"))


def gate(args, cond):
    """Unscored: native eager infer_action vs CUDA-graph replay on real observations (must be bit-exact)."""
    import numpy as np, torch
    from libero.libero import benchmark
    from experiments.libero.libero_utils import get_libero_env
    from evaluate_loopwam_libero import initial_states
    torch.set_num_threads(args.threads); torch.cuda.set_device(0)
    policy = Policy(cond["checkpoint"], cond["vae_path"]); adapter = make_adapter(cond)
    report = []
    for suite_name, task_id, seed in (("libero_10", 0, 42), ("libero_spatial", 3, 43), ("libero_goal", 7, 44)):
        task = benchmark.get_benchmark_dict()[suite_name]().get_task(task_id)
        context, mask = adapter.text(task.language)
        env, _ = get_libero_env(task, 256, seed)
        env.reset(); obs = env.set_init_state(initial_states(task)[0])
        for _ in range(SETTLING): obs, *_ = env.step(DUMMY)
        for q in range(args.gate_queries):
            image, proprio = adapter.observation(obs); out = {}
            for backend in ("eager", "cuda-graph", "eager", "cuda-graph"):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                act = policy(image, proprio, context, mask, seed, backend)
                torch.cuda.synchronize(); out.setdefault(backend, []).append((act.clone(), time.perf_counter() - t0))
            ref = out["eager"][0][0]
            report.append(dict(suite=suite_name, seed=seed, query=q,
                               bit_exact=all(torch.equal(ref, a) for a, _ in out["eager"] + out["cuda-graph"]),
                               max_abs_difference=max((ref - a).abs().max().item() for a, _ in out["cuda-graph"]),
                               eager_seconds=out["eager"][1][1], graph_seconds=out["cuda-graph"][1][1]))
            for act in libero_actions(adapter, ref)[:REPLAN]:
                obs, _, done, _ = env.step(act.tolist())
        env.close()
    result = dict(checkpoint_sha256=cond["sha256"], all_bit_exact=all(r["bit_exact"] for r in report),
                  queries=len(report), eager_ms_median=float(np.median([r["eager_seconds"] for r in report])) * 1e3,
                  graph_ms_median=float(np.median([r["graph_seconds"] for r in report])) * 1e3,
                  peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30, gpu=torch.cuda.get_device_name(0),
                  torch=torch.__version__, details=report)
    result["speedup"] = result["eager_ms_median"] / result["graph_ms_median"]
    write_json(args.gate_output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "details"}), flush=True)
    if not result["all_bit_exact"]:
        raise SystemExit("CUDA graph gate failed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--expected-sha", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--vae-path", required=True)
    p.add_argument("--text-cache-dir", required=True)
    p.add_argument("--backend", default="cuda-graph", choices=("cuda-graph", "eager"))
    p.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    p.add_argument("--workers-per-gpu", type=int, default=4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--suites", default=",".join(SUITES))
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    p.add_argument("--tasks", default=",".join(map(str, range(10))))
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--no-videos", dest="videos", action="store_false")
    p.add_argument("--gate", action="store_true", help="run the unscored CUDA-graph parity gate and exit")
    p.add_argument("--gate-queries", type=int, default=8)
    p.add_argument("--gate-output", default="")
    args = p.parse_args()
    args.token = f"{os.uname().nodename}-{os.environ.get('SLURM_JOB_ID', 'nojob')}-{os.getpid()}-{int(time.time())}"
    suites = args.suites.split(","); seeds = [int(x) for x in args.seeds.split(",")]
    tasks = [int(x) for x in args.tasks.split(",")]
    if any(s not in SUITES for s in suites) or not 1 <= args.episodes <= 50:
        raise SystemExit("invalid grid")
    cond = build_condition(args)
    if args.gate:
        return gate(args, cond)
    import torch
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    beat = lambda: write_json(out / "runners" / f"{args.token}.json", dict(time=time.time(), token=args.token))
    beat()
    groups = groups_for(suites, seeds, tasks)
    manifest = dict(cond, protocol=dict(name="lightwam_20261009_matched", suites=suites, seeds=seeds, tasks=tasks,
        episodes=args.episodes, max_policy_steps=MAX_STEPS, settling_steps=SETTLING, settling_action=DUMMY,
        settling_done="ignored", replan_steps=REPLAN, action_horizon=32, denoising_steps=STEPS, cfg=1.0,
        action_shift=1.0, seed_rule="global seed once per task/seed group; env.seed(base) at construction, env "
        "reused for 50 sequential initial states; every query uses the base seed", success="environment done flag",
        precision="FP32 model, BF16 autocast, BF16 VAE", camera=256, model_image="2x224 horizontal",
        gripper="sign(-(2*denormalized_gripper-1))", videos=args.videos),
        runner_sha256=sha256_file(__file__), adapter_source_sha256=sha256_file(ROOT / "scripts/evaluate_loopwam_libero.py"),
        code_root=str(ROOT), host=os.uname().nodename, slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        mujoco_gl=os.environ.get("MUJOCO_GL"), torch=torch.__version__)
    mpath = out / "manifest.json"
    if mpath.exists():
        old = json.loads(mpath.read_text())
        strip = lambda d: {k: v for k, v in d.items() if k != "videos"}  # videos never affect outcomes
        for k in ("sha256", "backend", "protocol", "vae_sha256", "runner_sha256", "text_cache_dir"):
            if (strip(old[k]) if k == "protocol" else old[k]) != (strip(manifest[k]) if k == "protocol" else manifest[k]):
                raise SystemExit(f"Resume refused: {k} differs")
    else:
        write_json(mpath, manifest)
    todo = [g for g in groups if not (out / "groups" / f"{g['id']}.json").exists()]
    t_start = time.time()
    print(json.dumps(dict(event="plan", run=args.run, pending_groups=len(todo), total_groups=len(groups),
                          gpus=args.gpus, workers_per_gpu=args.workers_per_gpu, token=args.token)), flush=True)
    if todo:
        gpus = args.gpus.split(","); nworkers = min(len(gpus) * args.workers_per_gpu, len(todo))
        ctx = mp.get_context("spawn"); work = ctx.Queue(); results = ctx.Queue()
        for g in todo: work.put(g)
        for _ in range(nworkers): work.put(None)
        procs = [ctx.Process(target=worker, args=(i, gpus[i % len(gpus)], args, cond, work, results), daemon=False)
                 for i in range(nworkers)]
        for pr in procs: pr.start()
        finished = done_workers = errors = eps = 0; last_beat = time.time()
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
                print(json.dumps(dict(msg, finished=finished, pending=len(todo) - finished, time=time.strftime("%H:%M:%S"),
                                      eps_per_hour=round(eps / (time.time() - t_start) * 3600))), flush=True)
            elif msg["event"] == "error":
                errors += 1; print(msg["error"], file=sys.stderr, flush=True)
            elif msg["event"] == "done":
                done_workers += 1
            else:
                print(json.dumps(msg), flush=True)
        for pr in procs: pr.join()
        write_json(out / "runs" / f"{args.token}.json", dict(token=args.token, started=t_start, finished=time.time(),
                   wall_seconds=time.time() - t_start, gpus=args.gpus, workers=nworkers, threads=args.threads,
                   groups_completed=finished, errors=errors, slurm_job_id=os.environ.get("SLURM_JOB_ID")))
        if errors:
            raise SystemExit(f"{errors} worker(s) failed")
    s = summarize(cond, groups, args.episodes)
    print(json.dumps(dict(event="summary", complete=s is not None, success_rate=s and s["success_rate"],
                          episodes=s and s["episodes"])), flush=True)


if __name__ == "__main__":
    main()
