#!/usr/bin/env python3
"""Matched LIBERO evaluation of LiLa-WAM trained on our LIBERO data (paper recipe), protocol lightwam_20261009_matched.

Protocol (identical to the LoopWAM/FastWAM matched runner evaluate_v0_matched.py, whose constants, claim/heartbeat
and summary code are imported unchanged):
- 4 suites x 10 tasks x official initial states 0..49 x seeds 42/43/44 = 6,000 episodes, 120 task/seed groups;
- one environment per group (get_libero_env(task, 256, seed): env.seed(seed) once) reused for its 50 initial states in
  order; set_global_seed(seed) once per group; 30 settling actions [0,0,0,0,0,0,-1] with done ignored; then policy
  control capped at 400 (Spatial/Object/Goal) or 700 (Long) steps; success = environment done flag.
LiLa-WAM specifics (no LIBERO eval exists upstream, so these are this runner's choices, recorded in the manifest):
- each suite uses its own policy: baselines/LiLa-WAM_LIBERO_ours/<suite>/stage2_epoch2_final.pt with that suite's
  config, norm stats and the task's task-condition vector;
- inputs from libero_data.libero_obs_to_inputs (both views rotated 180 deg, 256x256 INTER_AREA, [agentview | wrist]
  512x256; 8-D state = eef pos, axis-angle, gripper qpos), exactly the training format;
- public sampling procedure (robotwin_infer._predict_chunk): Gaussian start, 10 Euler flow steps, denormalize, B-spline
  smoothing of the 12-step chunk; every query of a group starts from the same noise (CPU generator seeded with the
  group seed: "every query uses the base seed");
- the first 6 of 12 actions are executed per query (half the chunk, as the public RoboTwin config executes 16 of 32);
- gripper: the policy outputs raw LIBERO gripper values (-1 open, +1 close); executed as sign(g) (protocol binarized
  gripper; an exact zero stays zero).
Speed: one batched inference server per GPU (bf16; DINOv3 once per batch, then each suite's action model on its rows)
and many CPU-only simulator workers (one environment each, EGL rendering spread over the GPUs). Batched bf16 inference
is not bit-identical to batch-1 inference; it is the same computation.
"""
from __future__ import annotations
import argparse, hashlib, json, multiprocessing as mp, os, queue, sys, time, traceback
from pathlib import Path

RUNNER_DIR = "/groups/yshang/an221229/checkpoints/LoopWAM_NT/v0_init_variants_work/eval"
ADAPTER = "/groups/yshang/an221229/code/lilawam_libero"
sys.path.insert(0, RUNNER_DIR)
import evaluate_v0_matched as M  # noqa: E402  (protocol constants + helpers; puts LOOPWAM_ROOT on sys.path)

REPLAN = 6
CHUNK = 12
TASKCOND = "/groups/yshang/an221229/data/LiLaWAM_libero_ours_taskcond"
SERVER_ORDER = ["libero_10", "libero_spatial", "libero_goal", "libero_object"]


CAMS = ("agentview_image", "robot0_eye_in_hand_image")


def camera_switch(env):
    """Swap the two camera sensors of a LIBERO env between the real renderer and a dummy. robosuite samples camera
    observables on a timer inside env.step; replacing only the sensor function keeps that timer logic identical, so
    observations are unchanged wherever the real sensor is installed. The runner installs it for the last 2 control
    steps before each query (verified bit-identical: images, state, done flags and qpos over 3 suites x 300 steps)."""
    import numpy as np
    obs_ = env.env._observables
    real = {c: obs_[c]._sensor for c in CAMS}
    dummy = {}
    for c in CAMS:
        f = lambda obs_cache: np.zeros((256, 256, 3), np.uint8)
        f.__modality__ = real[c].__modality__
        dummy[c] = f

    def cam(on):
        for c in CAMS:
            obs_[c]._sensor = real[c] if on else dummy[c]
    return cam


class StopRun(Exception):
    """--stop-at reached: abandon the unfinished group (nothing of it is written) and exit cleanly."""


def task_dir_name(text):
    import re
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def server(sidx, gpu, suites, cfgs, req_q, resp_qs, ready_q, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    try:
        import glob
        import numpy as np, torch
        from concurrent.futures import ThreadPoolExecutor
        sys.path.insert(0, ADAPTER)
        import libero_data  # noqa: F401  (LiLa-WAM repo on sys.path)
        import robotwin_infer as R
        from dataloader.dataset import _normalize_image
        torch.set_num_threads(args.server_threads)
        cpu_pool = ThreadPoolExecutor(args.server_threads)   # numpy releases the GIL: same function, run in parallel
        dt = torch.bfloat16
        infs = {s: R.RobotWinInference(config_path=cfgs[s]["config"], checkpoint_path=cfgs[s]["checkpoint"],
                                       norm_stats_path=cfgs[s]["stats"], device="cuda", dtype=dt) for s in suites}
        enc = infs[suites[0]].model          # frozen DINOv3 (identical weights in every suite's wrapper)
        smooth = {s: bool(infs[s].smooth_actions) for s in suites}
        tc_cache, noise_cache = {}, {}

        def task_cond(s, tdir):
            if (s, tdir) not in tc_cache:
                tc_cache[(s, tdir)] = torch.from_numpy(np.load(f"{TASKCOND}/{s}/{tdir}/task_cond.npy")).to("cuda", dt)
            return tc_cache[(s, tdir)]

        def noise(seed):
            if seed not in noise_cache:
                g = torch.Generator(device="cpu").manual_seed(seed)
                noise_cache[seed] = torch.randn((CHUNK, 7), generator=g).to("cuda", dt)
            return noise_cache[seed]

        ts = torch.linspace(0, 1, 10 + 1, device="cuda", dtype=dt)

        def encode(px):
            return enc.get_vision_features(px)

        def sample(m, feats, st, tc, x):
            # public robotwin_infer._predict_chunk: 10 Euler steps of the flow, then denormalize
            qpos = m.normalize_state(st); B = x.shape[0]
            for k in range(10):
                out = m.action_model(t=ts[k].repeat(B), noisy_actions=x, qpos_history=qpos,
                                     dino_features_list=feats, task_cond=tc)
                x = x + out["final_pred"] * (ts[k + 1] - ts[k])
            return m.denormalize_action(x)

        # CUDA graphs (one per batch bucket for the encoder, per suite and bucket for the sampler). Each graph is
        # checked bit-for-bit against eager execution of the same shapes; any mismatch disables graphs entirely.
        buckets = sorted({b for b in (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64) if b < args.max_batch} | {args.max_batch})
        bucket = lambda n: next(b for b in buckets if b >= n)
        enc_g, dit_g, graphs_ok, t_cap = {}, {}, False, time.time()
        if args.cuda_graphs:
            try:
                rng = torch.Generator(device="cuda").manual_seed(0)
                rnd = lambda shape: torch.randn(shape, device="cuda", generator=rng).to(dt)
                tc_dim = int(np.load(glob.glob(f"{TASKCOND}/{suites[0]}/*/task_cond.npy")[0]).shape[-1])

                def capture(fn, *static):
                    side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        for _ in range(2):
                            fn(*static)
                    torch.cuda.current_stream().wait_stream(side)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):        # private memory pool per graph: outputs never alias
                        out = fn(*static)
                    return g, out

                ok = True
                with torch.no_grad():
                    for b in reversed(buckets):
                        px_s = rnd((b, 3, 256, 512))
                        g, f_out = capture(encode, px_s)
                        ref = encode(px_s); g.replay(); torch.cuda.synchronize()
                        ok = ok and all(torch.equal(a, c) for a, c in zip(f_out, ref))
                        enc_g[b] = (g, px_s, f_out)
                        for s in suites:
                            m = infs[s].model
                            f_in = [rnd(f.shape) for f in f_out]
                            st_s, tc_s, x_s = rnd((b, 1, 8)), rnd((b, tc_dim)), rnd((b, CHUNK, 7))
                            g2, a_out = capture(lambda f, st, tc, x, m=m: sample(m, f, st, tc, x), f_in, st_s, tc_s, x_s)
                            ref = sample(m, f_in, st_s, tc_s, x_s); g2.replay(); torch.cuda.synchronize()
                            ok = ok and torch.equal(a_out, ref)
                            dit_g[(s, b)] = (g2, f_in, st_s, tc_s, x_s, a_out)
                graphs_ok = bool(ok)
            except Exception:
                ready_q.put(dict(event="server_warning", server=sidx, warning="CUDA graph capture failed; eager",
                                 error=traceback.format_exc()[-800:]))
                graphs_ok = False
            if not graphs_ok:
                enc_g, dit_g = {}, {}
                torch.cuda.empty_cache()
        ready_q.put(dict(event="server_ready", server=sidx, gpu=gpu, suites=suites, cuda_graphs=graphs_ok,
                         buckets=buckets, capture_seconds=round(time.time() - t_cap, 1)))
        stop, nb, nq, busy = False, 0, 0, 0.0
        while not stop:
            first = req_q.get()
            if first is None:
                break
            batch = [first]
            deadline = time.time() + args.batch_wait
            while len(batch) < args.max_batch:
                try:
                    r = req_q.get(timeout=max(0.0, deadline - time.time()))
                except queue.Empty:
                    break
                if r is None:
                    stop = True; break
                batch.append(r)
            t0 = time.perf_counter()
            with torch.no_grad():
                n = len(batch)
                px = torch.from_numpy(np.stack(list(cpu_pool.map(lambda r: _normalize_image(r["image"]), batch)))).to("cuda", dt)
                if graphs_ok:
                    g, px_s, f_out = enc_g[bucket(n)]
                    px_s[:n].copy_(px); g.replay()
                    feats = [f[:n] for f in f_out]
                else:
                    feats = encode(px)
                for s in sorted({r["suite"] for r in batch}):
                    idx = [i for i, r in enumerate(batch) if r["suite"] == s]
                    k = len(idx)
                    it = torch.tensor(idx, device="cuda")
                    f_s = [f.index_select(0, it) for f in feats]
                    st = torch.from_numpy(np.stack([batch[i]["state"] for i in idx]))[:, None].to("cuda", dt)
                    tc = torch.stack([task_cond(s, batch[i]["task_dir"]) for i in idx])
                    x = torch.stack([noise(batch[i]["seed"]) for i in idx])
                    if graphs_ok:
                        g2, f_in, st_s, tc_s, x_s, a_out = dit_g[(s, bucket(k))]
                        for fi, f in zip(f_in, f_s):
                            fi[:k].copy_(f)
                        st_s[:k].copy_(st); tc_s[:k].copy_(tc); x_s[:k].copy_(x)
                        g2.replay()
                        act = a_out[:k].float().cpu().numpy()
                    else:
                        act = sample(infs[s].model, f_s, st, tc, x).float().cpu().numpy()
                    for j, i in enumerate(idx):
                        a = R.bspline_smooth(act[j], degree=3, num_ctrl_pts=8) if smooth[s] else act[j]
                        resp_qs[batch[i]["worker"]].put(np.asarray(a, dtype=np.float32))
            busy += time.perf_counter() - t0; nb += 1; nq += len(batch)
            if nb % 500 == 0:
                ready_q.put(dict(event="server_stats", server=sidx, batches=nb, queries=nq, mean_batch=round(nq / nb, 2),
                                 busy_ms_per_batch=round(1000 * busy / nb, 1)))
        ready_q.put(dict(event="server_done", server=sidx, batches=nb, queries=nq, busy_seconds=round(busy, 1)))
    except BaseException:
        ready_q.put(dict(event="error", worker=f"server{sidx}", error=traceback.format_exc()))
        raise


def env_worker(index, egl_gpu, args, cond, work, results, req_qs, suite_server, resp_q):
    os.environ["CUDA_VISIBLE_DEVICES"] = egl_gpu
    os.environ["MUJOCO_EGL_DEVICE_ID"] = egl_gpu
    os.nice(args.worker_nice)   # the latency-critical inference servers must never wait for a CPU
    for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "SLURM_PROCID", "LOCAL_WORLD_SIZE"):
        os.environ.pop(k, None)
    current_claim = None
    try:
        import numpy as np
        sys.path.insert(0, ADAPTER)
        from libero_data import libero_obs_to_inputs
        from libero.libero import benchmark
        from experiments.libero.libero_utils import get_libero_env
        from evaluate_loopwam_libero import initial_states
        from fastwam.utils.pytorch_utils import set_global_seed
        out = Path(cond["output"])
        results.put(dict(event="ready", worker=index, gpu=egl_gpu))
        while True:
            group = work.get()
            if group is None:
                break
            cp = out / "claims" / f"{group['id']}.claim"
            if (out / "groups" / f"{group['id']}.json").exists() or not M.claim(cp, args.token, out):
                results.put(dict(event="skipped", group=group["id"], worker=index)); continue
            current_claim = cp
            t0 = time.perf_counter()
            suite_name, seed, task_id = group["suite"], group["seed"], group["task"]
            set_global_seed(seed, get_worker_init_fn=False)
            task = benchmark.get_benchmark_dict()[suite_name]().get_task(task_id)
            tdir = task_dir_name(task.language)
            if not Path(f"{TASKCOND}/{suite_name}/{tdir}/task_cond.npy").exists():
                raise FileNotFoundError(f"no task vector for {suite_name}/{tdir}")
            states = initial_states(task)
            env, description = get_libero_env(task, 256, seed)
            rq = req_qs[suite_server[suite_name]]
            rows = []
            try:
                for episode in range(args.episodes):
                    started = time.perf_counter(); qsec = 0.0; queries = 0
                    env.reset()
                    cam = camera_switch(env) if args.render_skip else (lambda on: None)
                    cam(False)
                    obs = env.set_init_state(states[episode])
                    for i in range(M.SETTLING):
                        cam(i >= M.SETTLING - 2)
                        obs, _, done, _ = env.step(M.DUMMY)
                    pending, steps, done = [], 0, False
                    while steps < M.MAX_STEPS[suite_name]:
                        if not pending:
                            img, st = libero_obs_to_inputs(obs)
                            q0 = time.perf_counter()
                            rq.put(dict(worker=index, suite=suite_name, task_dir=tdir, seed=seed, image=img, state=st))
                            a = resp_q.get(timeout=900)
                            qsec += time.perf_counter() - q0; queries += 1
                            if a.shape != (CHUNK, 7) or not np.isfinite(a).all():
                                raise ValueError(f"bad action chunk {a.shape}")
                            a = a[:REPLAN].astype(np.float64).copy()
                            a[:, -1] = np.sign(a[:, -1])
                            pending = a.tolist()
                        cam(len(pending) <= 2)
                        obs, _, done, _ = env.step(pending.pop(0))
                        steps += 1
                        if args.stop_at and time.time() > args.stop_at:
                            raise StopRun()
                        if done:
                            break
                    cam(True)   # real sensors back before the next env.reset()
                    rows.append(dict(id=group["id"], suite=suite_name, seed=seed, task=task_id, episode=episode,
                                     initial_state_index=episode, success=bool(done), policy_steps=steps, queries=queries,
                                     query_seconds=qsec, total_seconds=time.perf_counter() - started, video=None))
            finally:
                env.close()
            record = dict(group=group, task_description=description, checkpoint_sha256=cond["sha256"],
                          backend=cond["backend"], worker=index, gpu=egl_gpu, token=args.token,
                          elapsed_seconds=time.perf_counter() - t0, episodes=rows,
                          successes=sum(r["success"] for r in rows), completed_at=time.time())
            M.write_json(out / "groups" / f"{group['id']}.json", record)
            cp.unlink(missing_ok=True); current_claim = None
            results.put(dict(event="group", group=group["id"], successes=record["successes"], episodes=len(rows),
                             worker=index, seconds=round(record["elapsed_seconds"], 1),
                             policy_steps=sum(r["policy_steps"] for r in rows)))
        results.put(dict(event="done", worker=index))
    except StopRun:
        if current_claim is not None:
            Path(current_claim).unlink(missing_ok=True)   # unfinished group: free it for the next runner
        results.put(dict(event="done", worker=index, stopped=True))
    except BaseException:
        if current_claim is not None:
            Path(current_claim).unlink(missing_ok=True)   # let a retry take this group immediately
        results.put(dict(event="error", worker=index, error=traceback.format_exc()))
        raise


def build_condition(args):
    cfgs, shas = {}, {}
    for s in args.suites.split(","):
        d = sorted(Path(args.train_root, s, "stage2").glob("sft_*"))[-1]
        ck = d / "checkpoint_epoch_2.pt"
        if not ck.exists():
            raise SystemExit(f"missing final checkpoint for {s}: {ck}")
        cfgs[s] = dict(config=str(d / "config.yaml"), checkpoint=str(ck),
                       stats=f"/groups/yshang/an221229/data/LiLaWAM_libero_ours_stats_{s}.json",
                       hf_path=f"baselines/LiLa-WAM_LIBERO_ours/{s}/stage2_epoch2_final.pt")
        shas[s] = M.sha256_file(ck)
        cfgs[s]["sha256"] = shas[s]; cfgs[s]["stats_sha256"] = M.sha256_file(cfgs[s]["stats"])
    combined = hashlib.sha256(json.dumps(shas, sort_keys=True).encode()).hexdigest()
    return dict(run=args.run, output=args.output_dir, checkpoint="4 per-suite policies: " + ", ".join(
        f"{s}={c['hf_path']}" for s, c in cfgs.items()), sha256=combined, step="stage 2 epoch 2 (per suite)",
        backend="batched-bf16-server", suite_checkpoints=cfgs)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="lilawam_libero_ours_paper_recipe")
    p.add_argument("--train-root", default="/groups/yshang/an221229/checkpoints/wam_baselines/lilawam/libero_ours_paper_recipe")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    p.add_argument("--env-workers", type=int, default=8)
    p.add_argument("--max-batch", type=int, default=64)
    p.add_argument("--batch-wait", type=float, default=0.004)
    p.add_argument("--server-threads", type=int, default=4)
    p.add_argument("--worker-nice", type=int, default=10)
    p.add_argument("--no-cuda-graphs", dest="cuda_graphs", action="store_false")
    p.add_argument("--no-render-skip", dest="render_skip", action="store_false",
                   help="render the cameras on every step instead of only the 2 steps before each query")
    p.add_argument("--egl-gpus", default="", help="GPUs used for simulator rendering (default: --gpus); keeping "
                   "rendering off the inference GPUs avoids context time-slicing with the servers")
    p.add_argument("--suites", default=",".join(M.SUITES))
    p.add_argument("--seeds", default=",".join(map(str, M.SEEDS)))
    p.add_argument("--tasks", default=",".join(map(str, range(10))))
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--order", default="long-first", choices=("long-first", "short-first"),
                   help="group scheduling order only; per-group seeding and outcomes do not depend on it")
    p.add_argument("--stop-at", type=float, default=0, help="unix time: abandon unfinished groups and exit cleanly")
    args = p.parse_args()
    args.token = f"{os.uname().nodename}-{os.environ.get('SLURM_JOB_ID', 'nojob')}-{os.getpid()}-{int(time.time())}"
    suites = args.suites.split(","); seeds = [int(x) for x in args.seeds.split(",")]
    tasks = [int(x) for x in args.tasks.split(",")]
    cond = build_condition(args)
    import torch
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    beat = lambda: M.write_json(out / "runners" / f"{args.token}.json", dict(time=time.time(), token=args.token))
    beat()
    groups = M.groups_for(suites, seeds, tasks)
    manifest = dict(cond, protocol=dict(name="lightwam_20261009_matched", suites=suites, seeds=seeds, tasks=tasks,
        episodes=args.episodes, max_policy_steps=M.MAX_STEPS, settling_steps=M.SETTLING, settling_action=M.DUMMY,
        settling_done="ignored", replan_steps=REPLAN, action_horizon=CHUNK, denoising_steps=10,
        seed_rule="global seed once per task/seed group; env.seed(base) at construction, env reused for 50 sequential "
        "initial states; every query starts from the same noise drawn from a CPU generator seeded with the base seed",
        success="environment done flag", precision="bf16 (public LiLa-WAM inference dtype)", camera=256,
        model_image="[agentview | wrist] 2x256x256 side by side (512x256), both rotated 180 deg",
        state="eef pos (3), axis-angle (3), gripper qpos (2)", smoothing="public B-spline (degree 3, 8 control points)",
        gripper="sign(raw predicted gripper); policy trained on raw LIBERO gripper (-1 open, +1 close)",
        videos=False), runner_sha256=M.sha256_file(__file__),
        adapter_sha256=M.sha256_file(Path(ADAPTER) / "libero_data.py"), host=os.uname().nodename,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"), mujoco_gl=os.environ.get("MUJOCO_GL"), torch=torch.__version__,
        gpus=args.gpus, env_workers=args.env_workers, max_batch=args.max_batch, batch_wait=args.batch_wait,
        cuda_graphs_requested=args.cuda_graphs, egl_gpus=args.egl_gpus or args.gpus, render_skip=args.render_skip)
    mpath = out / "manifest.json"
    if mpath.exists():
        old = json.loads(mpath.read_text())
        for k in ("sha256", "protocol", "runner_sha256", "adapter_sha256"):
            if old[k] != manifest[k]:
                raise SystemExit(f"Resume refused: {k} differs")
    else:
        M.write_json(mpath, manifest)
    todo = [g for g in groups if not (out / "groups" / f"{g['id']}.json").exists()]
    if args.order == "short-first":
        todo = [g for g in todo if g["suite"] != "libero_10"] + [g for g in todo if g["suite"] == "libero_10"]
    t_start = time.time()
    print(json.dumps(dict(event="plan", run=args.run, pending_groups=len(todo), total_groups=len(groups),
                          gpus=args.gpus, env_workers=args.env_workers, token=args.token)), flush=True)
    if todo:
        gpus = args.gpus.split(",")
        used = [s for s in SERVER_ORDER if s in suites]
        suite_server = {s: i % len(gpus) for i, s in enumerate(used)}
        nworkers = min(args.env_workers, len(todo))
        ctx = mp.get_context("spawn")
        work, results = ctx.Queue(), ctx.Queue()
        req_qs = [ctx.Queue() for _ in gpus]
        resp_qs = [ctx.Queue() for _ in range(nworkers)]
        cfgs = cond["suite_checkpoints"]
        servers = [ctx.Process(target=server, args=(i, g, [s for s in used if suite_server[s] == i], cfgs, req_qs[i],
                                                       resp_qs, results, args)) for i, g in enumerate(gpus)
                   if any(suite_server[s] == i for s in used)]
        for sp in servers: sp.start()
        for g in todo: work.put(g)
        for _ in range(nworkers): work.put(None)
        egl = (args.egl_gpus or args.gpus).split(",")
        procs = [ctx.Process(target=env_worker, args=(i, egl[i % len(egl)], args, cond, work, results, req_qs,
                                                       suite_server, resp_qs[i])) for i in range(nworkers)]
        for pr in procs: pr.start()
        finished = done_workers = errors = eps = steps = 0; last_beat = time.time()
        while done_workers + errors < nworkers:
            if time.time() - last_beat > 60:
                beat(); last_beat = time.time()
            if any(not sp.is_alive() for sp in servers) and done_workers + errors < nworkers:
                print("server process died", file=sys.stderr, flush=True)
            try:
                msg = results.get(timeout=60)
            except queue.Empty:
                continue
            if msg["event"] == "skipped":
                finished += 1; continue
            if msg["event"] == "group":
                finished += 1; eps += msg["episodes"]; steps += msg["policy_steps"]
                el = time.time() - t_start
                print(json.dumps(dict(msg, finished=finished, pending=len(todo) - finished, time=time.strftime("%H:%M:%S"),
                                      eps_per_hour=round(eps / el * 3600), policy_steps_per_s=round(steps / el, 1))),
                      flush=True)
            elif msg["event"] == "error":
                errors += 1 if str(msg["worker"]).isdigit() or isinstance(msg["worker"], int) else 0
                print(msg["error"], file=sys.stderr, flush=True)
                if not isinstance(msg["worker"], int):   # a server failed: nothing can finish
                    for pr in procs: pr.terminate()
                    errors = nworkers - done_workers
            elif msg["event"] == "done":
                done_workers += 1
                if msg.get("stopped"):
                    print(json.dumps(dict(event="worker_stopped", worker=msg["worker"], time=time.strftime("%H:%M:%S"))), flush=True)
            else:
                print(json.dumps(msg), flush=True)
        for q in req_qs: q.put(None)
        for pr in procs: pr.join(timeout=60)
        for sp in servers: sp.join(timeout=60)
        while True:
            try:
                msg = results.get(timeout=2); print(json.dumps(msg) if "error" not in msg else msg["error"], flush=True)
            except queue.Empty:
                break
        M.write_json(out / "runs" / f"{args.token}.json", dict(token=args.token, started=t_start, finished=time.time(),
                     wall_seconds=time.time() - t_start, gpus=args.gpus, env_workers=nworkers, groups_completed=finished,
                     errors=errors, slurm_job_id=os.environ.get("SLURM_JOB_ID")))
        if errors:
            raise SystemExit(f"{errors} worker(s) failed")
    s = M.summarize(cond, groups, args.episodes)
    print(json.dumps(dict(event="summary", complete=s is not None, success_rate=s and s["success_rate"],
                          episodes=s and s["episodes"])), flush=True)


if __name__ == "__main__":
    main()
