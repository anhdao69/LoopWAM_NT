#!/usr/bin/env python
"""Batch-one LIBERO latency of a public WAM release, timed through the release's own evaluation code.

Each query is exactly one call of the repository's `experiments/libero/eval_libero_single._predict_action_chunk`:
raw simulator observation -> the repo's own preprocessing -> its native inference method -> denormalized,
gripper-converted host actions. The model is built the way that repo's `eval_single_process` builds it
(Hydra `sim_libero` config, `instantiate(cfg.model)`, its `_load_model_checkpoint`, its processor and dataset stats),
with no optimization. Observations: the 500 recorded raw LIBERO observations of the all-models latency sweep
(observations_500.pkl, sha256 5815c6fd...). torch.cuda.synchronize() runs before each clock start; the clock stops
when the host action array exists.

Prompt modes:
  native        what the repo's LIBERO evaluation does. FastWAM/FasterWAM re-encode the prompt with T5 on every
                query; Light-WAM pre-encodes it once per task (its default EVALUATION.preencode_prompt_context=true)
                and passes the cached context into every query.
  prompt_cached the T5 prompt context is computed once per task outside the clock (as in our matched protocol).
                For FastWAM/FasterWAM, model.encode_prompt is memoized per prompt string: same tensors, so the actions
                must be bit-identical to native (checked). For Light-WAM, cached is the same as native.

Usage: measure_public_latency.py --repo <root> --name <label> --ckpt <pt> --stats <json> --out <json>
       [--task <hydra task>] [--override k=v ...] [--modes native,prompt_cached] [--n 500] [--warmup 20]
"""
import argparse, hashlib, inspect, json, os, pickle, platform, socket, subprocess, sys, time
from pathlib import Path

OBS = "/groups/yshang/an221229/checkpoints/LoopWAM_NT/latency_20261030/observations_500.pkl"
OBS_SHA = "5815c6fdc22d2dfe0c241f89c687a6f2e83804f9bd298332867258eb7160e0f0"


def stats(ms):
    import numpy as np
    a = np.asarray(ms)
    return dict(n=len(a), mean_ms=float(a.mean()), median_ms=float(np.median(a)), p5_ms=float(np.percentile(a, 5)),
                p95_ms=float(np.percentile(a, 95)), std_ms=float(a.std()), min_ms=float(a.min()),
                max_ms=float(a.max()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True); p.add_argument("--name", required=True)
    p.add_argument("--ckpt", required=True); p.add_argument("--stats", required=True)
    p.add_argument("--out", required=True); p.add_argument("--task", default=None)
    p.add_argument("--config-name", default="sim_libero")
    p.add_argument("--override", action="append", default=[])
    p.add_argument("--modes", default="native,prompt_cached")
    p.add_argument("--n", type=int, default=500); p.add_argument("--warmup", type=int, default=20)
    a = p.parse_args()

    repo = Path(a.repo).resolve()
    raw = Path(OBS).read_bytes()
    if hashlib.sha256(raw).hexdigest() != OBS_SHA:
        raise SystemExit("observations file hash mismatch")
    records = pickle.loads(raw)[: a.n]

    os.chdir(repo)  # the repo's relative paths (configs, checkpoints/) resolve as in its own eval scripts
    sys.path[:0] = [str(repo / "src"), str(repo), str(repo / "experiments" / "libero")]  # as when run as a script
    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    import experiments.libero.eval_libero_single as E

    overrides = [f"ckpt={a.ckpt}", f"EVALUATION.dataset_stats_path={a.stats}",
                 f"EVALUATION.output_dir={Path(a.out).parent / ('evalout_' + Path(a.out).stem)}"] + a.override
    if a.task:
        overrides.insert(0, f"task={a.task}")
    with initialize_config_dir(config_dir=str(repo / "configs"), version_base=None):
        cfg = compose(config_name=a.config_name, overrides=overrides)
    if hasattr(E, "_maybe_apply_training_run_config"):  # Light-WAM merges the checkpoint's training config
        cfg = E._maybe_apply_training_run_config(cfg)

    # --- model setup: eval_single_process, verbatim order ---
    if cfg.get("seed") is not None:
        E.set_global_seed(int(cfg.seed), get_worker_init_fn=False)
    device = E._resolve_eval_device(cfg)
    dtype = E._mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    t_load = time.perf_counter()
    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    E._load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(device).eval()
    load_s = time.perf_counter() - t_load
    stats_path = E._resolve_dataset_stats_path(cfg)
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(E.load_dataset_stats_from_json(str(stats_path)))
    horizon = cfg.EVALUATION.get("action_horizon", None)
    horizon = int(cfg.data.train.num_frames) - 1 if horizon is None else int(horizon)
    video_size = cfg.data.train.get("video_size", [224, 224])
    in_h, in_w = int(video_size[0]), int(video_size[1])

    sig = inspect.signature(E._predict_action_chunk).parameters
    lightwam_cache = "cached_prompt_context" in sig
    languages = sorted({r["language"] for r in records})
    pre = {}
    if lightwam_cache:  # Light-WAM's own per-task pre-encoding (text encoder offloaded to CPU afterwards)
        for lang in languages:
            pre[lang] = E._maybe_preencode_task_prompt_context(lang, model, cfg, model_device=device)

    original_encode = model.encode_prompt
    memo = {}

    def memo_encode(prompt):
        if prompt not in memo:
            memo[prompt] = original_encode(prompt)
        return memo[prompt]

    def query(record):
        kw = dict(action_horizon=horizon, input_w=in_w, input_h=in_h, model_device=device)
        if lightwam_cache:
            c, m = pre[record["language"]]
            kw.update(cached_prompt_context=c, cached_prompt_context_mask=m)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        action, _imgs, _video = E._predict_action_chunk(record["obs"], record["language"], model, processor, cfg, **kw)
        dt = (time.perf_counter() - t0) * 1e3
        return np.asarray(action), dt

    result = dict(model=a.name, repo=str(repo), repo_commit=subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                  capture_output=True, text=True).stdout.strip(), checkpoint=a.ckpt, dataset_stats=str(stats_path),
                  task=a.task, overrides=a.override, model_dtype=str(dtype), action_horizon=horizon,
                  input_hw=[in_h, in_w], num_inference_steps=cfg.EVALUATION.get("num_inference_steps"),
                  action_infer_mode=cfg.EVALUATION.get("action_infer_mode", None),
                  preencode_prompt_context=bool(cfg.EVALUATION.get("preencode_prompt_context", False)),
                  seed=cfg.get("seed"), parameters=sum(p.numel() for p in model.parameters()),
                  text_encoder_loaded=getattr(model, "text_encoder", None) is not None, load_seconds=load_s,
                  host=socket.gethostname(), gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                  python=platform.python_version(), observations_sha256=OBS_SHA, n_observations=len(records),
                  warmup=a.warmup, slurm_job=os.environ.get("SLURM_JOB_ID"), modes={})
    actions = {}
    for mode in a.modes.split(","):
        if mode == "prompt_cached" and not lightwam_cache:
            model.encode_prompt = memo_encode
            for lang in languages:  # fill the memo outside the clock
                with torch.no_grad():
                    memo_encode(E.DEFAULT_PROMPT.format(task=lang))
        elif mode == "native":
            model.encode_prompt = original_encode
        elif mode == "prompt_cached" and lightwam_cache:
            model.encode_prompt = original_encode  # Light-WAM native already uses the per-task cache
        for r in records[: a.warmup]:
            query(r)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        times, acts = [], []
        for r in records:
            act, dt = query(r)
            times.append(dt); acts.append(act)
        actions[mode] = np.stack(acts)
        result["modes"][mode] = dict(**stats(times), peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
                                     per_query_ms=[round(t, 3) for t in times])
        print(json.dumps({"model": a.name, "mode": mode, **{k: v for k, v in result["modes"][mode].items()
                          if k != "per_query_ms"}}), flush=True)
    if "native" in actions and "prompt_cached" in actions:
        same = [bool(np.array_equal(x, y)) for x, y in zip(actions["native"], actions["prompt_cached"])]
        result["prompt_cached_identical_to_native"] = f"{sum(same)}/{len(same)}"
        print("prompt_cached identical to native:", result["prompt_cached_identical_to_native"], flush=True)
    result["action_chunk_shape"] = list(actions[next(iter(actions))].shape[1:])
    Path(a.out).write_text(json.dumps(result, indent=1))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
