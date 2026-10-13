#!/usr/bin/env python3
"""RT policy latency: batch 1, one CUDA graph per schedule, cached text K/V, 500 replayed raw observations.

Timing covers raw LIBERO observation -> preprocessing -> inference -> denormalized host action.
Run alone on an otherwise idle GPU.
"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", required=True, help="RUN:SCHEDULE,...")
    p.add_argument("--weights-root", required=True); p.add_argument("--vae-path", required=True)
    p.add_argument("--text-cache-dir", required=True); p.add_argument("--observations", type=int, default=500)
    p.add_argument("--warmup", type=int, default=20); p.add_argument("--output", required=True)
    a = p.parse_args()
    import numpy as np, torch
    from scripts.evaluate_loopwam_libero import ObservationAdapter, initial_states
    from fastwam.models.wan22.transport_inference import TransportInference
    from fastwam.models.wan22.residual_transport import SCHEDULES
    from libero.libero import benchmark
    from experiments.libero.libero_utils import get_libero_env
    # Record raw observations once (scripted settle + random small motions across suites), then replay for every model.
    observations = []; rng = np.random.default_rng(0)
    per = a.observations // 4
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        task = benchmark.get_benchmark_dict()[suite]().get_task(0)
        env, _ = get_libero_env(task, 256, 42); env.reset(); obs = env.set_init_state(initial_states(task)[0])
        for i in range(per):
            obs, *_ = env.step(list(rng.uniform(-0.3, 0.3, 6)) + [-1.0])
            observations.append((task.language, {k: np.array(v, copy=True) for k, v in obs.items()}))
        env.close()
    results = {}
    loaded = None
    for spec in a.conditions.split(","):
        run, sched = spec.split(":")
        if loaded != run:
            model = None; torch.cuda.empty_cache()
            folder = Path(a.weights_root) / run
            model = TransportInference(str(folder / "epoch_10.pt"), a.vae_path, device="cuda:0").eval()
            adapter = ObservationAdapter(json.loads((folder / "dataset_stats.json").read_text()), a.text_cache_dir, None)
            loaded = run
        model.schedule = sched
        texts = {lang: adapter.text(lang) for lang in {o[0] for o in observations}}
        times = []
        for i, (lang, obs) in enumerate(observations[:a.warmup] + observations):
            context, mask = texts[lang]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            image, proprio = adapter.observation(obs)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                act = model.infer_action(input_image=image, proprio=proprio, context=context, context_mask=mask,
                    action_horizon=32, num_inference_steps=len(SCHEDULES[sched]), text_cfg_scale=1.0, seed=42,
                    compile_action_infer=True)["action"]
            host = adapter.libero_actions(act)
            dt = time.perf_counter() - t0
            if i >= a.warmup: times.append(dt * 1e3)
        results[f"{run} {sched}"] = dict(p50_ms=float(np.percentile(times, 50)), p95_ms=float(np.percentile(times, 95)),
                                         mean_ms=float(np.mean(times)), n=len(times))
        print(json.dumps({f"{run} {sched}": results[f"{run} {sched}"]}), flush=True)
    Path(a.output).write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(0), batch=1, backend="cuda-graph",
        text="cached T5 context", observations=len(observations), timing="raw observation to denormalized host action",
        results=results), indent=1))


if __name__ == "__main__":
    main()
