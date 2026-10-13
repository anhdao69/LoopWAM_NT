#!/usr/bin/env python3
"""Unscored preflight: CUDA-graph vs eager RT actions on real LIBERO observations (must be bit-exact)."""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True); p.add_argument("--schedules", required=True)
    p.add_argument("--weights-root", required=True); p.add_argument("--vae-path", required=True)
    p.add_argument("--text-cache-dir", required=True); p.add_argument("--queries", type=int, default=12)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    import numpy as np, torch
    from scripts.evaluate_loopwam_libero import ObservationAdapter, initial_states
    from fastwam.models.wan22.transport_inference import TransportInference
    from fastwam.models.wan22.residual_transport import SCHEDULES
    from libero.libero import benchmark
    from experiments.libero.libero_utils import get_libero_env
    folder = Path(a.weights_root) / a.run
    model = TransportInference(str(folder / "epoch_10.pt"), a.vae_path, device="cuda:0").eval()
    adapter = ObservationAdapter(json.loads((folder / "dataset_stats.json").read_text()), a.text_cache_dir, None)
    report = []
    for suite_name, task_id, seed in (("libero_10", 0, 42), ("libero_spatial", 3, 43)):
        task = benchmark.get_benchmark_dict()[suite_name]().get_task(task_id)
        context, mask = adapter.text(task.language)
        env, _ = get_libero_env(task, 256, seed)
        env.reset(); obs = env.set_init_state(initial_states(task)[0])
        for _ in range(30): obs, *_ = env.step([0, 0, 0, 0, 0, 0, -1])
        for q in range(a.queries):
            image, proprio = adapter.observation(obs)
            out = {}
            for sched in a.schedules.split(","):
                model.schedule = sched
                for backend in ("eager", "graph"):
                    t0 = time.perf_counter()
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        act = model.infer_action(input_image=image, proprio=proprio, context=context, context_mask=mask,
                            action_horizon=32, num_inference_steps=len(SCHEDULES[sched]), text_cfg_scale=1.0,
                            seed=seed, compile_action_infer=backend == "graph")["action"]
                    torch.cuda.synchronize(); out[(sched, backend)] = (act.clone(), time.perf_counter() - t0)
                d = (out[(sched, "eager")][0] - out[(sched, "graph")][0]).abs().max().item()
                report.append(dict(suite=suite_name, seed=seed, query=q, schedule=sched, max_abs_difference=d,
                                   bit_exact=bool(torch.equal(out[(sched, "eager")][0], out[(sched, "graph")][0])),
                                   eager_seconds=out[(sched, "eager")][1], graph_seconds=out[(sched, "graph")][1]))
            # advance the environment with the first schedule's eager actions to get fresh observations
            chunk = adapter.libero_actions(out[(a.schedules.split(",")[0], "eager")][0])
            for act in chunk[:10]:
                obs, _, done, _ = env.step(act.tolist())
        env.close()
    ok = all(r["bit_exact"] for r in report)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(dict(run=a.run, schedules=a.schedules.split(","), all_bit_exact=ok,
                                              queries=report), indent=1))
    print(json.dumps(dict(run=a.run, all_bit_exact=ok, n=len(report),
                          eager_ms=np.median([r["eager_seconds"] for r in report]) * 1e3,
                          graph_ms=np.median([r["graph_seconds"] for r in report if r["query"] > 0]) * 1e3)))


if __name__ == "__main__":
    main()
