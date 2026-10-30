#!/usr/bin/env python3
"""Record one fixed set of raw LIBERO observations for the all-model latency benchmark.

Same recipe as scripts/measure_rt_latency.py (task 0 of each suite, seed 42, first initial state, then small random
end-effector motions from a fixed NumPy seed), but written once to a pickle so every model and every code branch
replays byte-identical inputs. CPU only (MuJoCo EGL rendering).
"""
import argparse, hashlib, pickle, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-suite", type=int, default=125)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    import numpy as np
    from scripts.evaluate_loopwam_libero import initial_states
    from libero.libero import benchmark
    from experiments.libero.libero_utils import get_libero_env
    rng = np.random.default_rng(0)
    records = []
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        task = benchmark.get_benchmark_dict()[suite]().get_task(0)
        env, _ = get_libero_env(task, 256, 42)
        env.reset()
        obs = env.set_init_state(initial_states(task)[0])
        for _ in range(a.per_suite):
            obs, *_ = env.step(list(rng.uniform(-0.3, 0.3, 6)) + [-1.0])
            records.append(dict(suite=suite, language=task.language,
                                obs={k: np.array(v, copy=True) for k, v in obs.items()}))
        env.close()
    blob = pickle.dumps(records, protocol=4)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_bytes(blob)
    print(f"{len(records)} observations, sha256 {hashlib.sha256(blob).hexdigest()}, {len(blob) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
