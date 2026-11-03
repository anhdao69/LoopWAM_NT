#!/usr/bin/env python3
"""Set up and check the CL-0 matched LIBERO evaluation environment (run on worker-2 with eval_env.sh sourced).

1. Writes the private LIBERO config ($LIBERO_CONFIG_PATH/config.yaml) pointing at $LIBERO_ROOT/libero/libero.
2. Prompt cache: every file of the training manifest's text_cache_files exists in $EVAL_TEXT with the recorded
   sha256; the three CL0 runs' data manifests are identical; the reference BASE10 E0 run has the same prompt files
   and valid-token counts (its bytes, regenerated on the other cluster, are counted, not required to match).
3. VAE sha256 equals every run's config.json asset hash.
4. LIBERO: the four suites have 10 tasks each, every task has >= 50 official initial states, bddl/init files exist.
5. Simulator smoke on each GPU given (GPU EGL, as the reference): build the runner's environment (get_libero_env, 256 px, env.seed), 30 settling
   steps, observation adapter output shapes (3x224x448 image in [-1,1], 8-D normalized state).
Writes one JSON report and exits non-zero on any failure.
"""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(os.environ["ELASTIC_ROOT"]); sys.path[:0] = [str(ROOT), str(ROOT / "src")]
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def smoke(gpu):
    """Child process: one environment on GPU `gpu` (as the runner's workers set it up)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu; os.environ["MUJOCO_EGL_DEVICE_ID"] = gpu
    import numpy as np, torch
    from libero.libero import benchmark
    from experiments.libero.libero_utils import get_libero_env
    from scripts.evaluate_loopwam_libero import ObservationAdapter, initial_states
    run = Path(os.environ["EVAL_RUNS"]) / "CL0-E0"
    data = json.loads((run / "data" / "data_manifest.json").read_text())
    adapter = ObservationAdapter(json.loads((run / "dataset_stats.json").read_text()), os.environ["EVAL_TEXT"],
                                 data["text_cache_files"])
    task = benchmark.get_benchmark_dict()["libero_10"]().get_task(0)
    context, mask = adapter.text(task.language)
    env, description = get_libero_env(task, 256, 42)
    env.reset(); obs = env.set_init_state(initial_states(task)[0])
    for _ in range(30):
        obs, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    image, state = adapter.observation(obs)
    from mujoco.egl import egl_ext as EGL
    egl_devices = len(EGL.eglQueryDevicesEXT())
    env.close()
    out = dict(gpu=gpu, description=description, image=list(image.shape), image_min=float(image.min()),
               image_max=float(image.max()), state=list(state.shape), context=list(context.shape),
               valid_tokens=int(mask.sum()), image_sum=float(image.double().sum()), egl_devices=egl_devices,
               mujoco_gl=os.environ.get("MUJOCO_GL"))
    print("SMOKE " + json.dumps(out), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference-e0", required=True, help="reference BASE10 E0 run folder (HF download)")
    p.add_argument("--gpus", default="0,1,2,3"); p.add_argument("--out", required=True)
    p.add_argument("--smoke-child", default=None)
    a = p.parse_args()
    if a.smoke_child is not None:
        return smoke(a.smoke_child)
    rep = dict(checks={}); fail = []
    def check(name, ok, **kw):
        rep["checks"][name] = dict(passed=bool(ok), **kw); print(json.dumps({name: rep["checks"][name]}, default=str), flush=True)
        if not ok: fail.append(name)
    # 1. LIBERO config
    lib = Path(os.environ["LIBERO_ROOT"]) / "libero" / "libero"
    cfg = dict(benchmark_root=str(lib), bddl_files=str(lib / "bddl_files"), init_states=str(lib / "init_files"),
               datasets=str(lib.parent / "datasets"), assets=str(lib / "assets"))
    cdir = Path(os.environ["LIBERO_CONFIG_PATH"]); cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "config.yaml").write_text("".join(f"{k}: {v}\n" for k, v in cfg.items()))
    check("libero_config", all(Path(cfg[k]).is_dir() for k in ("benchmark_root", "bddl_files", "init_states", "assets")),
          config=cfg)
    # 2. prompt cache
    runs = Path(os.environ["EVAL_RUNS"]); text = Path(os.environ["EVAL_TEXT"])
    manifests = {r: json.loads((runs / r / "data" / "data_manifest.json").read_text()) for r in ("CL0-E0", "CL0-E1", "CL0-E2")}
    ref = json.loads((Path(a.reference_e0) / "data" / "data_manifest.json").read_text())
    files = manifests["CL0-E0"]["text_cache_files"]
    bad = [n for n, v in files.items() if not (text / n).is_file() or sha256_file(text / n) != v["sha256"]]
    check("prompt_cache_bytes", not bad and len(files) == 40, files=len(files), mismatched=bad)
    check("cl0_manifests_identical", manifests["CL0-E0"] == manifests["CL0-E1"] == manifests["CL0-E2"])
    # The BASE10 elastic runs were trained and evaluated on another cluster with a regenerated prompt cache: same
    # prompt files and valid-token counts, different bytes. Each model is evaluated with its own training bytes
    # (--verify-text-hash); this records how the two caches relate.
    rt = ref["text_cache_files"]
    check("prompts_equal_reference_e0", set(rt) == set(files) and all(rt[k]["valid_tokens"] == files[k]["valid_tokens"] for k in files)
          and ref["normalization_sha256"] == manifests["CL0-E0"]["normalization_sha256"],
          reference_text_dir=ref.get("text_cache_dir"), files_with_different_bytes=sum(rt[k]["sha256"] != files[k]["sha256"] for k in files))
    # 3. VAE
    vae = sha256_file(os.environ["ELASTIC_VAE"])
    cfgs = {r: json.loads((runs / r / "config.json").read_text())["asset_sha256"]["vae"] for r in manifests}
    check("vae", all(v == vae for v in cfgs.values()), vae_sha256=vae)
    # 4. LIBERO suites / init states
    from libero.libero import benchmark
    from scripts.evaluate_loopwam_libero import initial_states
    counts = {}
    for s in SUITES:
        suite = benchmark.get_benchmark_dict()[s]()
        counts[s] = [len(initial_states(suite.get_task(t))) for t in range(suite.n_tasks)]
    check("libero_suites", all(len(v) == 10 and min(v) >= 50 for v in counts.values()), init_states=counts)
    # 5. simulator smoke per GPU (separate processes, as the runner's spawned workers)
    smokes = {}
    for g in a.gpus.split(","):
        r = subprocess.run([sys.executable, __file__, "--reference-e0", a.reference_e0, "--out", a.out, "--smoke-child", g],
                           capture_output=True, text=True, timeout=900)
        line = [l for l in r.stdout.splitlines() if l.startswith("SMOKE ")]
        smokes[g] = json.loads(line[-1][6:]) if line else dict(error=(r.stderr or r.stdout)[-2000:])
    ok = all("error" not in v and v["mujoco_gl"] == "egl" and v["image"] == [3, 224, 448] and v["state"] == [8] and -1 <= v["image_min"]
             and v["image_max"] <= 1 for v in smokes.values())
    sums = {v.get("image_sum") for v in smokes.values()}
    # Device isolation: the job's cgroup must hide the GR1 job's GPUs from EGL (only this job's four are enumerable).
    try:
        open("/dev/nvidia4", "rb").close(); hidden = False
    except OSError:
        hidden = True
    check("gpu_isolation", hidden and all(v.get("egl_devices") == len(a.gpus.split(",")) for v in smokes.values()),
          dev_nvidia4_blocked=hidden, egl_devices=[v.get("egl_devices") for v in smokes.values()])
    check("simulator_smoke", ok and len(sums) == 1, per_gpu=smokes, identical_render_across_gpus=len(sums) == 1)
    rep["passed"] = not fail
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    print("SETUP PASSED" if not fail else f"SETUP FAILED: {fail}")
    sys.exit(0 if not fail else 1)


if __name__ == "__main__":
    main()
