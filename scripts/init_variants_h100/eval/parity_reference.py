#!/usr/bin/env python3
"""Harness parity: the released v0-video4-action4 policy (the campaign's "LoopWAM full 4/4", export 1c659a7c) through
the unchanged evaluate_v0_matched.py worker on LIBERO-Long seed 42 (10 task groups x 50 initial states), compared
episode by episode with the campaign's own outcomes in evaluation_report/final/episodes.csv. Unscored for the init
variants. Text caches: this cluster's regenerated files (same names and valid-token counts as the release).
"""
import argparse, csv, json, multiprocessing as mp, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_v0_matched as ev

REFERENCE = Path("/groups/yshang/an221229/checkpoints/ChronoLoop/parent/libero-all-suites/v0-video4-action4")
EXPORT_SHA = "1c659a7c2f6970c3ea5972a8fba4abd9db35857c83a67a053c0ee35555fb602b"
CAMPAIGN = Path("/lustre/fs1/home/an221229/code/LoopWAM_v042/evaluation_report/final/episodes.csv")
CLUSTER_DATA = Path("/groups/yshang/an221229/checkpoints/LoopWAM_NT/v0_init_variants/core_random/train/data/data_manifest.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True); p.add_argument("--vae-path", required=True)
    p.add_argument("--text-cache-dir", required=True); p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--seed", type=int, default=42); p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    sha = ev.sha256_file(REFERENCE / "policy.pt")
    if sha != EXPORT_SHA:
        raise SystemExit(f"reference export sha {sha} != {EXPORT_SHA}")
    vae = ev.sha256_file(args.vae_path)
    if vae != "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981":
        raise SystemExit("VAE differs from the campaign VAE")
    release = json.loads((REFERENCE / "data_manifest.json").read_text())
    cluster = json.loads(CLUSTER_DATA.read_text())["text_cache_files"]
    if ({k: v.get("valid_tokens") for k, v in release["text_cache_files"].items()}
            != {k: v.get("valid_tokens") for k, v in cluster.items()}):
        raise SystemExit("Cluster text caches differ from the release in names or valid tokens")
    release["text_cache_files"] = cluster
    ev.write_json(out / "data_manifest_cluster_text.json", release)
    cond = dict(run="reference_v0_video4_action4", checkpoint=str(REFERENCE / "policy.pt"), sha256=sha, step=21700,
                stats=str(REFERENCE / "dataset_stats.json"), data_manifest=str(out / "data_manifest_cluster_text.json"),
                vae_path=args.vae_path, vae_sha256=vae, text_cache_dir=args.text_cache_dir, backend="cuda-graph",
                output=str(out))
    args.token = f"{os.uname().nodename}-{os.environ.get('SLURM_JOB_ID', 'nojob')}-{os.getpid()}-{int(time.time())}"
    args.videos = False; args.episodes = 50; args.gate_queries = 4; args.gate_output = str(out / "gate.json")
    ev.write_json(out / "runners" / f"{args.token}.json", dict(time=time.time(), token=args.token))
    # Same CUDA-graph gate as production, on this checkpoint.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = args.gpus.split(",")[0]
    ev.gate(args, cond)
    groups = [g for g in ev.groups_for(["libero_10"], [args.seed], range(10))
              if not (out / "groups" / f"{g['id']}.json").exists()]
    t0 = time.time()
    if groups:
        gpus = args.gpus.split(","); ctx = mp.get_context("spawn"); work = ctx.Queue(); results = ctx.Queue()
        for g in groups: work.put(g)
        for _ in groups: work.put(None)
        procs = [ctx.Process(target=ev.worker, args=(i, gpus[i % len(gpus)], args, cond, work, results))
                 for i in range(len(groups))]
        for pr in procs: pr.start()
        ended = 0
        while ended < len(procs):
            msg = results.get()
            if msg["event"] in ("done", "error"):
                ended += 1
            if msg["event"] != "ready":
                print(json.dumps(msg), flush=True)
        for pr in procs: pr.join()
    ours = {}
    for f in (out / "groups").glob("*.json"):
        for r in json.loads(f.read_text())["episodes"]:
            ours[(r["task"], r["episode"])] = r
    theirs = {(int(r["task"]), int(r["episode"])): r for r in csv.DictReader(CAMPAIGN.open())
              if r["model"] == "LoopWAM full 4/4" and r["suite"] == "libero_10" and int(r["seed"]) == args.seed}
    keys = sorted(theirs)
    if sorted(ours) != keys:
        raise SystemExit(f"episode keys differ: ours {len(ours)} campaign {len(keys)}")
    s_ours = sum(ours[k]["success"] for k in keys); s_theirs = sum(theirs[k]["success"] == "True" for k in keys)
    agree = sum(ours[k]["success"] == (theirs[k]["success"] == "True") for k in keys)
    steps_equal = sum(ours[k]["policy_steps"] == int(theirs[k]["policy_steps"]) for k in keys)
    per_task = {t: dict(h100=sum(ours[(t, e)]["success"] for e in range(50)),
                        campaign=sum(theirs[(t, e)]["success"] == "True" for e in range(50))) for t in range(10)}
    report = dict(reference=str(REFERENCE), export_sha256=sha, suite="libero_10", seed=args.seed, episodes=len(keys),
                  h100_successes=s_ours, campaign_successes=s_theirs, h100_success_rate=s_ours / len(keys),
                  campaign_success_rate=s_theirs / len(keys), outcome_agreement=agree / len(keys),
                  identical_policy_steps=steps_equal / len(keys), per_task=per_task,
                  wall_seconds=time.time() - t0, runner_sha256=ev.sha256_file(ev.__file__))
    ev.write_json(out / "parity.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "per_task"}), flush=True)


if __name__ == "__main__":
    main()
