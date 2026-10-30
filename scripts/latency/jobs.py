#!/usr/bin/env python3
"""Job list of the all-model latency benchmark: every LIBERO policy of this repository, at the inference
configurations its reported success rates used. Writes JSON lines consumed by run_jobs.sh.

Paths: HF exports (anhdao69/LoopWAM_NT, fixed revision) are read from $LAT_HF (node-local download); unpublished
checkpoints (RT runs, elastic runs, init-variant stepwise, the released FastWAM) from /groups.
"""
import json, os, sys

CODE = "/lustre/fs1/home/an221229/code"
MAIN = f"{CODE}/LoopWAM_v042"            # origin/LoopWAM_NT 56e3328 (clean detached worktree)
KV = f"{CODE}/LoopWAM_kv"                # KV_concat_v042 8484ec2
PARTITION = f"{CODE}/LoopWAM_partition"  # origin/partition_grid_20261026 a584631
INITVAR = f"{CODE}/LoopWAM_initvar"      # v0-init-variants evaluation code 4ebf38a
RT = f"{CODE}/LoopWAM_rt"                # residual_trans
ELASTIC = f"{CODE}/LoopWAM_elastic"      # elastic
CHRONO = f"{CODE}/LoopWAM_chrono"        # chrono 43e6a2f
CK = "/groups/yshang/an221229/checkpoints"
VAE21 = f"{CK}/RollingForcing/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
VAE22 = f"{CK}/FastWAM/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
FLAT9 = [f"flat_k{k}_n{n}" for k in (1, 2, 4) for n in (2, 5, 10)]
# One model per distinct architecture + inference configuration runs first (the queue is ordered by tier); the rest
# share an architecture with a tier-1 model and should match it up to measurement noise.
TIER1 = {"FastWAM (released, libero_uncond_2cam224)", "all-suites v0 4/4", "all-suites v0 4/2", "all-suites v0 4/1",
         "all-suites v0 2/2", "all-suites v0 1/4", "all-suites Dense-S12", "all-suites Dense-S30",
         "all-suites v0 4/1 kv-mix", "long v0 4/1 kv-concat", "long v1 4/4", "long v2 4/4", "partition p0c6s6_k4",
         "RT-A", "RT-A2", "RT-B4", "RT-Pa", "RT-B2a", "elastic E0", "elastic E1", "elastic E2", "elastic SP11",
         "elastic F0-PB",
         "ChronoLoop CL-0_no-memory_a4", "ChronoLoop CL-A_mem16-learned-loopwrite_a4"}


def jobs(hf):
    out = []

    def add(name, group, family, code, checkpoint, configs=("native",), stats=None, vae=VAE21, backends=None,
            extra=None):
        out.append(dict(name=name, group=group, tier=1 if name in TIER1 else 2, family=family, code_root=code,
                        checkpoint=checkpoint,
                        configs=list(configs), stats=stats, vae=vae,
                        backends=backends or ("eager,compile,graph,fast" if family == "fastwam" else "eager,graph,fast"),
                        extra=extra or {}))

    add("FastWAM (released, libero_uncond_2cam224)", "baseline", "fastwam", MAIN,
        f"{CK}/FastWAM/fastwam_release/libero_uncond_2cam224.pt",
        stats=f"{CK}/FastWAM/fastwam_release/libero_uncond_2cam224_dataset_stats.json", vae=VAE22)
    for d, label in (("v0-video4-action4", "v0 4/4"), ("v0-video4-action2", "v0 4/2"), ("v0-video4-action1", "v0 4/1"),
                     ("v0-video2-action2", "v0 2/2"), ("v0-video1-action4", "v0 1/4"),
                     ("dense-s12", "Dense-S12"), ("dense-s30", "Dense-S30")):
        # Upstream's own accelerated path (Inductor reduce-overhead) is also timed on the v0 4/4 reference.
        add(f"all-suites {label}", "all-suites", "loopwam", MAIN, f"{hf}/libero-all-suites/{d}/policy.pt",
            backends="eager,compile,graph,fast" if d == "v0-video4-action4" else None)
    add("all-suites v0 4/1 kv-mix", "all-suites", "loopwam", KV,
        f"{hf}/libero-all-suites/v0-video4-action1-kv-mix/policy.pt")
    for v in ("core16_21", "core_mean", "core_random", "ffn_magnitude", "no_video_pretraining"):
        add(f"init variant {v}", "init-variants", "loopwam", INITVAR,
            f"{hf}/libero-all-suites/init_variants/{v}/policy.pt")
    add("init variant stepwise", "init-variants", "loopwam", INITVAR,
        f"{CK}/LoopWAM_NT/v0_init_variants/stepwise/train/latest.pt",
        stats=f"{CK}/LoopWAM_NT/v0_init_variants/stepwise/train/data/dataset_stats.json")
    for p in ("p0c6s6_k4", "p1c6s5_k4", "p5c6s1_k4", "p6c6s0_k4", "p1c9s2_k3", "p4c3s5_k7"):
        add(f"partition {p}", "partition-grid", "loopwam", PARTITION,
            f"{hf}/libero-all-suites/partition-grid-centered-20261026/{p}/policy.pt")
    for d, label in (("v0-video4-action4-original", "v0 4/4 original"), ("v0-video4-action4-repeat", "v0 4/4 repeat"),
                     ("v0-video4-action2", "v0 4/2"), ("v0-video4-action1", "v0 4/1"), ("v0-video2-action2", "v0 2/2"),
                     ("v0-video1-action4", "v0 1/4"), ("v1-video4-action4", "v1 4/4"), ("v2-video4-action4", "v2 4/4"),
                     ("dense-s12", "Dense-S12"), ("dense-s30", "Dense-S30")):
        add(f"long {label}", "libero-long", "loopwam", MAIN, f"{hf}/libero-long/{d}/policy.pt")
    for d, label in (("v0-video4-action1-kv-concat", "v0 4/1 kv-concat"), ("v0-video4-action1-kv-mix", "v0 4/1 kv-mix")):
        add(f"long {label}", "libero-long", "loopwam", KV, f"{hf}/libero-long/{d}/policy.pt")
    rt = {"RT-A": ("S1", "S2", "S3", "S4"), "RT-A2": ("S1", "S2", "S5", "S6", "S7", "S8"), "RT-B2a": ("parent2",),
          "RT-B4": ("S1", "S2"), "RT-Pa": ("S1", "S2"), "RT-T1ft": ("S1", "S2"), "RT-TF": ("S1", "S2")}
    for run, scheds in rt.items():
        add(run, "residual-transport", "rt", RT, f"{CK}/LoopWAM_NT/{run}/epoch_10.pt", scheds)
    elastic = {"E0": FLAT9, "E1": FLAT9, "E2": FLAT9, "SP11": ("flat_k1_n2",),
               "F0-PB": ("S8", "S7", "parent2", "S6", "S5", "S2", "parent5", "S1", "parent10")}
    for run, scheds in elastic.items():
        add(f"elastic {run}", "elastic", "rt", ELASTIC, f"{CK}/LoopWAM_NT/elastic/{run}/epoch_10.pt", scheds)
    for run in ("CL-0_no-memory_a4", "CL-A_mem16-learned-loopwrite_a4", "CL-REG_mem16-reset-registers_a4",
                "CL-W2_mem16-learned-external-updater_a4", "CL-FRAME_history-frame-k3_a4", "CL-0-at1_no-memory_a1",
                "CL-A-at1_mem16-learned-loopwrite_a1"):
        # The ChronoLoop shared data dir is not on this cluster; the all-suites export's normalization statistics
        # stand in (normalization does not change the computation).
        add(f"ChronoLoop {run}", "chronoloop", "chrono", CHRONO, f"{CK}/ChronoLoop/runs/{run}/checkpoints/epoch_10.pt",
            stats=f"{hf}/libero-all-suites/v0-video4-action4/dataset_stats.json", backends="eager,fast")
    return out


if __name__ == "__main__":
    hf = os.environ["LAT_HF"]
    # Longest jobs first within a tier (shortest makespan for parallel workers): cost ~ configurations, plus the
    # slow FastWAM load and Inductor compilation.
    cost = lambda r: len(r["configs"]) + 4 * (r["family"] == "fastwam") + ("compile" in r["backends"])
    rows = sorted(jobs(hf), key=lambda r: (r["tier"], -cost(r)))
    for i, r in enumerate(rows):
        r["id"] = f"{i:02d}"
        print(json.dumps(r))
    print(f"{len(rows)} jobs, {sum(len(r['configs']) for r in rows)} configurations", file=sys.stderr)
