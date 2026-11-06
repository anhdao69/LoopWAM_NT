#!/usr/bin/env python3
"""Pre-evaluation audit of the CL-0 elastic exports (CPU only; run on worker-2 with plans/elastic_cl0/env.sh).

For each run and export label it checks, and writes one JSON record:
- format/version/loop metadata as the LIBERO runner's checkpoint_policy_spec requires (v0, 4/4);
- the frozen parent is bit-equal, in key set, dtype and value, to the FP32 CL-0 teacher that training loaded: every
  `mot` tensor outside `mixtures.action.*` and the whole proprio_encoder (so the --bf16-frozen speed option, which
  stored frozen Linears in BF16 only in GPU memory, did not leak into the export);
- export_epoch writes the merged student into `mot["mixtures.action.*"]`: those tensors must equal `student_action`
  (P-c has no LoRA, so merging is the identity), be FP32, and differ from the teacher's action expert (it trained);
- student_action / step_condition / transport presence matches the run's transport kind;
- transport_config equals config.json "config" (the runner refuses otherwise), its schedules are the flat9 grid,
  and the parent recorded in it is CL-0;
- training_state: epoch, update, windows_seen, next_micro and the contract (world 4, microbatch 32, global batch 128,
  277,713 windows, 21,700 planned updates, normalization d5ad351e, seed 42), plus the speed options used;
- config.json VAE hash equals the VAE file, dataset_stats.json hashes to the contract's normalization hash.
"""
import argparse, hashlib, json, math, os
from pathlib import Path
import torch

FLAT9 = ["flat_k1_n2", "flat_k1_n5", "flat_k1_n10", "flat_k2_n2", "flat_k2_n5", "flat_k2_n10",
         "flat_k4_n2", "flat_k4_n5", "flat_k4_n10"]
NORM_SHA = "d5ad351eb3272f416d34082e8af5483ba1118ceca355a37d9f268fdb269fb3e4"
CL0_SHA = "137f0a839b5a1e2363621420ab599cd36a5693000f784a04a11b9bea2bea2cae"
WINDOWS, UPDATES = 277713, 21700


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def same_tensors(a, b):
    if set(a) != set(b):
        return False, f"key sets differ ({len(set(a) ^ set(b))} keys)"
    bad = [k for k in a if a[k].dtype != b[k].dtype or a[k].shape != b[k].shape or not torch.equal(a[k], b[k])]
    return not bad, (f"{len(bad)} tensors differ, e.g. {bad[:3]}" if bad else f"{len(a)} tensors bit-equal")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", required=True); p.add_argument("--runs", default="CL0-E0,CL0-E1,CL0-E2")
    p.add_argument("--labels", default="epoch_10"); p.add_argument("--out", required=True)
    a = p.parse_args()
    teacher_path, vae = os.environ["ELASTIC_TEACHER"], os.environ["ELASTIC_VAE"]
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=False, mmap=True)
    teacher_record = json.loads(Path(teacher_path).with_suffix(".json").read_text())
    vae_sha = sha256_file(vae)
    report = dict(teacher=teacher_path, teacher_sha256=sha256_file(teacher_path), vae_sha256=vae_sha, exports={})
    ok_all = teacher_record["source_sha256"] == CL0_SHA and teacher_record["sha256"] == report["teacher_sha256"]
    for run in a.runs.split(","):
        folder = Path(a.runs_root) / run
        config = json.loads((folder / "config.json").read_text())
        stats_sha = sha256_file(folder / "dataset_stats.json")
        for label in a.labels.split(","):
            path = folder / f"{label}.pt"
            x = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            st = x.get("training_state") or {}; c = st.get("contract") or {}
            tc = x.get("transport_config") or {}
            checks = {}
            checks["format"] = (x.get("format_version") == "loopwam-s-v1" and x.get("version") == "v0"
                                and x.get("inference_loops") == 4 and x.get("trained_max_loops") == 4
                                and x.get("video_loops", 4) == 4 and x.get("action_loops", 4) == 4)
            A = "mixtures.action."
            frozen = {k: v for k, v in x["mot"].items() if not k.startswith(A)}
            tfrozen = {k: v for k, v in teacher["mot"].items() if not k.startswith(A)}
            ok, why = same_tensors(frozen, tfrozen); checks["frozen_parent_equal_fp32_teacher"] = ok
            dtypes = sorted({str(v.dtype) for v in x["mot"].values()})
            ok2, why2 = same_tensors(x["proprio_encoder"], teacher["proprio_encoder"]); checks["proprio_equal_teacher"] = ok2
            kind = config["config"]["kind"]
            checks["parts"] = "student_action" in x and x.get("step_condition") is not None
            sa = x["student_action"]
            action = {k[len(A):]: v for k, v in x["mot"].items() if k.startswith(A)}
            taction = {k[len(A):]: v for k, v in teacher["mot"].items() if k.startswith(A)}
            checks["mot_action_equals_student"] = same_tensors(action, sa)[0]
            checks["student_fp32"] = all(v.dtype == torch.float32 for v in sa.values() if v.is_floating_point())
            changed = sum(not torch.equal(sa[k], taction[k]) for k in sa if k in taction)
            checks["student_trained"] = set(sa) == set(taction) and changed > 0
            student_info = dict(student_tensors=len(sa), student_changed_vs_teacher=changed,
                                transport_tensors=len(x.get("transport") or {}),
                                transport_numel=sum(v.numel() for v in (x.get("transport") or {}).values()))
            checks["transport_config_equals_config_json"] = tc == config["config"]
            checks["flat9_schedules"] = sorted(tc.get("schedules", [])) == sorted(FLAT9)
            parent = tc.get("parent") or {}
            checks["parent_is_cl0"] = parent.get("chronoloop_sha256") == CL0_SHA
            epoch = st.get("epoch"); want = math.ceil(WINDOWS / 128) * epoch if isinstance(epoch, int) else None
            checks["training_state"] = (x.get("step") == want and st.get("update") == want
                                        and st.get("windows_seen") == epoch * WINDOWS)
            checks["contract"] = (c.get("world") == 4 and c.get("microbatch") == 32 and c.get("global_batch") == 128
                                  and c.get("windows") == WINDOWS and c.get("planned_updates") == UPDATES
                                  and c.get("epochs") == 10 and c.get("normalization_sha256") == NORM_SHA
                                  and c.get("config") == tc and c.get("seed", 42) == 42)
            checks["vae_hash"] = config["asset_sha256"]["vae"] == vae_sha
            checks["stats_hash"] = stats_sha == NORM_SHA
            rec = dict(path=str(path), sha256=sha256_file(path), size=path.stat().st_size, checks=checks,
                       passed=all(v is True for v in checks.values()), parent_mot=why, parent_mot_dtypes=dtypes,
                       proprio=why2, kind=kind, scope=tc.get("scope"), step_cond=tc.get("step_cond"),
                       epoch=epoch, step=x.get("step"), windows_seen=st.get("windows_seen"),
                       next_micro=st.get("next_micro"), speed_options=c.get("speed_options"),
                       contract_world=c.get("world"), contract_microbatch=c.get("microbatch"), **student_info)
            report["exports"][f"{run}/{label}"] = rec
            ok_all &= rec["passed"]
            print(json.dumps({f"{run}/{label}": {k: rec[k] for k in ("sha256", "passed", "checks", "parent_mot",
                              "parent_mot_dtypes", "student_changed_vs_teacher", "transport_numel", "epoch", "step", "speed_options")}}, default=str), flush=True)
            del x
    report["passed"] = bool(ok_all)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, default=str))
    print("ALL PASSED" if ok_all else "FAILED")


if __name__ == "__main__":
    main()
