#!/usr/bin/env python
"""Append a results section (training losses, uploaded checkpoints, open-loop check, latency) to the HF model card
of baselines/LiLa-WAM_LIBERO_ours and upload it together with the open-loop and latency result files.
Every number is read from the result files. Run after all suites are uploaded."""
import json, shutil
from pathlib import Path
from huggingface_hub import CommitOperationAdd, HfApi

RUN = Path("/groups/yshang/an221229/checkpoints/wam_baselines/lilawam/libero_ours_paper_recipe")
LATR = Path("/groups/yshang/an221229/checkpoints/wam_baselines/latency/results/lilawam_libero_evc103.json")
REPO, P = "anhdao69/LoopWAM_NT", "baselines/LiLa-WAM_LIBERO_ours"
SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
NAME = {"libero_10": "Long (libero_10)", "libero_goal": "Goal", "libero_object": "Object", "libero_spatial": "Spatial"}
base = RUN / "hf_staging" / "README_base.md"
if not base.exists():
    shutil.copy(RUN / "hf_staging" / "README.md", base)
man = json.loads((RUN / "manifest.json").read_text())
md = ["", "## Results so far (generated from the result files)", "",
      "### Training", "",
      "| Suite | Stage-1 epoch mean loss (5 epochs) | Stage-2 epoch mean loss (2 epochs) | Final file | sha256 (HF LFS verified) |",
      "|---|---|---|---|---|"]
for s in SUITES:
    e = man["suites"].get(s)
    if not e:
        md.append(f"| {NAME[s]} | not uploaded | | | |"); continue
    up = json.loads((RUN / f"uploaded_{s}.json").read_text())
    final = [k for k in e["checkpoints"] if "stage2_epoch2_final" in k or "INCOMPLETE" in k][0]
    md.append(f"| {NAME[s]} | {' → '.join(f'{x:.4f}' for x in e.get('stage1_epoch_avg_losses', []))} | "
              f"{' → '.join(f'{x:.4f}' for x in e.get('stage2_epoch_avg_losses', []))} | `{final.split('/', 2)[2]}` | "
              f"`{e['checkpoints'][final]['sha256'][:16]}…` ({'match' if up['verify'][final]['match'] else 'MISMATCH'}) |")
md += ["", "### Open-loop check on held-in demonstration frames (not a success rate)", "",
       "512 random training samples per suite, seed 0, public sampling procedure (10 Euler steps). MAE in raw action "
       "units over the 6 end-effector deltas; the baseline always predicts the suite's mean action.", "",
       "| Suite | Checkpoint | MAE model | MAE baseline | Gripper sign agreement model / baseline |", "|---|---|---|---|---|"]
ops = []
for s in SUITES:
    for stage in ("stage1", "stage2"):
        f = RUN / "openloop" / f"{s}_{stage}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        md.append(f"| {NAME[s]} | {'stage-1 epoch 5' if stage == 'stage1' else 'final'} | {r['raw']['mae_xyzrot']:.4f} | "
                  f"{r['baseline_mean_action']['mae_xyzrot']:.4f} | {r['raw']['gripper_sign_agreement']:.3f} / "
                  f"{r['baseline_mean_action']['gripper_sign_agreement']:.3f} |")
        ops.append(CommitOperationAdd(f"{P}/results/openloop_{s}_{stage}.json", str(f)))
if LATR.exists():
    r = json.loads(LATR.read_text()); n = r["modes"]["native"]
    md += ["", "### Latency", "",
           f"Batch 1 on an H100 80GB (evc103, GPU otherwise idle), 500 recorded raw LIBERO observations after 20 warmup "
           f"queries, from raw observation to the smoothed host action chunk ({r['action_chunk_shape'][0]}×"
           f"{r['action_chunk_shape'][1]}): mean {n['mean_ms']:.1f} ms, p50 {n['median_ms']:.1f}, p95 {n['p95_ms']:.1f}, "
           f"peak {n['peak_allocated_gb']:.2f} GB; {r['parameters'] / 1e9:.3f}B parameters including the frozen DINOv3 "
           f"({r['parameters_action_model'] / 1e6:.1f}M trainable action model)."]
    ops.append(CommitOperationAdd(f"{P}/results/latency_lilawam_libero_evc103.json", str(LATR)))
summ_p = RUN / "eval_matched" / "summary.json"
if summ_p.exists():
    sm = json.loads(summ_p.read_text())
    md += ["", "### LIBERO success under our matched protocol", "",
           "Protocol lightwam_20261009_matched: 4 suites × 10 tasks × official initial states 0–49 × seeds 42/43/44; "
           "30 settling steps; at most 400 policy steps (700 for Long); success = environment done flag. Each suite "
           "uses its own final policy. 6 of 12 predicted actions are executed per query, and the gripper is executed as "
           "sign(g). Inference is batched, in bf16.", "",
           "| Suite | Successes / episodes | Success % | Seeds 42 / 43 / 44 % |", "|---|---|---|---|"]
    for s in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        v = sm["per_suite"].get(s)
        if v:
            md.append(f"| {NAME[s]} | {v['successes']} / {v['episodes']} | {100 * v['success_rate']:.2f} | "
                      + " / ".join(f"{100 * v['seeds'][k]['success_rate']:.1f}" for k in sorted(v["seeds"])) + " |")
    md.append(f"| **All** | {sm['successes']} / {sm['episodes']} | **{100 * sm['success_rate']:.2f}** | "
              + " / ".join(f"{100 * sm['per_seed'][k]['success_rate']:.1f}" for k in sorted(sm["per_seed"])) + " |")
    ops.append(CommitOperationAdd(f"{P}/results/eval_matched_summary.json", str(summ_p)))
    ops.append(CommitOperationAdd(f"{P}/results/eval_matched_manifest.json", str(RUN / "eval_matched" / "manifest.json")))
else:
    md += ["", "LIBERO success rates under our matched protocol (4 suites × 10 tasks × 50 trials × 3 seeds): **pending**.", ""]
text = base.read_text().rstrip() + "\n" + "\n".join(md)
(RUN / "hf_staging" / "README.md").write_text(text)
ops.append(CommitOperationAdd(f"{P}/README.md", str(RUN / "hf_staging" / "README.md")))
api = HfApi(token=open("/groups/yshang/an221229/cache/huggingface/token").read().strip())
info = api.create_commit(REPO, operations=ops, commit_message="LiLa-WAM LIBERO baseline: model card results, open-loop and latency files")
print("uploaded", len(ops), "files; commit", info.oid[:12])
