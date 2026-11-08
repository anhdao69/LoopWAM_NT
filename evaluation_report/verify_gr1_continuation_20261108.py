#!/usr/bin/env python3
"""Independent recomputation of the RoboCasa-GR1 epochs 6-10 continuation results from the evidence stored in this folder.

Usage: python3 verify_gr1_continuation_20261108.py   (run from anywhere; reads the folders next to this script)
Writes robocasa_gr1_continuation_20261108/independent_check.json.

Checks, for each continuation epoch 6-10 (24 tasks x 50 episodes):
  per-episode successes (eval/official_epochNN/n12/*.json) == summary.json == official_epochNN_validation.json
  == official_results_continuation.md (task cells and average) == the orchestrator's validation line == README.md;
  eval config (checkpoint, expected step, seed 0, 50 episodes, n_action_steps 12, 720 steps, n_envs 1, EGL) and the
  checkpoint/step reported by the policy server for every task.
Also: the table's H100 epoch 4/5 columns == the per-episode evidence in ../robocasa_gr1_v0_20261101; training metrics
cover updates 117,581-235,160 once each and agree with COMPLETE; the GR1-E0 teacher selection follows its rule from
these results; the Hugging Face receipts for epoch_09.pt / epoch_10.pt match training/checkpoints_sha256.json.
"""
import gzip, json, re
from pathlib import Path

ER = Path(__file__).resolve().parent
C = ER / "robocasa_gr1_continuation_20261108"
V0 = ER / "robocasa_gr1_v0_20261101"
PER, SUFFIX = 23516, "_GR1ArmsAndWaistFourierHands_Env"
rec = dict(epochs={}, checks={}, failures=[])
fail = rec["failures"].append


def table(md):
    lines = [l for l in md.read_text().splitlines() if l.startswith("|")]
    head = [x.strip() for x in lines[0].strip("|").split("|")]
    return {c[0]: dict(zip(head[1:], c[1:])) for c in ([x.strip().strip("*") for x in l.strip("|").split("|")] for l in lines[2:])}


def run_dir(d, epoch=None):
    per, served = {}, set()
    for f in sorted((d / "n12").glob("*.json")):
        x = json.loads(f.read_text())
        assert (x["n_action_steps"], x["n_episodes"], x["seed"], x["n_envs"], x["max_episode_steps"]) == (12, 50, 0, 1, 720), f
        assert len(x["successes"]) == 50 and sum(bool(s) for s in x["successes"]) == x["successes_first_n"], f
        per[f.stem.replace(SUFFIX, "")] = x["successes_first_n"]
        m = x["server_metadata"]
        served.add((Path(m["checkpoint"]).name, m["step"], m["normalization_sha256"]))
    s = json.loads((d / "summary.json").read_text())
    summ = s["by_n_action_steps"]["12"]
    same = {t.replace(SUFFIX, ""): v["successes"] for t, v in summ["per_task"].items()} == per \
        and all(v["episodes"] == 50 for v in summ["per_task"].values())
    return per, summ, same, s, served


tab = table(C / "eval" / "official_results_continuation.md")
orch = (C / "record" / "orchestrator.log").read_text()
readme = (C / "README.md").read_text()
ck = json.loads((C / "training" / "checkpoints_sha256.json").read_text())["checkpoints"]
norm = set()
for e in range(6, 11):
    d = C / "eval" / f"official_epoch{e:02d}"
    per, summ, same, s, served = run_dir(d, e)
    k, step = sum(per.values()), e * PER
    mean = 100 * k / 1200
    cfg = json.loads((d / "eval_config.json").read_text())
    cfg_ok = (Path(cfg["checkpoint"]).name == f"epoch_{e:02d}.pt" and cfg["expected_step"] == step and cfg["seed"] == 0
              and cfg["episodes"] == 50 and cfg["steps"] == "12" and cfg["max_episode_steps"] == 720 and cfg["n_envs"] == 1
              and cfg["render"] == "egl" and len(cfg["tasks"]) == 24 and Path(s["checkpoint"]).name == f"epoch_{e:02d}.pt")
    served_ok = len(served) == 1 and next(iter(served))[:2] == (f"epoch_{e:02d}.pt", step)
    norm |= {x[2] for x in served}
    v = json.loads((C / "eval" / f"official_epoch{e:02d}_validation.json").read_text())
    val_ok = (v["valid"] is True and v["problems"] == [] and v["step"] == step and abs(v["mean_success"] * 100 - mean) < 1e-9
              and {t.replace(SUFFIX, ""): x["successes"] for t, x in v["per_task"].items()} == per)
    col = f"ep{e}"
    task_ok = all(int(tab[t][col]) == 2 * n for t, n in per.items()) and len(tab) == 25
    avg_ok = abs(float(tab["Average"][col]) - round(mean, 2)) < 0.006
    orch_ok = re.search(rf"validation epoch {e} rc=0: \S+official_epoch{e:02d}\s+step {step}\s+tasks 24\s+mean success {mean:.2f}%\s+"
                        rf"\({k}/1200 episodes\)\s+summary mean {mean:.2f}% VALID", orch) is not None
    row = re.search(rf"^\| {e} \| {step:,} \| `{ck[f'epoch_{e:02d}.pt']['sha256'][:12]}` \| [^|]+ \| \**{mean:.2f}% \({k}/1200\)\** \|$",
                    readme, re.M) is not None
    ok = (len(per) == 24 and same and summ["complete"] and summ["tasks"] == 24 and abs(summ["mean_success"] * 100 - mean) < 1e-9
          and cfg_ok and served_ok and val_ok and task_ok and avg_ok and orch_ok and row)
    rec["epochs"][str(e)] = dict(step=step, tasks=len(per), episodes=50 * len(per), successes=k, mean_percent=round(mean, 4),
                                 summary_json_equal=same, eval_config=cfg_ok, served_checkpoint=served_ok, validation_equal=val_ok,
                                 table_tasks_equal=task_ok, table_average_equal=avg_ok, orchestrator_line=orch_ok, readme_row=row, passed=ok)
    if not ok:
        fail(f"epoch {e}")
rec["checks"]["one normalization across epochs 6-10"] = len(norm) == 1

# H100 epoch 4/5 columns of the continuation table == the v0 per-episode evidence
for e in (4, 5):
    per, summ, same, _, _ = run_dir(V0 / "eval_h100" / f"official_epoch{e:02d}")
    col = f"H100 ep{e}"
    ok = same and all(int(tab[t][col]) == 2 * n for t, n in per.items()) and abs(float(tab["Average"][col]) - round(100 * sum(per.values()) / 1200, 2)) < 0.006
    rec["checks"][f"table {col} == robocasa_gr1_v0_20261101 evidence"] = ok
    norm_v0 = {json.loads(f.read_text())["server_metadata"]["normalization_sha256"] for f in (V0 / "eval_h100" / f"official_epoch{e:02d}" / "n12").glob("*.json")}
    rec["checks"][f"epoch {e} normalization == continuation"] = norm_v0 == norm

# training records
rows = [json.loads(l) for l in gzip.open(C / "training" / "metrics.jsonl.gz", "rt") if l.strip()]
ups = [r["update"] for r in rows]
done = json.loads((C / "training" / "COMPLETE").read_text())
man = json.loads((C / "training" / "manifest.json").read_text())["contract"]
rec["checks"]["metrics cover updates 117,581-235,160 once each"] = ups == list(range(5 * PER + 1, 10 * PER + 1))
rec["checks"]["metrics end == COMPLETE"] = (done["status"], done["update"], done["planned_updates"], done["windows_seen"]) == \
    ("complete", 10 * PER, 10 * PER, rows[-1]["windows_seen"]) and rows[-1]["epochs_complete"] == 10
rec["checks"]["contract: 23,516 updates/epoch, phase from 117,580 to 235,160"] = \
    (man["updates_per_epoch"], man["phase_start_update"], man["planned_updates"], man["warmup_fraction"], man["lr_backbone"]) == (PER, 5 * PER, 10 * PER, 0.0, 3e-05)
rec["checks"]["one training and one evaluation attempt per epoch"] = all(
    len(re.findall(rf"\] epoch {e} attempt 1 rc=0 timing ", orch)) == 1 and len(re.findall(rf"\] eval epoch {e} attempt 1 rc=0: ", orch)) == 1
    and re.search(rf"\] (eval )?epoch {e} attempt [2-9]", orch) is None for e in range(6, 11))

# GR1-E0 teacher selection and Hugging Face receipts
sel = json.loads((C / "followup_gr1_e0" / "teacher_selection.json").read_text())
s9, s10 = rec["epochs"]["9"]["successes"], rec["epochs"]["10"]["successes"]
rec["checks"]["teacher selection follows its rule"] = (sel["chosen_epoch"] == (10 if s10 > s9 else 9) and sel["evidence"]["9"]["successes"] == s9
                                                       and sel["evidence"]["10"]["successes"] == s10
                                                       and sel["teacher_sha256"] == ck[f"epoch_{sel['chosen_epoch']:02d}.pt"]["sha256"])
receipts = {e: json.loads((C / "hf" / f"epoch_{e:02d}.pt.uploaded.json").read_text()) for e in (9, 10)}
log = (C / "hf" / "hf_upload_epoch09_10.log").read_text(errors="replace")
rec["checks"]["HF receipts == checkpoint sha256/size/step/successes"] = all(
    r["sha256"] == ck[f"epoch_{e:02d}.pt"]["sha256"] and r["size"] == ck[f"epoch_{e:02d}.pt"]["bytes"] and r["step"] == e * PER
    and r["official_successes"] == rec["epochs"][str(e)]["successes"] and r["path"] == f"Robocasa-GR1/epoch_{e:02d}.pt"
    and f'"sha256": "{r["sha256"]}"' in log for e, r in receipts.items()) and len({r["revision"] for r in receipts.values()}) == 1 \
    and "RECEIPTS COMMITTED" in log and "GR1_EXIT=0" in log
for k, v in rec["checks"].items():
    if not v:
        fail(k)
rec["passed"] = not rec["failures"]
rec["description"] = __doc__.split("Checks, ", 1)[1].strip()
(C / "independent_check.json").write_text(json.dumps(rec, indent=1) + "\n")
print(json.dumps(dict(passed=rec["passed"], failures=rec["failures"],
                      epochs={e: (r["successes"], r["passed"]) for e, r in rec["epochs"].items()}, checks=rec["checks"]), indent=1))
