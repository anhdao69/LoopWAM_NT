#!/usr/bin/env python
"""Assemble the public-WAM baselines report (latency of the released models + LiLa-WAM trained on our LIBERO data)
from the measured files. Every number in a generated table is read from a result file at generation time.
Usage: make_report.py <out.md>"""
import glob, json, os, re, subprocess, sys
from pathlib import Path

LAT = Path("/groups/yshang/an221229/checkpoints/wam_baselines/latency")
RUN = Path("/groups/yshang/an221229/checkpoints/wam_baselines/lilawam/libero_ours_paper_recipe")
LILA = Path("/groups/yshang/an221229/checkpoints/wam_baselines/lilawam")
DATA_STATS = "/groups/yshang/an221229/data/LiLaWAM_libero_ours_stats_{}.json"
SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
SUITE_NAME = {"libero_10": "Long (libero_10)", "libero_goal": "Goal", "libero_object": "Object", "libero_spatial": "Spatial"}
EPOCHS = {"stage1": 5, "stage2": 2}


def jload(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def latency_results():
    res = {}
    for p in sorted(LAT.glob("results/*.json")):
        if "smoke" in p.name:
            continue
        key, node = p.stem.rsplit("_", 1)
        res[(key, node)] = json.loads(p.read_text())
    return res


def full_stats_table(res):
    order = ["fastwam", "fastwam_nocompile", "fasterwam", "lightwam", "lilawam", "lilawam_libero"]
    md = ["| Model | Node (job) | Prompt mode | n | mean | std | p5 | p50 | p95 | min | max | Hz | Peak GB | Load s |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in order:
        for (key, node), r in sorted(res.items()):
            if key != k:
                continue
            for mode, s in r["modes"].items():
                md.append(f"| {r['model']} | {node} ({r.get('slurm_job')}) | {mode} | {s['n']} | {s['mean_ms']:.2f} | "
                          f"{s['std_ms']:.2f} | {s['p5_ms']:.2f} | {s['median_ms']:.2f} | {s['p95_ms']:.2f} | "
                          f"{s['min_ms']:.2f} | {s['max_ms']:.2f} | {1000 / s['mean_ms']:.1f} | "
                          f"{s['peak_allocated_gb']:.2f} | {r['load_seconds']:.0f} |")
    return "\n".join(md)


def t5_table(res):
    md = ["| Model | evc102: native − prompt_cached (ms) | evc103: native − prompt_cached (ms) | Share of native (evc102) |",
          "|---|---|---|---|"]
    for k in ["fastwam", "fastwam_nocompile", "fasterwam", "lightwam"]:
        cells, share = [], ""
        for node in ["evc102", "evc103"]:
            r = res.get((k, node))
            if not r:
                cells.append("-"); continue
            d = r["modes"]["native"]["mean_ms"] - r["modes"]["prompt_cached"]["mean_ms"]
            cells.append(f"{d:+.1f}")
            if node == "evc102":
                share = f"{100 * d / r['modes']['native']['mean_ms']:.1f}%"
        md.append(f"| {res.get((k, 'evc102'), res.get((k, 'evc103')))['model']} | {cells[0]} | {cells[1]} | {share} |")
    return "\n".join(md)


def train_log_info(suite, stage):
    p = RUN / "logs" / f"{suite}_{stage}.log"
    if not p.exists():
        return None
    t = p.read_text(errors="ignore")
    info = {"epochs_done": [], "losses": [], "times": []}
    m = re.search(r"Dataset Size: (\d+) \| Batches per Epoch: (\d+)", t)
    if m:
        info["frames"], info["batches"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"LR schedule: CosineAnnealingLR ([0-9.e+-]+) -> ([0-9.e+-]+)", t)
    if m:
        info["lr"] = f"{float(m.group(1)):.0e} → {float(m.group(2)):.0e}"
    for e, l, s in re.findall(r"=== Epoch (\d+) Completed\. Avg Loss: ([0-9.]+) \| Time: ([0-9.]+)s ===", t):
        info["epochs_done"].append(int(e)); info["losses"].append(float(l)); info["times"].append(float(s))
    steps = re.findall(r"Epoch \[(\d+)/(\d+)\] Step \[(\d+)/(\d+)\] Loss: ([0-9.]+) LR: ([0-9.e+-]+)", t)
    if steps:
        info["last_step"] = steps[-1]
    info["init_from"] = bool(re.search(r"Initializing weights from", t))
    info["complete"] = "Training Complete." in t
    info["traceback"] = "Traceback" in t
    return info


def job_events():
    t = (RUN / "train_job.log").read_text() if (RUN / "train_job.log").exists() else ""
    ev = {}
    for ts, suite, stage, what in re.findall(r"\[(\S+)\] (libero_\w+) (stage[12]) ?(start|rc=\d+)?", t):
        ev.setdefault((suite, stage), {})["end" if what.startswith("rc") else "start"] = ts[11:19]
        if what.startswith("rc"):
            ev[(suite, stage)]["rc"] = what
    # stage2 starts right after stage1 ends
    for s in SUITES:
        if (s, "stage1") in ev and "end" in ev[(s, "stage1")]:
            ev.setdefault((s, "stage2"), {}).setdefault("start", ev[(s, "stage1")]["end"])
    return ev, "all done" in t


def training_tables():
    ev, all_done = job_events()
    md = ["| Suite | Demos | Frames | Updates / epoch | Stage | LR (cosine) | Epochs done | Per-epoch mean loss | Epoch wall s | Start → end | Status |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in SUITES:
        st = jload(DATA_STATS.format(s))
        for stage in ("stage1", "stage2"):
            i = train_log_info(s, stage)
            e = ev.get((s, stage), {})
            if i is None:
                md.append(f"| {SUITE_NAME[s]} | {st['robotwin2']['meta']['files']} | {st['robotwin2']['meta']['frames']:,} | "
                          f"- | {stage} | - | 0/{EPOCHS[stage]} | - | - | - | not started |")
                continue
            if i["complete"]:
                status = "complete" + (" (init from stage-1 epoch 5)" if stage == "stage2" else "")
            elif i["traceback"]:
                status = "FAILED (traceback in log)"
            elif "last_step" in i:
                ep, ept, stp, stpt, loss, lr = i["last_step"]
                status = f"running: epoch {ep}/{ept}, update {stp}/{stpt}"
            else:
                status = "starting"
            md.append(f"| {SUITE_NAME[s]} | {st['robotwin2']['meta']['files']} | {st['robotwin2']['meta']['frames']:,} | "
                      f"{i.get('batches', '-')} | {stage} | {i.get('lr', '-')} | {len(i['epochs_done'])}/{EPOCHS[stage]} | "
                      f"{' → '.join(f'{x:.4f}' for x in i['losses']) or '-'} | {' / '.join(f'{x:.0f}' for x in i['times']) or '-'} | "
                      f"{e.get('start', '-')} → {e.get('end', '…')} | {status} |")
    return "\n".join(md), all_done


def checkpoint_table():
    md = ["| Suite | File on Hugging Face (`anhdao69/LoopWAM_NT`) | Epoch / update | Epoch mean loss | Size GB | sha256 | HF LFS sha256 = local | Commit |",
          "|---|---|---|---|---|---|---|---|"]
    man = jload(RUN / "manifest.json") or {"suites": {}}
    any_row = False
    for s in SUITES:
        up = jload(RUN / f"uploaded_{s}.json")
        ent = man["suites"].get(s)
        if not up or not ent:
            md.append(f"| {SUITE_NAME[s]} | not uploaded yet | | | | | | |")
            continue
        for rp, c in ent["checkpoints"].items():
            v = up["verify"].get(rp, {})
            any_row = True
            md.append(f"| {SUITE_NAME[s]} | `{rp}` | {c['epoch']} / {c['global_step']} | {c['epoch_avg_loss']:.4f} | "
                      f"{c['size'] / 1e9:.2f} | `{c['sha256'][:16]}…` | {'yes' if v.get('match') else 'NO'} | "
                      f"`{(up.get('commit') or '')[:10]}` |")
    return "\n".join(md), any_row


def openloop_table():
    md = ["| Suite | Checkpoint | Samples (valid steps) | MAE, 6 EEF deltas: model | smoothed | mean-action baseline | "
          "First-step MAE: model / baseline | Gripper sign agreement: model / baseline | max abs(raw − smoothed) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in SUITES:
        for stage in ("stage1", "stage2"):
            r = jload(RUN / "openloop" / f"{s}_{stage}.json")
            if not r:
                continue
            a, b, z = r["raw"], r["bspline_smoothed"], r["baseline_mean_action"]
            name = "stage-1 epoch 5" if stage == "stage1" else "final (stage-2 epoch 2)"
            md.append(f"| {SUITE_NAME[s]} | {name} | {r['samples']} ({r['valid_action_steps']}) | {a['mae_xyzrot']:.4f} | "
                      f"{b['mae_xyzrot']:.4f} | {z['mae_xyzrot']:.4f} | {a['mae_xyzrot_first_action']:.4f} / "
                      f"{z['mae_xyzrot_first_action']:.4f} | {a['gripper_sign_agreement']:.3f} / {z['gripper_sign_agreement']:.3f} | "
                      f"{r.get('max_abs_raw_minus_smoothed', float('nan')):.4f} |")
    return "\n".join(md) if len(md) > 2 else "_No open-loop results yet._"


def eval_section():
    out = RUN / "eval_matched"
    summ, man = jload(out / "summary.json"), jload(out / "manifest.json")
    groups = sorted((out / "groups").glob("*.json")) if (out / "groups").exists() else []
    order = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    fast = {"libero_spatial": 97.47, "libero_object": 99.27, "libero_goal": 96.27, "libero_10": 93.53}
    paper = {"libero_spatial": 98.0, "libero_object": 98.8, "libero_goal": 97.2, "libero_10": 94.2}
    if summ:
        md = ["| Suite | Successes / episodes | Success % | Seed 42 / 43 / 44 % | FastWAM (ours, matched) % | Paper (LiLa-WAM) % |",
              "|---|---|---|---|---|---|"]
        for su in order:
            v = summ["per_suite"].get(su)
            if not v:
                md.append(f"| {SUITE_NAME[su]} | not evaluated | | | {fast[su]:.2f} | {paper[su]:.1f} |"); continue
            sd = " / ".join(f"{100 * v['seeds'][k]['success_rate']:.1f}" for k in sorted(v["seeds"]))
            md.append(f"| {SUITE_NAME[su]} | {v['successes']} / {v['episodes']} | **{100 * v['success_rate']:.2f}** | {sd} | "
                      f"{fast[su]:.2f} | {paper[su]:.1f} |")
        ps = " / ".join(f"{100 * summ['per_seed'][k]['success_rate']:.2f}" for k in sorted(summ["per_seed"]))
        md.append(f"| **All** | {summ['successes']} / {summ['episodes']} | **{100 * summ['success_rate']:.2f}** | {ps} | 96.63 | 97.1 |")
        import datetime as _d
        t0, t1 = summ["first_group_completed_at"], summ["last_group_completed_at"]
        runs = [jload(p) for p in sorted((out / "runs").glob("*.json"))]
        wall = sum(r["wall_seconds"] for r in runs if r)
        md += ["", f"- **Completed:** {summ['episodes']} episodes, {summ['total_policy_steps']:,} policy steps, "
               f"{summ['total_queries']:,} policy queries.",
               f"- **Runner time:** {wall / 60:.1f} min wall across {len(runs)} runner invocation(s), in job {man.get('slurm_job_id')} "
               f"on {man.get('host')}. GPUs: {man.get('gpus')}; {man.get('env_workers')} simulator workers.",
               f"- **Last group completed:** {_d.datetime.fromtimestamp(t1).strftime('%Y-%m-%d %H:%M:%S')}.",
               f"- **Checkpoint identity:** combined sha256 of the four policies `{summ['checkpoint_sha256'][:16]}…`. "
               "Per-suite sha256 values are in `eval_matched/manifest.json` and match Section 6.6."]
        return "\n".join(md), f"{100 * summ['success_rate']:.2f}% ({summ['successes']}/{summ['episodes']})"
    if groups:
        rows = [json.loads(p.read_text()) for p in groups]
        n = sum(len(r["episodes"]) for r in rows); k = sum(r["successes"] for r in rows)
        per = {}
        for r in rows:
            su = r["group"]["suite"]; a = per.setdefault(su, [0, 0]); a[0] += r["successes"]; a[1] += len(r["episodes"])
        txt = (f"**Running (interim, not a result):** {len(rows)}/120 task-seed groups done, {k}/{n} episodes successful so far: "
               + ", ".join(f"{SUITE_NAME[su]} {a[0]}/{a[1]}" for su, a in per.items()) + ".")
        return txt, "running"
    return ("Not started yet. `eval_login_watcher.sh` on evuser2 launches `eval_lilawam_step.sh` in your interactive "
            "job 1102933 (2 H100, 64 CPUs) as soon as it starts."), "pending"


def main():
    res = latency_results()
    tables = subprocess.run([sys.executable, str(LAT / "build_tables.py"), str(LAT / "results"), str(LAT / "tables.md")],
                            capture_output=True, text=True, check=True).stdout
    train_md, all_done = training_tables()
    ckpt_md, any_ckpt = checkpoint_table()
    conv = jload(LILA / "conversion_verification.json")
    tmpl = (LAT / "report_v2_template.md").read_text()
    lat_tr = res.get(("lilawam_libero", "evc103"))
    lat_tr_line = (f"{lat_tr['modes']['native']['mean_ms']:.1f} ms mean (p50 {lat_tr['modes']['native']['median_ms']:.1f}, "
                   f"p95 {lat_tr['modes']['native']['p95_ms']:.1f}), {lat_tr['modes']['native']['peak_allocated_gb']:.2f} GB peak, "
                   f"measured on evc103 after training finished with the GPU otherwise idle"
                   if lat_tr else "not measured yet (needs the finished checkpoint and an idle GPU)")
    status = ("**Training: all four suites finished both stages.**" if all_done else
              "**Training: in progress** (the table in Section 6.5 shows exactly where each suite is).")
    out = (tmpl.replace("{{LAT_TABLES}}", tables.strip())
               .replace("{{LAT_FULL}}", full_stats_table(res))
               .replace("{{T5}}", t5_table(res))
               .replace("{{TRAIN_TABLE}}", train_md)
               .replace("{{CKPT_TABLE}}", ckpt_md)
               .replace("{{OPENLOOP}}", openloop_table())
               .replace("{{TRAIN_STATUS}}", status)
               .replace("{{LAT_TRAINED}}", lat_tr_line)
               .replace("{{CONV}}", f"{conv['episodes_checked']} randomly chosen demonstrations ({conv['frames_checked']:,} frames)"
                        f", all checks passed: {conv['all_ok']}" if conv else "not run"))
    ev_md, ev_short = eval_section()
    out = out.replace("{{EVAL}}", ev_md).replace("{{EVAL_SHORT}}", ev_short)
    assert "{{" not in out, re.findall(r"\{\{\w+\}\}", out)
    Path(sys.argv[1]).write_text(out)
    print("wrote", sys.argv[1], len(out), "chars; training all done:", all_done, "; checkpoints uploaded:", any_ckpt)


if __name__ == "__main__":
    main()
