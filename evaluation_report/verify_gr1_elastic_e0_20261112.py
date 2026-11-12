#!/usr/bin/env python3
"""Independent recomputation of the GR1-E0 (elastic RoboCasa-GR1 student) results from the evidence stored in this report.

Usage: python3 verify_gr1_elastic_e0_20261112.py   (run from anywhere; reads the folders next to this script)
Writes robocasa_gr1_elastic_e0_20261112/independent_check.json.

It shares no code with robocasa_gr1_elastic_e0_20261112/analysis/analyze_gr1_e0.py.

Checks, for each of the 25 evaluations (24 tasks x 50 episodes):
  per-episode successes (eval/eEE_<schedule>/n12/*.json) == summary.json == eEE_<schedule>_validation.json
  == the run's record/RESULTS.md == the orchestrator's "attempt 1 rc=0" line == analysis/analysis.json == README.md;
  eval config (checkpoint, expected step, schedule, seed 0, 50 episodes, n_action_steps 12, 720 steps, n_envs 1, EGL,
  GPUs 4-7, 2 servers per GPU, 24 clients); the policy server of every task reports that epoch's checkpoint path, step
  and sha256 (== training/checkpoints_sha256.json), the schedule, its loop list and step count, noise seed 42 and the
  teacher's sha256; the validator's verdict is "valid" except for the two documented single-episode flags.
Also: paired statistics against the teacher (net successes and McNemar z over the 1,200 paired episodes), the epoch 4 -> 5
changes and the pooled loop counts are recomputed here and compared with analysis/analysis.json and the README tables;
analysis/per_task_successes.csv equals the per-episode evidence; the teacher's counts equal the per-episode evidence in
../robocasa_gr1_continuation_20261108 and the teacher selection follows its rule; training metrics hold updates
1-235,160 once each, in order, and agree with status.json and config.json; every epoch export passed its verification;
the Hugging Face receipt matches the epoch-5 checkpoint hash and size and the upload log; every stored file equals the
SHA-256 and size recorded in SOURCES.tsv (the .gz files after decompression).
"""
import csv, gzip, hashlib, json, math, re
from pathlib import Path

ER = Path(__file__).resolve().parent
F = ER / "robocasa_gr1_elastic_e0_20261112"
TE = ER / "robocasa_gr1_continuation_20261108" / "eval"
PER, SUFFIX, PREFIX = 47032, "_GR1ArmsAndWaistFourierHands_Env", "gr1_unified/"
B = "/mnt/data/vmo-ai-task/anhdh35/elastic_gr1"
rec = dict(evaluations={}, checks={}, failures=[])
fail = rec["failures"].append
FLAGS = {"e01_flat_k1_n2": "PnPMilkToMicrowaveClose", "e05_flat_k1_n5": "PosttrainPnPNovelFromPlateToCardboardboxSplitA"}
FLAG_TEXT = ": a final-step-success video is not an any-step success in the JSON"


def episodes(d):
    out = {}
    for f in sorted((d / "n12").glob("*.json")):
        x = json.loads(f.read_text())
        out[f.stem.replace(SUFFIX, "")] = x
    return out


def mcnemar(a, b):
    up = down = 0
    for t in a:
        for x, y in zip(a[t]["successes"], b[t]["successes"]):
            up += (not x) and bool(y)
            down += bool(x) and (not y)
    return up - down, ((up - down) / math.sqrt(up + down) if up + down else 0.0)


def rows(md, header_start):
    """Rows of the markdown table whose header line starts with header_start."""
    lines = md.splitlines()
    i = next(n for n, l in enumerate(lines) if l.startswith(header_start))
    out = []
    for l in lines[i + 2:]:
        if not l.startswith("|"):
            break
        out.append([c.strip() for c in l.strip().strip("|").split("|")])
    return out


ck = json.loads((F / "training" / "checkpoints_sha256.json").read_text())["checkpoints"]
sel = json.loads((F / "record" / "teacher_selection.json").read_text())
ana = json.loads((F / "analysis" / "analysis.json").read_text())
ana_ev = {(e["epoch"], e["schedule"]): e for e in ana["evaluations"]}
readme = (F / "README.md").read_text()
results_md = (F / "record" / "RESULTS.md").read_text()
orch = (F / "record" / "orchestrator.log").read_text()
teacher = {e: episodes(TE / f"official_epoch{e:02d}") for e in (9, 10)}
tsum = {e: sum(sum(x["successes"]) for x in t.values()) for e, t in teacher.items()}
T = teacher[9]

# README tables
main = {r[0].strip("`"): r for r in rows(readme, "| setting | loops | steps | block passes | epoch 1 |")}
versus = {r[0].strip("`"): r for r in rows(readme, "| setting | block passes | epoch 4 | net vs teacher (z) |")}
pooled_rows = {r[0]: r for r in rows(readme, "| loops | successes of 7,200 |")}

data = {}
dirs = sorted(d for d in (F / "eval").iterdir() if d.is_dir())
for d in dirs:
    name = d.name
    m = re.fullmatch(r"e(\d\d)_(flat_k(\d+)_n(\d+))", name)
    e, sched, k, n = int(m[1]), m[2], int(m[3]), int(m[4])
    label = f"[{k},{k}]" if n == 2 else f"[{k}]x{n}"
    ep = episodes(d)
    data[(e, sched)] = ep
    succ = {t: sum(bool(s) for s in x["successes"]) for t, x in ep.items()}
    total = sum(succ.values())
    r = dict(epoch=e, schedule=sched, label=label, step=e * PER, tasks=len(ep), episodes=sum(len(x["successes"]) for x in ep.values()),
             successes=total, percent=round(100 * total / 1200, 4))
    r["protocol_fields"] = len(ep) == 24 and sorted(ep) == sorted(T) and all(
        x["seed"] == 0 and x["n_episodes"] == 50 and x["n_envs"] == 1 and x["n_action_steps"] == 12 and x["max_episode_steps"] == 720
        and len(x["successes"]) == 50 and x["successes_first_n"] == succ[t] and x["env_name"] == PREFIX + t + SUFFIX
        and x["env"]["MUJOCO_GL"] == "egl" for t, x in ep.items())
    sha = ck[f"epoch_{e:02d}.pt"]["sha256"]
    r["served_checkpoint"] = all(
        (sm["checkpoint"], sm["step"], sm["export_sha256"], sm["schedule"], sm["loops"], sm["num_inference_steps"], sm["noise_seed"],
         sm["teacher_sha256"], sm["teacher_epoch"], sm["transport_kind"], sm["n_action_steps"], sm["host"])
        == (f"{B}/runs/GR1-E0/epoch_{e:02d}.pt", e * PER, sha, sched, [k] * n, n, 42, sel["teacher_sha256"], 9, "T0", 12, "worker-2")
        for sm in (x["server_metadata"] for x in ep.values()))
    cfg = json.loads((d / "eval_config.json").read_text())
    r["eval_config"] = ((cfg["checkpoint"], cfg["expected_step"], cfg["elastic_schedule"], cfg["seed"], cfg["episodes"], cfg["steps"],
                         cfg["n_envs"], cfg["max_episode_steps"], cfg["render"], cfg["gpus"], cfg["servers_per_gpu"], cfg["parallel"],
                         len(cfg["tasks"]), cfg["out"])
                        == (f"{B}/runs/GR1-E0/epoch_{e:02d}.pt", e * PER, sched, 0, 50, "12", 1, 720, "egl", "4,5,6,7", 2, 24, 24, f"{B}/eval/{name}"))
    sm = json.loads((d / "summary.json").read_text())
    s12 = sm["by_n_action_steps"]["12"]
    r["summary_json_equal"] = (s12["complete"] is True and s12["tasks"] == 24 and abs(s12["mean_success"] - total / 1200) < 1e-12
                               and sm["checkpoint"] == cfg["checkpoint"]
                               and {t.replace(SUFFIX, ""): v["successes"] for t, v in s12["per_task"].items()} == succ
                               and all(v["episodes"] == 50 for v in s12["per_task"].values()))
    v = json.loads((F / "eval" / f"{name}_validation.json").read_text())
    r["validation_equal"] = (v["successes"] == total and v["episodes"] == 1200 and v["step"] == e * PER and v["schedule"] == sched
                             and v["export_sha256"] == sha and {t: x["successes"] for t, x in v["per_task"].items()} == succ)
    if name in FLAGS:
        r["validator"] = "flag"
        r["validator_as_documented"] = v["valid"] is False and v["problems"] == [FLAGS[name] + SUFFIX + FLAG_TEXT]
    else:
        r["validator"] = "valid"
        r["validator_as_documented"] = v["valid"] is True and v["problems"] == []
    line = next((l for l in results_md.splitlines() if l.startswith(f"| {e} | {e * PER} | {sched} ")), "")
    r["run_results_md_row"] = f"| {n} | {total}/1200 | {100 * total / 1200:.2f}% | {'PROBLEMS' if name in FLAGS else 'VALID'} |" in line
    olines = [l for l in orch.splitlines() if f"eval epoch {e} {sched} attempt " in l]
    r["orchestrator_line"] = (len(olines) == 1 and "attempt 1 rc=0:" in olines[0]
                              and abs(json.loads(olines[0].split("rc=0: ", 1)[1])["mean"] - total / 1200) < 1e-12)
    net, z = mcnemar(T, ep)
    r["vs_teacher"] = dict(net=net, z=round(z, 4))
    a = ana_ev[(e, sched)]
    r["analysis_json_equal"] = ((a["successes"], a["loops"], a["steps"], a["block_passes"], a["vs_teacher"]["net"], a["export_sha256"], a["validator_valid"])
                                == (total, k, n, n * (6 + 6 * k), net, sha, name not in FLAGS) and abs(a["vs_teacher"]["z"] - z) < 1e-9)
    cell = main[label][3 + e]
    r["readme_cell"] = (cell == f"{total} ({100 * total / 1200:.2f}%)" + (" *" if name in FLAGS else "")
                        and main[label][1:4] == [str(k), str(n), str(n * (6 + 6 * k))])
    if e in (4, 5):
        col = 2 if e == 4 else 4
        r["readme_vs_teacher_row"] = (versus[label][col] == f"{total} ({100 * total / 1200:.2f}%)"
                                      and versus[label][col + 1] == f"{net:+d} (z {z:+.2f})" and versus[label][1] == str(n * (6 + 6 * k)))
    r["passed"] = all(v for kk, v in r.items() if isinstance(v, bool))
    if not r["passed"]:
        fail(name)
    rec["evaluations"][name] = r

C = rec["checks"]
C["25 evaluations, the expected epoch/schedule set"] = (len(data) == 25 and sorted(data) == sorted(
    [(e, "flat_k1_n2") for e in (1, 2, 3)] + [(e, s) for e in (2, 3) for s in ("flat_k2_n2", "flat_k4_n2")]
    + [(e, f"flat_k{k}_n{n}") for e in (4, 5) for k in (1, 2, 4) for n in (2, 5, 10)]))
C["README 'not run' cells are exactly the settings without an evaluation"] = all(
    (main[lab][3 + e] == "not run") == ((e, f"flat_k{k}_n{n}") not in data)
    for k in (1, 2, 4) for n in (2, 5, 10) for lab in [f"[{k},{k}]" if n == 2 else f"[{k}]x{n}"] for e in range(1, 6))
C["validator flags are exactly the two documented ones"] = sorted(n for n, r in rec["evaluations"].items() if r["validator"] == "flag") == sorted(FLAGS)

# teacher
C["teacher counts == per-episode evidence == teacher selection == analysis.json == README"] = (
    tsum == {9: 626, 10: 620} and sel["evidence"]["9"]["successes"] == tsum[9] and sel["evidence"]["10"]["successes"] == tsum[10]
    and ana["teacher"] == {"9": tsum[9], "10": tsum[10]} and f"**{tsum[9]} ({100 * tsum[9] / 1200:.2f}%)**" in readme)
C["teacher selection follows its rule"] = (sel["chosen_epoch"] == (10 if tsum[10] > tsum[9] else 9) == 9 and sel["step"] == 211644
                                           and sel["teacher"].endswith("/epoch_09.pt"))
C["teacher protocol == student protocol (seed 0, 50 episodes, 12 action steps, 720 steps, noise seed 42)"] = all(
    x["seed"] == 0 and x["n_episodes"] == 50 and x["n_action_steps"] == 12 and x["max_episode_steps"] == 720
    and x["server_metadata"]["noise_seed"] == 42 and x["server_metadata"]["step"] == 211644 for x in T.values())

# epoch 4 -> 5, pooled loops
ok = True
tot = {4: 0, 5: 0}
up = down = 0
for k in (1, 2, 4):
    for n in (2, 5, 10):
        s = f"flat_k{k}_n{n}"
        lab = f"[{k},{k}]" if n == 2 else f"[{k}]x{n}"
        net, z = mcnemar(data[(4, s)], data[(5, s)])
        g = ana["grid"][s]
        ok &= g["epoch4_to_5"]["net"] == net and abs(g["epoch4_to_5"]["z"] - z) < 1e-9 and versus[lab][6] == f"{net:+d} (z {z:+.2f})"
        for e in (4, 5):
            tot[e] += rec["evaluations"][f"e{e:02d}_{s}"]["successes"]
        for t in data[(4, s)]:
            for x, y in zip(data[(4, s)][t]["successes"], data[(5, s)][t]["successes"]):
                up += (not x) and bool(y)
                down += bool(x) and (not y)
zall = (up - down) / math.sqrt(up + down)
ok &= (ana["grid"]["epoch4_all_nine"], ana["grid"]["epoch5_all_nine"], ana["grid"]["all_nine_epoch4_to_5"]["net"]) == (tot[4], tot[5], up - down)
ok &= abs(ana["grid"]["all_nine_epoch4_to_5"]["z"] - zall) < 1e-9
ok &= f"total {tot[4]:,} and {tot[5]:,} of 10,800 (paired net {up - down:+d}, z {zall:+.2f})" in readme
C["epoch 4 -> 5 changes (nine settings and their total) == analysis.json == README"] = bool(ok)
rec["epoch4_vs_epoch5"] = dict(epoch4=tot[4], epoch5=tot[5], net=up - down, z=round(zall, 4))

pool, ok = {}, True
for k in (1, 2, 4):
    pool[k] = sum(rec["evaluations"][f"e{e:02d}_flat_k{k}_n{n}"]["successes"] for e in (4, 5) for n in (2, 5, 10))
    ok &= ana["pooled"]["epoch 4 and 5"][f"loops_{k}"] == pool[k] and pooled_rows[str(k)][1] == f"{pool[k]} ({100 * pool[k] / 7200:.2f}%)"
rec["pooled_loops_epochs_4_and_5"] = dict(successes_of_7200=pool)
for a, b in ((1, 2), (2, 4)):
    up = down = 0
    for e in (4, 5):
        for n in (2, 5, 10):
            A_, B_ = data[(e, f"flat_k{a}_n{n}")], data[(e, f"flat_k{b}_n{n}")]
            for t in A_:
                for x, y in zip(A_[t]["successes"], B_[t]["successes"]):
                    up += (not x) and bool(y)
                    down += bool(x) and (not y)
    z = (up - down) / math.sqrt(up + down)
    p = ana["pooled"]["epoch 4 and 5"][f"loops_{a}_to_{b}"]
    ok &= p["net"] == up - down == pool[b] - pool[a] and abs(p["z"] - z) < 1e-9 and pooled_rows[str(b)][2] == f"{up - down:+d} (z {z:+.2f})"
    rec["pooled_loops_epochs_4_and_5"][f"loops_{a}_to_{b}"] = dict(net=up - down, z=round(z, 4))
C["pooled loop counts and their paired changes == analysis.json == README"] = bool(ok)

# every paired contrast in analysis.json that names two single runs
ok, n_checked = True, 0
for key, p in ana["contrasts"].items():
    m = re.fullmatch(r"(flat_k\d+_n\d+) epoch (\d) -> (\d)", key)
    m2 = re.fullmatch(r"epoch (\d) loops (\d) steps (\d+) -> (\d+)", key)
    m3 = re.fullmatch(r"epoch (\d) steps (\d+) loops (\d) -> (\d)", key)
    if m:
        a, b = data[(int(m[2]), m[1])], data[(int(m[3]), m[1])]
    elif m2:
        a, b = data[(int(m2[1]), f"flat_k{m2[2]}_n{m2[3]}")], data[(int(m2[1]), f"flat_k{m2[2]}_n{m2[4]}")]
    elif m3:
        a, b = data[(int(m3[1]), f"flat_k{m3[3]}_n{m3[2]}")], data[(int(m3[1]), f"flat_k{m3[4]}_n{m3[2]}")]
    else:
        continue
    net, z = mcnemar(a, b)
    ok &= p["net"] == net and abs(p["z"] - z) < 1e-9
    n_checked += 1
C[f"all {n_checked} single-run paired contrasts in analysis.json recomputed"] = bool(ok) and n_checked == 22 + 18 + 24

# per-task csv
with open(F / "analysis" / "per_task_successes.csv") as f:
    table = list(csv.reader(f))
head = table[0]
ok = len(table) == 26 and table[-1][0].startswith("TOTAL")
cols = {"teacher e9": T, "teacher e10": teacher[10]}
for (e, s), ep in data.items():
    k, n = (int(x[1:]) for x in s.split("_")[1:])
    cols[f"e{e} " + (f"[{k},{k}]" if n == 2 else f"[{k}]x{n}")] = ep
ok &= sorted(head[1:]) == sorted(cols)
for row in table[1:-1]:
    ok &= all(int(row[i]) == sum(bool(x) for x in cols[head[i]][row[0]]["successes"]) for i in range(1, len(head)))
ok &= all(int(table[-1][i]) == sum(sum(bool(x) for x in v["successes"]) for v in cols[head[i]].values()) for i in range(1, len(head)))
C["analysis/per_task_successes.csv == per-episode evidence (27 columns x 24 tasks and totals)"] = bool(ok) and len(head) == 28

# training
count, prev, good = 0, 0, True
last = None
with gzip.open(F / "training" / "metrics.jsonl.gz", "rt") as f:
    for line in f:
        last = json.loads(line)
        good &= last["update"] == prev + 1 and last["epoch"] == (last["update"] - 1) // PER + 1
        prev = last["update"]
        count += 1
status = json.loads((F / "training" / "status.json").read_text())
cfg = json.loads((F / "training" / "config.json").read_text())
launches = [json.loads(l) for l in (F / "training" / "launches.jsonl").read_text().splitlines()]
C["training metrics: updates 1-235,160 once each, in order"] = bool(good) and count == prev == 5 * PER
C["training status/config agree (complete, 5 epochs, 235,160 updates, batch 128, 6,020,058 windows, teacher, revision)"] = (
    (status["status"], status["epoch"], status["update"], status["windows_seen"], status["scheduler"]["total"], status["scheduler"]["warmup"])
    == ("complete", 5, 5 * PER, 5 * 6020058, 5 * PER, 7055) and last["windows_seen"] == 5 * 6020058
    and (cfg["epochs"], cfg["planned_updates"], cfg["global_batch"], cfg["world"], cfg["microbatch"], cfg["windows"]) == (5, 5 * PER, 128, 4, 32, 6020058)
    and math.ceil(6020058 / 128) == PER and cfg["asset_sha256"]["teacher"] == sel["teacher_sha256"] == cfg["config"]["parent"]["sha256"]
    and cfg["config"]["parent"]["epoch"] == 9 and cfg["git_revision"] == (F / "record" / "REVISION").read_text().strip()
    and cfg["config"]["schedules"] == [f"flat_k{k}_n{n}" for k in (1, 2, 4) for n in (2, 5, 10)]
    and cfg["config"]["end_prefix"] == 12 and cfg["config"]["gripper_loss"] is False
    and cfg["asset_sha256"]["elastic_init"] == json.loads((F / "training" / "checkpoints_sha256.json").read_text())["init"]["gr1_common_init_v1.pt"]["sha256"])
C["six launches, all at the pinned revision, targets 1,2,3,4,5,5"] = ([l["target_epoch"] for l in launches] == [1, 2, 3, 4, 5, 5]
                                                                     and {l["revision"] for l in launches} == {cfg["git_revision"]}
                                                                     and [l["resume"] for l in launches] == [False] + [True] * 5)
C["final learning rates are 1% of 1e-3 and 3e-5"] = all(abs(a / b - 1) < 1e-9 for a, b in zip(last["lr"], (1e-5, 3e-7)))
ver = [json.loads((F / "record" / "validation" / f"verify_GR1-E0_e{e:02d}.json").read_text()) for e in range(1, 6)]
C["every epoch export passed its verification"] = all(v["passed"] is True and v["pairs"][f"epoch_{e:02d}.pt"]["passed"] is True
                                                      and all(s["bit_equal"] for s in v["server"]) for e, v in enumerate(ver, 1))

# Hugging Face receipt
r = json.loads((F / "hf" / "GR1-E0.epoch_05.pt.uploaded.json").read_text())
log = (F / "hf" / "publish.log").read_text(errors="replace")
C["HF receipt == epoch_05.pt sha256/size/step, commit in the upload log, result listed at upload is [1,1] only"] = (
    (r["repo"], r["path"], r["sha256"], r["size"], r["step"]) == ("anhdao69/LoopWAM_NT", "Robocasa-GR1/Elastic/E0/epoch_05.pt",
                                                                 ck["epoch_05.pt"]["sha256"], ck["epoch_05.pt"]["bytes"], 5 * PER)
    and r["evals_at_upload"] == {"flat_k1_n2": rec["evaluations"]["e05_flat_k1_n2"]["successes"]}
    and f'"commit": "{r["revision"]}"' in log and f'"sha256": "{r["sha256"]}"' in log
    and r["files_verified"]["Robocasa-GR1/Elastic/E0/epoch_05.pt"] == r["size"]
    and f"commit `{r['revision'][:8]}`" in readme)

# SOURCES.tsv against the stored bytes
src = [l.split("\t") for l in (F / "SOURCES.tsv").read_text().splitlines()[1:]]
bad = []
for path, _host, _src, sha, size, stored, _chk in src:
    b = (F / path).read_bytes()
    if stored == "gzip":
        b = gzip.decompress(b)
    if hashlib.sha256(b).hexdigest() != sha or len(b) != int(size):
        bad.append(path)
stored_files = {str(p.relative_to(F)) for p in F.rglob("*") if p.is_file()}
generated = {"README.md", "SOURCES.tsv", "excluded_files.tsv", "independent_check.json", "eval/media_manifest.tsv.gz",
             "training/checkpoints_sha256.json"} | {f"analysis/{n}" for n in (
                 "analyze_gr1_e0.py", "results.md", "per_task.md", "per_task_successes.csv", "training_curves.md", "analysis.json",
                 "video_flag_check.py", "video_flag_check_output.md")}
C[f"SOURCES.tsv: {len(src)} stored files equal the recorded SHA-256 and size"] = not bad and len(src) == 1736
C["folder contents == SOURCES.tsv rows + the listed generated files"] = stored_files - {"independent_check.json"} == ({s[0] for s in src} | generated) - {"independent_check.json"}
with gzip.open(F / "eval" / "media_manifest.tsv.gz", "rt") as f:
    media = f.read().splitlines()[1:]
C["media manifest: 1,200 videos per evaluation (30,000) + 4 smoke-test videos"] = (
    len(media) == 30004 and all(sum(1 for l in media if f"/eval/{d.name}/videos/n12/" in l) == 1200 for d in dirs))
rec["media"] = dict(videos=len(media), bytes=sum(int(l.split("\t")[1]) for l in media))
C["README media figures"] = f"{len(media):,} rollout videos ({rec['media']['bytes'] / 1e9:.1f} GB)" in readme

for k, v in C.items():
    if not v:
        fail(k)
rec["passed"] = not rec["failures"]
rec["description"] = __doc__.split("Checks, ", 1)[1].strip()
(F / "independent_check.json").write_text(json.dumps(rec, indent=1) + "\n")
print(json.dumps(dict(passed=rec["passed"], failures=rec["failures"],
                      evaluations={n: (r["successes"], r["vs_teacher"]["net"], r["passed"]) for n, r in rec["evaluations"].items()},
                      checks=C), indent=1))
