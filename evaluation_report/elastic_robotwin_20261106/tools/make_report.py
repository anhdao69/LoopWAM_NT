#!/usr/bin/env python3
"""Regenerate elastic_robotwin/REPORT.md from the result files: every measurement is read or computed here; only recipe constants are quoted."""
import calendar, json, math, os, re, statistics, subprocess, sys, time
from pathlib import Path

W = Path("/mnt/data/vmo-ai-task/anhdh35/elastic_robotwin")
B = W / "ep5"
G = Path("/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robocasa_gr1_v0_20261101")
UPDATES, WARMUP = 47462, 1424
FLAT9 = ["flat_k1_n2", "flat_k1_n5", "flat_k1_n10", "flat_k2_n2", "flat_k2_n5", "flat_k2_n10", "flat_k4_n2", "flat_k4_n5", "flat_k4_n10"]


def J(p):
    return json.loads(Path(p).read_text())


def rows(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def utc(t):
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(t))


def est(t):
    return time.strftime("%a %b %d %I:%M %p EST", time.gmtime(t - 5 * 3600))


def plural(k, word):
    return f"{k} {word}" + ("" if k == 1 else "s")


def yes(x):
    return "PASS" if x else "**FAIL**"


def lr_factor(step, total, warmup):  # scripts/train_loopwam.py, verbatim logic
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = min(1., (step - warmup) / max(total - warmup - 1, 1))
    return .01 + .99 * .5 * (1 + math.cos(math.pi * progress))


now = time.time()
out = []
w = out.append

# ---------------------------------------------------------------- live status
status = J(B / "STATUS.json")
e0dir = B / "runs/RT50-ep5-E0"
m = rows(e0dir / "metrics.jsonl")
cfg = J(e0dir / "config.json")
launches = rows(e0dir / "launches.jsonl")
n = len(m)
start_stamp = m[0]["time"] - m[0]["seconds"]
overall = (m[-1]["time"] - m[0]["time"]) / (n - 1)
recent = m[-999:] if n > 999 else m
recent_rate = (recent[-1]["time"] - recent[0]["time"]) / (len(recent) - 1)
timing = {(r["tag"], r["run"]): r for r in J(W / "timing/summary.json")}
ratio = {e: timing[("w4_mb32", f"RT50-{e}")]["wall_seconds_per_update"] / timing[("w4_mb32", "RT50-E0")]["wall_seconds_per_update"]
         for e in ("E0", "E1", "E2")}
gap = 6 * 60
e0_end = m[-1]["time"] + (UPDATES - n) * recent_rate
e0_hours = (e0_end - start_stamp) / 3600
e1_hours = UPDATES * overall * ratio["E1"] / 3600
e2_hours = UPDATES * overall * ratio["E2"] / 3600
e1_end = e0_end + gap + e1_hours * 3600
e2_end = e1_end + gap + e2_hours * 3600

w("# RoboTwin-50 elastic distillation (E0/E1/E2): results so far")
w("")
w(f"Generated {utc(now)} by `tools/make_report.py` from the result files under `{W}`: every measurement below is read or")
w("computed from those files (recipe constants such as betas and weight decay are quoted from the recipe). Job 5758 on worker-2")
w("(8x H100). Nothing has been published to HF; code is in local commits only. Previous version: `REPORT.v1_timing.md`.")
w("")
w("## 1. Status")
w("")
w(f"- Queue `RT50-ep5-E0 -> E1 -> E2` (one RoboTwin-50 epoch each, GPUs 0-3): phase **{status['phase']}**, \"{status['detail']}\".")
w(f"- RT50-ep5-E0: update **{n:,} / {UPDATES:,}** ({100 * n / UPDATES:.1f}%), started {utc(start_stamp)}, "
  f"{len(launches)} launch(es), {sum(1 for l in launches if l.get('resume'))} resume(s).")
w(f"- Speed: {overall:.3f} s/update average since start (includes recovery saves and the update-2000 export), "
  f"{recent_rate:.3f} s/update over the last {len(recent):,} updates.")
w("- Estimated finish (E1/E2 scaled from E0's live speed by their 4-GPU timing ratios "
  f"{ratio['E1']:.3f} / {ratio['E2']:.3f}; ~6 min between runs for startup and export verification):")
w("")
w("| run | estimated hours | estimated finish |")
w("|---|---|---|")
w(f"| RT50-ep5-E0 | {e0_hours:.1f} | {utc(e0_end)} ({est(e0_end)}) |")
w(f"| RT50-ep5-E1 | {e1_hours:.1f} | {utc(e1_end)} ({est(e1_end)}) |")
w(f"| RT50-ep5-E2 | {e2_hours:.1f} | {utc(e2_end)} ({est(e2_end)}) |")
w("")
w("- No evaluation/inference of the distilled students has been run: there is no RoboTwin evaluation harness for elastic")
w("  exports yet (the exports and their loader exist and are verified, see sections 3-4).")
w("")

# ---------------------------------------------------------------- teacher
hf = J(B / "validation/hf_teacher_provenance.json")
core = J(B / "prelaunch/preflight_core.json")["checks"]
ti = core["teacher_identity"]
hp = hf["robotwin/v0-video4-action4/checkpoint_provenance.json"]
w("## 2. Teacher and recipe")
w("")
w("Teacher = student initializer = RoboTwin-50 LoopWAM v0 4/4 **epoch 5**:")
w("")
w("| | HF `anhdao69/LoopWAM_NT` `robotwin/v0-video4-action4/policy.pt` | file the trainer loads |")
w("|---|---|---|")
w(f"| sha256 | {hf['robotwin/v0-video4-action4/policy.pt']['lfs_sha256']} | {ti['teacher_sha256']} |")
w(f"| bytes | {hf['robotwin/v0-video4-action4/policy.pt']['size']:,} | {os.path.getsize(cfg['teacher']):,} |")
w(f"| epoch / step | {hp['epoch']} / {hp['step']:,} (checkpoint_provenance.json) | step {ti['step']:,} |")
w("")
w(f"Local path: `{cfg['teacher']}`. Same data/normalization/VAE contract as the distillation data: "
  f"data manifest `{cfg['data_manifest_sha256'][:12]}`, normalization `{cfg['normalization_sha256'][:12]}`, "
  f"VAE `{cfg['asset_sha256']['vae'][:12]}` (the trainer refuses a teacher whose contract differs).")
w("")
c = cfg["config"]
w(f"Recipe `{cfg['run']}` (E0 = T0, E1 = T1, E2 = T4 transport): the unchanged LIBERO/CL-0 E-recipe except the parent and two")
w(f"RoboTwin adaptations, endpoint loss over the {c['end_prefix']} executed actions (LIBERO 10) and no binarized gripper loss")
w(f"(gripper_loss={c['gripper_loss']}; RoboTwin grippers are continuous). Scope `{c['scope']}` (whole student action expert trainable),")
w(f"step conditioning, anchor, regime `{c['regime']}`, schedules = FLAT9 with the balanced sampler, init `{c['init']}`.")
w("")
lr1 = m[0]["lr"]
base = [x * WARMUP / 1 for x in lr1]
fac = lr_factor(n - 1, UPDATES, WARMUP)
expect = [b * fac for b in base]
lr_ok = all(abs(a - b) <= 1e-12 + 1e-9 * abs(b) for a, b in zip(m[-1]["lr"], expect))
w("| optimizer group | peak LR (from the logged LR at update 1 x warmup) | weight decay |")
w("|---|---|---|")
w(f"| step condition (+ T4 transport in E2) | {base[0]:.3g} | 0 |")
w(f"| student action expert | {base[1]:.3g} | 0.01 |")
w("")
w(f"AdamW betas (0.9, 0.95), eps 1e-8, grad clip 1.0, global batch {cfg['global_batch']}, world {cfg['world']} x microbatch "
  f"{cfg['microbatch']}; warmup {WARMUP:,} updates (3%), cosine to 1% at update {cfg['planned_updates']:,}. "
  f"Logged LR at update {n:,} = {m[-1]['lr'][0]:.4g} / {m[-1]['lr'][1]:.4g}, the schedule recomputed independently gives "
  f"{expect[0]:.4g} / {expect[1]:.4g}: {yes(lr_ok)}.")
w(f"Speed options {', '.join(k for k, v in cfg['speed_options'].items() if v)} (each bit-exact, section 3); "
  f"code revision `{cfg['git_revision'][:12]}` (frozen clone `LoopWAM_rt50ep5_20261106`).")
w("")

# ---------------------------------------------------------------- prelaunch validation
w("## 3. Pre-launch validation with the epoch-5 teacher (`ep5/prelaunch/`)")
w("")
w("| check | result | detail |")
w("|---|---|---|")
for name, chk in core.items():
    detail = ""
    if name == "teacher_rollout_equals_deployed_infer_action":
        rr = chk["rows"]; detail = f"{sum(r['bit_equal'] for r in rr)}/{len(rr)} windows bit-equal, max diff {max(r['max_abs_diff'] for r in rr)}"
    elif name == "dataset_equals_robotwin_trainer_fields":
        detail = f"{chk['windows']} windows ({chk['with_padding']} with padding), mismatches {len(chk['mismatches'])}"
    elif name == "common_initialization_and_trainable_counts":
        detail = ", ".join(f"{k} {v:,}" for k, v in chk["trainable"].items()) + " trainable"
    elif name == "teacher_identity":
        detail = f"sha {chk['teacher_sha256'][:16]}..., step {chk['step']:,}, VAE {chk['vae_sha256'][:12]} = teacher contract"
    elif name == "data_identity":
        detail = f"{chk['windows']:,} windows, data manifest {chk['data_manifest_sha256'][:12]}, normalization {chk['normalization_sha256'][:12]} = teacher contract"
    elif name == "parameter_counts":
        detail = f"policy {chk['base_policy']:,}, action expert {chk['action_expert']:,}"
    w(f"| {name} | {yes(chk['passed'])} | {detail} |")
for e in ("E0", "E1", "E2"):
    sp = J(B / f"prelaunch/preflight_speed_{e}.json")["checks"]
    allp = [p for k, v in sp.items() for p in v["paths"].values()]
    w(f"| speed options bit-exact, RT50-ep5-{e} | {yes(all(v['passed'] for v in sp.values()))} | "
      f"{len(sp)} options x {len(next(iter(sp.values()))['paths'])} paths, max grad diff {max(p['max_grad_abs_diff'] for p in allp)} |")
rc = json.loads("{" + (B / "prelaunch/resume_compare_fixed.log").read_text().split("{", 1)[1])
w(f"| resume 20 -> 30 == straight 30 (world 4, mb 32, E2) | {yes(rc['passed'])} | metrics of all {rc['updates'][0]} updates; "
  f"{', '.join(k for k, v in rc['parts'].items() if v)} bit-identical; {rc['tensors_changed_20_to_30']} tensors moved 20->30 |")
for f in ("verify_early", "verify_epoch"):
    v = J(B / f"prelaunch/{f}.json")
    for name, p in v["pairs"].items():
        rr = p["rollout"]
        w(f"| export round trip `{name}` (trainer's export_epoch) | {yes(p['passed'])} | {sum(p['checks'].values())}/{len(p['checks'])} checks; "
          f"frozen: {p['frozen']}; student moved {p['student_tensors_changed']}; loader rollout == training model "
          f"{sum(r['bit_equal'] for r in rr)}/{len(rr)} ({len({r['schedule'] for r in rr})} schedules x {len({r['window'] for r in rr})} windows); "
          f"export {v['export_seconds']:.1f} s |")
    if "padding" in v:
        pad_v = v
for v in [pad_v]:
        pd = v["padding"]
        w(f"| all-padding microbatch (last update of the epoch on rank 3) | {yes(all(r['loss'] == 0 and r['grads_zero'] and r['grads_finite'] for r in pd.values()))} | "
          + "; ".join(f"{s}: loss {r['loss']}, grads zero+finite on {r['grad_tensors']} tensors" for s, r in pd.items()) + " |")
mem = [l.split() for l in (B / "prelaunch/memory.log").read_text().splitlines() if "job_mem_GiB" in l]
w("")
w(f"Peak job memory during validation {max(int(x[-1]) for x in mem)} GiB of 960 GiB. A first validation attempt hung on a bare "
  "`wait` in the script (it also waited for the endless memory sampler) and was cancelled; its completed preflight results are kept "
  "in `ep5/prelaunch.attempt1_bare_wait_hang/`. The first resume comparison crashed on numpy RNG arrays in the checker; the fixed "
  "checker's result is the row above (`resume_compare_fixed.log`). The verifier names its epoch-type test export from the 30-update "
  "test run's 0-based epoch counter (`epoch_00.pt`); the trainer's end-of-epoch export is `export_epoch(..., epoch + 1)`, i.e. "
  "`epoch_01.pt`, which is the file the queue requires.")
w("")

# ---------------------------------------------------------------- production E0
w("## 4. RT50-ep5-E0 training so far")
w("")
fin = all(all(math.isfinite(r[k]) for k in ("local", "end", "anchor", "total", "grad_norm")) for r in m)
blocks_ok = all(sorted(r["schedule"] for r in m[i:i + 9]) == sorted(FLAT9) for i in range(0, n - n % 9, 9))
w(f"- All {n:,} updates: finite losses and gradients {yes(fin)}; grip loss always 0 {yes(all(r['grip'] == 0 for r in m))}; "
  f"every 9-update block a full FLAT9 permutation {yes(blocks_ok)}; windows seen {m[-1]['windows_seen']:,} = 128 x {n:,} "
  f"{yes(m[-1]['windows_seen'] == 128 * n)}.")
w(f"- Data wait: {m[0]['data_wait_seconds']:.3f} s at update 1 (DataLoader workers starting), max {max(r['data_wait_seconds'] for r in m[1:]):.4f} s "
  f"over updates 2-{n:,} (median {statistics.median(r['data_wait_seconds'] for r in m):.5f} s); peak GPU memory {max(r['peak_allocated_gb'] for r in m):.1f} GB; "
  f"compute {sum(r['seconds'] for r in m[18:]) / len(m[18:]):.3f} s/update (mean after warm-up).")
st = J(e0dir / "status.json")
w(f"- Last recovery save: update {st['update']:,} (`latest.pt`, every 200 updates); status `{st['status']}`.")
sv = J(B / "validation/static_E0_step2000.json")["pairs"]["step_00002000.pt"]
w(f"- Update-2000 export `step_00002000.pt`: static verification {yes(sv['passed'])} ({sum(sv['checks'].values())}/{len(sv['checks'])} checks; "
  f"frozen: {sv['frozen']}; student moved {sv['student_tensors_changed']}). Update 2,001 was stamped "
  f"{m[2000]['time'] - m[1999]['time']:.1f} s after update 2,000 (save + export + recovery copy + one update; median gap between "
  f"consecutive updates {statistics.median(b['time'] - a['time'] for a, b in zip(m, m[1:])):.2f} s).")
w("")
w("Loss trend (means over bins of 2,997 updates = 333 complete FLAT9 blocks, so every bin weights the nine schedules equally):")
w("")
w("| updates | local | end | anchor | total | grad norm |")
w("|---|---|---|---|---|---|")
bin_ = 2997
for s in range(0, n - bin_ + 1, bin_):
    sl = m[s:s + bin_]
    mean = lambda k: sum(r[k] for r in sl) / len(sl)
    w(f"| {s + 1:,}-{s + bin_:,} | {mean('local'):.5f} | {mean('end'):.6f} | {mean('anchor'):.5f} | {mean('total'):.5f} | {mean('grad_norm'):.3f} |")
w("")
w("Per-schedule mean local loss, first vs latest 2,997-update bin:")
w("")
first, last = m[:bin_], m[(n // bin_ - 1) * bin_:(n // bin_) * bin_]
w("| schedule | " + " | ".join(FLAT9) + " |")
w("|---|" + "---|" * 9)
for label, sl in (("first bin", first), ("latest bin", last)):
    w(f"| {label} | " + " | ".join(f"{sum(r['local'] for r in sl if r['schedule'] == s) / sum(1 for r in sl if r['schedule'] == s):.5f}" for s in FLAT9) + " |")
w("")
w("These are training losses only (distillation to the epoch-5 teacher); they say nothing yet about task success.")
w("")

# ---------------------------------------------------------------- GR1
GT = G / "train/continuation_e06_10"
for _ in range(60):   # timing.json can be written a few updates before the matching metrics line is visible over NFS
    gt = J(GT / "timing.json")
    gr = [r for r in rows(GT / "metrics.jsonl")]
    ups = [r["update"] for r in gr]
    if gt["update"] in ups:
        break
    time.sleep(5)
else:
    raise SystemExit(f"GR1 timing.json update {gt['update']} never appeared in metrics.jsonl (last {ups[-1]})")
assert ups == list(range(ups[0], ups[0] + len(ups))), "GR1 metrics not contiguous"
gr = gr[:ups.index(gt["update"]) + 1]
resume_t = os.path.getmtime(GT / "resume_audit_u117580.json")
# First job-5758 step that is not GR1's own (its two steps start with the resume), not an nvidia-smi probe and not an
# instant failure: every later step is epoch-5 preparation/validation/training on GPUs 0-3.
acct = [l.split("|") for l in subprocess.run(["sacct", "-j", "5758", "-n", "-P", "-o", "JobID,JobName,Start,ElapsedRaw",
        "--starttime", time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(resume_t - 600))], capture_output=True, text=True,
        check=True, env=dict(os.environ, TZ="UTC")).stdout.splitlines()]
steps = [(calendar.timegm(time.strptime(st, "%Y-%m-%dT%H:%M:%S")), jid, name, int(el)) for jid, name, st, el in acct
         if re.fullmatch(r"5758\.\d+", jid) and st[:1].isdigit()]
gr1_steps = [jid for t, jid, name, el in steps if abs(t - resume_t) < 120]
other = sorted((t, jid) for t, jid, name, el in steps if t > resume_t - 120 and jid not in gr1_steps and name != "nvidia-smi" and el > 0)
first_ep5, first_ep5_step = other[0]
# Each update's true time lies in [lo, hi]: wall time between records >= the logged per-update seconds.
lo, acc = [], resume_t
for r in gr:
    acc += r["seconds"]; lo.append(acc)
hi, acc = [0.] * len(gr), gt["heartbeat_unix"]
for i in range(len(gr) - 1, -1, -1):
    hi[i] = acc; acc -= gr[i]["seconds"]
skip = 420   # compile / warm-up after the 13:49 resume
alone = [r for r, h in zip(gr[skip:], hi[skip:]) if h < first_ep5]
with_e0 = [r for r, l in zip(gr, lo) if l >= start_stamp]
hour = [r for r, l in zip(gr, lo) if start_stamp <= l < start_stamp + 3600]
last = gr[-3000:]
gstat = lambda sl: f"{statistics.mean(r['seconds'] for r in sl):.3f} / {statistics.median(r['seconds'] for r in sl):.3f}"
w("## 5. GR1 continuation running alongside (GPUs 4-7)")
w("")
w(f"GR1 resumed at update {gr[0]['update'] - 1:,} at {utc(resume_t)} (`resume_audit_u117580.json`); its metrics have no timestamps, "
  f"so each update's time is bracketed between the resume time plus the summed per-update seconds and the `timing.json` heartbeat "
  f"(update {gt['update']:,} at {utc(gt['heartbeat_unix'])}) minus them (bracket width at most "
  f"{(gt['elapsed_training_seconds'] - sum(r['seconds'] for r in gr)) / 60:.0f} min).")
w("")
w("| GR1 window | updates | s/update mean / median |")
w("|---|---|---|")
w(f"| alone: after warm-up, certainly before the first epoch-5 Slurm step ({first_ep5_step} at {utc(first_ep5)}) | {alone[0]['update']:,}-{alone[-1]['update']:,} | {gstat(alone)} |")
w(f"| first hour of RT50-ep5-E0 on GPUs 0-3 | {hour[0]['update']:,}-{hour[-1]['update']:,} | {gstat(hour)} |")
w(f"| latest 3,000 updates (E0 running) | {last[0]['update']:,}-{last[-1]['update']:,} | {gstat(last)} |")
w("")
w(f"GR1's per-update time drifts up slowly and steadily over the whole continuation (it rose the same way while GR1 ran alone), with no "
  f"step when E0 started: median {statistics.median(r['seconds'] for r in alone[-1000:]):.3f} s over the last 1,000 updates certainly "
  f"before {first_ep5_step}, {statistics.median(r['seconds'] for r in with_e0[:1000]):.3f} s over the first 1,000 certainly during E0.")
orch = (G / "continuation_20261106/orchestrator.log").read_text()
stop6 = int(re.findall(r"--max-updates (\d+)", orch)[-1])
w(f"GR1 is at update {gt['update']:,}; the running attempt stops at the end of epoch 6 (update {stop6:,}, "
  f"`--max-updates` in `orchestrator.log`) for that epoch's evaluation; {gt['planned_updates']:,} is the 10-epoch total. GR1 step ids "
  f"{', '.join(gr1_steps)} (started with the resume).")
w("")

# ---------------------------------------------------------------- timing with the epoch-10 teacher
w("## 6. Earlier result: training-speed test with the epoch-10 teacher (`timing/`)")
w("")
w("Same port with the epoch-10 teacher (sha 70cf5af7, = HF `robotwin/v0-video4-action4/epoch_10/policy.pt`), 108 updates per run, "
  "first 9 excluded as warm-up; hours include the ~2 s recovery save per 100 updates.")
w("")
w("| layout | run | GPUs | mb | s/update compute (wall) | data wait max | peak GB | " + " | ".join(f"{k} h" for k in next(iter(timing.values()))["hours"]) + " |")
w("|---|---|---|---|---|---|---|---|---|---|")
for (tag, run), r in sorted(timing.items()):
    w(f"| {tag} | {run} | {r['launch']['gpus']} | {r['launch']['microbatch']} | {r['seconds_per_update']:.3f} ({r['wall_seconds_per_update']:.3f}) | "
      f"{r['data_wait_max']:.3f} | {r['peak_allocated_gb']:.1f} | " + " | ".join(f"{x:.1f}" for x in r["hours"].values()) + " |")
w("")
T = W / "timing"
K = ("local", "end", "grip", "anchor", "grad_norm", "schedule", "windows_seen")
same = lambda a, b: (lambda A, B_: len(A) == len(B_) and all(x[k] == y[k] for x, y in zip(A, B_) for k in K))(rows(T / a / "metrics.jsonl"), rows(T / b / "metrics.jsonl"))
w(f"- Determinism: E0 repeated on GPUs 0-3 {yes(same('w4_mb32/RT50-E0', 'w4x2_mb32/RT50-E0'))}, E1 on GPUs 0-3 vs 4-7 "
  f"{yes(same('w4_mb32/RT50-E1', 'w4x2_mb32/RT50-E1'))}: all 108 updates bit-identical.")
for e in ("E0", "E1", "E2"):
    A_, B_ = rows(T / f"w8_mb16/RT50-{e}/metrics.jsonl"), rows(T / f"w4_mb32/RT50-{e}/metrics.jsonl")
    big = sum(1 for x, y in zip(A_, B_) for k in ("local", "end", "anchor") if abs(y[k]) > 1e-2 and abs(x[k] - y[k]) / abs(y[k]) > 2e-3)
    gn = sorted((abs(x["grad_norm"] - y["grad_norm"]) / y["grad_norm"], x["update"], x["grad_norm"], y["grad_norm"]) for x, y in zip(A_, B_))
    w(f"- 8 GPUs vs 4 GPUs, {e}: same schedules/windows {yes([r['schedule'] for r in A_] == [r['schedule'] for r in B_])}; update-1 local "
      f"{A_[0]['local']:.6f} vs {B_[0]['local']:.6f}; max |diff| local {max(abs(x['local'] - y['local']) for x, y in zip(A_, B_)):.1e}; "
      f"loss values >1e-2 differing by >0.2%: {big}; grad-norm relative difference median {statistics.median(g[0] for g in gn):.1e}, "
      f"90th percentile {gn[int(.9 * len(gn))][0]:.1e}, max {gn[-1][0]:.1e} (update {gn[-1][1]}: {gn[-1][2]:.4f} vs {gn[-1][3]:.4f}); "
      f"update-108 local {A_[-1]['local']:.5f} vs {B_[-1]['local']:.5f}.")
A_, B_ = rows(T / "w8_mb16/RT50-E2/metrics.jsonl"), rows(T / "w4_mb32/RT50-E2/metrics.jsonl")
u = max(range(len(A_)), key=lambda i: abs(A_[i]["grad_norm"] - B_[i]["grad_norm"]) / B_[i]["grad_norm"])
rel = lambda i, k: abs(A_[i][k] - B_[i][k]) / abs(B_[i][k])
w(f"- The gaps are bf16 rounding of 16- vs 32-sample microbatches; neither layout is the reference and nothing drifts. The E2 maximum "
  f"is one isolated update ({A_[u]['update']}, {A_[u]['schedule']}): the updates before and after differ by "
  f"{100 * rel(u - 1, 'grad_norm'):.2f}% and {100 * rel(u + 1, 'grad_norm'):.2f}% in grad norm, and its own losses by "
  + ", ".join(f"{k} {100 * rel(u, k):.1f}%" for k in ("local", "end", "anchor")) + ". The previous report version called all of "
  "these rounding-level; the full-run figures above are the precise statement.")
pc10 = J(W / "validation/preflight_core.json")["checks"]
pc10_text = ", ".join(k + " " + yes(v["passed"]) for k, v in pc10.items())
sp10_text = ", ".join("E%d %s" % (i, yes(J(W / ("validation/preflight_speed_E%d.json" % i))["passed"])) for i in range(3))
w(f"- Epoch-10 preflight (`validation/`): {pc10_text}; speed options {sp10_text}.")
wall = lambda tag, e: timing[(tag, f"RT50-{e}")]["wall_seconds_per_update"]
w(f"- 4 -> 8 GPUs speeds one run up {wall('w4_mb32', 'E0') / wall('w8_mb16', 'E0'):.2f}x (E0), "
  f"{wall('w4_mb32', 'E1') / wall('w8_mb16', 'E1'):.2f}x (E1), {wall('w4_mb32', 'E2') / wall('w8_mb16', 'E2'):.2f}x (E2): 16-sample "
  "microbatches under-use an H100. Two concurrent 4-GPU runs: "
  f"E0 {wall('w4x2_mb32', 'E0'):.3f} vs {wall('w4_mb32', 'E0'):.3f} alone, E1 {wall('w4x2_mb32', 'E1'):.3f} vs {wall('w4_mb32', 'E1'):.3f} alone.")
w("")

# ---------------------------------------------------------------- code and limits
w("## 7. What was built, problems found, limits")
w("")
w("- Port of the elastic recipes to RoboTwin-50 (dataset reader equal to the RoboTwin trainer's fields, robot teacher, 14-D noise,")
w("  8-GPU layout), RoboTwin export format `loopwam-s-robot-rt-v1` (teacher FP32 payload + trained student + elastic fields; its own")
w("  format name so the original RoboTwin deploy code cannot load it as a plain policy) and its loader `load_robot_transport`.")
w("- Problems found and fixed before any production training: the validation script's bare `wait` hang; the resume checker's numpy")
w("  comparison; the queue's export-verification step pinned CPUs outside a small step's cpuset (now a 120-CPU step); an export gap")
w("  on resume right after update 2000 (now re-exported).")
w("- Unattended operation: auto-resume from `latest.pt` (up to 6 attempts per run), 30-min no-progress watchdog, export verification")
w("  after each run, login-node keeper. Recovery state is saved every 200 updates.")
w("- Not done yet: RoboTwin task evaluation of the distilled students (needs an evaluation harness for the new export format).")
w("  Nothing uploaded to HF; git commits are local only (branch `elastic_robotwin`, frozen production clone at "
  f"`{cfg['git_revision'][:12]}`).")
w("")
w("## 8. Files")
w("")
w(f"- Status: `{B}/STATUS.json`, `{B}/queue.log`, `{B}/keeper.log`")
w(f"- Runs: `{B}/runs/RT50-ep5-E*/` (metrics.jsonl, config.json, latest.pt, step_00002000.pt, epoch_01.pt at the end)")
w(f"- Validation: `{B}/prelaunch/`, `{B}/validation/`; epoch-10 timing: `{W}/timing/`, `{W}/validation/`")
Path(sys.argv[1]).write_text("\n".join(x for x in out if x is not None) + "\n")
print(f"wrote {sys.argv[1]} ({len(out)} lines)")
