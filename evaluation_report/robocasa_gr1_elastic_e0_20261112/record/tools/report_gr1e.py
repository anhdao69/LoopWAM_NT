#!/usr/bin/env python3
"""RESULTS.md of the GR1-E0 x5 campaign: teacher choice and every validated official eval (epoch x schedule).
Usage: report_gr1e.py RECORD_DIR"""
import json, sys, time
from pathlib import Path

B = Path(sys.argv[1]); EV = B / "eval"
LOOPS = {"flat_k1_n2": "[1, 1]", "flat_k4_n2": "[4, 4]", "flat_k4_n5": "[4]x5"}
lines = [f"# GR1-E0 x5 (elastic E0 on RoboCasa-GR1) - official evaluation results", "",
         f"Updated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}. Protocol: 24 tasks x 50 episodes, n_action_steps 12, "
         "max 720 steps, seed 0, EGL (the GR1 v0 continuation protocol). Rates are any-step success over 1200 episodes.", ""]
try:
    sel = json.loads((B / "teacher_selection.json").read_text())
    ev = sel["evidence"]
    lines += [f"Teacher: GR1 v0 continuation epoch {sel['chosen_epoch']} (sha256 {sel['teacher_sha256'][:16]}...) - {sel['reason']}.",
              f"Teacher official results: epoch 9 {ev['9']['successes']}/1200"
              + (f", epoch 10 {ev['10']['successes']}/1200" + ("" if ev['10']['valid'] else " (not valid)") if ev['10'].get('present') else ""), ""]
except Exception as e:
    lines += [f"Teacher selection: not made yet ({type(e).__name__})", ""]
lines += ["| epoch | update | schedule | NFE | success | rate | valid |", "|---|---|---|---|---|---|---|"]
for f in sorted(EV.glob("e*_validation.json")):
    d = json.loads(f.read_text()); epoch = int(f.name[1:3])
    lines.append(f"| {epoch} | {d['step']} | {d['schedule']} {LOOPS.get(d['schedule'], '')} | {int(d['schedule'].split('_n')[1])} "
                 f"| {d['successes']}/{d['episodes']} | {100 * d['mean_success']:.2f}% | {'VALID' if d['valid'] else 'PROBLEMS'} |")
try:
    st = json.loads((B / "STATUS.json").read_text()); lines += ["", f"Status: {st['phase']} - {st['detail']} ({st['utc']})"]
except Exception:
    pass
(B / "RESULTS.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
