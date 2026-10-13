#!/usr/bin/env python3
"""Fill epoch-10 rows of Residual_Transport_Results.{csv,md} from h100_eval/conditions.json (complete conditions only)."""
import csv, io, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERF = ROOT / "plans/performance"
conds = {k: v for k, v in json.loads((ROOT / "plans/residual_transport_evidence/h100_eval/conditions.json").read_text()).items()
         if v["complete"]}
REMOTE_REPORT = "evaluation_report/campaign"
H100_REPORT = "plans/performance/Residual_Transport_H100_Evaluation_20261013.md"


def pct(v):
    return f"{v:.2f}"


# CSV: fill the epoch_10, fixed-noise row of each complete condition.
path = PERF / "Residual_Transport_Results.csv"
rows = list(csv.DictReader(io.StringIO(path.read_text())))
fields = list(rows[0].keys())
for r in rows:
    c = conds.get(f"{r['run']} {r['schedule']}")
    if not c or r["checkpoint"] != "epoch_10" or r["noise_mode"] != "fixed":
        continue
    s = c["per_suite"]
    r.update(checkpoint_sha256=c["checkpoint_sha256"], spatial_success_pct=pct(s["libero_spatial"]["success_pct"]),
             object_success_pct=pct(s["libero_object"]["success_pct"]), goal_success_pct=pct(s["libero_goal"]["success_pct"]),
             long_success_pct=pct(s["libero_10"]["success_pct"]), four_suite_mean_pct=pct(c["four_suite_mean_pct"]),
             gpu_model=c["gpu"] if c["source"] == "remote campaign" else "NVIDIA H100 80GB HBM3",
             eval_report=REMOTE_REPORT if c["source"] == "remote campaign" else H100_REPORT)
    note = f"{c['successes']}/6000 episodes; seed std {c['seed_std_pct']:.2f} pp"
    keep = [x for x in r["notes"].split("; ") if x and "/6000 episodes" not in x and not x.startswith("seed std")]
    r["notes"] = "; ".join(keep + [note])
out = io.StringIO(); w = csv.DictWriter(out, fieldnames=fields, lineterminator="\n"); w.writeheader(); w.writerows(rows)
path.write_text(out.getvalue())

# Markdown summary tables: Final checkpoint/hash, Long fixed %, Four-suite mean %, Report.
md = PERF / "Residual_Transport_Results.md"
lines = md.read_text().split("\n")
for i, line in enumerate(lines):
    m = re.match(r"^\| (RT[^|]*?) \| (\S+) \| (\d+) \|", line)
    if not m:
        continue
    c = conds.get(f"{m.group(1).strip()} {m.group(2)}")
    if not c:
        continue
    cells = [x.strip() for x in line.strip("|").split("|")]
    if len(cells) != 10:
        continue
    cells[3] = f"epoch_10 / {c['checkpoint_sha256'][:12]}"
    cells[6] = pct(c["per_suite"]["libero_10"]["success_pct"])
    cells[8] = pct(c["four_suite_mean_pct"])
    cells[9] = ("[remote campaign](../../evaluation_report/campaign/report.html)" if c["source"] == "remote campaign"
                else "[H100](Residual_Transport_H100_Evaluation_20261013.md)")
    lines[i] = "| " + " | ".join(cells) + " |"
md.write_text("\n".join(lines))
print("filled", sorted(conds))
