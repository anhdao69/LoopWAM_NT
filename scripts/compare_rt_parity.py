#!/usr/bin/env python3
"""Compare local RT-A S2 parity groups with the remote campaign's per-episode records."""
import csv, json, sys
from pathlib import Path
local_dir, csv_path, model = Path(sys.argv[1]), sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "RT-A S2"
remote = {(r["suite"], int(r["seed"]), int(r["task"]), int(r["episode"])): r
          for r in csv.DictReader(open(csv_path)) if r["model"] == model}
rows = []
for p in sorted((local_dir / "groups").glob("*.json")):
    for e in json.loads(p.read_text())["episodes"]:
        r = remote[(e["suite"], e["seed"], e["task"], e["episode"])]
        rows.append((e, r))
n = len(rows)
if not n: sys.exit("no completed groups yet")
ls = sum(e["success"] for e, _ in rows); rs = sum(r["success"] == "True" for _, r in rows)
agree = sum(e["success"] == (r["success"] == "True") for e, r in rows)
same_steps = sum(e["policy_steps"] == int(r["policy_steps"]) for e, r in rows)
print(json.dumps(dict(episodes=n, local_success=ls, remote_success=rs, local_sr=ls / n, remote_sr=rs / n,
                      outcome_agreement=agree / n, identical_policy_steps=same_steps / n,
                      local_only=sum(e["success"] and r["success"] != "True" for e, r in rows),
                      remote_only=sum((not e["success"]) and r["success"] == "True" for e, r in rows)), indent=1))
by = {}
for e, r in rows:
    k = (e["suite"], e["task"]); b = by.setdefault(k, [0, 0, 0])
    b[0] += e["success"]; b[1] += r["success"] == "True"; b[2] += 1
for k, v in sorted(by.items()): print(k, "local", v[0], "remote", v[1], "of", v[2])
