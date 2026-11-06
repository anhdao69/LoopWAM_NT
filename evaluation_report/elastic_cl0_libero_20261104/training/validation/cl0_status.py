"""CL-0 campaign status: per-run progress, loss/grad means, rate, nonfinite check, ETA (UTC and EST)."""
import json, sys, time, datetime, os
from pathlib import Path
runs = Path(sys.argv[1]); order = ["CL0-E0", "CL0-E1", "CL0-E2"]; total = 21700; now = time.time()
est = datetime.timezone(datetime.timedelta(hours=-5))
fmt = lambda t: (datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%m-%d %H:%M UTC") + " / " +
                 datetime.datetime.fromtimestamp(t, est).strftime("%m-%d %I:%M %p EST"))
eta = None
for name in order:
    m = runs / name / "metrics.jsonl"
    if not m.exists() or m.stat().st_size == 0:
        print(name, "not started"); continue
    rows = [json.loads(l) for l in open(m)]
    last = rows[-1]; secs = [r["seconds"] for r in rows[20:]] or [r["seconds"] for r in rows]; tail = secs[-500:]
    start = json.loads(open(runs / name / "launches.jsonl").readline())["time"]
    wall = now - datetime.datetime.fromisoformat(start).timestamp()
    w = rows[-100:]; mean = lambda k: sum(r[k] for r in w) / len(w)
    bad = [r["update"] for r in rows if not all(v == v and abs(v) < 1e6 for v in (r["local"], r["end"], r["grip"], r["anchor"], r["grad_norm"]))]
    print(f"{name}: update {last['update']}/{total} epoch {last['epoch']} | s/update mean {sum(secs)/len(secs):.3f} "
          f"last500 {sum(tail)/len(tail):.3f} wall {wall/len(rows):.3f} | last100 local {mean('local'):.4f} end {mean('end'):.4f} "
          f"grip {mean('grip'):.4f} anchor {mean('anchor'):.4f} grad {mean('grad_norm'):.3f} | peak {max(r['peak_allocated_gb'] for r in rows):.1f} GB | nonfinite {bad[:3]}")
    if last["update"] < total:
        rate = wall / len(rows); finish = now + (total - last["update"]) * rate
        print(f"  {name} finish ~ {fmt(finish)}")
        later = [n for n in order[order.index(name) + 1:]]
        for n in later:  # following runs at the same rate plus ~3 min startup/export
            finish += total * rate * (1.08 if n == "CL0-E2" else 1.0) + 180
            print(f"  {n} finish ~ {fmt(finish)} (projected)")
        break
