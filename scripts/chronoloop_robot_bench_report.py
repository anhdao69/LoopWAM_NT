#!/usr/bin/env python3
"""Summarize a ChronoLoop robot training-speed benchmark and project full training time.

python scripts/chronoloop_robot_bench_report.py BENCH_DIR SCHEDULE_JSON [--skip 15]

Steady state = updates after --skip (compile warm-up and recompiles). Projection uses valid
windows per second, because masked slots are dropped (time scales with valid windows), and
also reports the plain s/update projection for comparison.
"""
import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('bench_dir')
    p.add_argument('schedule')
    p.add_argument('--skip', type=int, default=15)
    args = p.parse_args()
    bench = Path(args.bench_dir)
    rows = [json.loads(l) for l in (bench / 'metrics.jsonl').read_text().splitlines() if l.strip()]
    rows = [r for r in rows if 'update' in r and 'seconds' in r]
    steady = [r for r in rows if r['update'] > args.skip]
    sec = np.asarray([r['seconds'] for r in steady]); val = np.asarray([r['valid_windows'] for r in steady])
    sched = json.loads(Path(args.schedule).read_text())
    summary = sched['summary']
    updates, valid_total = summary['updates'], summary['valid_windows']
    per_epoch_updates = summary['epoch_complete_update']['1']
    windows_per_epoch = valid_total // sched['epochs']
    wps = val.sum() / sec.sum()
    # GPU telemetry inside the steady-state wall window (metrics carry no timestamps: use timing order).
    gpu = []
    gpath = bench / 'gpu.csv'
    if gpath.exists():
        with gpath.open() as f:
            for row in csv.reader(f):
                try:
                    t = dt.datetime.strptime(row[0].strip(), '%Y/%m/%d %H:%M:%S.%f')
                    gpu.append((t, int(row[1]), float(row[2].split()[0]), float(row[3].split()[0]), float(row[4].split()[0])))
                except (ValueError, IndexError):
                    pass
    tail_seconds = float(sec.sum())
    if gpu:
        end = max(g[0] for g in gpu)
        window = [g for g in gpu if (end - g[0]).total_seconds() <= tail_seconds and g[2] > 0 or False]
        busy = [g for g in gpu if (end - g[0]).total_seconds() <= tail_seconds]
    out = dict(
        bench=str(bench), steady_updates=len(steady), skip=args.skip,
        s_per_update_mean=round(float(sec.mean()), 3), s_per_update_median=round(float(np.median(sec)), 3),
        s_per_update_p90=round(float(np.percentile(sec, 90)), 3),
        valid_windows_per_update_steady=round(float(val.mean()), 1),
        valid_windows_per_update_schedule=round(valid_total / updates, 1),
        valid_windows_per_s=round(float(wps), 2),
        peak_allocated_gb=max(r.get('peak_allocated_gb', 0) for r in rows),
        final_loss_video=steady[-1].get('loss_video'), final_loss_action=steady[-1].get('loss_action'),
        nonfinite=any(not np.isfinite(r.get('loss', 0)) or not np.isfinite(r.get('grad_norm', 0)) for r in rows),
        schedule=dict(updates_10_epochs=updates, valid_windows_10_epochs=valid_total,
                      updates_per_epoch=per_epoch_updates, windows_per_epoch=windows_per_epoch),
        projection_hours=dict(
            per_epoch_by_windows=round(windows_per_epoch / wps / 3600, 2),
            ten_epochs_by_windows=round(valid_total / wps / 3600, 2),
            ten_epochs_by_s_per_update=round(updates * float(sec.mean()) / 3600, 2)),
    )
    if gpu:
        for idx in sorted({g[1] for g in busy}):
            mine = [g for g in busy if g[1] == idx]
            out[f'gpu{idx}_util_mean'] = round(float(np.mean([g[2] for g in mine])), 1)
            out[f'gpu{idx}_power_w_mean'] = round(float(np.mean([g[3] for g in mine])), 1)
    print(json.dumps(out, indent=1))
    (bench / 'bench_summary.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
