#!/usr/bin/env python3
"""Completed-episode throughput of GR1 eval dirs: only <uuid>_success{0,1}.mp4 are finished episodes; per-client
steady-state episode time = gaps between consecutive finished episodes of the same task (one client per task)."""
import glob, os, statistics as st, sys, time
for d in sys.argv[1:]:
    t0 = os.path.getmtime(f'{d}/eval_config.json'); gaps = []; done = []; succ = 0
    for td in glob.glob(f'{d}/videos/n12/*'):
        m = sorted(os.path.getmtime(v) for v in glob.glob(f'{td}/*_success[01].mp4'))
        succ += len(glob.glob(f'{td}/*_success1.mp4')); done += m; gaps += [b - a for a, b in zip(m, m[1:])]
    end = max(done) if done else t0
    print(f'{d.split("/eval/")[-1]}: {len(done)} finished eps ({succ} success) in {(end - t0)/60:.1f} min; '
          f'first-episode median {st.median(sorted(done)[:24]) - t0 if len(done) >= 24 else float("nan"):.0f}s; '
          f'steady gap median {st.median(gaps) if gaps else float("nan"):.0f}s (n={len(gaps)})')
