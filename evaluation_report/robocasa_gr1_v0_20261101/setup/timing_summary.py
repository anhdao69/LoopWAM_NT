import json, statistics, sys
R = sys.argv[1]
for m in (8, 16):
    try:
        rows = [json.loads(l) for l in open(f"{R}/train/timing_micro{m}/metrics.jsonl")]
    except FileNotFoundError:
        continue
    s = [r["seconds"] for r in rows if r["update"] > 20]
    if s:
        print(f"micro{m} n={len(rows)} median={statistics.median(s):.3f} mean={statistics.fmean(s):.3f} p90={sorted(s)[int(.9*len(s))]:.3f} "
              f"max_wait={max(r['data_wait_seconds'] for r in rows[20:]):.3f} first={rows[0]['seconds']:.1f}")
    for r in rows[:2] + rows[9:10] + rows[49:50] + rows[99:100] + rows[-1:]:
        print(" ", r["update"], round(r["seconds"], 2), "loss", round(r["loss"], 3), "v", round(r["loss_video"], 3), "a", round(r["loss_action"], 3),
              "gn", round(r["grad_norm"], 2), "lr", f"{r['lr_backbone']:.2e}", "alloc", round(r["peak_allocated_gb"], 1), "res", round(r["peak_reserved_gb"], 1))
