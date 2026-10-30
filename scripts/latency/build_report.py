#!/usr/bin/env python3
"""Aggregate the latency sweep (measure_latency.py result files) into latency_results.json/.csv and the Markdown
tables of the all-models latency report, joined with each model's reported LIBERO success rate (success_rates.py).

Usage: build_report.py --results <dir> --jobs <jobs.jsonl> --out <data dir> [--control <dir>]
                       [--elastic-results <elastic eval results.json>] [--local-sha <json>] [--hf-sha <json>]
Writes <out>/latency_results.json, latency_results.csv and tables.md (sections pasted into the report).
Speedups are ratios of mean latencies. 'exact' counts compared queries whose raw action chunk equals the eager chunk
bit for bit (compared on the observations eager was timed on).
"""
import argparse, csv, json, statistics
from pathlib import Path

import success_rates

BACKENDS = ("eager", "compile", "graph", "fast")


def fmt(x, d=1):
    return "" if x is None else f"{x:.{d}f}"


def label(depths):
    """Schedule label in the style of the elastic evaluation report: [1,1], [4]x5, [4]+[1]x9, [4,1,1,1,1]."""
    if not depths:
        return ""
    n = len(depths)
    if len(set(depths)) == 1 and n > 2:
        return f"[{depths[0]}]x{n}"
    if n > 5 and len(set(depths[1:])) == 1:
        return f"[{depths[0]}]+[{depths[1]}]x{n - 1}"
    return "[" + ",".join(map(str, depths)) + "]"


def arch(info):
    if info.get("loader") == "FastWAM":
        return "FastWAM"
    if info.get("loader") == "TransportInference":
        return "RT policy"
    if info.get("loader") == "create_chronoloop":
        return "ChronoLoop"
    v, a = info.get("video_loops"), info.get("action_loops")
    kv = info.get("action_kv_mode")
    s = f"{info.get('version') or ''} v{v}/a{a}".strip()
    return s + (f" kv-{kv}" if kv and kv not in ("aligned", "none") else "")


def rows_from(res_dir, jobs, sr):
    rows, missing = [], []
    for j in jobs:
        f = res_dir / f"{j['id']}.json"
        if not f.is_file():
            missing.append(j)
            continue
        r = json.loads(f.read_text())
        for config, entry in r["configs"].items():
            info = entry.get("config", {})
            row = dict(id=j["id"], group=j["group"], tier=j["tier"], model=j["name"], family=j["family"],
                       config=config, depths=info.get("depths"), label=label(info.get("depths")),
                       arch=arch(r.get("info", {})), parameters=r.get("info", {}).get("parameters")
                       or r.get("info", {}).get("policy_parameters"), gpu=r["gpu"], host=r["host"],
                       torch=r["torch"], code_commit=r["code_commit"], checkpoint=r["checkpoint"],
                       observations=r["observations"], eager_stride=r.get("eager_stride"),
                       graph_stride=r.get("graph_stride"), fast_converted_linear=r.get("fast_converted_linear_modules"))
            s = sr.get((j["name"], config))
            if s:
                row.update(sr_pct=s["success_pct"], sr_episodes=s["episodes"], sr_protocol=s["protocol"],
                           sr_source=s["source"])
            for b in BACKENDS:
                e = entry.get(b)
                if not e:
                    continue
                if "error" in e:
                    row[f"{b}_error"] = e["error"][:300]
                    continue
                row.update({f"{b}_mean_ms": e["mean_ms"], f"{b}_median_ms": e["median_ms"], f"{b}_p5_ms": e["p5_ms"],
                            f"{b}_p95_ms": e["p95_ms"], f"{b}_std_ms": e["std_ms"], f"{b}_min_ms": e["min_ms"],
                            f"{b}_n": e["n"], f"{b}_peak_gb": e["peak_allocated_gb"],
                            f"{b}_preprocess_ms": e["preprocess"]["mean_ms"]})
                if "exact_vs_eager" in e:
                    row[f"{b}_exact"] = e["exact_vs_eager"]
                    row[f"{b}_compared"] = e["compared_vs_eager"]
                    row[f"{b}_max_abs_diff"] = e["max_abs_diff_vs_eager"]
            for b in ("graph", "compile"):
                if row.get("eager_mean_ms") and row.get(f"{b}_mean_ms"):
                    row[f"{b}_speedup_vs_eager"] = row["eager_mean_ms"] / row[f"{b}_mean_ms"]
            if row.get("fast_mean_ms"):
                if row.get("eager_mean_ms"):
                    row["fast_speedup_vs_eager"] = row["eager_mean_ms"] / row["fast_mean_ms"]
                if row.get("graph_mean_ms"):
                    row["fast_speedup_vs_graph"] = row["graph_mean_ms"] / row["fast_mean_ms"]
                row["fast_hz"] = 1e3 / row["fast_mean_ms"]
            rows.append(row)
        if r.get("all_fast_graphs_resident"):
            rows[-1]["_resident"] = r["all_fast_graphs_resident"]
    return rows, missing


def exact_cell(r, b):
    return f"{r[b + '_exact']}/{r[b + '_compared']}" if f"{b}_exact" in r else ("error" if f"{b}_error" in r else "")


def sr_cell(r):
    if "sr_pct" not in r:
        return "not evaluated"
    tag = {"matched-6k": "", "matched-long": " L1500", "long-100": " L100"}[r["sr_protocol"]]
    return f"{r['sr_pct']:.2f}{tag}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True)
    p.add_argument("--jobs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--control")
    p.add_argument("--elastic-results")
    p.add_argument("--local-sha")
    p.add_argument("--hf-sha")
    a = p.parse_args()
    jobs = [json.loads(l) for l in Path(a.jobs).read_text().splitlines() if l.strip()]
    sr = success_rates.load(a.elastic_results)
    rows, missing = rows_from(Path(a.results), jobs, sr)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    resident = {r["model"]: r.pop("_resident") for r in rows if "_resident" in r}

    # Checkpoint identity: the timed file vs the file each success rate was scored with.
    ident, eval_sha = {}, dict(success_rates.EVAL_SHA)
    if a.elastic_results:
        for c in json.loads(Path(a.elastic_results).read_text())["conditions"].values():
            if c.get("checkpoint_sha256"):
                eval_sha[f"elastic {c['run']}"] = c["checkpoint_sha256"][:12]
    if a.local_sha and Path(a.local_sha).is_file():
        for path, v in json.loads(Path(a.local_sha).read_text()).items():
            want = eval_sha.get(v["model"])
            ident[v["model"]] = dict(sha256=v["sha256"], eval_sha256_prefix=want,
                                     match=None if want is None else v["sha256"].startswith(want))
    control = []
    if a.control and Path(a.control).is_dir():
        crow, _ = rows_from(Path(a.control), jobs, sr)
        base = {(r["id"], r["config"]): r for r in rows}
        for c in crow:
            b = base.get((c["id"], c["config"]))
            if b:
                control.append(dict(model=c["model"], config=c["config"], **{
                    f"{k}_{s}": (b.get(f"{k}_{s}"), c.get(f"{k}_{s}")) for k in ("eager", "graph", "fast")
                    for s in ("mean_ms", "median_ms", "p95_ms")}))

    (out / "latency_results.json").write_text(json.dumps(dict(
        rows=rows, missing=[m["name"] for m in missing], all_fast_graphs_resident=resident, control=control,
        checkpoint_identity=ident), indent=1))
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("id", "group", "model", "config"), k))
    with open(out / "latency_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    md = []
    # Headline numbers.
    fast = [r for r in rows if r.get("fast_mean_ms")]
    tot = lambda b, k: sum(r.get(f"{b}_{k}", 0) for r in rows)
    sp_e = [r["fast_speedup_vs_eager"] for r in fast if r.get("fast_speedup_vs_eager")]
    sp_g = [r["fast_speedup_vs_graph"] for r in fast if r.get("fast_speedup_vs_graph")]
    mem = [(r["eager_peak_gb"], r["fast_peak_gb"]) for r in fast if r.get("eager_peak_gb")]
    md.append("## 4. Headline numbers\n")
    md.append(f"- Measured: {len({r['model'] for r in rows})} models, {len(rows)} model x configuration rows, on "
              f"{', '.join(sorted({r['gpu'] for r in rows}))} ({', '.join(sorted({r['host'] for r in rows}))}).")
    if missing:
        md.append(f"- Not measured (sweep ran out of allocation time): {', '.join(m['name'] for m in missing)}.")
    md.append(f"- Bit-exact vs eager: fast {tot('fast', 'exact')}/{tot('fast', 'compared')}, graph "
              f"{tot('graph', 'exact')}/{tot('graph', 'compared')}, Inductor compile "
              f"{tot('compile', 'exact')}/{tot('compile', 'compared')} compared queries.")
    if sp_e:
        md.append(f"- Fast vs eager (mean latency): median {statistics.median(sp_e):.2f}x, range "
                  f"{min(sp_e):.2f}x-{max(sp_e):.2f}x over {len(sp_e)} rows.")
    if sp_g:
        md.append(f"- Fast vs the campaign CUDA-graph backend: median {statistics.median(sp_g):.2f}x, range "
                  f"{min(sp_g):.2f}x-{max(sp_g):.2f}x over {len(sp_g)} rows.")
    if mem:
        md.append(f"- Peak allocated memory, eager -> fast: median {statistics.median(e for e, _ in mem):.2f} -> "
                  f"{statistics.median(f for _, f in mem):.2f} GB.")
    errs = [(r["model"], r["config"], b, r[f"{b}_error"]) for r in rows for b in BACKENDS if f"{b}_error" in r]
    if errs:
        md.append(f"- Backend errors: " + "; ".join(f"{m} {c} {b}: {e[:120]}" for m, c, b, e in errs))
    md.append("")

    # Main table.
    md.append("## 5. All models\n")
    md.append("SR: four-suite matched success % (6,000 episodes) unless tagged L1500 (matched LIBERO-Long only, 1,500 "
              "episodes) or L100 (older single-seed Long evaluation, 100 episodes). Latency in ms, batch 1, mean "
              "(p50 / p95 for fast). Speedups are ratios of means.\n")
    md.append("| Group | Model | Architecture | Config | SR % | Eager mean | Graph mean | Fast mean (p50 / p95) | "
              "Fast Hz | Fast vs eager | Fast vs graph | Exact graph / fast | Peak GB eager -> fast |")
    md.append("|" + "---|" * 13)
    for r in rows:
        cfg = r["config"] if r["config"] != "native" else "native"
        if r["label"]:
            cfg = f"{cfg} {r['label']}"
        md.append("| {} | {} | {} | {} | {} | {} | {} | {} ({} / {}) | {} | {} | {} | {} / {} | {} -> {} |".format(
            r["group"], r["model"], r["arch"], cfg, sr_cell(r), fmt(r.get("eager_mean_ms")),
            fmt(r.get("graph_mean_ms")) or ("n/a" if r["family"] == "chrono" else ""), fmt(r.get("fast_mean_ms")),
            fmt(r.get("fast_median_ms")), fmt(r.get("fast_p95_ms")), fmt(r.get("fast_hz"), 2),
            (fmt(r.get("fast_speedup_vs_eager"), 2) + "x") if r.get("fast_speedup_vs_eager") else "",
            (fmt(r.get("fast_speedup_vs_graph"), 2) + "x") if r.get("fast_speedup_vs_graph") else "",
            exact_cell(r, "graph") or "n/a", exact_cell(r, "fast"), fmt(r.get("eager_peak_gb"), 2),
            fmt(r.get("fast_peak_gb"), 2)))
    md.append("")

    # Compile comparison.
    comp = [r for r in rows if r.get("compile_mean_ms") or r.get("compile_error")]
    if comp:
        md.append("## 6. Upstream torch.compile (Inductor, reduce-overhead) vs fast\n")
        md.append("| Model | Eager mean | Compile mean (p95) | Exact compile | Max abs diff | Graph mean | "
                  "Fast mean (p95) | Exact fast |")
        md.append("|" + "---|" * 8)
        for r in comp:
            md.append(f"| {r['model']} | {fmt(r.get('eager_mean_ms'))} | {fmt(r.get('compile_mean_ms'))} "
                      f"({fmt(r.get('compile_p95_ms'))}) | {exact_cell(r, 'compile')} | "
                      f"{r.get('compile_max_abs_diff', r.get('compile_error', ''))} | {fmt(r.get('graph_mean_ms'))} | "
                      f"{fmt(r.get('fast_mean_ms'))} ({fmt(r.get('fast_p95_ms'))}) | {exact_cell(r, 'fast')} |")
        md.append("")

    # Elastic operating points.
    el = [r for r in rows if r["group"] == "elastic"]
    if el:
        md.append("## 7. Elastic checkpoints: success and latency per operating point\n")
        md.append("| Run | Schedule | Depths | Action-expert block passes | SR % | Fast mean (p95) | Graph mean | "
                  "Eager mean |")
        md.append("|" + "---|" * 8)
        for r in el:
            md.append(f"| {r['model'].replace('elastic ', '')} | {r['config']} | {r['label']} | "
                      f"{sum(r['depths']) if r['depths'] else ''} | {sr_cell(r)} | {fmt(r.get('fast_mean_ms'))} "
                      f"({fmt(r.get('fast_p95_ms'))}) | {fmt(r.get('graph_mean_ms'))} | {fmt(r.get('eager_mean_ms'))} |")
        md.append("")
    if resident:
        md.append("### 7.1 Memory with every operating point's fast graph resident\n")
        md.append("| Model | Captured graphs | Allocated GB | Peak GB | Reserved GB |")
        md.append("|---|---|---|---|---|")
        for m, v in resident.items():
            md.append(f"| {m} | {v['graphs']} | {v['allocated_gb']:.2f} | {v['peak_allocated_gb']:.2f} | "
                      f"{v['reserved_gb']:.2f} |")
        md.append("")

    # Pareto front over matched four-suite rows.
    cand = sorted((r for r in fast if r.get("sr_protocol") == "matched-6k"), key=lambda r: r["fast_mean_ms"])
    front, best = [], -1.0
    for r in cand:
        if r["sr_pct"] > best:
            front.append(r)
            best = r["sr_pct"]
    if front:
        md.append("## 8. Success-latency Pareto front, matched four-suite models\n")
        md.append("Each row is faster than every row below it and more successful than every row above it.\n")
        md.append("| Model | Config | SR % | Fast mean ms | Fast p95 ms |")
        md.append("|---|---|---|---|---|")
        for r in front:
            md.append(f"| {r['model']} | {r['config']} {r['label']} | {r['sr_pct']:.2f} | {r['fast_mean_ms']:.1f} | "
                      f"{r['fast_p95_ms']:.1f} |")
        md.append("")

    if control:
        md.append("## 9. Interference control\n")
        md.append("The same job re-measured on one GPU with the node's other three GPUs idle, vs the 4-worker sweep.\n")
        md.append("| Model | Config | Backend | Sweep mean / p95 | Alone mean / p95 | Difference of means |")
        md.append("|---|---|---|---|---|---|")
        for c in control:
            for b in ("eager", "graph", "fast"):
                (sm, am), (sp, ap) = c[f"{b}_mean_ms"], c[f"{b}_p95_ms"]
                if sm and am:
                    md.append(f"| {c['model']} | {c['config']} | {b} | {sm:.1f} / {sp:.1f} | {am:.1f} / {ap:.1f} | "
                              f"{100 * (sm - am) / am:+.1f}% |")
        md.append("")

    if ident:
        md.append("## 10. Checkpoint identity\n")
        md.append("sha256 of each locally stored timed checkpoint vs the checkpoint its success rate was scored with "
                  "(HF downloads are checked against their export records separately).\n")
        md.append("| Model | Timed file sha256 | Evaluated sha256 prefix | Match |")
        md.append("|---|---|---|---|")
        for m, v in ident.items():
            md.append(f"| {m} | {v['sha256'][:16]} | {v['eval_sha256_prefix'] or '-'} | "
                      f"{'yes' if v['match'] else ('no' if v['match'] is False else 'n/a')} |")
        md.append("")
    (out / "tables.md").write_text("\n".join(md) + "\n")
    print(f"{len(rows)} rows from {len(jobs) - len(missing)} jobs; missing: {[m['name'] for m in missing]}")


if __name__ == "__main__":
    main()
