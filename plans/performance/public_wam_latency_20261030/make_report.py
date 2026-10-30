#!/usr/bin/env python
"""Assemble the public-WAM latency report (released models through their own code and weights) from the measured
files. Every number in a generated table is read from a result file at generation time.
Usage: make_report.py <out.md>"""
import json, re, subprocess, sys
from pathlib import Path

LAT = Path("/groups/yshang/an221229/checkpoints/wam_baselines/latency")
ORDER = ["fastwam", "fastwam_nocompile", "fasterwam", "lightwam", "lilawam"]


def jload(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def latency_results():
    res = {}
    for p in sorted(LAT.glob("results/*.json")):
        if "smoke" in p.name:
            continue
        key, node = p.stem.rsplit("_", 1)
        if key in ORDER:
            res[(key, node)] = json.loads(p.read_text())
    return res


def full_stats_table(res):
    order = ORDER
    md = ["| Model | Node (job) | Prompt mode | n | mean | std | p5 | p50 | p95 | min | max | Hz | Peak GB | Load s |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in order:
        for (key, node), r in sorted(res.items()):
            if key != k:
                continue
            for mode, s in r["modes"].items():
                md.append(f"| {r['model']} | {node} ({r.get('slurm_job')}) | {mode} | {s['n']} | {s['mean_ms']:.2f} | "
                          f"{s['std_ms']:.2f} | {s['p5_ms']:.2f} | {s['median_ms']:.2f} | {s['p95_ms']:.2f} | "
                          f"{s['min_ms']:.2f} | {s['max_ms']:.2f} | {1000 / s['mean_ms']:.1f} | "
                          f"{s['peak_allocated_gb']:.2f} | {r['load_seconds']:.0f} |")
    return "\n".join(md)


def t5_table(res):
    md = ["| Model | evc102: native − prompt_cached (ms) | evc103: native − prompt_cached (ms) | Share of native (evc102) |",
          "|---|---|---|---|"]
    for k in ["fastwam", "fastwam_nocompile", "fasterwam", "lightwam"]:
        cells, share = [], ""
        for node in ["evc102", "evc103"]:
            r = res.get((k, node))
            if not r:
                cells.append("-"); continue
            d = r["modes"]["native"]["mean_ms"] - r["modes"]["prompt_cached"]["mean_ms"]
            cells.append(f"{d:+.1f}")
            if node == "evc102":
                share = f"{100 * d / r['modes']['native']['mean_ms']:.1f}%"
        md.append(f"| {res.get((k, 'evc102'), res.get((k, 'evc103')))['model']} | {cells[0]} | {cells[1]} | {share} |")
    return "\n".join(md)


def main():
    res = latency_results()
    tables = subprocess.run([sys.executable, str(LAT / "build_tables.py"), str(LAT / "results"), str(LAT / "tables.md")],
                            capture_output=True, text=True, check=True).stdout
    tmpl = (LAT / "report_v2_template.md").read_text()
    out = (tmpl.replace("{{LAT_TABLES}}", tables.strip())
               .replace("{{LAT_FULL}}", full_stats_table(res))
               .replace("{{T5}}", t5_table(res)))
    assert "{{" not in out, re.findall(r"\{\{\w+\}\}", out)
    Path(sys.argv[1]).write_text(out)
    print("wrote", sys.argv[1], len(out), "chars")


if __name__ == "__main__":
    main()
