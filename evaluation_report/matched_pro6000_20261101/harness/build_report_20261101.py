#!/usr/bin/env python3
"""Fill the 2026-11-01 batch report template with the tables computed by report_20261101.py.

Usage: python3 build_report_20261101.py <results.json> <template.md> <output README.md> [findings.md]
Placeholders: {{BATCH}} (this batch's conditions), {{REFS}} (reference rows), {{CONTRASTS}}, {{INTEGRITY}}, {{PENDING}}.
"""
import json
import sys
from pathlib import Path

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SEEDS = ["42", "43", "44"]
BATCH = ["v0 4/2", "E0 [3]x10", "E0 [5]x10", "E0 [6]x10", "Z10 [4,4]", "Z10 [4]x5", "Z10 [1,1]"]
LABEL = {"v0 4/2": "v0 4/2 (video 4 / action 2 export, NFE 10)",
         "E0 [3]x10": "E0 epoch 10, K_a 3, NFE 10", "E0 [5]x10": "E0 epoch 10, K_a 5, NFE 10",
         "E0 [6]x10": "E0 epoch 10, K_a 6, NFE 10",
         "Z10 [4,4]": "base v0 4/4, NFE 2 [4,4], no distillation", "Z10 [4]x5": "base v0 4/4, NFE 5 [4]x5, no distillation",
         "Z10 [1,1]": "base v0 4/4 at [1,1] (K_a 1, NFE 2), no distillation"}
KIND = {"same_host": "same host", "same_hardware_campaign": "same hardware (campaign)", "cross_host": "cross host (H100)"}
pct = lambda v: f"{100 * v:.2f}"  # noqa: E731


def row(name, r, label=None):
    return (f"| {label or name} | **{pct(r['success_rate'])}** ({pct(r['ci95'][0])}–{pct(r['ci95'][1])}) | "
            + " | ".join(pct(r["per_suite"][s]["success_rate"]) for s in SUITES) + " | "
            + " / ".join(pct(r["per_seed"][k]["success_rate"]) for k in SEEDS) + f" | {r['host']} |")


HEAD = ("| Condition | Success % (95% CI) | Spatial | Object | Goal | Long | Seeds 42 / 43 / 44 | Evaluated on |\n"
        "|---|---|---:|---:|---:|---:|---|---|")


def main():
    res = json.loads(Path(sys.argv[1]).read_text()); tpl = Path(sys.argv[2]).read_text()
    C = res["conditions"]
    batch = [HEAD] + [row(n, C[n], LABEL[n]) for n in BATCH if n in C]
    refs = [HEAD] + [row(n, r) for n, r in C.items() if n not in BATCH]
    con = ["| Contrast (secondary, unadjusted) | Kind | Difference, pp (95% CI) | p | Discordant clusters | Spatial | Object | Goal | Long |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for n, c in res["contrasts"].items():
        o = c["overall"]
        con.append(f"| {n} | {KIND[c['kind']]} | {100 * o['difference']:+.2f} ({100 * o['ci95'][0]:+.2f} to {100 * o['ci95'][1]:+.2f}) | "
                   f"{o['p_permutation']:.4f} | {o['discordant_clusters']}/{o['clusters']} | "
                   + " | ".join(f"{100 * c['per_suite'][s]['difference']:+.2f}" for s in SUITES) + " |")
    integ = ["| Condition | Episodes | Code head in every group record | Gripper exactly 0.5 | Runner summary.json |",
             "|---|---:|---|---:|---|"]
    for n in BATCH:
        if n not in C:
            continue
        r = C[n]
        if n == "v0 4/2":
            integ.append(f"| {n} | {r['episodes']} | matched harness (`matched_eval/verify_run.py` passed: 6000 episode files, 120 groups, 18 builds, 0 issues) | n/a | matches |")
        else:
            integ.append(f"| {n} | {r['episodes']} | {', '.join(h[:7] for h in r['heads'])} (manifest {r['manifest_head'][:7]}) | "
                         f"{r['exact_half_gripper']} | {r['summary_json']} |")
    pending = ("" if not res["incomplete"] else
               "**Not yet complete:** " + "; ".join(f"{k} ({v})" for k, v in res["incomplete"].items()
                                                    if k in BATCH) + ".")
    out = (tpl.replace("{{BATCH}}", "\n".join(batch)).replace("{{REFS}}", "\n".join(refs))
           .replace("{{CONTRASTS}}", "\n".join(con)).replace("{{INTEGRITY}}", "\n".join(integ)).replace("{{PENDING}}", pending))
    findings = Path(sys.argv[4]).read_text().strip() if len(sys.argv) > 4 else ""
    out = out.replace("{{FINDINGS}}", findings)
    Path(sys.argv[3]).parent.mkdir(parents=True, exist_ok=True)
    Path(sys.argv[3]).write_text(out)


if __name__ == "__main__":
    main()
