"""Reported LIBERO success rates of the benchmarked models, with the protocol and the source of every number.

Protocols:
  matched-6k   lightwam_20261009_matched (evaluation_report/audit_setup.py): 4 suites x 10 tasks x 50 initial states
               x seeds 42/43/44 = 6,000 episodes. Elastic, RT, init-variant and all-suites rows.
  matched-long the same protocol on LIBERO-Long (libero_10) only: 1,500 episodes.
  long-100     the older single-seed Long evaluation: libero_10, 10 tasks x 10 episodes, seed 42 (100 episodes); the
               HF export's evaluation_summary.json.
Models absent from SR were not evaluated under any of these protocols.
"""
import json
from pathlib import Path

FINAL = "evaluation_report/final/results.json (origin/LoopWAM_NT)"
INITVAR = "plans/performance/LoopWAM_v0_Init_Variants_Results_20261028.md (origin/v0-init-variants)"
RT = "plans/performance/Residual_Transport_Results.csv (origin/residual_trans)"
CAMPAIGN = "evaluation_report/campaign/run_summary.csv (origin/LoopWAM_NT)"
HFEVAL = "HF export evaluation_summary.json (anhdao69/LoopWAM_NT@393310af)"

# (model, config) -> (successes, episodes, protocol, source)
SR = {
    ("FastWAM (released, libero_uncond_2cam224)", "native"): (5798, 6000, "matched-6k", FINAL),
    ("all-suites v0 4/4", "native"): (5695, 6000, "matched-6k", FINAL),
    ("all-suites v0 4/1", "native"): (5640, 6000, "matched-6k", FINAL),
    ("all-suites v0 2/2", "native"): (5547, 6000, "matched-6k", FINAL),
    ("all-suites v0 1/4", "native"): (5501, 6000, "matched-6k", FINAL),
    ("all-suites v0 4/1 kv-mix", "native"): (5667, 6000, "matched-6k", FINAL),
    ("init variant core_mean", "native"): (5569, 6000, "matched-6k", INITVAR),
    ("init variant ffn_magnitude", "native"): (5527, 6000, "matched-6k", INITVAR),
    ("init variant stepwise", "native"): (5488, 6000, "matched-6k", INITVAR),
    ("init variant no_video_pretraining", "native"): (5139, 6000, "matched-6k", INITVAR),
    ("init variant core_random", "native"): (4932, 6000, "matched-6k", INITVAR),
    ("long v0 4/4 original", "native"): (1166, 1500, "matched-long", FINAL),
    ("long v0 4/4 repeat", "native"): (1192, 1500, "matched-long", FINAL),
    ("long v0 4/1", "native"): (1110, 1500, "matched-long", FINAL),
    ("long v0 4/1 kv-mix", "native"): (1133, 1500, "matched-long", FINAL),
    ("long v0 4/1 kv-concat", "native"): (1118, 1500, "matched-long", FINAL),
    ("long v0 4/2", "native"): (86, 100, "long-100", HFEVAL),
    ("long v0 2/2", "native"): (81, 100, "long-100", HFEVAL),
    ("long v0 1/4", "native"): (86, 100, "long-100", HFEVAL),
    ("long v1 4/4", "native"): (82, 100, "long-100", HFEVAL),
    ("long v2 4/4", "native"): (81, 100, "long-100", HFEVAL),
    ("long Dense-S12", "native"): (81, 100, "long-100", HFEVAL),
    ("long Dense-S30", "native"): (86, 100, "long-100", HFEVAL),
    # RT four-suite means (matched-6k; RTX PRO 6000 remote campaign or H100 rt_eval_h100, see the CSV)
    ("RT-A", "S1"): (5668, 6000, "matched-6k", RT), ("RT-A", "S2"): (5647, 6000, "matched-6k", RT),
    ("RT-A", "S3"): (5643, 6000, "matched-6k", RT), ("RT-A", "S4"): (5632, 6000, "matched-6k", RT),
    ("RT-B4", "S1"): (5659, 6000, "matched-6k", RT),
    ("RT-T1ft", "S1"): (5674, 6000, "matched-6k", RT), ("RT-T1ft", "S2"): (5651, 6000, "matched-6k", RT),
    ("RT-Pa", "S1"): (5637, 6000, "matched-6k", RT),
    ("RT-TF", "S1"): (5665, 6000, "matched-6k", RT), ("RT-TF", "S2"): (5648, 6000, "matched-6k", RT),
    ("RT-A2", "S1"): (5646, 6000, "matched-6k", RT), ("RT-A2", "S2"): (5593, 6000, "matched-6k", RT),
    ("RT-A2", "S5"): (5699, 6000, "matched-6k", RT), ("RT-A2", "S6"): (5669, 6000, "matched-6k", RT),
    ("RT-A2", "S7"): (5696, 6000, "matched-6k", RT), ("RT-A2", "S8"): (5672, 6000, "matched-6k", RT),
    ("RT-B4", "S2"): (5566, 6000, "matched-6k", CAMPAIGN), ("RT-Pa", "S2"): (5575, 6000, "matched-6k", CAMPAIGN),
    ("RT-B2a", "parent2"): (5674, 6000, "matched-6k", CAMPAIGN),
    ("ChronoLoop CL-0_no-memory_a4", "native"): (5766, 6000, "matched-6k", CAMPAIGN),
    ("ChronoLoop CL-A_mem16-learned-loopwrite_a4", "native"): (5771, 6000, "matched-6k", CAMPAIGN),
    ("ChronoLoop CL-REG_mem16-reset-registers_a4", "native"): (5758, 6000, "matched-6k", CAMPAIGN),
    ("ChronoLoop CL-W2_mem16-learned-external-updater_a4", "native"): (5761, 6000, "matched-6k", CAMPAIGN),
    ("ChronoLoop CL-FRAME_history-frame-k3_a4", "native"): (5754, 6000, "matched-6k", CAMPAIGN),
    ("ChronoLoop CL-0-at1_no-memory_a1", "native"): (5742, 6000, "matched-6k", CAMPAIGN),
    ("ChronoLoop CL-A-at1_mem16-learned-loopwrite_a1", "native"): (5720, 6000, "matched-6k", CAMPAIGN),
}
# sha256 of the checkpoint each SR was scored with, where the source records it (checked against the timed file).
EVAL_SHA = {
    "RT-A": "15ba0de18c94", "RT-B4": "7ec1dedc3444", "RT-T1ft": "efb19b697478", "RT-Pa": "17f089aed0a4",
    "RT-TF": "c6cdba96a98a", "RT-A2": "25ad8ca4705e", "RT-B2a": "25175895948",
    "ChronoLoop CL-0_no-memory_a4": "137f0a839b5a", "ChronoLoop CL-A_mem16-learned-loopwrite_a4": "3b480febd188",
    "ChronoLoop CL-REG_mem16-reset-registers_a4": "6de1bbf80ab0",
    "ChronoLoop CL-W2_mem16-learned-external-updater_a4": "d08096facd67",
    "ChronoLoop CL-FRAME_history-frame-k3_a4": "b322ae52c816", "ChronoLoop CL-0-at1_no-memory_a1": "77cbd77a5216",
    "ChronoLoop CL-A-at1_mem16-learned-loopwrite_a1": "391d3b59bcd2",
    "FastWAM (released, libero_uncond_2cam224)": "1000437cfcf5",
}


def load(elastic_results=None):
    """SR table including the elastic campaign's own results JSON (scripts/elastic_eval_report.py output)."""
    table = {k: dict(successes=s, episodes=n, success_pct=100 * s / n, protocol=p, source=src)
             for k, (s, n, p, src) in SR.items()}
    if elastic_results:
        data = json.loads(Path(elastic_results).read_text())
        for row in data["conditions"].values():
            if row.get("complete") and row.get("episodes") == 6000:
                table[(f"elastic {row['run']}", row["schedule"])] = dict(
                    successes=row["successes"], episodes=6000, success_pct=100 * row["successes"] / 6000,
                    protocol="matched-6k", source="elastic campaign results.json (this report)")
    return table
