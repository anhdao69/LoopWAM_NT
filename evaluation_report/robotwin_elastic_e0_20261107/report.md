# Elastic E0 — full RoboTwin parallel evaluation

Updated 2026-11-08T19:09:56+07:00.

| Schedule | Host / workers | Status | Successes / episodes | Live SR | Verified cells / 100 |
| --- | --- | --- | ---: | ---: | ---: |
| [1,1] · NFE2 | Local6000 / 4 | complete; audited | 8684/10000 | 86.84% | 100 |
| [4,4] · NFE2 | 137:6000 / 4 | complete; audited | 9025/10000 | 90.25% | 100 |
| [4]×5 · NFE5 | Local6000 / 4 | complete; audited | 9096/10000 | 90.96% | 100 |

Remote sync: successful; last transfer-attempt age 0s; remote progress-snapshot age 10076s. Snapshots older than180seconds are explicitly marked stale.

**Protocol:** all50tasks, clean + randomized,100episodes/task/setting,seed42,10,000episodes per schedule;30,000total. Four video loops; action depths/NFE as listed; FP32weights/BF16compute; predict32/execute24; original normalization, camera packing, scene admission, unseen instructions and RT32/depth8 renderer. Smoke and interrupted attempts are excluded.

Live SR includes completed episodes in active attempts and remains provisional. Final scores require100 complete task/setting cells and a successful video/trace/runtime audit. Historical Python instruction-template randomness remains unseeded, so exact trajectory pairing is not claimed.

**Parallel execution:** [4,4] is assigned exclusively to server137; the local server finishes[1,1], then runs[4]×5. The original evaluator/code/runtime remains frozen. Only its scheduling supervisor was replaced; per-cell results are not mixed across hosts.

**Host validation:** server137 and local both use RTXPRO6000/driver580.178.04. Checkpoint, source, assets, checked package versions and protocol are checked before launch. Native/graph and cross-host action/T5 parity results are recorded in [remote verification](parallel137_20261108/report.md). Server189 is not used for scored evaluation.

[Original detailed setup](setup_report.md) · [handoff receipt](parallel137_20261108/handoff.json) · [schedule ownership](parallel137_20261108/assignments.json) · [remote report](parallel137_20261108/remote_mirror/report.md) · [machine-readable progress](current_results.json)
