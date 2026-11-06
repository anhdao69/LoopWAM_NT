# RoboTwin-10 H100 retrains — evaluation progress

Updated 2026-11-07T01:49:54+07:00 (Vietnam time). Remote report sync: successful.

Both models: epoch5/update45445; seed42; NFE10; ten tasks; clean and randomized;100 accepted episodes/task/setting;2,000 per model. Smoke and interrupted attempts excluded. Partial SR is provisional.

| Model | Host | Queue stage | Episodes | Successes | SR |
| --- | --- | --- | ---: | ---: | ---: |
| v0 4/4 H100 retrain | Local96GB /4workers | complete | 2000/2000 | 1573 | 78.65% |
| v0 4/2 H100 retrain | Remote189 48GB /2workers | complete | 2000/2000 | 1616 | 80.80% |

- [4/4 detailed setup](../robotwin10_h100_v0_4x4_20261106/setup_report.md) · [live report](../robotwin10_h100_v0_4x4_20261106/report.md) · [final report when ready](../robotwin10_h100_v0_4x4_20261106/final_report.md)
- [4/2 detailed setup](../robotwin10_h100_v0_4x2_remote189_20261106/setup_report.md) · [live report](../robotwin10_h100_v0_4x2_remote189_20261106/report.md) · [final report when ready](../robotwin10_h100_v0_4x2_remote189_20261106/final_report.md)

Native model, simulator, task limits and renderer quality are preserved. Each checkpoint must pass strict loading, source/runtime hashes, exact native/graph comparison and simulator smoke tests before scored evaluation. Final reports require all videos/traces and renderer logs to pass audit. Training donor-artifact/data-manifest hash differences from the original releases are disclosed in setup. Instruction-template sampling remains unseeded as in the original harness; this is not exact trajectory pairing.
