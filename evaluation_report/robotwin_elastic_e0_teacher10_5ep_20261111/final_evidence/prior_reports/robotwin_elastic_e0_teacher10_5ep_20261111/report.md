# Elastic E0 teacher epoch10 / student five epochs — full RoboTwin evaluation

Updated 2026-11-12T04:00:40+07:00.

| Schedule | Host | Stage | Successes / episodes | Live SR | E0 final SR |
| --- | --- | --- | ---: | ---: | ---: |
| [1,1] / NFE 2 | local | complete; audited | 8928/10000 | 89.28% | 86.84% |
| [2,2] / NFE 2 | remote137 | complete; audited | 9060/10000 | 90.60% | pending |

New E0 queue: local RTX PRO6000=[1,1]; server137 RTX PRO6000=[2,2]. Both use NFE2 and four simulator workers, shared T5 and CUDA graphs. Each host waits for at least60000MiB free. Native/graph parity, cross-host parity and a four-worker smoke test must pass before scored evaluation. Other workloads remain untouched.

Full 50 tasks × clean/randomized × 100 episodes = 10,000 total; seed42. Four video loops, native T0 (no transport), FP32 policy weights/BF16 compute, predict32/execute24, original normalization/cameras/scene admission/horizons and RT32/depth8 rendering.

Live SR includes completed episodes in active attempts and is provisional. Partial new E0 SR must not be compared directly with the older E0 score until task coverage matches. Final scores require full video/trace/runtime audits. Python instruction selection remains unseeded as in E0, so exact scene/instruction pairing is not promised.

Remote report transfer: successful. Remote snapshot age: 3.9457480907440186 seconds; snapshots older than180seconds are flagged.

[Detailed setup and fairness report](setup_report.md) · [Server137](remote137/report.md) · [Machine-readable progress](current_results.json)
