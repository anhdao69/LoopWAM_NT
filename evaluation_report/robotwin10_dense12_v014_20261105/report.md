# RoboTwin-10: Dense-s12 and v0 1/4

Updated 2026-11-06T02:05:25+07:00.

Both models: epoch 5, seed 42, NFE 10, all ten tasks, clean and randomized, 100 accepted episodes per task/setting; **2,000 scored episodes each**. Both selected machines have 96 GB RTX PRO 6000 Blackwell Max-Q GPUs.

| Model | Host | Queue stage | Selected workers | Scored episodes | SR |
| --- | --- | --- | ---: | ---: | ---: |
| [Dense-s12](../robotwin10_dense_s12_20261105/report.md) | local | complete | 4 | 2000/2,000 | 72.20% |
| [v0 1/4](../robotwin10_v0_1x4_remote137_20261105/report.md) | 10.148.182.137 | complete | 4 | 2000/2,000 | 73.15% |

**Recovery notice:** The ten-worker attempts were invalidated for sustained GPU/OIDN renderer failures. Their counts are excluded; current counts below belong only to fresh four-worker reruns.

SR is provisional until each final audit passes. Calibration and interrupted preflight episodes are excluded. The remote evidence is synchronized approximately every 60 seconds. No previous model result is overwritten.

## Dense-s12

[Detailed setup](../robotwin10_dense_s12_20261105/setup_report.md) · [Queue status](../robotwin10_dense_s12_20261105/queue_status.json) · [Final report](../robotwin10_dense_s12_20261105/final_report.md) when complete.

Native/graph action equality: passed exactly. Short model-only timings: native 81.5 ms/query, CUDA graph 61.1 ms/query. These are not complete simulator throughput measurements.

| Backend | Workers | Episodes | Wall time | GPU headroom |
| --- | ---: | ---: | ---: | ---: |
| native | 4 | 20 | 353.5 s | 46,877 MiB |
| cuda-graph | 4 | 20 | 356.5 s | 52,625 MiB |
| cuda-graph | 8 | 20 | 278.0 s | 31,398 MiB |
| cuda-graph | 10 | 20 | 266.2 s | 24,037 MiB |

Selected 4 workers. Selection basis: Four-worker operational recovery after sustained ten-worker GPU/render failures; short-profile throughput no longer determines concurrency.. Calibration includes startup and instruction/scene variation.

Skipped speed_graph_w12: Insufficient free GPU memory for measured projection plus reserve.
Final audit: **passed**; failed attempts: 0.

## v0 1/4

[Detailed setup](../robotwin10_v0_1x4_remote137_20261105/setup_report.md) · [Queue status](../robotwin10_v0_1x4_remote137_20261105/queue_status.json) · [Final report](../robotwin10_v0_1x4_remote137_20261105/final_report.md) when complete.

Native/graph action equality: passed exactly. Short model-only timings: native 157.4 ms/query, CUDA graph 104.1 ms/query. These are not complete simulator throughput measurements.

| Backend | Workers | Episodes | Wall time | GPU headroom |
| --- | ---: | ---: | ---: | ---: |
| native | 4 | 20 | 371.6 s | 55,157 MiB |
| cuda-graph | 4 | 20 | 377.6 s | 51,648 MiB |
| cuda-graph | 8 | 20 | 272.0 s | 31,335 MiB |
| cuda-graph | 10 | 20 | 266.0 s | 20,443 MiB |

Selected 4 workers. Selection basis: Four-worker operational recovery after sustained ten-worker GPU/render failures; short-profile throughput no longer determines concurrency.. Calibration includes startup and instruction/scene variation.

Skipped speed_graph_w12: Insufficient free GPU memory for measured projection plus reserve.
Final audit: **passed**; failed attempts: 0.

[Execution plan](execution_plan.md) · [Host inspection](host_selection.json) · [Speed-control review](speed_review.json)

The released Python instruction-template RNG is unseeded, so same-seed runs do not guarantee paired instruction strings. The scored protocol matches previous runs; all actual episode seeds and instructions are logged.
