# Elastic E0: epoch-10 teacher, five-epoch student — full RoboTwin-50

Prepared 2026-11-11T12:15:57+07:00 (Vietnam time). This is a new evaluation of the specified release, with two independently owned schedules on the two requested RTX PRO 6000 GPUs. See [live report](report.md) and [machine-readable status](current_results.json) for current stage. Both hosts passed CPU/source/numerical/review gates and their eight-episode simulator smoke checks. Full scoring began at approximately12:20 on November11,Vietnam time, on bothservers. The20,000-episode benchmark remains in progress; final results require both final audits.

## Release identity

Source: [E0-epoch10-teacher-5ep](https://huggingface.co/anhdao69/LoopWAM_NT/tree/b72f355505d98c473fb91e45d10e16d4cc7ce536/robotwin/Elastic/E0-epoch10-teacher-5ep), pinned HF revision `b72f355505d98c473fb91e45d10e16d4cc7ce536`.

| Field | Verified value |
|---|---|
| Student checkpoint | `epoch_05.pt` |
| Student training | Five epochs; update **237,310** |
| Teacher | Full RoboTwin v0 4/4, epoch **10**, update **474,620** |
| Transport | **T0**, no learned transport |
| Format | `loopwam-s-robot-rt-v1` |
| Checkpoint SHA256 | `07d24ff77410e516ac90d24d76d651e2102b09b0458d9d5d86ea18dff39e823e` |
| Checkpoint size | 2,340,331,885 bytes |
| Teacher SHA256 | `70cf5af73db5fe582c50c5f9fb68296d5271a9c0fea51509b3cdb5f9696be1ea` |
| Training revision | `2c78ff6f186c1d9073068b8eee5b328d19dba826` |
| Model metadata | v0, 3/6/3 architecture, four video loops, history0, memory0, 14D action/state |

The folder name identifies the teacher's epoch. The released student is epoch 5; it must not be mislabeled as an epoch-10 student. This fresh five-epoch distillation differs from the earlier one-epoch E0 distilled from the epoch-5 teacher.

All 13 published files were downloaded on server137 using verified HTTPS, checked against HF LFS/Git hashes and SHA256SUMS, then copied locally and rechecked. Both hosts use the same checkpoint bytes. The local CPU check compares all **339 frozen video tensors plus 2 proprioception tensors** directly to the verified epoch-10 teacher: all341 are bit-equal. Server137 verifies the identical checkpoint hash and that teacher-comparison receipt. The trained student matches the action state stored in the export; all checked floating weights are finite FP32. Normalization equals the prior full50 protocol and the training contract.

## Schedule ownership and scoring protocol

| Host | GPU | Assigned schedule | Action core loops per denoising step | NFE | Scored episodes |
|---|---|---|---|---:|---:|
| Local | RTX PRO6000 Blackwell Max-Q, 96GB | `flat_k1_n2` | **[1,1]** | **2** | **10,000** |
| `10.148.182.137` | RTX PRO6000 Blackwell Max-Q, 96GB | `flat_k2_n2` | **[2,2]** | **2** | **10,000** |

The schedules run independently in parallel once their checks and memory admission pass. Each schedule covers all **50 native RoboTwin tasks**, both `demo_clean` and `demo_randomized`, and **100 accepted episodes per task/setting**. Combined campaign total: **20,000 scored episodes**. Every scored run uses **seed42 only**.

Other fixed settings: four video loops; trained step condition **0.5** for each of the two denoising steps; CFG1; FP32 policy weights/BF16 compute; predict32 actions and execute24 before replanning; original 14D absolute qpos output; original z-score statistics; FastWAM PIL uint8 image resizing and384×320camera packing; CPU seed42 noise reset per query; unseen instruction templates; original expert scene-admission checks and task horizons; RT32/depth8 rendering with the previously validated OIDN compatibility setup. Videos, accepted scene seeds, instructions and action-query hashes are retained.

Python instruction-template selection remains unseeded as in the previous native runs. Matching evaluation seed/settings does not guarantee identical scene/instruction trajectories across separate evaluations. Scene rejections are handled by the original expert-admission rules and do not count as accepted scored episodes. No task scores are cherry-picked or replaced using SR.

## Source, environment and numerical checks

The evaluator is an isolated copy of the previously validated E0 deployment. Native model/inference mathematics are unchanged: all **19 native model files** match the training revision exactly. Production adapter changes are limited to explicit[2,2] schedule support, its CLI choice, and strict validation of the new student/teacher identity. The full source manifest contains **984 entries** including task code, evaluator, policies, test references and configuration. Runtime freeze also verifies inherited host assets/files (22,066 local,22,072 remote) and binds the new checkpoint, launcher and gate receipts. Existing runtime manifests cannot be silently replaced when resuming.

Both hosts: Python3.10.22, PyTorch2.7.1+cu128, NumPy1.26.4, Pillow12.0.0, SAPIEN3.0.0b1, transforms3d0.4.2, Transformers4.49.0, Gymnasium0.29.1, NVIDIA driver580.178.04. Native imports use the checked source and existing validated environment.

Checks completed before queue launch:

- Release digest/size verification and strict student/teacher metadata checks.
- CPU frozen tensor, student-state, finite-weight and normalization checks.
- **96 scoped inference/adapter/queue/report tests passed on each host**.
- Both[1,1] and[2,2] native-vs-CUDA-graph outputs match exactly on **eight recorded real simulator inputs**, including revisiting inputs in reverse order to catch stale graph buffers; zero tolerance and maximum difference0.
- Actual native velocity calls confirm the requested loop depths and step condition0.5.
- All **16 cross-host output pairs** match exactly; saved reference artifacts have identical SHA256.
- Independent review found one Important status-reporting defect: errors could retain a running or unaudited-complete label. The fix passed regression tests after four failing cases were demonstrated. No Critical or unresolved Important findings remain.

Test scope: an initial broader selection exposed six missing continuation-training helper dependencies in the inherited inference-only snapshot. Seven training-controller tests are excluded from the scoped suite; none is imported by this evaluator. This is not claimed as a green full-repository training test suite. A remote-only test failure caused by a local absolute reference path was fixed by copying the byte-identical FastWAM source as a portable test fixture. These changes do not alter production image arithmetic.

## Speed, admission and required smoke

Use **four simulator workers per GPU**, shared T5 resources and validated CUDA graphs. Higher sustained concurrency previously caused renderer/memory failures, so no short-profile-based concurrency increase is used here. Each queue requires at least **60,000MiB free GPU memory** before simulator work and does not stop another project's processes. Initial numerical gates ran with more than12,000MiB free and less than4,000MiB reserved by the gate. Their contended timings are not used as benchmark-throughput claims.

Each host must complete **eight unscored smoke episodes** before scoring: adjust_bottle and open_microwave, both settings, two episodes per cell, at the full four-worker concurrency. These16campaign smoke episodes are excluded from all20,000scored episodes. A smoke/validation failure stops the host queue and appears as an error in reports.

The earlier full50 E0[1,1] run took **15.02hours**. An initial planning allowance is approximately **15–20hours per schedule after actual scoring starts**, depending on task success, simulation duration and contention. Running bothhosts in parallel avoids summing the two durations. This is an estimate; update it from scored task timings. GPU waiting time and retries are additional.

## Automatic execution and audit

Each host has a detached supervisor with exclusive queue lock, PID/start-time ownership checks, a fixed host/schedule assignment and a frozen source/runtime manifest. Per-cell maximum attempts:3; no-progress timeout:1,800seconds. Resume only accepts complete cells with matching checkpoint/protocol and verified videos/traces. Failed or interrupted partial attempts remain preserved but excluded from valid totals. The controller survives terminal disconnect; machine-reboot restart is not configured.

Finalization requires **100 complete task/settings and10,000 verified episode/video records per schedule**, rechecks source/runtime/checkpoint hashes, recomputes success totals and records excluded attempts. Only a passed final audit receives a complete label. Preliminary SR is provisional.

Local deployment: `/data14t/anhdh35/loopwam_robotwin_elastic_e0_teacher10_5ep_20261111`.
Remote deployment: `/media/vmo-perception/disk_2/anhdh35/loopwam_robotwin_elastic_e0_teacher10_5ep_20261111`.
Report root: `/storage/anhdh35/LoopWAM_NT/evaluation_report/robotwin_elastic_e0_teacher10_5ep_20261111`.

The local sync process refreshes both host reports every60seconds and flags stale unaudited snapshots. Host `report.md`, `current_results.json`, queue state, gate receipts and eventual schedule-specific `final_report.md`, task/episode CSVs and final audits are preserved beneath `local/` and `remote137/`. Source snapshots and download/validation records are retained for review.

## Interpretation of comparisons

[1,1] versus[2,2] uses the same new student checkpoint, seed protocol, task coverage, runtime versions and validated arithmetic, differing in action loop schedule and host. Cross-host numerical parity is verified on recorded inputs; this does not establish perfectly paired simulator episodes. Comparison to the older E0 also changes teacher epoch and student training duration, so any SR difference cannot be attributed to loop depth alone. The older E0 has no completed[2,2] reference in this campaign.
