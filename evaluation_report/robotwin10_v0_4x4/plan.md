# RoboTwin-10 v0 4/4 evaluation plan

**Goal:** After the current GPU evaluation finishes, download and evaluate the requested RoboTwin-10 v0 release using the same scored protocol as dense-s30.

**Architecture:** Reuse the frozen evaluator at 7a2181d and its native checkpoint loader. Keep this release, queue state, smoke results, scored results and reports in separate directories. A persistent tmux queue owns the wait, pinned download, validation, calibration, full evaluation and final audit.

**Spec:** User request of 2026-11-03 and planned_protocol.json.

## Constraints and review focus

- User clarification: start when GPU memory allows; captured SpatialForcing process barrier removed. Do not stop other jobs. Dense-s30 already finished and passed its final audit.
- Download after verified dense completion; verify ten files against the pinned HF revision and release SHA256SUMS.
- Require matching source hashes, native strict load, epoch 5/update 45445, 584567886 policy parameters and video/action loops 4/4.
- Seed 42; NFE 10; 10 tasks; clean/randomized; 100 episodes per pair; total 2000. Preserve original simulator acceptance, horizons, image processing and instruction RNG.
- Numeric graph gate and unscored simulator gates must pass. Calibrate available 1/2/4-worker profiles with GPU memory margin. Never choose eight workers: earlier long evaluations hit renderer/VRAM limits.
- Freeze all runtime inputs before scoring. Preserve failed attempts and exclude their partial episodes.
- Produce task and episode CSVs and verified final Markdown after all 20 pairs complete.

## Tasks

- [x] Inspect release metadata, previous completion and current GPU work; compare nine source hashes.
- [x] Prepare queue scripts, run 38 existing tests with CUDA input preprocessing, and verify hash/admission logic.
- [x] Start persistent queue; user authorized downloading now and GPU admission by available memory.
- [x] Download after predecessor completion; verify all ten files and strictly load native v0 checkpoint.
- [ ] Automatically validate graph equivalence and measure available concurrency.
- [ ] Automatically run 2000 episodes and audit all results/videos; publish final report.

The user already authorized this queue; no additional approval is needed. This is operational configuration of the existing evaluator; no model or simulator changes are planned.
