# RoboTwin evaluation and continuation implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Automatically evaluate epoch5 across full RoboTwin, then train epochs6–10 at cosine3e-5→3e-7 with every epoch saved.
**Architecture:** Isolated original-v0 adapter, resumable simulator scheduler, phase-aware trainer, persistent allocation controller.
**Tech Stack:** Python, PyTorchDDP, Slurm, SAPIEN3/CuRobo, existing RoboTwin harness.
**Spec:** docs/superpowers/specs/2026-10-30-robotwin-eval-continue-design.md

## Global constraints
Original v0 only; existing training source/output untouched;8GPUs and real global128;retain optimizer;save6,7,8,9,10;eval10000episodes before continuation;all persistent files/mutations /mnt/data.

## Review focus
Wrong epoch/model checkpoint, incomplete eval counted complete, duplicated/missing windows on continuation, LR restart/resume off-by-one, killed subprocess or stale controller causing duplicate launches.

### Task1: continuation contract and trainer
- [x] Add failing tests for epoch5 boundary, immutable contract rejection, optimizer preservation, LR endpoints and absolute epoch numbering.
- [x] Add explicit continuation argument/helpers; preserve original5epoch path; save6–10.
- [ ] Run CPU tests and native8GPU smoke from a frozen trained checkpoint fixture, validate resume and tail.

### Task2: original-v0 RoboTwin policy and full evaluator
- [x] Install isolated pinned simulator/assets; validate imports and renderer.
- [x] Tests for policy normalization/input/inference routing and evaluation completeness/retry guards.
- [x] Implement adapter+resumable50task×2setting manager, per-episode records/videos and strict final summary.
- [x] Run real clean/randomized simulator smoke with epoch4 weights; measure throughput/memory; select concurrency.

### Task3: persistent queue and handoff
- [x] Test mocked-stage orchestration success, incomplete parent/eval rejection, duplicate lock and resume.
- [x] Implement controller verifying epoch5 weights/optimizer equality then eval then continuation; launch detached allocation step and verify waiting heartbeat.
- [x] Independent review; address correctness findings, run regression; commit and push source/evidence/report.

Progress ledger: source branch created from04ea0fa. User approved cosine3e-5→3e-7 and epoch6–10 saving. Simulator absent on H100; installation ongoing.

Progress: contract/policy CPU tests15passed; pure pipeline tests6passed including duplicate-stage lock and child failure recording. Original training unaffected. Assets downloaded; SAPIEN3/CuRobo install ongoing, graphics userspace runtime absent in container and being prepared privately. Builder review requested exact checkpoint tensor comparison, optimizer/cursor guards, 10000 verified videos, stage adoption/locks; implemented, final code review pending.

Final setup: CPU337passed/24skipped, GPU344passed/17skipped; clean/randomized original-v0 epoch4 smoke passed serially and concurrently on two different H100s, including verified videos. All50task modules import. SAPIEN3.0.0b1 preserved; H100 raster fallback and Mesa PCI selection are explicit. Runtime0703c9d; CPU controller5489.54 is waiting for epoch5. Native8GPU continuation smoke/resume is intentionally scheduled after full epoch5 evaluation and before production epochs6–10, so that gate is not marked complete here. No claim of final benchmark results or completed continuation.
