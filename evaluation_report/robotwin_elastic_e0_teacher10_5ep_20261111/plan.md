# E0 teacher10 five-epoch evaluation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Implement inline, then one independent full review.

**Goal:** Start or durably queue the two verified full50 evaluations on the two requested GPUs.
**Architecture:** Copy the validated E0 evaluator into an isolated snapshot; minimally add flat_k2_n2 and the new strict release identity. Reuse checked host queue/report/final audit components. Source and runtime fingerprints guard the independent deployments.
**Tech Stack:** Python3.10, PyTorch, native RoboTwin/SAPIEN, CUDAgraphs, SSH/rsync.
**Spec:** spec.md

## Global Constraints

Use only the specified release. Seed42;50tasks;2settings;100episodes;[1,1] local and[2,2]137;NFE2;fourworkers;predict32/execute24;T0. No modifications to other campaigns or GPU jobs. No score mixing. Preserve all verification and report artifacts.

## Review Focus

- Wrong parent/epoch accidentally accepted: test strict release guard with old teacher/student/T4 inputs.
- [2,2] silently mapped to[4,4] or wrong step condition: test native schedule mapping and instrument actual native velocity calls in graph gate.
- Wrong host/schedule or duplicate supervisor: test host assignment and retained process ownership checks; independent queue locks.
- Runtime/source rebinding on resume: existing publish_runtime rejects changed manifests; tests exercise rejection; source equality and checkpoint digests required on bothhosts.
- Failed partial episodes reused or finalization skipped: retain complete evidence and artifact validation; final audit checks every cell/video and excluded attempts.

## Tasks

- [ ] Task1: Verify release metadata/download and create isolated evaluator; record plan/spec/provenance. Download bothhosts; verify against published SHA256 and older protocol.
- [ ] Task2: Add strict new export identity and [2,2] adapter via failing tests, then run native adapter/evaluator/transport suites. Keep native model math unchanged.
- [ ] Task3: Adapt host queues, CPU verification, graph gate, source freeze, finalizer and report sync to exact assignments. Test assignment/contract/resume behavior. Compare frozen teacher weights and native source hashes.
- [ ] Task4: Deploy identical code on bothhosts; run CPU/source/tests and native/graph/cross-host equivalence. Request independent review; fix substantive findings with tests before launching queues.
- [ ] Task5: Launch detached queues and report sync; verify live ownership and accurate waiting/running status. Observe full scoring if GPU admission is currently available, otherwise verify the durable waiting state. Write setup and launch receipts.

Verification commands and results go in progress.md; retain artifacts rather than delete review evidence. Queue completion means automatic scheduling established; full benchmark result remains pending until final audit.
