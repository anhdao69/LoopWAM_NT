# E0 [4,4] continuation implementation plan

Goal: queue one full RoboTwin50 [4,4],NFE2 evaluation automatically after both current runs pass their audits.
Architecture: isolated queue scripts, reused immutable model/runtime, audited dependency gate, existing fourworker evaluator and finalization.
Spec: spec.md

- [x] Test [4,4] acceptance and rejection of altered seed/depth/checkpoint/protocol.
- [x] Test dependency waiting, full-audit requirements, wrong-checkpoint rejection.
- [x] Extend only copied queue/contract scripts and add dependency and graph gates.
- [x] Run campaign tests, inspect code changes, get an independent review.
- [ ] Start detached queue, verify PID identity and progress; retain logs/report.

Review focus: no premature start on incomplete/missing audit; no duplicate launch; transient SSH errors; wrong checkpoint or schedule; accurate queued/error/final status.

Migration: test remote host assignment and transferred predecessor integrity; compare source/checkpoint hashes, fresh numerical/smoke gates on137, synchronize report to controller. Local queue must remain stopped.
