# Elastic E0 full RoboTwin: setup, scope and validation

> Execution update, 2026-11-08: the queue now runs in parallel at the user’s request. Server `10.148.182.137` owns `[4,4]` / NFE 2; the local server preserves its active `[1,1]` evaluator and then runs `[4]×5`. The protocol below remains unchanged. See the [parallel execution and verification report](parallel137_20261108/report.md) and [live results](report.md).

## Checkpoint and source

- Hugging Face: `anhdao69/LoopWAM_NT`, pinned revision `cf5bc4fd986bc6877d613c4adea7c3b293ae0075`, folder `robotwin/Elastic/E0`.
- Checkpoint `epoch_01.pt`, SHA256 `abc3c0075a6e1f3ce380be8d6c7b6b9d8bb697ac21cf3f2d40a93efb821b539a`, 2,340,331,629 bytes. All13 release files verified.
- Native format `loopwam-s-robot-rt-v1`; one completed distillation epoch;47,462 updates;6,075,103 windows. Recipe T0, scope pc, trained step conditioning. Parent is original full RoboTwin v0 epoch5,237,310 updates, SHA256 `38975220e9441303878d2dfb9956d57e6416c6a906c99c43d9bca015b092fc51`.
- Main checkout `/storage/anhdh35/LoopWAM_NT` fast-forwarded from43ef539 to50d5bfe. All1,062 existing local report files preserved and hash-verified, with local and upstream report backups.
- Evaluation uses isolated `elastic_robotwin` source94c7d55 at `/data14t/anhdh35/loopwam_robotwin_elastic_e0_20261107/repo`. All25 native model source files equal training revision2c78ff6. Native model mathematics and weights are unchanged.
- Simulator and shared text helper copied from the original frozen full50 evaluation;704 matched source/configuration/instruction files. Frozen historical evaluator was not changed.

## Original sequential queue (superseded by parallel assignment)

| Order | Schedule | Action loops at each step | NFE | Scored episodes |
| --- | --- | --- | ---: | ---: |
| 1 | flat_k1_n2 | [1,1] | 2 |10,000|
| 2 | flat_k4_n2 | [4,4] | 2 |10,000|
| 3 | flat_k4_n5 | [4,4,4,4,4] | 5 |10,000|

Four video loops are retained for every query. These settings adjust action loops and the number of denoising steps. Step-conditioning value is0.5 for NFE2 and0.2 for NFE5; action time grids are[1000,500] and[1000,800,600,400,200], with native uniform flow deltas.

Each schedule uses all50 RoboTwin tasks × clean/randomized ×100 accepted episodes. Seed42 controls the original scene-seed protocol and fixed CPU action noise for every query; candidate scene seeds begin at4,300,000. Python instruction-template generation remains unseeded exactly as in the historical setup. Saved scene seeds and instruction strings allow checking actual pairing, but identical pairing is not presumed.

The original expert planning and success filter admits scenes, original task horizons apply, and unseen instructions are used. State/action normalization exactly matches training and the prior epoch5 evaluation. Inputs are the released FastWAM PIL uint8 resized head/wrist pack384×320, converted to BF16 before2/255−1. Policy parameters remain FP32; inference uses BF16 autocast; VAE follows native BF16. Predict32 joint-position actions and execute24 before replanning.

## Speed and hardware

Local RTX PRO6000 Blackwell Max-Q96GB; four concurrent simulator workers and one shared T5 process. Video recording and RT32/depth8 renderer remain enabled. OIDN2.3.3 compatibility runtime matches the established local baseline. No changes to success logic, action horizon, scene admission or precision are used for speed.

| Schedule | Native query | CUDA graph query |
| --- | ---: | ---: |
| [1,1] |44.3 ms|38.3 ms|
| [4,4] |59.7 ms|48.1 ms|
| [4]×5 |101.3 ms|69.5 ms|

These are measured single-worker policy-query times on eight recorded observations; they exclude simulator setup, planning, rendering and T5. Four-worker end-to-end throughput and completion time depend on task horizons and success frequency. The recent epoch10 full benchmark took16.52 hours with four workers at NFE10; lower query latency alone does not determine the E0 wall time.

## Validation and recovery

52 targeted tests passed: native Elastic schedule/export behavior, RoboTwin14D adapter, original image arithmetic, scoring/renderer contracts, restart invariants, and recovery regressions. Graph results matched native output bit-for-bit on all3 schedules across8 real simulator observations with noise seeds42,43,44, checked in forward and reverse replay order. Loaded state tensors match the pinned release exactly.

Independent read-only review found queue-cleanup/reporting recovery issues, all addressed and regression-tested. Child process groups are recorded by PID and start identity; cleanup is concurrent; restarts reject live duplicate supervisors and clean owned orphan workers. Bound runtime-manifest bytes are preserved on restart. Completed results are revalidated and audited without rerunning their episodes.

Four exploratory smoke episodes completed before review fixes and are archived in `smoke_pre_review`; they never count toward SR. The production queue repeats four smoke episodes per schedule on the final frozen code (12 additional episodes), all excluded. Valid policy failures in smoke are allowed; crashes, malformed outputs, missing video/trace or renderer/resource errors fail the gate.

A completed scored cell requires100 unique admitted scene seeds, consistent cumulative success counts, readable nonempty videos, one logged instruction per episode, contiguous policy queries and action hashes. OIDN/CUDA/Vulkan resource errors invalidate the whole attempt. Three attempts per cell and1,800-second no-progress timeout; failed/interrupted partial attempts are excluded and inventoried. Each schedule finalizer checks all10,000 videos and episode traces, checkpoint/source/runtime hashes, per-task sums, clean/randomized totals and final SR before publishing the final audit.

## Artifacts and operation

- Bulk checkpoint/log/video directory: `/data14t/anhdh35/loopwam_robotwin_elastic_e0_20261107`.
- Persistent launcher: `bash /data14t/anhdh35/loopwam_robotwin_elastic_e0_20261107/start_queue.sh` (only one controller may hold its lock).
- Live overview: [report.md](report.md), [current_results.json](current_results.json).
- Review: [review.md](review.md), [review gate](review_gate.json), [tests](final_tests.log).
- Provenance: [download receipt](download_receipt.json), [source check](source_check.json), [runtime manifest](runtime_manifest.json), [graph gate](graph_gate.json), [evaluation code snapshot](evaluation_code.tar.gz).
- Per-schedule final outputs appear automatically in `flat_k1_n2`, `flat_k4_n2`, `flat_k4_n5`: task_results.csv, episodes.csv, final_results.json, final_audit.json, failed_attempts.json and final_report.md.

These are new E0 evaluations, kept separate from the historical v0 epoch5/epoch6/epoch10 and RoboTwin10 results. Final benchmark SR will be available only after the corresponding schedule finishes.
