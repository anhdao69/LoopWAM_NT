# LoopWAM partition grid implementation plan

Goal: train seven v0 aligned symmetric partitions on full four-suite LIBERO, then publish verified final weights.

Design: keep the exact twelve canonical donor blocks in their existing physical order (0,1,2,9,10,11,12,13,14,27,28,29). Change only which contiguous blocks repeat. Re-selecting donors by stage would confound the partition comparison. Each setup has 12 stored blocks and 30 applications per expert. Legacy defaults remain 3/6/3, four loops. Explicit partition metadata must survive checkpoint loading and be checked against the training contract. Zero prelude/coda are valid. Extend loop capacity only for explicit v0 partitions that require it. New partitions are aligned v0 only.

Runs: (0,6,6)x4; (1,6,5)x4; (3,6,3)x4 fresh control; (5,6,1)x4; (6,6,0)x4; (1,9,2)x3; (4,3,5)x7. Seed42, global128, fresh donor/optimizer, FP32 parameters/moments and BF16 compute, original full-suite normalization/cache, 277713 windows/epoch, 21700 updates/10 epochs. Preserve existing 3/6/3 weights. Training-only queues, no rollout evaluation between runs, matching the latest queue instruction. Final weights go into distinct date-stamped folders under anhdao69/LoopWAM_NT/libero-all-suites; retain local resumable latest checkpoint. No home storage, no other allocation changes.

Implementation:
- [ ] Tests first: all seven schedules versus explicit dense unroll, gradients, checkpointed gradients, cached inference and future masking; default exact equivalence; validation and checkpoint mode/partition mismatch guards.
- [ ] Add partition validation and loop capacity in loop_mot.py, factory/save/load/exit weights in loopwam.py. Keep initialization unchanged.
- [ ] Thread CLI through trainer/benchmark; record contract/manifest; update evaluator guards.
- [ ] Run existing tests and new tests on CPU/GPU, archive evidence.
- [ ] Benchmark representative DDP layouts with global128; smoke each partition from scratch on full dataset and verify fairness/finite losses/inference reload before queue launch.
- [ ] Launch two sequential queues inside allocation5248, two GPUs each. Fail closed on errors. Validate every finished checkpoint and export weights-only with provenance to HF; compare remote LFS SHA with local export. Upload failure recorded/retried without discarding training output.
- [ ] Record start evidence and measured ETA; sync source/report locally.

Review focus: empty stages, loops7 loss/serialization, unchanged donor identity, mismatched saved partitions, allocation GPU isolation, incomplete uploads/checkpoints, source/data provenance.
