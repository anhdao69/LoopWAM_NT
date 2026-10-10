# Full four-suite LoopWAM KV-mix training

## Authorized experiment

Fresh v0 with four video loops, one action loop, learned all-loop KV mixing.
All four available LIBERO suites, same full-suite normalization/cache and seed 42.
Ten epochs, global batch 128, 277,713 real windows per epoch, 21,700 total
updates and 2,777,130 real windows. FP32 parameters/Adam moments, BF16 compute.
Canonical donor initialization and fresh optimizer. Training only; no simulator
inference or evaluation is scheduled.

Use the owner's idle interactive allocation 4798 (two H100s on worker-3) for
preflight throughput tests. Submit a dependent two-GPU batch job after gates
pass; verify the submission before releasing only allocation 4798.

## Checkpoint retention

The new opt-in `--retain-epochs-from 5` preserves completed epochs 5, 6, 7, 8,
9 and 10 as `epoch_005.pt` through `epoch_010.pt`. `latest.pt` continues to be
saved each epoch, including epochs 1--4 for recovery. Archived checkpoints use
hard links: the model writer atomically replaces `latest.pt`, leaving older
inodes intact. This avoids serializing the same checkpoint twice. Under ZeRO,
the associated native optimizer tags are marked and exempted from pruning.
Defaults are unchanged for existing runs. DDP epoch checkpoints include weights,
optimizer, both rank RNG states, update/window count, and training contract.

Six DDP checkpoints require approximately 42.1 GB; final `latest.pt` shares
storage with epoch 10. The shared filesystem initially had only 86 GB free.
The runner checks free capacity before launching training. Future shared usage
by other jobs remains outside this run's control.

## Verification

- Red test: original trainer rejects `retain_epoch`, as expected (two failures).
- First GPU test launch failed because login `/tmp` is not shared with workers;
  preserved its failure log, moved tests to shared source, then reran.
- Nine CPU checkpoint/backend tests pass when executed on worker-3, including
  archive survival across atomic replacement and native ZeRO pruning.
- Benchmark DDP, ZeRO-1 and ZeRO-2 at global batch 128 with increasing
  microbatches; expected out-of-memory trials are recorded and rejected.
- Selected setup must pass ten fresh native updates with finite losses and
  gradients, precision checks, cache provenance and full-suite fairness check.
- Production verifies all 21,700 updates and every retained epoch on completion.

## Source isolation

Existing live training source `source_kv_3a89bfa` is unchanged. This run uses an
independent snapshot in `runs/loopwam_nt/mix_full_20261010/source`, derived from
commit `3a89bfac5cfccad66bfebbe00014f7f739a78adc`, with checkpoint-retention changes
and the dedicated training runner. Release hashes bind all Python model/trainer
source and the selected configuration before production begins.

Final configuration, job ID and measured estimate will be appended after gates.
