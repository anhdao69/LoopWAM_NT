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
storage with epoch 10. The shared filesystem initially reported only 86 GiB free (`df -h`); after the native
smoke checkpoint, Python reported approximately 84.2 GB (78.4 GiB) free.
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

## Submitted training job

**4800**, `test_training`, worker-3, two H100s, 30 CPUs, 256 GiB host memory,
72-hour wall-time. Submitted after all gates passed with `afterany:4798`;
confirmed the owner/resource/dependency before cancelling interactive job 4798.
Slurm then started job 4800 on worker-3. Existing jobs 4770, 4795 and 4796 were
not modified.

Output: `/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/mix_full_job4800/`.
Slurm log: `runs/loopwam_nt/mix_full_20261010/slurm-4800.log`.
Training log and checkpoints are under the output directory (`train.log`,
`train/latest.pt`, and, starting after epoch 5, `train/epoch_005.pt` etc.).

## Throughput selection

Ten-update trials, mean over updates 3--10; each trial starts from the canonical
donor. Global batch remains 128 throughout. These short measurements are
configuration comparisons, not complete production runtimes.

| Backend | Microbatch/GPU | Accumulation | Activation checkpointing | Seconds/update |
|---|---:|---:|---|---:|
| **DDP (selected)** | **8** | **8** | **No** | **3.623** |
| ZeRO-1 | 8 | 8 | No | 3.734 |
| ZeRO-2 | 8 | 8 | No | 3.907 |
| DDP | 16 | 4 | Yes | 4.399 |
| DDP | 32 | 2 | Yes | 3.963 |
| DDP | 64 | 1 | Yes | 3.924 |

Microbatch 16 without activation checkpointing ran out of memory for each of
DDP, ZeRO-1 and ZeRO-2; these expected capacity trials were rejected. Lower
accumulation did not produce the fastest update time. The selected native
training smoke measured 3.622 seconds/update, projecting **21.83 training hours**.
Allow **22--24 hours from production start**, including checkpoint overhead and
normal variation. No evaluation time is included or scheduled.

One observed GPU sample during the selected benchmark was 98% utilization on
both GPUs and 60,107 MiB used per GPU. This is a snapshot, not a utilization
average. Benchmark allocator peak reserved memory was approximately 61.34 GB.

## Release and evidence

Local implementation commit: `713c32f`. Remote independent source snapshot:
`a52d351cd78253820c146e4215ad612883f71f2d`. The release binds all source file
hashes, selected configuration, successful tests and native fairness evidence.
Free shared space at release was 80.30 GB; the runner requires at least 63.14 GB
before starting, based on six 7.016 GB archives plus checkpoint-write headroom.

Raw preflight records, OOM logs, the initial test-launch failure, native training
metrics/manifests, fairness result, source hashes, and verified Slurm submission
are synced in `plans/evidence/kv_concat_20261008/full_mix_20261010/`.

## Production startup verified

At 2026-10-10 01:39:57 UTC, Slurm job 4800 was RUNNING on worker-3.
Twelve persisted update records were checked with finite video/action losses
and gradient norms; the concurrently read timing file had advanced to update
13 at a mean 3.617 seconds/update. Cache misses were zero. The production
manifest confirms fresh initialization (`resume=null`), full-suite budget,
global batch 128, and `retain_epochs_from=5`. Production started around
01:38 UTC (October 9, 9:38 p.m. EDT). The 22--24 hour allowance places completion
around **October 10, 7:40--9:40 p.m. EDT**. This is an estimate, not a completed
training result.
