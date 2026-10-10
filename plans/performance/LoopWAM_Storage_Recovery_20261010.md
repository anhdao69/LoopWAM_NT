# Full-suite training recovery — October 10, 2026

## Incident and recovery plan

Jobs 4796 (2/2) and 4800 (full 4/1 KV mix) ended with exit code 1 at
03:35 UTC. Their logs end abruptly without a Python exception. At the same
check, `/mnt/data` reported 0 bytes free. The user removed weights; `/mnt/data`
now reports 11 TB free. Job 4770's trainer process is gone and its last
checkpoint is intact, but its paused campaign controller holds the allocation.
Job 4795 is the obsolete dependent continuation of 4770.

Continue with the existing output filesystem and exact remaining experiment
contract. Never delete or overwrite the failed-run evidence. Recover from the
latest verified DDP checkpoints for 4/1 and 2/2. Restart mix from the canonical
Wan donor because it failed before its first epoch checkpoint. Finish all
remaining models in the original chains with no evaluation between stages.

| Replacement plan | First stage | Remaining stages | Checkpoint used |
|---|---|---|---|
| concat | Resume aligned v0 4/1 | Fresh aligned 3/3 → Dense-S12 | update 15,190 of 21,700 |
| mix | Resume aligned v0 2/2 | Fresh Dense-S30 | update 4,340 of 21,700 |
| kv_mix | Fresh v0 mix 4/1 | none | canonical donor; epoch archives 5–10 |

All stages preserve the four-suite data, seed 42, global batch 128, ten epochs,
DDP and originally selected microbatch. The two resumes restore optimizer and
per-rank RNG from the saved checkpoint. All stages write new unique output
folders under `/mnt/data/.../storage_recovery_20261010/`; original outputs are
read-only recovery inputs. Mix retains each epoch 5–10 as a hard-linked
checkpoint, plus latest.pt. Completion gates validate all updates and windows,
finite metrics, checkpoint optimizer/RNG state, resumed log continuity, training
contract and full-suite data/normalization fairness.

Training output is written only after a 500 GiB free-space gate. The latent
cache under `/mnt/data` is reused read-only in practice; the recovery runner
requires every cache entry to already be valid. The currently observed 11 TB
free provides ample headroom. No job belonging to another user is modified.

## Preserved recovery inputs

- 4/1: `kv_campaign_job4770/campaign_3a89bfa/full_41/train/latest.pt`, step
  15,190; matching checkpoint and metrics are copied to a new recovery-input
  folder before replacing the stale job.
- 2/2: `training_only_job4796/full_22/train/latest.pt`, step 4,340; matching
  checkpoint and metrics are copied to the recovery-input folder.
- Mix 4/1: previous run reached update 1,914 with no checkpoint. Restart from
  the canonical donor with the original seed and training contract.

## Operational evidence

The `storage_recovery_20261010` server directory records source checkpoint
hashes, Slurm IDs, per-stage logs, manifests, timing, fairness checks, and final
checkpoint hashes. The implementation runner uses the previously benchmarked
DDP configurations and performs only training.

## Resubmission status — 2026-10-10 04:01 UTC

After the user cleared space, the two resume checkpoints were copied to
`storage_recovery_20261010/input_checkpoints/` and verified byte-for-byte
against their sources. Their DDP optimizer states and two per-rank RNG states
were present. The staged SHA-256 values are recorded in
`recovery_inputs.json`. `/mnt/data` had 12 TiB free at submission.

| Slurm job | Queue | Initial stage | State at 04:01 UTC | GPUs |
|---|---|---|---|---:|
| 4823 | concat | Resume 4/1 at update 15,190; then 3/3 and Dense-S12 | RUNNING | 2 |
| 4824 | mix | Resume 2/2 at update 4,340; then Dense-S30 | RUNNING | 2 |
| 4825 | kv_mix | Fresh mix 4/1 | RUNNING | 2 |

Each command requests the full 21,700-update budget with global batch 128 and
has no `--max-updates` cap. The trainer's `--smoke` option performs gradient
and initialization checks; it does not limit the training duration. Job logs
and all outputs are rooted under `/mnt/data/.../loopwam_nt/`. Slurm stdout is
explicitly directed to `recovery_operations_20261010/`; no training outputs
are directed to `/home`. The stale controller job 4770 and obsolete dependent
job 4795 were canceled after replacement jobs were submitted; their prior run
directories and checkpoints remain preserved.

Using the prior measured update rates, the two sequential queues each have an
estimated 33–34 hours of training remaining. The fresh mix run is estimated at
about 22 hours. If allocation and throughput remain steady, the longest queue
should finish around October 11, 14:00 UTC (10:00 EDT). This estimate excludes
queue delays and is not a measured completion time.

## Wrapper failure diagnosis — 2026-10-10

Slurm job 4823 exited with code 1 after its resumed 4/1 trainer had completed
all 21,700 updates and 2,777,130 windows. Its final checkpoint is present and
has SHA-256
`e8c7882ea5149ba7c45c0692b0e5f813af88953beb0780bffd6978795f7f96ef`.
Manual validation against the full-suite baseline passed the data, split,
normalization, update-count, and checkpoint checks. The failure came from the
recovery wrapper expecting `manifest.json`; resumed outputs are named
`resume_manifest.json`. The same review found that the wrapper omitted the
expected resume path when checking provenance.

Both wrapper issues are fixed on `KV_concat` (commits `af3e88a`, `f7636a1`,
`defb87e`). Job 4841 is queued to run the remaining 3/3 and Dense-S12 stages.
Job 4824 is still training its resumed 2/2 stage. Job 4842 is queued with an
`afterany:4824` dependency; it will validate job 4824's completed checkpoint
and then run Dense-S30. Job 4824 may itself exit nonzero in its older in-memory
wrapper after training completes; job 4842 validates and continues from the
completed output.
