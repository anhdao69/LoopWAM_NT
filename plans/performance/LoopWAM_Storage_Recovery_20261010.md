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
