# Training-only continuation

User explicitly requests resubmitting Slurm and removing all inference between
stages. Preserve the active full-suite training, data/optimizer contracts, and
all prior evidence. Do not repeat completed Long runs.

1. Validate existing allocations and immutable source3a89bfa; prepare a standalone
   continuation runner importing the existing training command and completion gates.
2. Submit two dependent two-H100 jobs: after4770 run full3/3 then Dense-S12;
   after4771 run full2/2 then Dense-S30. No evaluation, latency or peer barriers.
3. Pause only each old campaign controller PID, leaving its torchrun child alive.
   A CPU-only handoff monitor validates the current final training checkpoint and
   then cancels only its own old allocation. New jobs require a successful handoff
   record and the completed predecessor training before starting.
4. Verify PID ownership/command/job, source hashes, checkpoint completeness,
   optimizer state and finite full training logs. Fail closed on any mismatch.
5. Review the whole operational change, submit, verify pending dependencies and
   ongoing current training, record job IDs and estimated training-only completion.

Ruling: continue live current training instead of immediately resuming an older
checkpoint, because current saves lag behind live updates. This avoids discarded
work. New jobs begin at the next model and use fresh canonical initialization.
Ruling: the user's latest request supersedes all future simulator/latency stages;
completed evaluation artifacts stay intact. No changes to training/model code.

Completed: five local/server unit tests, parent-only SIGSTOP subprocess check, independent review fixes, pinned-source dry runs, jobs4795/4796 submission, both guards armed, live child progress verified. Monitoring errors preserve live training and require operator recovery. No optimizer/training change, no unsaved-update rollback.
