# RT execution ledger
Plan: docs/superpowers/plans/2026-10-10-residual-transport.md
Ruling: User's 10 epochs and epoch 8/9/10 retention override old 6000-update budget; keep optional 2000-update early checkpoint.
Ruling: User explicitly preserves evaluation gates. RT-A/RT-B4 eligible; no invented gate result.
Ruling: Work in user-specified remote directory on newly created residual_trans, preserving the prior branch and untracked specification.
Allocation: 4843 on worker-0, 2 H100 80GB, unlimited walltime, idle as checked 2026-10-10. 4825 is unrelated storage recovery; do not interrupt.
Source baseline: 3eac53a. Relevant available remote branches: LoopWAM_NT and KV_concat.

Verification: 328 project tests passed, 7 skipped; all 20 recipes native smoke passed.
Resume: ten resumed updates bit-exact. Graphs: 26 cases x 50 queries, zero error.
Exports: unmerged trained state preserved for BF16 bit-exact inference; B2a roundtrip exact.
Layout: 2/4 logical ranks on 2 physical GPUs passed 2% tolerance; no 4-device NCCL claim.
Final benchmark: deterministic global128/micro64, no block checkpoint, 73.07GB/GPU.
Implementation ready for pinned snapshot and eligible production launch.

Launch: RT-A step4843.16 running; RT-B4 job4846 pending resources; RT-A recovery4847 pending dependency.
Source f669184242b5 pushed and pinned. First20 production updates finite and full training contract confirmed.
HF public metadata uploaded and downloaded back with exact hash match; trained weights pending milestones.
All currently eligible launches complete. External gates, actual four-device NCCL and undefined R-FM remain explicitly outstanding.
