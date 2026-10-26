# Approved centered-donor grid

Supersedes the fixed-donor partition experiment. User approved beginning prelude, centered core with lower-index tie, ending coda; requested only six models and authorized stopping SimpleMemVLN. No fresh3/6/3 production run.

Donors, zero-based inclusive: pre0..P-1; core floor((30-C)/2)..floor((30-C)/2)+C-1; coda30-S..29. Six-core uses12..17; nine-core10..18; three-core13..15. Preserve canonical default behavior behind explicit `--donor-strategy stage_centered`; record selected donors in architecture and training contract; guard mismatched checkpoint strategies. Both video/action experts take corresponding converted native donor tensors; heads/proprio remain artifact seed42.

Production:0/6/6x4,1/6/5x4,5/6/1x4,6/6/0x4,1/9/2x3,4/3/5x7. Full four-suite LIBERO,1712local demonstrations,277713windows/epoch,21700updates,10epochs,global128,seed42,fresh AdamW,FP32 weights/moments,BF16 compute. Original full-suite normalization and VAE cache unchanged. Three sequential experiments per allocation; use all four H100s of each allocation4925/5248. Stop only identified SimpleMemVLN step4925.194 and the superseded old LoopWAM step5248.63; preserve allocations, original shells and outputs.

Validation: tiny exact donor selection/metadata guards; native all679 tensors per model equal chosen donor-bank tensors; full CPU/GPU regression; independent review; compare DDP/ZeRO1/ZeRO2 and checkpointed microbatches with FP32/finite/global-budget gates; choose fastest verified four-GPU layout; native ten-update smokes plus inference reload for all six; source/config/fairness gates before release. Training-only production, final checkpoint HF uploads with source/export/remote hash validation. New outputs/HF paths include centered to distinguish original fixed-donor runs.

Progress: initial21 donor tests and native six-model tensor audit passed. Initial CPU471pass16skip, GPU487pass. Review identified benchmark acceptance checks and interrupted-export retry robustness; fixed with red/green tests. Benchmark and final tests underway. Code changes stay on partition_grid_20261026 branch; no merge into older branches.

Completed: final CPU473pass16skip, GPU489pass; all six native10-update/fairness/reload/inference gates passed; clean DDP/ZeRO sweep selected four-GPU DDP mb8 acc4. Production launched5248.70/4925.201 at18:50UTC, verified both atupdate29 with finite metrics. Final HF uploads remain scheduled after each ten-epoch completion. See plans/performance/LoopWAM_Centered_Donor_Grid_20261026.md and machine-readable evidence.
