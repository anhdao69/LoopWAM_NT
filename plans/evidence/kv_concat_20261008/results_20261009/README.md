# Completed Long results — October 9, 2026

Synced metadata, all 600 three-seed concat/mix episodes, both expanded 500-episode
grids, full training metrics, final eager latency, diagnostics plots, and generated
reports. Expanded seed42 overlaps the original first ten states per task: do not
pool those executions as independent evidence. One training seed42 only.

The original 300-episode decision is inconclusive; no new threshold is applied to
the expanded result. Native BF16 Inductor failed numerical equivalence; inference
and latency here are eager. Checkpoints, videos and latent tensors stay on server.

`verify_results.py` independently recomputes reported counts, paired tests,
confidence intervals and training loss summaries using only Python's standard
library. `verification.json` records its checks and local file SHA-256 hashes.
`current_training_snapshot.json` is a dated status snapshot, not final full-suite
results. The historical launch snapshot is retained in sibling campaign_3a89bfa.
