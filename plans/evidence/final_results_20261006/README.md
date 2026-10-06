# Final LIBERO-Long results — October 6, 2026

See the [detailed report](../../performance/LoopWAM_Final_Results_20261006.md).

Four fresh, completed 10-epoch runs: v0 96/100, v1 82/100, Dense-S12 81/100, v2 81/100. Each final checkpoint was evaluated on 100 episodes with videos.

## Contents

- `raw/`: original training/evaluation manifests, timing, metrics, episode records and logs. Server paths and original status fields are preserved.
- `remote_audit.json`: final checkpoint hashes/contracts, Slurm completion evidence, and SHA-256/frame-count checks for all 400 videos.
- `analysis.json`, `models.csv`, `epochs.csv`, `tasks.csv`, `episodes.csv`, `paired_outcomes.csv`, `failed_episodes.csv`, `runtime_summary.json`: computed summaries.
- `figures/`: success/runtime, learning curves and per-task scores.
- `local_integrity.json`, `files.sha256`: local report/data verification and file inventory.

**Videos remain on the H100 server at the user's request.** Model checkpoints are also retained on the server. Every episode CSV includes its video's absolute server path. No videos, checkpoints, optimizer shards, latent caches or credentials are committed here.

`v1_bs128_job4691/job_status.json` contains a stale `exit_code: 1` from its rejected microbatch16 benchmark. The final job and evaluation completed successfully, as explained in the report and Slurm audit.

## Reproduce

From the repository root, with Python and Matplotlib available:

```bash
python plans/performance/analysis/summarize_final_results.py
python plans/performance/analysis/write_final_report.py
python plans/performance/analysis/verify_local_archive.py --metadata-only
```

To verify archived data without rebuilding:

```bash
cd plans/evidence/final_results_20261006
shasum -a 256 -c files.sha256
```
