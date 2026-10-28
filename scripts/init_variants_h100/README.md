# H100 wrappers for the v0 initialization-variant runs (October 2026)

Copies of the scripts that trained, evaluated and published the H100 init-variant runs; they ran from
`/groups/yshang/an221229/checkpoints/LoopWAM_NT/v0_init_variants_work` against this branch at `4ebf38a` (unmodified).
Results: [`plans/performance/LoopWAM_v0_Init_Variants_Results_20261028.md`](../../plans/performance/LoopWAM_v0_Init_Variants_Results_20261028.md).

Training (`scripts/train_loopwam.py` of this branch, unchanged):
- `env.sh`: Python/paths; `h100_build_variant.py` runs `scripts/build_init_variant.py` with this cluster's canonical
  donor artifact hash (`a990a583...`; the branch pins the HiPerGator build `a3bf2097...`).
- `train_variant_v1_2gpu.sh` (core_random, ffn_magnitude: 2 GPUs x microbatch 8 x accumulation 8) and `train_variant.sh`
  (same plus NPROC/WORKERS; stepwise used 4 x 8 x 4): stage the latent cache on node-local /tmp, train, then verify.
- `job_variant.sh` (run lock, retries), `smoke.sh`, `step_variant4.sh` (4-GPU smoke + run inside an srun step),
  `stepwise_watcher.sh` (waits for a new interactive job, falls back to the remaining time of another), `seed_latent_cache.py`.
- `h100_verify_variant.py`: the branch fairness check; the only accepted difference is the regenerated T5 text-cache file
  bytes (same names and valid-token counts).

Evaluation (`eval/`): `evaluate_v0_matched.py` (SHA-256 `812893704471...`, byte-identical to the copy that ran) implements
the campaign's matched LIBERO protocol with a CUDA-graph replay that must pass a bit-exact `--gate`; `eval_step.sh` /
`eval_step_hf.sh` launch it inside an srun step; `parity_reference.py` runs the released v0 4/4 through the same runner
and compares with the campaign's per-episode record. Publishing: `publish_initvar.py`. Report: `build_initvar_report.py`.
