# LoopWAM v0 initialization variants: training and matched LIBERO evaluation

Generated 2026-10-28 from the run records by [`scripts/init_variants_h100/build_initvar_report.py`](../../scripts/init_variants_h100/build_initvar_report.py); machine-readable data in [`plans/evidence/v0_init_variants_20261028`](../../plans/evidence/v0_init_variants_20261028).

Every variant is LoopWAM v0 with video/action loops 4/4 (3 prelude + 6 shared core x 4 loops + 3 coda blocks per expert, 584,536,135 policy parameters), trained for ten epochs on all four LIBERO suites with the v0 release recipe: 1,712 demonstrations / 277,713 windows, seed 42, global batch 128, 21,700 updates, AdamW (0.9, 0.95), weight decay 0.01, LR 1e-4 with 5% warm-up and cosine decay, clip 1.0, FP32 master weights with BF16 compute. Only the student initialization differs (prelude = Wan blocks 0-2 and coda = Wan blocks 27-29 in every variant that uses Wan weights).

## Results (6,000 episodes per model)

| Initialization | All four suites | Spatial | Object | Goal | Long | Seed 42 | Seed 43 | Seed 44 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default (v0 release) | **94.92%** (5,695) | 96.93 | 98.73 | 96.13 | 87.87 | 95.55 | 94.95 | 94.25 |
| `core_mean` | **92.82%** (5,569) | 96.27 | 96.47 | 94.40 | 84.13 | 94.00 | 92.95 | 91.50 |
| `ffn_magnitude` | **92.12%** (5,527) | 95.80 | 98.27 | 92.93 | 81.47 | 93.00 | 91.50 | 91.85 |
| `stepwise` | **91.47%** (5,488) | 94.93 | 95.73 | 93.87 | 81.33 | 92.90 | 91.75 | 89.75 |
| `no_video_pretraining` | **85.65%** (5,139) | 91.93 | 92.60 | 90.00 | 68.07 | 86.00 | 88.15 | 82.80 |
| `core_random` | **82.20%** (4,932) | 92.00 | 91.07 | 85.13 | 60.60 | 85.20 | 85.75 | 75.65 |
| `core16_21` | not evaluated | | | | | | | |

Success rates in percent; each suite has 1,500 episodes (10 tasks x 50 initial states x 3 seeds), each seed 2,000. The default row is the campaign result for the released `libero-all-suites/v0-video4-action4` (evaluation_report/final, checkpoint export `1c659a7c2f69...`); every other row was evaluated on H100 in this work with the same protocol.

Observations. Each variant was trained once (seed 42), so training-seed variation is not measured; the binomial 95% interval is about +/-0.7-1.0 points on a model total and +/-1.8-2.5 on Long (episodes are not fully independent, so treat this as a lower bound). Gaps of under one point, such as `core_mean` vs `ffn_magnitude`, are not resolved.

- No variant beats the v0 initialization (94.92%). Ranking: `core_mean` 92.82%, `ffn_magnitude` 92.12%, `stepwise` 91.47%, `no_video_pretraining` 85.65%, `core_random` 82.20%.
- Every variant whose core comes from pretrained Wan blocks (`core_mean`, `ffn_magnitude`, `stepwise`) scores 91.47-92.82%, 2.1-3.5 points below v0. Without pretrained core weights the score drops further: `no_video_pretraining` (no Wan weights at all) 85.65% and `core_random` (pretrained prelude and coda around a freshly initialized core) 82.20%.
- Long has the largest drop for every model: `core_mean` 3.7, `ffn_magnitude` 6.4, `stepwise` 6.5, `no_video_pretraining` 19.8, `core_random` 27.3 points below v0's 87.87%. On the other three suites the pretrained-core variants stay within 3.2 points of v0.
- Seed spread (max-min over seeds 42/43/44) is at most 5.35 points for every model except `core_random` (10.10 points, lowest at seed 44); v0 is 1.30. The protocol reuses one noise vector for every query of a task/seed, so a seed effect can be shared across episodes.
- Final-epoch action loss orders the variants `stepwise` 0.0160 < `core_mean` 0.0161 < `ffn_magnitude` 0.0173 < `no_video_pretraining` 0.0267 < `core_random` 0.0325; the two highest-loss variants are also the two lowest-scoring.

### LIBERO-Long per task (successes out of 150)

| Task | default | `core_mean` | `ffn_magnitude` | `stepwise` | `no_video_pretraining` | `core_random` |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 138 | 136 | 124 | 124 | 94 | 95 |
| 1 | 146 | 144 | 149 | 148 | 131 | 137 |
| 2 | 142 | 127 | 134 | 136 | 127 | 104 |
| 3 | 147 | 143 | 139 | 145 | 134 | 132 |
| 4 | 128 | 119 | 120 | 102 | 54 | 82 |
| 5 | 148 | 150 | 148 | 142 | 129 | 97 |
| 6 | 94 | 83 | 81 | 80 | 79 | 40 |
| 7 | 147 | 137 | 119 | 110 | 106 | 98 |
| 8 | 90 | 90 | 82 | 106 | 53 | 49 |
| 9 | 138 | 133 | 126 | 127 | 114 | 75 |

## Training

| Initialization | Init rule | Trained on | Layout (GPUs x micro x accum) | Hours | Segments | Final-epoch loss video / action | Fairness | Hugging Face |
|---|---|---|---|---:|---:|---|---|---|
| `default` | core = Wan blocks 9-14 (the v0 release) | v0 release (earlier campaign), NVIDIA H100 80GB HBM3 | 4 x 8 x 4 | 12.1 | — | — | reference | [libero-all-suites/v0-video4-action4](https://huggingface.co/anhdao69/LoopWAM_NT/tree/main/libero-all-suites/v0-video4-action4) |
| `core_mean` | core block i = element-wise mean of Wan blocks 3+i, 9+i, 15+i, 21+i | HiPerGator, NVIDIA B200 | 2 x 16 x 4 | 12.8 | 1 | 0.0528 / 0.0161 | passed | [libero-all-suites/init_variants/core_mean](https://huggingface.co/anhdao69/LoopWAM_NT/tree/main/libero-all-suites/init_variants/core_mean) |
| `ffn_magnitude` | core = Wan blocks 9-14; each video-expert FFN keeps the 6,144 of 8,960 Wan neurons with the largest norm(W1 row) x norm(W2 column) (v0: 6,144 evenly spaced neurons) | H100 job 990547, NVIDIA H100 80GB HBM3 | 2 x 8 x 8 | 22.1 | 1 | 0.0555 / 0.0173 | passed (text-cache exception) | [libero-all-suites/init_variants/ffn_magnitude](https://huggingface.co/anhdao69/LoopWAM_NT/tree/main/libero-all-suites/init_variants/ffn_magnitude) |
| `stepwise` | core = Wan blocks 4, 8, 12, 16, 20, 24 | H100 job 1080498, NVIDIA H100 80GB HBM3 | 4 x 8 x 4 | 11.4 | 1 | 0.0559 / 0.0160 | passed (text-cache exception) | not uploaded |
| `no_video_pretraining` | every Wan-derived tensor from Wan2.1 from-scratch init (no video pretraining) | H100 jobs 990527 -> 990547, NVIDIA H100 80GB HBM3 | 2 x 8 x 8 | 23.1 | 2 | 0.0704 / 0.0267 | passed (text-cache exception) | [libero-all-suites/init_variants/no_video_pretraining](https://huggingface.co/anhdao69/LoopWAM_NT/tree/main/libero-all-suites/init_variants/no_video_pretraining) |
| `core_random` | core = freshly initialized default DiT blocks (seed 42), both experts | H100 job 990547, NVIDIA H100 80GB HBM3 | 2 x 8 x 8 | 22.1 | 1 | 0.0630 / 0.0325 | passed (text-cache exception) | [libero-all-suites/init_variants/core_random](https://huggingface.co/anhdao69/LoopWAM_NT/tree/main/libero-all-suites/init_variants/core_random) |
| `core16_21` | core = Wan blocks 15-20 | HiPerGator, NVIDIA B200 | 2 x 16 x 4 | 12.8 | 1 | 0.0519 / 0.0164 | passed | [libero-all-suites/init_variants/core16_21](https://huggingface.co/anhdao69/LoopWAM_NT/tree/main/libero-all-suites/init_variants/core16_21) |

Every run used the same data manifest, normalization (`d5ad351eb327...`), Wan2.1 VAE (`38071ab59bd9...`), schedule and precision. The branch fairness check (`scripts/v0_init_variant_verify.py`) passed for every B200 and H100 init-variant run, and the same field-by-field comparison with the v0 release passed for `no_video_pretraining`, with one documented exception on this cluster: the 40 T5 text-embedding files were regenerated (identical prompts, file names and valid-token counts; different bytes). The global batch is 128 in every run, but the device layout differs (B200 2 x 16 x 4, H100 2 x 8 x 8 or 4 x 8 x 4, v0 release 4 x 8 x 4), so the sampler order and noise streams differ between runs. Hours are training time only (the H100 runs shared nodes with other jobs at times). H100 init artifacts were built from this cluster's canonical donor artifact (`a990a583...`; the branch pins the HiPerGator build `a3bf2097...`, same Wan2.1 source).

`stepwise` ran on all four H100s of interactive job 1080498 (4 x 8 x 4, the v0 release layout) at 1.87 s/update; `core_random` and `ffn_magnitude` ran side by side on job 990547 (GPUs 0,1 and 2,3). `no_video_pretraining` hours include the updates repeated after its first job ended (two segments). `no_video_pretraining` was trained from LoopWAM_NT `56e3328` plus the scratch-initialization patch (branch `scratch_init_v0`); its inference code is identical to this branch.

## Evaluation protocol and checks

The campaign protocol `lightwam_20261009_matched` (`evaluation_report/audit_setup.py`): four suites x ten tasks x official initial states 0-49 x evaluation seeds 42/43/44. One environment per task/seed (env.seed(base seed) at construction) is reused for its 50 initial states in order; the global Python/NumPy/Torch seed is set once per task/seed and every action query uses the base seed. 30 settling actions (done ignored), then at most 400 (Spatial/Object/Goal) or 700 (Long) policy steps; 32-action chunks with ten executed per query; success is the environment done flag. Native LoopWAM inference: FP32 weights, BF16 autocast, BF16 Wan2.1 VAE, ten denoising steps, CFG 1, action shift 1; observation adapter and normalization from `scripts/evaluate_loopwam_libero.py`; gripper sign(-(2g-1)).

- Runner: [`scripts/init_variants_h100/eval/evaluate_v0_matched.py`](../../scripts/init_variants_h100/eval/evaluate_v0_matched.py) (SHA-256 `812893704471...`, identical for every model), adapted from the residual_trans H100 runner; task/seed groups are claimed atomically so several runners can share the work, and finished groups are never rerun.
- Speed: one CUDA graph replays the unchanged native video-prefill + action-denoising computation (no Inductor, no reduced steps). Before scoring each checkpoint, `--gate` compared it with native eager `infer_action` on real simulator observations from three suites: all 24 queries were bit-identical for every model.
- Harness parity: the released v0 4/4 policy through the same runner scored 454/500 on Long seed 42 versus 452/500 in the campaign record; 91.2% of episodes have the same outcome (trajectories diverge numerically across machines).
- `core_mean` was trained on HiPerGator, whose text-cache bytes differ from this cluster's; its evaluation checks the cluster files (same 40 names and valid-token counts), recorded in `core_mean/eval_text_cache_substitution.json`.
- Videos were not saved (they do not affect outcomes). 16 worker processes on 4 H100s and 16 CPU cores; the simulator is CPU-bound at about 18 policy steps per second per worker.

| Model | Evaluated checkpoint SHA-256 | Graph / eager query (ms) | Wall time | Policy steps | Queries |
|---|---|---:|---:|---:|---:|
| `core_mean` | `7613e262c47b...` | 250 / 492 | 1.29 h | 1,050,965 | 107,477 |
| `ffn_magnitude` | `455bc19ea802...` | 102 / 246 | 1.26 h | 1,071,853 | 109,563 |
| `stepwise` | `0af0b9174fdd...` | 102 / 250 | 1.30 h | 1,092,303 | 111,615 |
| `no_video_pretraining` | `e0bfdfa67447...` | 254 / 504 | 1.43 h | 1,253,586 | 127,542 |
| `core_random` | `cedc3a02f249...` | 103 / 254 | 1.86 h | 1,352,408 | 137,315 |

Gate timings for `no_video_pretraining` and `core_mean` were measured while another evaluation shared the GPUs. `core_random` started on GPUs 0,1 at 10:57 and gained GPUs 2,3 at 11:40, when the `ffn_magnitude` training released them. Wall time runs from the first runner start to the last runner end; all evaluations ran on 2026-10-28.

## Artifacts

- Evidence: [`plans/evidence/v0_init_variants_20261028`](../../plans/evidence/v0_init_variants_20261028): `results.json`, `episodes.csv` (every scored episode plus the parity run), `suite_seed_results.csv`, and per model the evaluation summary/manifest, CUDA-graph gate, runner sessions, training timing, fairness and initialization manifest.
- Scripts: [`scripts/init_variants_h100`](../../scripts/init_variants_h100).
- Weights: Hugging Face `anhdao69/LoopWAM_NT`, `libero-all-suites/init_variants/<variant>` (core16_21, core_mean, core_random, ffn_magnitude, no_video_pretraining); `stepwise` is not uploaded. On this cluster: `/groups/yshang/an221229/checkpoints/LoopWAM_NT/v0_init_variants/<variant>` (H100 training runs and all five evaluations, `eval_matched/`).
