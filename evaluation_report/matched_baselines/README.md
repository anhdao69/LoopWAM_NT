# Matched LIBERO evaluation: public WAM baselines and LoopWAM exports (updated 2026-10-30)

Every number here uses the campaign protocol `lightwam_20261009_matched`: 4 suites x 10 tasks x official initial states 0-49 x evaluation seeds 42/43/44 = **6,000 episodes per model** (1,500 per suite, 2,000 per seed).

- **This evaluation:** 14 models evaluated on a local 2x RTX PRO 6000 Blackwell host (eager native inference), each re-verified on 2026-10-30 against its full 6,000-episode grid (Section 5). This is where FasterWAM, Light-WAM, Dense-S12/S30, the v0 test-time action-depth sweep, the six centered partition-grid exports and the `core16_21` initialization variant were scored.
- **Other branches (pulled 2026-10-30):** every four-suite matched result reported on the other branches (Elastic, Residual Transport, ChronoLoop, init variants, campaign rows), as reported there; the documents themselves are in [`../branch_reports`](../branch_reports).
- FastWAM was scored by the campaign (`evaluation_report/final`). LiLa-WAM has no matched result (Section 4).

## 1. All four-suite matched results (6,000 episodes each), highest first

| # | Model | Success % | Evaluated |
|---:|---|---:|---|
| 1 | **FasterWAM** | **98.72** | this evaluation (local RTX PRO 6000 Blackwell) |
| 2 | FastWAM (released, libero_uncond_2cam224) | 96.63 | H100, branch `elastic` all-models report (baseline) |
| 3 | ChronoLoop CL-A_mem16-learned-loopwrite_a4 | 96.18 | H100, branch `elastic` all-models report (chronoloop) |
| 4 | ChronoLoop CL-0_no-memory_a4 | 96.10 | H100, branch `elastic` all-models report (chronoloop) |
| 5 | ChronoLoop CL-W2_mem16-learned-external-updater_a4 | 96.02 | H100, branch `elastic` all-models report (chronoloop) |
| 6 | ChronoLoop CL-REG_mem16-reset-registers_a4 | 95.97 | H100, branch `elastic` all-models report (chronoloop) |
| 7 | ChronoLoop CL-FRAME_history-frame-k3_a4 | 95.90 | H100, branch `elastic` all-models report (chronoloop) |
| 8 | elastic F0-PB parent2 [4,4] | 95.88 | H100, branch `elastic` all-models report (elastic) |
| 9 | **Light-WAM** | **95.77** | this evaluation (local RTX PRO 6000 Blackwell) |
| 10 | elastic F0-PB parent5 [4]x5 | 95.75 | H100, branch `elastic` all-models report (elastic) |
| 11 | ChronoLoop CL-0-at1_no-memory_a1 | 95.70 | H100, branch `elastic` all-models report (chronoloop) |
| 12 | elastic E1 flat_k4_n5 [4]x5 | 95.65 | H100, branch `elastic` all-models report (elastic) |
| 13 | **LoopWAM Dense-S30** | **95.63** | this evaluation (local RTX PRO 6000 Blackwell) |
| 14 | elastic E0 flat_k4_n5 [4]x5 | 95.45 | H100, branch `elastic` all-models report (elastic) |
| 15 | ChronoLoop CL-A-at1_mem16-learned-loopwrite_a1 | 95.33 | H100, branch `elastic` all-models report (chronoloop) |
| 16 | elastic E2 flat_k4_n5 [4]x5 | 95.28 | H100, branch `elastic` all-models report (elastic) |
| 17 | elastic E2 flat_k4_n2 [4,4] | 95.27 | H100, branch `elastic` all-models report (elastic) |
| 18 | elastic E1 flat_k4_n2 [4,4] | 95.22 | H100, branch `elastic` all-models report (elastic) |
| 19 | elastic E2 flat_k2_n5 [2]x5 | 95.07 | H100, branch `elastic` all-models report (elastic) |
| 20 | RT-A2 S5 [4,2,2,2,2] | 94.98 | H100, branch `elastic` all-models report (residual-transport) |
| 21 | elastic E0 flat_k4_n2 [4,4] | 94.97 | H100, branch `elastic` all-models report (elastic) |
| 22 | RT-A2 S7 [4,2] | 94.93 | H100, branch `elastic` all-models report (residual-transport) |
| 23 | all-suites v0 4/4 | 94.92 | H100, branch `elastic` all-models report (all-suites) |
| 24 | elastic E0 flat_k4_n10 [4]x10 | 94.87 | H100, branch `elastic` all-models report (elastic) |
| 25 | elastic E0 flat_k2_n5 [2]x5 | 94.85 | H100, branch `elastic` all-models report (elastic) |
| 26 | elastic E2 flat_k2_n2 [2,2] | 94.75 | H100, branch `elastic` all-models report (elastic) |
| 27 | elastic E2 flat_k4_n10 [4]x10 | 94.70 | H100, branch `elastic` all-models report (elastic) |
| 28 | elastic F0-PB S7 [4,2] | 94.70 | H100, branch `elastic` all-models report (elastic) |
| 29 | elastic F0-PB parent10 [4]x10 | 94.68 | H100, branch `elastic` all-models report (elastic) |
| 30 | **LoopWAM v0 Ka5** | **94.60** | this evaluation (local RTX PRO 6000 Blackwell) |
| 31 | elastic E1 flat_k2_n2 [2,2] | 94.58 | H100, branch `elastic` all-models report (elastic) |
| 32 | RT-B2a parent2 [4,4] | 94.57 | H100, branch `elastic` all-models report (residual-transport) |
| 33 | RT-T1ft S1 [4]+[2]x9 | 94.57 | H100, branch `elastic` all-models report (residual-transport) |
| 34 | elastic E0 flat_k1_n5 [1]x5 | 94.53 | H100, branch `elastic` all-models report (elastic) |
| 35 | elastic E0 flat_k2_n2 [2,2] | 94.53 | H100, branch `elastic` all-models report (elastic) |
| 36 | RT-A2 S8 [4,1] | 94.53 | H100, branch `elastic` all-models report (residual-transport) |
| 37 | elastic E1 flat_k2_n5 [2]x5 | 94.52 | H100, branch `elastic` all-models report (elastic) |
| 38 | elastic F0-PB S8 [4,1] | 94.50 | H100, branch `elastic` all-models report (elastic) |
| 39 | RT-A2 S6 [4,1,1,1,1] | 94.48 | H100, branch `elastic` all-models report (residual-transport) |
| 40 | RT-A S1 [4]+[2]x9 | 94.47 | H100, branch `elastic` all-models report (residual-transport) |
| 41 | elastic E1 flat_k4_n10 [4]x10 | 94.45 | H100, branch `elastic` all-models report (elastic) |
| 42 | all-suites v0 4/1 kv-mix | 94.45 | H100, branch `elastic` all-models report (all-suites) |
| 43 | RT-TF S1 [4]+[2]x9 | 94.42 | H100, branch `elastic` all-models report (residual-transport) |
| 44 | **LoopWAM v0 partition 0/6/6** | **94.42** | this evaluation (local RTX PRO 6000 Blackwell) |
| 45 | elastic E0 flat_k1_n2 [1,1] | 94.33 | H100, branch `elastic` all-models report (elastic) |
| 46 | RT-B4 S1 [4]+[2]x9 | 94.32 | H100, branch `elastic` all-models report (residual-transport) |
| 47 | elastic E1 flat_k1_n5 [1]x5 | 94.30 | H100, branch `elastic` all-models report (elastic) |
| 48 | elastic E2 flat_k1_n5 [1]x5 | 94.25 | H100, branch `elastic` all-models report (elastic) |
| 49 | elastic F0-PB S5 [4,2,2,2,2] | 94.25 | H100, branch `elastic` all-models report (elastic) |
| 50 | RT-T1ft S2 [4]+[1]x9 | 94.18 | H100, branch `elastic` all-models report (residual-transport) |
| 51 | RT-TF S2 [4]+[1]x9 | 94.13 | H100, branch `elastic` all-models report (residual-transport) |
| 52 | RT-A S2 [4]+[1]x9 | 94.12 | H100, branch `elastic` all-models report (residual-transport) |
| 53 | RT-A2 S1 [4]+[2]x9 | 94.10 | H100, branch `elastic` all-models report (residual-transport) |
| 54 | elastic E2 flat_k1_n2 [1,1] | 94.07 | H100, branch `elastic` all-models report (elastic) |
| 55 | elastic E2 flat_k2_n10 [2]x10 | 94.07 | H100, branch `elastic` all-models report (elastic) |
| 56 | RT-A S3 [4,3,2,2,2,1,1,1,1,1] | 94.05 | H100, branch `elastic` all-models report (residual-transport) |
| 57 | elastic E1 flat_k2_n10 [2]x10 | 94.00 | H100, branch `elastic` all-models report (elastic) |
| 58 | elastic E2 flat_k1_n10 [1]x10 | 94.00 | H100, branch `elastic` all-models report (elastic) |
| 59 | all-suites v0 4/1 | 94.00 | H100, branch `elastic` all-models report (all-suites) |
| 60 | RT-Pa S1 [4]+[2]x9 | 93.95 | H100, branch `elastic` all-models report (residual-transport) |
| 61 | RT-A S4 [4,4,1,1,1,1,1,1,1,1] | 93.87 | H100, branch `elastic` all-models report (residual-transport) |
| 62 | elastic E0 flat_k2_n10 [2]x10 | 93.83 | H100, branch `elastic` all-models report (elastic) |
| 63 | elastic SP11 flat_k1_n2 [1,1] | 93.77 | H100, branch `elastic` all-models report (elastic) |
| 64 | elastic F0-PB S6 [4,1,1,1,1] | 93.75 | H100, branch `elastic` all-models report (elastic) |
| 65 | elastic E1 flat_k1_n2 [1,1] | 93.73 | H100, branch `elastic` all-models report (elastic) |
| 66 | elastic E0 flat_k1_n10 [1]x10 | 93.68 | H100, branch `elastic` all-models report (elastic) |
| 67 | **LoopWAM v0 partition 1/6/5** | **93.67** | this evaluation (local RTX PRO 6000 Blackwell) |
| 68 | elastic E1 flat_k1_n10 [1]x10 | 93.48 | H100, branch `elastic` all-models report (elastic) |
| 69 | elastic F0-PB S1 [4]+[2]x9 | 93.38 | H100, branch `elastic` all-models report (elastic) |
| 70 | RT-A2 S2 [4]+[1]x9 | 93.22 | H100, branch `elastic` all-models report (residual-transport) |
| 71 | **LoopWAM v0 partition 5/6/1** | **93.17** | this evaluation (local RTX PRO 6000 Blackwell) |
| 72 | **LoopWAM v0 init core16_21** | **93.12** | this evaluation (local RTX PRO 6000 Blackwell) |
| 73 | RT-Pa S2 [4]+[1]x9 | 92.92 | H100, branch `elastic` all-models report (residual-transport) |
| 74 | init variant core_mean | 92.82 | H100, branch `elastic` all-models report (init-variants) |
| 75 | RT-B4 S2 [4]+[1]x9 | 92.77 | H100, branch `elastic` all-models report (residual-transport) |
| 76 | all-suites v0 2/2 | 92.45 | H100, branch `elastic` all-models report (all-suites) |
| 77 | init variant ffn_magnitude | 92.12 | H100, branch `elastic` all-models report (init-variants) |
| 78 | elastic F0-PB S2 [4]+[1]x9 | 92.05 | H100, branch `elastic` all-models report (elastic) |
| 79 | **LoopWAM v0 partition 1/9/2 k3** | **91.78** | this evaluation (local RTX PRO 6000 Blackwell) |
| 80 | all-suites v0 1/4 | 91.68 | H100, branch `elastic` all-models report (all-suites) |
| 81 | init variant stepwise | 91.47 | H100, branch `elastic` all-models report (init-variants) |
| 82 | **LoopWAM v0 partition 4/3/5 k7** | **90.75** | this evaluation (local RTX PRO 6000 Blackwell) |
| 83 | **LoopWAM v0 partition 6/6/0** | **90.27** | this evaluation (local RTX PRO 6000 Blackwell) |
| 84 | **LoopWAM Dense-S12** | **89.98** | this evaluation (local RTX PRO 6000 Blackwell) |
| 85 | **LoopWAM v0 Ka3** | **89.47** | this evaluation (local RTX PRO 6000 Blackwell) |
| 86 | **LoopWAM v0 Ka6** | **89.20** | this evaluation (local RTX PRO 6000 Blackwell) |
| 87 | init variant no_video_pretraining | 85.65 | H100, branch `elastic` all-models report (init-variants) |
| 88 | init variant core_random | 82.20 | H100, branch `elastic` all-models report (init-variants) |

Bold rows are this evaluation. With 6,000 episodes the binomial 95% interval is about +/-0.6 points for a model near 95% success (+/-0.8 near 90%); episodes within a task/seed group share the seed, so treat it as a lower bound. Rows from other branches are copied from their reports and were not re-verified here.

## 2. This evaluation, per suite

Success % with the sample standard deviation over the three seeds; per-seed columns average all four suites.

| Model | Spatial | Object | Goal | Long | **All** | Seed 42 | Seed 43 | Seed 44 | What was evaluated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FasterWAM | 99.00 ± 0.40 | 99.60 ± 0.35 | 98.20 ± 0.40 | 98.07 ± 0.42 | **98.72** | 98.85 | 98.65 | 98.65 | public release, HF `hustvl/FasterWAM` `libero/step_021700.pt` (one model for all suites) |
| Light-WAM | 96.67 ± 0.50 | 98.87 ± 0.12 | 96.60 ± 0.40 | 90.93 ± 1.10 | **95.77** | 95.60 | 95.65 | 96.05 | public release, HF `l1ziang/lightwam-checkpoints` (one checkpoint per suite) |
| LoopWAM Dense-S30 | 97.27 ± 0.70 | 98.60 ± 0.60 | 95.87 ± 0.12 | 90.80 ± 1.04 | **95.63** | 95.90 | 96.00 | 95.00 | HF `libero-all-suites/dense-s30` |
| LoopWAM v0 Ka5 | 94.47 ± 0.46 | 98.87 ± 0.31 | 96.20 ± 0.60 | 88.87 ± 1.42 | **94.60** | 94.90 | 94.50 | 94.40 | same v0 export, test-time action depth 5 (core loops 1,1,2,3,4) |
| LoopWAM v0 partition 0/6/6 | 97.20 ± 0.35 | 98.73 ± 0.64 | 95.27 ± 0.31 | 86.47 ± 1.36 | **94.42** | 94.90 | 94.55 | 93.80 | HF `partition-grid-centered-20261026/p0c6s6_k4` |
| LoopWAM v0 partition 1/6/5 | 97.27 ± 0.46 | 98.13 ± 1.47 | 95.93 ± 0.12 | 83.33 ± 2.05 | **93.67** | 94.55 | 93.20 | 93.25 | HF `partition-grid-centered-20261026/p1c6s5_k4` |
| LoopWAM v0 partition 5/6/1 | 96.73 ± 0.70 | 97.40 ± 1.06 | 93.60 ± 0.69 | 84.93 ± 2.93 | **93.17** | 94.40 | 93.10 | 92.00 | HF `partition-grid-centered-20261026/p5c6s1_k4` |
| LoopWAM v0 init core16_21 | 96.20 ± 0.35 | 97.27 ± 0.83 | 94.60 ± 0.35 | 84.40 ± 3.02 | **93.12** | 94.15 | 92.85 | 92.35 | HF `libero-all-suites/init_variants/core16_21` |
| LoopWAM v0 partition 1/9/2 k3 | 97.53 ± 0.50 | 98.33 ± 0.46 | 93.47 ± 2.10 | 77.80 ± 2.51 | **91.78** | 92.10 | 92.35 | 90.90 | HF `partition-grid-centered-20261026/p1c9s2_k3` (3 loops) |
| LoopWAM v0 partition 4/3/5 k7 | 93.33 ± 1.79 | 97.87 ± 1.29 | 92.47 ± 1.50 | 79.33 ± 1.67 | **90.75** | 91.55 | 90.95 | 89.75 | HF `partition-grid-centered-20261026/p4c3s5_k7` (7 loops) |
| LoopWAM v0 partition 6/6/0 | 95.07 ± 1.29 | 92.67 ± 2.21 | 91.27 ± 2.44 | 82.07 ± 2.39 | **90.27** | 92.30 | 90.05 | 88.45 | HF `partition-grid-centered-20261026/p6c6s0_k4` |
| LoopWAM Dense-S12 | 94.80 ± 1.06 | 93.27 ± 5.02 | 92.13 ± 1.80 | 79.73 ± 1.47 | **89.98** | 91.65 | 90.65 | 87.65 | HF `libero-all-suites/dense-s12` |
| LoopWAM v0 Ka3 | 95.53 ± 0.90 | 92.60 ± 3.75 | 92.93 ± 1.17 | 76.80 ± 4.98 | **89.47** | 92.00 | 89.35 | 87.05 | HF `libero-all-suites/v0-video4-action4`, test-time action depth 3 (core loops 2,3,4) |
| LoopWAM v0 Ka6 | 88.13 ± 2.57 | 96.93 ± 2.39 | 93.73 ± 1.01 | 78.00 ± 2.95 | **89.20** | 87.95 | 90.80 | 88.85 | same v0 export, test-time action depth 6 (core loops 1,1,1,2,3,4) |

For reference, the campaign scored the same v0 export at action depth 4 (its trained depth, "LoopWAM full 4/4") at **94.92%** (96.93 / 98.73 / 96.13 / 87.87), and FastWAM at 96.63% (`evaluation_report/final`).

Per-task results, McNemar paired tests against the campaign's LoopWAM full 4/4 and every episode are in [`data/`](data).

## 3. Gaps these results fill

The `elastic` all-models latency report (copied in `../branch_reports/elastic/plans/performance/`) lists these models as "not evaluated"; their matched results are now:

| Model (as named there) | Matched success % (this evaluation) | Long % |
|---|---:|---:|
| all-suites v0 4/2 | not evaluated | |
| all-suites Dense-S12 | 89.98 | 79.73 |
| all-suites Dense-S30 | 95.63 | 90.80 |
| partition p0c6s6_k4 | 94.42 | 86.47 |
| init variant core16_21 | 93.12 | 84.40 |
| partition p1c6s5_k4 | 93.67 | 83.33 |
| partition p5c6s1_k4 | 93.17 | 84.93 |
| partition p6c6s0_k4 | 90.27 | 82.07 |
| partition p1c9s2_k3 | 91.78 | 77.80 |
| partition p4c3s5_k7 | 90.75 | 79.33 |

`core16_21` is also the one variant marked "not evaluated" in `LoopWAM_v0_Init_Variants_Results_20261028.md` (branch `v0-init-variants`).

## 4. LiLa-WAM

LiLa-WAM has no matched LIBERO result. Its only public weights are a RoboTwin 2.0 model (one head camera, dual-arm state and actions), which cannot run LIBERO; its RoboTwin latency is in `../branch_reports/elastic/plans/performance/Public_WAM_Latency_20261030.md`. A LiLa-WAM trained on our LIBERO data with the paper recipe was evaluated on branch `elastic` and then withdrawn there (commit `e88defe`, 2026-10-30: "Remove LiLa-WAM LIBERO training and evaluation results"), so it is not reported here.

## 5. Checkpoints and verification (this evaluation)

`matched_eval/verify_run.py` (in `harness/matched_eval_source.tar.gz`) re-read every run on 2026-10-30: the full grid of 6,000 episode files (each file name vs its contents, initial state = episode index, checkpoint sha256), the 120 atomically written task/seed group files against the episode files, the manifest protocol, every model build report (checkpoint sha256, loop schedule, NFE 10, CFG 1, 30 settling steps, replan 10, nothing left unloaded, strict loads for LoopWAM), and the summary totals. Outputs: [`verification/`](verification).

| Model | Checkpoint file (under the eval root) | sha256 | Action core schedule | Builds | Verified |
|---|---|---|---|---:|---|
| FasterWAM | `checkpoints/fasterwam_release/libero/step_021700.pt` | `01be4efca565d189…` | - | 7 | passed (6000 episodes) |
| Light-WAM | `checkpoints/lightwam-checkpoints/lightwam_libero_10/checkpoints/weights/step_080000.pt`<br>`checkpoints/lightwam-checkpoints/lightwam_libero_goal/checkpoints/weights/step_060000.pt`<br>`checkpoints/lightwam-checkpoints/lightwam_libero_object/checkpoints/weights/step_012500.pt`<br>`checkpoints/lightwam-checkpoints/lightwam_libero_spatial/checkpoints/weights/step_055000.pt` | `0c40ca8159536308…`<br>`421e954b0a1af5e9…`<br>`57a83d3041299681…`<br>`93c816ed1e751271…` | - | 15 | passed (6000 episodes) |
| LoopWAM Dense-S30 | `checkpoints/loopwam_hf/dense-s30/policy.pt` | `9726eec3f4aafa74…` | - | 4 | passed (6000 episodes) |
| LoopWAM Dense-S12 | `checkpoints/loopwam_hf/dense-s12/policy.pt` | `a94f1839457abecc…` | - | 5 | passed (6000 episodes) |
| LoopWAM v0 Ka3 | `checkpoints/loopwam_hf/v0-video4-action4/policy.pt` | `1c659a7c2f6970c3…` | 2,3,4 | 5 | passed (6000 episodes) |
| LoopWAM v0 Ka5 | `checkpoints/loopwam_hf/v0-video4-action4/policy.pt` | `1c659a7c2f6970c3…` | 1,1,2,3,4 | 3 | passed (6000 episodes) |
| LoopWAM v0 Ka6 | `checkpoints/loopwam_hf/v0-video4-action4/policy.pt` | `1c659a7c2f6970c3…` | 1,1,1,2,3,4 | 12 | passed (6000 episodes) |
| LoopWAM v0 partition 0/6/6 | `checkpoints/loopwam_hf/partition-grid-centered-20261026/p0c6s6_k4/policy.pt` | `9d6a754a894b0977…` | 1,2,3,4 | 6 | passed (6000 episodes) |
| LoopWAM v0 partition 1/6/5 | `checkpoints/loopwam_hf/partition-grid-centered-20261026/p1c6s5_k4/policy.pt` | `26992d6c0d2e788f…` | 1,2,3,4 | 3 | passed (6000 episodes) |
| LoopWAM v0 partition 5/6/1 | `checkpoints/loopwam_hf/partition-grid-centered-20261026/p5c6s1_k4/policy.pt` | `2fa44095e0ef4ead…` | 1,2,3,4 | 2 | passed (6000 episodes) |
| LoopWAM v0 partition 6/6/0 | `checkpoints/loopwam_hf/partition-grid-centered-20261026/p6c6s0_k4/policy.pt` | `2dabf8e01924836e…` | 1,2,3,4 | 3 | passed (6000 episodes) |
| LoopWAM v0 partition 1/9/2 k3 | `checkpoints/loopwam_hf/partition-grid-centered-20261026/p1c9s2_k3/policy.pt` | `4f0dc3ac79201763…` | 1,2,3 | 2 | passed (6000 episodes) |
| LoopWAM v0 partition 4/3/5 k7 | `checkpoints/loopwam_hf/partition-grid-centered-20261026/p4c3s5_k7/policy.pt` | `39f574b8c35b0b52…` | 1,2,3,4,5,6,7 | 10 | passed (6000 episodes) |
| LoopWAM v0 init core16_21 | `checkpoints/loopwam_hf/init_variants/core16_21/policy.pt` | `7810a43430d3bd28…` | 1,2,3,4 | 3 | passed (6000 episodes) |

## 6. Protocol (identical for every model)

- Suites `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`; tasks 0-9; seeds 42, 43, 44; official
  initial states 0-49.
- Per task/seed group: `set_global_seed(seed)`, one environment `get_libero_env(task, 256, seed)` (env.seed) reused
  for its 50 episodes in order; each episode `reset()` + `set_init_state(state[i])`.
- Light-WAM's unchanged `run_single_episode` (sha256 `c7f5c738…`) drives every model: 30 settling actions
  `[0,0,0,0,0,0,-1]` (done ignored), then at most 400 (Spatial/Object/Goal) or 700 (Long) policy steps, 32-action
  chunks with 10 executed per query, success = the environment `done` flag.
- Every action query receives the group's base seed (no per-episode or per-query reseeding).
- Policies run their own native inference: FasterWAM `one_pass_future_cache`, 10 steps, shift 5, BF16; Light-WAM its
  training-run config, 10 steps; LoopWAM the campaign's written rule with the native pieces (strict
  `create_loopwam`, FP32 policy, BF16 VAE, `infer_action` under BF16 autocast, 10 steps, CFG 1, action shift 1,
  gripper `sign(-(2g-1))`). The prompt is encoded once per task with each family's own encoder (cached actions
  verified bit-identical to online prompt encoding).
- Rendering `MUJOCO_GL=egl`. Groups are claimed from a shared queue and written atomically, so the number of workers
  is not part of the protocol, and an interrupted group always reruns from episode 0.

The full specification, the model-construction gates (init independence, strict loading, repeat identity) and every
decision taken during the runs are in [`harness/README_MATCHED_EVAL.md`](harness/README_MATCHED_EVAL.md).

## 7. Setup and provenance

- **Hardware:** one host with 2x NVIDIA RTX PRO 6000 Blackwell (97.9 GB), shared with other users' jobs; runs used
  1-3 workers per model on GPU 0 and/or 1. Eager native inference (no CUDA graph, no compile).
- **Software:** Python 3.10.21, torch 2.7.1+cu128, mujoco 3.3.2, robosuite 1.4.0, LIBERO `8f1084e`.
- **Model code:** Light-WAM `db7d05e`, FasterWAM `8366781`; LoopWAM from this repository: `chrono` @ `277c635` (v0,
  Dense), `partition_grid_20261026` @ `a584631` (partition exports), `v0-init-variants` @ `4ebf38a` (`core16_21`, its
  training revision). The eight training code files recorded in each partition/init export's manifest are
  byte-identical to the checkout it runs on.
- **Checkpoints:** downloaded from Hugging Face at pinned revisions and checked against their LFS sha256 (and every
  other file against its git blob id); logs in [`evidence/download_logs`](evidence/download_logs). Partition exports:
  `bd59888` (p0c6s6, p1c6s5), `906fcb1` (p5c6s1, p6c6s0), `ac02c3f` (p1c9s2_k3, p4c3s5_k7); `core16_21`: `9257a7e`.
- **Per-run records:** [`runs/<model>/`](runs) holds each run's manifest (protocol, source hashes, code revisions),
  summary and first build report per suite.

## 8. Agreement with the H100 campaign

The harness was validated on the released v0 4/4 export (`1c659a7c…`, campaign "LoopWAM full 4/4") before any new
model was scored:

- Gate on 40 campaign episodes: 40/40 outcomes agree, 39/40 trajectories identical
  (`evidence/gates/gate_loopwam_v0_full44_libero_spatial.json`).
- A 200-episode diagnostic run (task 2, seed 43, all four suites): 200/200 outcomes agree with the campaign's
  per-episode records (193/200 successes in both).
- The partition-code and init-variant-code controls reproduced the v0 gate (all successes identical); the only
  differences lie in episodes where the same code varies between reruns on this host (next section).

So the local results and the H100 results above are directly comparable at the success-rate level.

## 9. Caveats and run history

- **Episode-level nondeterminism.** Rerunning campaign group `libero_10:9:42` with identical code and weights on this
  host changed the action digest (sometimes the step count) of episodes among 3-9 of that group from run to run, while
  success never changed (`evidence/gates/chrono_rerun_variation.json`). Success rates are stable; per-episode trajectories are not
  guaranteed bit-identical across reruns.
- **Seed effect.** Seed 44 is the lowest of the three seeds for 11 of the 14 models here (Section 2); all six
  four-suite models with per-episode records in `../final/episodes.csv` decline from seed 42 to 43 to 44. Every query
  of a task/seed group uses the same base seed, so a seed effect is shared by all 50 episodes of a group.
- **Timing is not comparable across models.** The GPUs were shared with other users' jobs and up to 9 of our workers
  ran together, so the mean seconds per query in `data/` (0.12-0.63 s) reflect contention: the five v0 4/4-schedule
  exports (partition 0/6/6, 1/6/5, 5/6/1, 6/6/0 and `core16_21`) share one architecture yet range from 0.20 to
  0.38 s.
  Use `../branch_reports/elastic/plans/performance/LoopWAM_All_Models_Latency_20261030.md` for latency.
- **Interruptions never changed a completed result.** A worker hang (p5c6s1, 2026-10-29), two host reboots
  (2026-10-29 19:25 during p6c6s0, 2026-10-30 18:01 after all runs had finished), a GPU that needed a reset after the
  first reboot, and memory-pressure pauses / one host OOM kill of our own worker (p4c3s5_k7) only interrupted
  unfinished groups, which reran from episode 0 with the same seed; completed groups are written atomically and
  re-verified (Section 5). Details and timestamps: [`evidence/run_parallel_queue.log`](evidence/run_parallel_queue.log)
  and `harness/README_MATCHED_EVAL.md`.
- **Removed from the queue by request (2026-10-28, before they started):** `core_mean` and `no_video_pretraining`;
  both have H100 matched results on branch `v0-init-variants` (Section 1).

## 10. Files

| Path | Content |
|---|---|
| `data/summary_table.md` | Table written by the report script at the end of the queue (includes the campaign rows it compares with) |
| `data/suite_results.csv`, `data/task_results.csv` | Per suite (Wilson 95% interval, per seed) and per task |
| `data/paired_results.csv` | McNemar paired tests per suite against the campaign's LoopWAM full 4/4 |
| `data/episodes.csv`, `data/results.json` | Every scored episode (steps, queries, action digest, checkpoint sha256) and the full results |
| `verification/` | `verify_run.py` output per model (`.txt`) and as JSON |
| `runs/<model>/` | Manifest, summary and build reports per model |
| `evidence/` | Model-construction gates, checkpoint download logs, the queue log |
| `harness/` | Evaluation code (`matched_eval_source.tar.gz`) and its README |
| `build_readme.py`, `README_body.md` | Build this README from the files above and `../branch_reports` |

