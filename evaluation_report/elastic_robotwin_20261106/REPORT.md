# RoboTwin-50 elastic distillation (E0/E1/E2): results so far

Generated 2026-11-06 20:42 UTC by `tools/make_report.py` from the result files under `/mnt/data/vmo-ai-task/anhdh35/elastic_robotwin`: every measurement below is read or
computed from those files (recipe constants such as betas and weight decay are quoted from the recipe). Job 5758 on worker-2
(8x H100). Nothing has been published to HF; code is in local commits only. Previous version: `REPORT.v1_timing.md`.

## 1. Status

- Queue `RT50-ep5-E0 -> E1 -> E2` (one RoboTwin-50 epoch each, GPUs 0-3): phase **training**, "RT50-ep5-E0: attempt 1/6 (fresh)".
- RT50-ep5-E0: update **18,323 / 47,462** (38.6%), started 2026-11-06 15:39 UTC, 1 launch(es), 0 resume(s).
- Speed: 0.993 s/update average since start (includes recovery saves and the update-2000 export), 0.994 s/update over the last 999 updates.
- Estimated finish (E1/E2 scaled from E0's live speed by their 4-GPU timing ratios 0.992 / 1.009; ~6 min between runs for startup and export verification):

| run | estimated hours | estimated finish |
|---|---|---|
| RT50-ep5-E0 | 13.1 | 2026-11-07 04:45 UTC (Fri Nov 06 11:45 PM EST) |
| RT50-ep5-E1 | 13.0 | 2026-11-07 17:50 UTC (Sat Nov 07 12:50 PM EST) |
| RT50-ep5-E2 | 13.2 | 2026-11-08 07:08 UTC (Sun Nov 08 02:08 AM EST) |

- No evaluation/inference of the distilled students has been run: there is no RoboTwin evaluation harness for elastic
  exports yet (the exports and their loader exist and are verified, see sections 3-4).

## 2. Teacher and recipe

Teacher = student initializer = RoboTwin-50 LoopWAM v0 4/4 **epoch 5**:

| | HF `anhdao69/LoopWAM_NT` `robotwin/v0-video4-action4/policy.pt` | file the trainer loads |
|---|---|---|
| sha256 | 38975220e9441303878d2dfb9956d57e6416c6a906c99c43d9bca015b092fc51 | 38975220e9441303878d2dfb9956d57e6416c6a906c99c43d9bca015b092fc51 |
| bytes | 2,338,751,333 | 2,338,751,333 |
| epoch / step | 5 / 237,310 (checkpoint_provenance.json) | step 237,310 |

Local path: `/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robotwin_full_v0_20261028/release_v2/production/checkpoints/epoch_05.pt`. Same data/normalization/VAE contract as the distillation data: data manifest `ac26ba644a10`, normalization `38ed9c8d4663`, VAE `38071ab59bd9` (the trainer refuses a teacher whose contract differs).

Recipe `RT50-ep5-E0` (E0 = T0, E1 = T1, E2 = T4 transport): the unchanged LIBERO/CL-0 E-recipe except the parent and two
RoboTwin adaptations, endpoint loss over the 24 executed actions (LIBERO 10) and no binarized gripper loss
(gripper_loss=False; RoboTwin grippers are continuous). Scope `pc` (whole student action expert trainable),
step conditioning, anchor, regime `fu`, schedules = FLAT9 with the balanced sampler, init `common_v1`.

| optimizer group | peak LR (from the logged LR at update 1 x warmup) | weight decay |
|---|---|---|
| step condition (+ T4 transport in E2) | 0.001 | 0 |
| student action expert | 3e-05 | 0.01 |

AdamW betas (0.9, 0.95), eps 1e-8, grad clip 1.0, global batch 128, world 4 x microbatch 32; warmup 1,424 updates (3%), cosine to 1% at update 47,462. Logged LR at update 18,323 = 0.0007058 / 2.117e-05, the schedule recomputed independently gives 0.0007058 / 2.117e-05: PASS.
Speed options bf16_frozen, teacher_graph, gc_freeze, no_fill_uninitialized (each bit-exact, section 3); code revision `2c78ff6f186c` (frozen clone `LoopWAM_rt50ep5_20261106`).

## 3. Pre-launch validation with the epoch-5 teacher (`ep5/prelaunch/`)

| check | result | detail |
|---|---|---|
| teacher_identity | PASS | sha 38975220e9441303..., step 237,310, VAE 38071ab59bd9 = teacher contract |
| data_identity | PASS | 6,075,103 windows, data manifest ac26ba644a10, normalization 38ed9c8d4663 = teacher contract |
| dataset_equals_robotwin_trainer_fields | PASS | 450 windows (91 with padding), mismatches 0 |
| parameter_counts | PASS | policy 584,567,886, action expert 105,285,134 |
| teacher_rollout_equals_deployed_infer_action | PASS | 4/4 windows bit-equal, max diff 0.0 |
| common_initialization_and_trainable_counts | PASS | RT50-ep5-E0 105,685,646, RT50-ep5-E1 105,685,646, RT50-ep5-E2 105,762,094 trainable |
| speed options bit-exact, RT50-ep5-E0 | PASS | 4 options x 8 paths, max grad diff 0.0 |
| speed options bit-exact, RT50-ep5-E1 | PASS | 4 options x 8 paths, max grad diff 0.0 |
| speed options bit-exact, RT50-ep5-E2 | PASS | 4 options x 8 paths, max grad diff 0.0 |
| resume 20 -> 30 == straight 30 (world 4, mb 32, E2) | PASS | metrics of all 30 updates; weights, optimizer, ema, rng, contract, update, epoch, next_micro, windows_seen, status, scheduler bit-identical; 352/352 tensors moved 20->30 |
| export round trip `step_00002000.pt` (trainer's export_epoch) | PASS | 15/15 checks; frozen: 339 tensors bit-equal; student moved 338/338; loader rollout == training model 27/27 (9 schedules x 3 windows); export 4.0 s |
| export round trip `epoch_00.pt` (trainer's export_epoch) | PASS | 14/14 checks; frozen: 339 tensors bit-equal; student moved 338/338; loader rollout == training model 27/27 (9 schedules x 3 windows); export 4.0 s |
| all-padding microbatch (last update of the epoch on rank 3) | PASS | flat_k1_n2: loss 0.0, grads zero+finite on 352 tensors; flat_k4_n10: loss 0.0, grads zero+finite on 342 tensors |

Peak job memory during validation 56 GiB of 960 GiB. A first validation attempt hung on a bare `wait` in the script (it also waited for the endless memory sampler) and was cancelled; its completed preflight results are kept in `ep5/prelaunch.attempt1_bare_wait_hang/`. The first resume comparison crashed on numpy RNG arrays in the checker; the fixed checker's result is the row above (`resume_compare_fixed.log`). The verifier names its epoch-type test export from the 30-update test run's 0-based epoch counter (`epoch_00.pt`); the trainer's end-of-epoch export is `export_epoch(..., epoch + 1)`, i.e. `epoch_01.pt`, which is the file the queue requires.

## 4. RT50-ep5-E0 training so far

- All 18,323 updates: finite losses and gradients PASS; grip loss always 0 PASS; every 9-update block a full FLAT9 permutation PASS; windows seen 2,345,344 = 128 x 18,323 PASS.
- Data wait: 0.643 s at update 1 (DataLoader workers starting), max 0.0011 s over updates 2-18,323 (median 0.00033 s); peak GPU memory 70.5 GB; compute 0.979 s/update (mean after warm-up).
- Last recovery save: update 18,200 (`latest.pt`, every 200 updates); status `running`.
- Update-2000 export `step_00002000.pt`: static verification PASS (14/14 checks; frozen: 339 tensors bit-equal; student moved 338/338). Update 2,001 was stamped 9.7 s after update 2,000 (save + export + recovery copy + one update; median gap between consecutive updates 0.92 s).

Loss trend (means over bins of 2,997 updates = 333 complete FLAT9 blocks, so every bin weights the nine schedules equally):

| updates | local | end | anchor | total | grad norm |
|---|---|---|---|---|---|
| 1-2,997 | 0.01058 | 0.002426 | 0.00383 | 0.01371 | 0.211 |
| 2,998-5,994 | 0.00211 | 0.000807 | 0.00251 | 0.00377 | 0.115 |
| 5,995-8,991 | 0.00170 | 0.000696 | 0.00242 | 0.00326 | 0.109 |
| 8,992-11,988 | 0.00149 | 0.000578 | 0.00235 | 0.00295 | 0.103 |
| 11,989-14,985 | 0.00134 | 0.000582 | 0.00229 | 0.00278 | 0.097 |
| 14,986-17,982 | 0.00121 | 0.000527 | 0.00223 | 0.00259 | 0.092 |

Per-schedule mean local loss, first vs latest 2,997-update bin:

| schedule | flat_k1_n2 | flat_k1_n5 | flat_k1_n10 | flat_k2_n2 | flat_k2_n5 | flat_k2_n10 | flat_k4_n2 | flat_k4_n5 | flat_k4_n10 |
|---|---|---|---|---|---|---|---|---|---|
| first bin | 0.01268 | 0.02253 | 0.03364 | 0.00341 | 0.00633 | 0.00975 | 0.00168 | 0.00213 | 0.00308 |
| latest bin | 0.00146 | 0.00205 | 0.00292 | 0.00065 | 0.00088 | 0.00128 | 0.00045 | 0.00051 | 0.00073 |

These are training losses only (distillation to the epoch-5 teacher); they say nothing yet about task success.

## 5. GR1 continuation running alongside (GPUs 4-7)

GR1 resumed at update 117,580 at 2026-11-06 13:49 UTC (`resume_audit_u117580.json`); its metrics have no timestamps, so each update's time is bracketed between the resume time plus the summed per-update seconds and the `timing.json` heartbeat (update 138,139 at 2026-11-06 20:42 UTC) minus them (bracket width at most 4 min).

| GR1 window | updates | s/update mean / median |
|---|---|---|
| alone: after warm-up, certainly before the first epoch-5 Slurm step (5758.76 at 2026-11-06 15:13 UTC) | 118,001-121,573 | 1.180 / 1.177 |
| first hour of RT50-ep5-E0 on GPUs 0-3 | 123,136-126,159 | 1.190 / 1.182 |
| latest 3,000 updates (E0 running) | 135,140-138,139 | 1.199 / 1.193 |

GR1's per-update time drifts up slowly and steadily over the whole continuation (it rose the same way while GR1 ran alone), with no step when E0 started: median 1.176 s over the last 1,000 updates certainly before 5758.76, 1.181 s over the first 1,000 certainly during E0.
GR1 is at update 138,139; the running attempt stops at the end of epoch 6 (update 141,096, `--max-updates` in `orchestrator.log`) for that epoch's evaluation; 235,160 is the 10-epoch total. GR1 step ids 5758.71, 5758.72 (started with the resume).

## 6. Earlier result: training-speed test with the epoch-10 teacher (`timing/`)

Same port with the epoch-10 teacher (sha 70cf5af7, = HF `robotwin/v0-video4-action4/epoch_10/policy.pt`), 108 updates per run, first 9 excluded as warm-up; hours include the ~2 s recovery save per 100 updates.

| layout | run | GPUs | mb | s/update compute (wall) | data wait max | peak GB | 1 RoboTwin epoch (47,462 updates) h | LIBERO-recipe update count (21,700) h | 10 RoboTwin epochs (474,620 updates) h |
|---|---|---|---|---|---|---|---|---|---|
| w4_mb32 | RT50-E0 | 0,1,2,3 | 32 | 0.975 (0.978) | 0.001 | 70.5 | 13.2 | 6.0 | 131.6 |
| w4_mb32 | RT50-E1 | 0,1,2,3 | 32 | 0.965 (0.970) | 0.000 | 70.5 | 13.0 | 6.0 | 130.3 |
| w4_mb32 | RT50-E2 | 0,1,2,3 | 32 | 0.985 (0.986) | 0.001 | 70.4 | 13.3 | 6.1 | 133.1 |
| w4x2_mb32 | RT50-E0 | 0,1,2,3 | 32 | 0.974 (0.976) | 0.001 | 70.5 | 13.1 | 6.0 | 131.1 |
| w4x2_mb32 | RT50-E1 | 4,5,6,7 | 32 | 0.970 (0.973) | 0.001 | 70.5 | 13.1 | 6.0 | 131.1 |
| w8_mb16 | RT50-E0 | 0,1,2,3,4,5,6,7 | 16 | 0.749 (0.754) | 0.001 | 37.0 | 10.2 | 4.7 | 102.3 |
| w8_mb16 | RT50-E1 | 0,1,2,3,4,5,6,7 | 16 | 0.760 (0.765) | 0.002 | 37.0 | 10.3 | 4.7 | 103.2 |
| w8_mb16 | RT50-E2 | 0,1,2,3,4,5,6,7 | 16 | 0.767 (0.771) | 0.001 | 37.0 | 10.5 | 4.8 | 104.6 |

- Determinism: E0 repeated on GPUs 0-3 PASS, E1 on GPUs 0-3 vs 4-7 PASS: all 108 updates bit-identical.
- 8 GPUs vs 4 GPUs, E0: same schedules/windows PASS; update-1 local 0.143525 vs 0.143525; max |diff| local 1.4e-04; loss values >1e-2 differing by >0.2%: 2; grad-norm relative difference median 3.8e-04, 90th percentile 1.9e-03, max 1.6e-02 (update 99: 0.6622 vs 0.6731); update-108 local 0.05412 vs 0.05415.
- 8 GPUs vs 4 GPUs, E1: same schedules/windows PASS; update-1 local 0.066118 vs 0.066118; max |diff| local 3.3e-04; loss values >1e-2 differing by >0.2%: 3; grad-norm relative difference median 5.2e-04, 90th percentile 2.5e-03, max 7.4e-03 (update 50: 0.8889 vs 0.8823); update-108 local 0.25981 vs 0.25978.
- 8 GPUs vs 4 GPUs, E2: same schedules/windows PASS; update-1 local 0.051817 vs 0.051817; max |diff| local 2.1e-04; loss values >1e-2 differing by >0.2%: 1; grad-norm relative difference median 4.8e-04, 90th percentile 2.2e-03, max 1.0e-01 (update 101: 0.3115 vs 0.3479); update-108 local 0.12528 vs 0.12549.
- The gaps are bf16 rounding of 16- vs 32-sample microbatches; neither layout is the reference and nothing drifts. The E2 maximum is one isolated update (101, flat_k2_n2): the updates before and after differ by 0.12% and 0.16% in grad norm, and its own losses by local 0.1%, end 1.4%, anchor 0.2%. The previous report version called all of these rounding-level; the full-run figures above are the precise statement.
- Epoch-10 preflight (`validation/`): teacher_identity PASS, data_identity PASS, dataset_equals_robotwin_trainer_fields PASS, parameter_counts PASS, teacher_rollout_equals_deployed_infer_action PASS, common_initialization_and_trainable_counts PASS; speed options E0 PASS, E1 PASS, E2 PASS.
- 4 -> 8 GPUs speeds one run up 1.30x (E0), 1.27x (E1), 1.28x (E2): 16-sample microbatches under-use an H100. Two concurrent 4-GPU runs: E0 0.976 vs 0.978 alone, E1 0.973 vs 0.970 alone.

## 7. What was built, problems found, limits

- Port of the elastic recipes to RoboTwin-50 (dataset reader equal to the RoboTwin trainer's fields, robot teacher, 14-D noise,
  8-GPU layout), RoboTwin export format `loopwam-s-robot-rt-v1` (teacher FP32 payload + trained student + elastic fields; its own
  format name so the original RoboTwin deploy code cannot load it as a plain policy) and its loader `load_robot_transport`.
- Problems found and fixed before any production training: the validation script's bare `wait` hang; the resume checker's numpy
  comparison; the queue's export-verification step pinned CPUs outside a small step's cpuset (now a 120-CPU step); an export gap
  on resume right after update 2000 (now re-exported).
- Unattended operation: auto-resume from `latest.pt` (up to 6 attempts per run), 30-min no-progress watchdog, export verification
  after each run, login-node keeper. Recovery state is saved every 200 updates.
- Not done yet: RoboTwin task evaluation of the distilled students (needs an evaluation harness for the new export format).
  Nothing uploaded to HF; git commits are local only (branch `elastic_robotwin`, frozen production clone at `2c78ff6f186c`).

## 8. Files

- Status: `/mnt/data/vmo-ai-task/anhdh35/elastic_robotwin/ep5/STATUS.json`, `/mnt/data/vmo-ai-task/anhdh35/elastic_robotwin/ep5/queue.log`, `/mnt/data/vmo-ai-task/anhdh35/elastic_robotwin/ep5/keeper.log`
- Runs: `/mnt/data/vmo-ai-task/anhdh35/elastic_robotwin/ep5/runs/RT50-ep5-E*/` (metrics.jsonl, config.json, latest.pt, step_00002000.pt, epoch_01.pt at the end)
- Validation: `/mnt/data/vmo-ai-task/anhdh35/elastic_robotwin/ep5/prelaunch/`, `/mnt/data/vmo-ai-task/anhdh35/elastic_robotwin/ep5/validation/`; epoch-10 timing: `/mnt/data/vmo-ai-task/anhdh35/elastic_robotwin/timing/`, `/mnt/data/vmo-ai-task/anhdh35/elastic_robotwin/validation/`
