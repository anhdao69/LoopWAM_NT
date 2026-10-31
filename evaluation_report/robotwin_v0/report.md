# LoopWAM v0 RoboTwin — final evaluation report

**Complete: 10,000/10,000 episodes; 100/100 task/setting pairs.** Finished 2026-10-31 20:06:56 (Asia/Ho_Chi_Minh, UTC+07:00).
Final artifact audit passed at 2026-10-31 20:16:08. No pending pairs or active workers remain.

## Final success rates

| Setting | Successes | Failures | Episodes | Success rate |
| --- | ---: | ---: | ---: | ---: |
| Clean | 4,580 | 420 | 5,000 | **91.60%** |
| Randomized | 4,593 | 407 | 5,000 | **91.86%** |
| Combined | 9,173 | 827 | 10,000 | **91.73%** |

SR is successful accepted episodes divided by accepted scored episodes. Every task has 100 episodes per setting, so the per-setting micro-average equals the mean of the 50 task SRs. Combined SR uses all 10,000 episodes. Smoke tests, concurrency calibration, rejected candidate scenes and interrupted attempts are excluded.
This is one evaluation seed (42). Clean and randomized results differ by 0.26 percentage points; this alone does not establish a robustness improvement or statistical significance.

## Model, seed and protocol

| Item | Evaluated value |
| --- | --- |
| Model | Original stateless LoopWAM v0, video 4 / action 4; partition 3/6/3; no history or memory |
| Checkpoint | RoboTwin `v0-video4-action4`, epoch 5, optimizer step 237,310 |
| Parameters | 584,567,886 policy parameters; shared T5 and VAE are separate components |
| Tasks | All 50 RoboTwin tasks, clean and randomized |
| Episodes | 100 accepted episodes per task per setting; 10,000 total |
| Evaluation seed | 42; first candidate scene seed 4,300,000 |
| Action noise | CPU RNG with fixed seed 42 on every inference query |
| Instruction split | Unseen; actual strings saved per episode |
| Model inference | NFE 10; CFG 1; predict 32 actions, execute 24 before replanning |
| Robot state and control | 14D joint state and absolute 14D `qpos` actions; training z-score statistics |
| Precision | FP32 policy weights, BF16 inference/VAE; no quantization |
| Execution | CUDA graphs; shared T5 server; initial 8 workers, recovery 4 workers |
| Renderer | RT, 32 samples/pixel, depth 8; OIDN 2.3.3 on RTX PRO 6000 Blackwell Max-Q 96 GB |

Release: [pinned Hugging Face checkpoint](https://huggingface.co/anhdao69/LoopWAM_NT/tree/4544aebb8c72b41af043efd97373dd0803e80a76/robotwin/v0-video4-action4).
Policy SHA256: `38975220e9441303878d2dfb9956d57e6416c6a906c99c43d9bca015b092fc51`.
See [setup and comparability details](setup_report.md) for dependency versions, normalization/VAE hashes, source provenance and rendering adaptations.

## Task-by-task final results

All rows are complete. Each setting has 100 episodes, so its success count is also its SR in percent. Combined SR uses 200 episodes per task.

| Task | Clean successes / 100 | Randomized successes / 100 | Combined SR |
| --- | ---: | ---: | ---: |
| adjust_bottle | 100 | 99 | 99.50% |
| beat_block_hammer | 100 | 98 | 99.00% |
| blocks_ranking_rgb | 100 | 99 | 99.50% |
| blocks_ranking_size | 93 | 97 | 95.00% |
| click_alarmclock | 99 | 100 | 99.50% |
| click_bell | 100 | 100 | 100.00% |
| dump_bin_bigbin | 99 | 97 | 98.00% |
| grab_roller | 100 | 100 | 100.00% |
| handover_block | 97 | 94 | 95.50% |
| handover_mic | 100 | 100 | 100.00% |
| hanging_mug | 65 | 59 | 62.00% |
| lift_pot | 100 | 100 | 100.00% |
| move_can_pot | 88 | 98 | 93.00% |
| move_pillbottle_pad | 99 | 100 | 99.50% |
| move_playingcard_away | 100 | 100 | 100.00% |
| move_stapler_pad | 85 | 77 | 81.00% |
| open_laptop | 92 | 97 | 94.50% |
| open_microwave | 33 | 41 | 37.00% |
| pick_diverse_bottles | 87 | 83 | 85.00% |
| pick_dual_bottles | 97 | 96 | 96.50% |
| place_a2b_left | 94 | 96 | 95.00% |
| place_a2b_right | 96 | 89 | 92.50% |
| place_bread_basket | 93 | 97 | 95.00% |
| place_bread_skillet | 89 | 88 | 88.50% |
| place_burger_fries | 96 | 99 | 97.50% |
| place_can_basket | 67 | 59 | 63.00% |
| place_cans_plasticbox | 99 | 93 | 96.00% |
| place_container_plate | 97 | 99 | 98.00% |
| place_dual_shoes | 88 | 82 | 85.00% |
| place_empty_cup | 100 | 100 | 100.00% |
| place_fan | 97 | 95 | 96.00% |
| place_mouse_pad | 91 | 94 | 92.50% |
| place_object_basket | 78 | 77 | 77.50% |
| place_object_scale | 100 | 97 | 98.50% |
| place_object_stand | 91 | 97 | 94.00% |
| place_phone_stand | 99 | 98 | 98.50% |
| place_shoe | 94 | 96 | 95.00% |
| press_stapler | 95 | 97 | 96.00% |
| put_bottles_dustbin | 94 | 95 | 94.50% |
| put_object_cabinet | 86 | 94 | 90.00% |
| rotate_qrcode | 85 | 78 | 81.50% |
| scan_object | 89 | 88 | 88.50% |
| shake_bottle | 100 | 100 | 100.00% |
| shake_bottle_horizontally | 100 | 100 | 100.00% |
| stack_blocks_three | 95 | 98 | 96.50% |
| stack_blocks_two | 100 | 99 | 99.50% |
| stack_bowls_three | 87 | 86 | 86.50% |
| stack_bowls_two | 94 | 98 | 96.00% |
| stamp_seal | 94 | 98 | 96.00% |
| turn_switch | 58 | 71 | 64.50% |

## Timeline and measured performance

All timestamps below are Vietnam time (UTC+07:00).

| Event | Time / duration |
| --- | --- |
| Initial launch, 8 workers | 2026-10-30 22:19:28 |
| Initial run stopped after 95 complete pairs | 2026-10-31 18:30:08 |
| Initial run elapsed | 20h 10m 40s |
| Downtime before recovery launch | 0h 32m 50s |
| Recovery evaluator launch, 4 workers | 2026-10-31 19:02:58 |
| Final completion | 2026-10-31 20:06:56 |
| Recovery elapsed, including saved-episode revalidation | 1h 3m 58s |
| Total wall time, including downtime | 21h 47m 28s |

These are controller timestamps. Initial environment setup, downloads, smoke tests and calibration preceded the initial launch and are not included. Summing task durations would double-count concurrent execution.

The accepted attempts logged **105,068 inference queries**. Mean measured query time was **0.3794 s**, median **0.3758 s**, and p95 **0.5672 s**. The timer surrounds model inference plus CUDA synchronization; it excludes image packing, text encoding, action denormalization and simulation. These measurements mix 8-worker and 4-worker contention and include warmup. They are not isolated model latency or end-to-end control latency.

The separate graph gate measured native 0.1644 s versus CUDA graph 0.1300 s per query on eight captured observations with exact output equality. Short 16-episode concurrency trials measured 281.14 s (2 workers), 185.75 s (4), and 165.67 s (8). Those trials underestimated full-run memory pressure: eight workers repeatedly reached the GPU memory limit. Recovery at four workers completed without a new failed attempt. No claim of a globally optimal concurrency setting is made.

## Failures, retries and recovery

There were **77 interrupted attempts**: 16 with CUDA out-of-memory errors, 58 with rendering-buffer allocation errors, and 3 other exits. The three other exits were SIGTERM (-15) before any scored episode; this is consistent with the manager's no-progress timeout, but the failure records do not label the termination cause explicitly.
Interrupted attempts contain **2,923 logged episode completions**, all excluded from the scored 10,000. These can repeat scenes across retries and must not be treated as additional independent benchmark episodes.

The initial limit was three attempts per pair. After five pairs exhausted that limit, recovery reduced workers from eight to four and raised the total allowed attempts to five. Each missing pair completed on attempt 4; attempt 5 was unnecessary. All 95 already-completed result files remained byte-identical and passed revalidation.

| Accepted attempt | Completed pairs |
| --- | ---: |
| attempt_01 | 44 |
| attempt_02 | 40 |
| attempt_03 | 11 |
| attempt_04 | 5 |

| Recovery pair | Final successes / 100 |
| --- | ---: |
| move_can_pot/demo_randomized | 98 |
| pick_diverse_bottles/demo_randomized | 83 |
| pick_dual_bottles/demo_clean | 97 |
| place_bread_skillet/demo_randomized | 88 |
| place_burger_fries/demo_randomized | 99 |

[Recovery details](recovery_20261031.md) · [All interrupted attempts](failed_attempts.csv).

## Candidate filtering and comparability limits

The frozen FastWAM harness accepts scenes only after expert planning and an expert success check. It also catches exceptions during expert execution or rollout setup, advances the candidate seed, and continues until 100 accepted scored episodes are obtained. An exception during a policy rollout can instead terminate the attempt and trigger a whole-pair retry. These paths must be distinguished.
The accepted logs contain **357 caught setup/expert exception markers across 33 pairs**: 251 missing-target-pose assertions, 102 index errors and 4 NoneType-subscript errors. None of these accepted logs contains a CUDA OOM or `cannot create buffer` message. These skips follow the unchanged harness behavior; they are not policy episode failures. The candidate audit records seed ranges, skipped-candidate totals and exception counts per pair.
Image processing follows the frozen released FastWAM PIL uint8/BF16 adapter. It differs from the uploaded LoopWAM H100 evaluator's torchvision float resize. Native LoopWAM retains its trained true-token text mask; FastWAM uses a different padding-mask convention. Rendering uses RT and a Blackwell-compatible OIDN upgrade; it is not claimed pixel-identical to OIDN 2.0.1 or the H100 raster fallback.
The harness does not explicitly seed Python instruction-template selection. Seed 42 therefore does not guarantee identical instruction strings across independent reruns or retries. Actual instructions, accepted scene seeds and action-trace hashes are exported in episodes.csv. Use these records for any paired comparison; do not infer pairing from the seed label alone.
This run evaluates LoopWAM only. It does not supply a newly evaluated RoboTwin FastWAM baseline or a head-to-head significance test. Artifact integrity and protocol checks cannot prove every simulator behavior is error-free.

## Latest pull versus evaluated source

Main checkout `/storage/anhdh35/LoopWAM_NT` was fast-forwarded from `56e33285a82c6abe075565b3d60c26a5ba7a148f` to `43ef539b4744ebfec1bb7b3864904f3d48465903`. The fetched update changes 126 report files and no executable code. All pre-existing tracked local modifications were preserved, and the previous local RoboTwin reports were archived before editing.
The completed evaluation used `0a0c6e57c56db40d3ff244b6857621874cbc38c9` in `/storage/anhdh35/LoopWAM-robotwin-eval`, based on upstream `0f9e494`. That worktree remains frozen. Its upstream branch now points to `dc5e75839f0e0fe0dd1f2c288be7dbf45ce2c3e1`; the intervening changes are report-only. The latest pull is not retroactively assigned as the evaluation's code version.

## Final verification and evidence

The final audit checked the checkpoint SHA256, all **782 source/manifest-bound files**, all **21,132 runtime asset/dependency files**, all **10,000 video files** through ffprobe and SHA256, and all episode outcomes, unique per-pair scene seeds, instructions and query-trace digests. Every completed cell matches its saved result.json; summary.json and COMPLETE.json agree. The 95 pre-recovery result files remain unchanged.
Video verification checks readable stream metadata and file integrity; it is not a manual review of every frame. The original 26 preflight tests and four smoke episodes are archived; they are not newly rerun or counted as scored episodes during this report-only update.

| Artifact | Purpose |
| --- | --- |
| [final_audit.json](final_audit.json) / [log](final_audit.log) | Fresh verification results, counts, hashes and timings |
| [final_results.json](final_results.json) / [task_results.csv](task_results.csv) | Final aggregate and per-task/setting metrics, accepted attempts and durations |
| [episodes.csv](episodes.csv) | All 10,000 accepted episode records, instructions, scene seeds and video/trace hashes |
| [candidate_scene_audit.csv](candidate_scene_audit.csv) / [summary](candidate_scene_audit.json) | Accepted seed ranges and candidate-skip evidence |
| [COMPLETE.json](COMPLETE.json) | Copy of the evaluator completion artifact |
| [protocol.json](protocol.json) / [runtime_manifest.json](runtime_manifest.json) | Frozen evaluation and dependency contracts |
| [launch.json](launch.json) | Completed recovery controller state |
| [initial_run_final_state.json](initial_run_final_state.json) / [initial snapshot](initial_launch_snapshot.json) | Initial failed run and historical launch evidence |
| [setup_report.md](setup_report.md) / [recovery report](recovery_20261031.md) | Detailed setup and recovery decisions |
| [audit_final.py](audit_final.py) / [build_final_report.py](build_final_report.py) | Reproducible audit and report generation |

Completion artifact SHA256: `9d463a458a4f5c52adbfc6925cc48841c1e27c23fff6543d174dcf7c51216129`.
Protocol manifest SHA256: `56fa331d8eddd0838aa364559ee94df734ed6205e193e6c3419748c576c4e31d`.
Bulk simulator logs, commands and videos remain under `/data14t/anhdh35/loopwam_robotwin_v0_20261030/results`. CSV video paths refer to these local files.
