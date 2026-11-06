# LoopWAM v0 4/4 epoch 10 — full RoboTwin — final evaluation

Completed 2026-11-06T18:32:47.413490+07:00. All 100 task/setting pairs and 10,000 scored episodes passed artifact verification.

| Setting | Successes / episodes | SR |
| --- | ---: | ---: |
| demo_clean | 4603/5000 | 92.06% |
| demo_randomized | 4584/5000 | 91.68% |
| combined | 9187/10000 | 91.87% |

| Task | Clean successes / 100 | Randomized successes / 100 | Combined SR |
| --- | ---: | ---: | ---: |
| adjust_bottle | 100 | 99 | 99.50% |
| beat_block_hammer | 100 | 96 | 98.00% |
| blocks_ranking_rgb | 99 | 99 | 99.00% |
| blocks_ranking_size | 98 | 96 | 97.00% |
| click_alarmclock | 99 | 100 | 99.50% |
| click_bell | 100 | 100 | 100.00% |
| dump_bin_bigbin | 100 | 98 | 99.00% |
| grab_roller | 100 | 100 | 100.00% |
| handover_block | 94 | 95 | 94.50% |
| handover_mic | 100 | 100 | 100.00% |
| hanging_mug | 57 | 45 | 51.00% |
| lift_pot | 100 | 100 | 100.00% |
| move_can_pot | 93 | 99 | 96.00% |
| move_pillbottle_pad | 100 | 100 | 100.00% |
| move_playingcard_away | 100 | 100 | 100.00% |
| move_stapler_pad | 93 | 85 | 89.00% |
| open_laptop | 92 | 93 | 92.50% |
| open_microwave | 36 | 50 | 43.00% |
| pick_diverse_bottles | 86 | 85 | 85.50% |
| pick_dual_bottles | 100 | 96 | 98.00% |
| place_a2b_left | 97 | 94 | 95.50% |
| place_a2b_right | 96 | 92 | 94.00% |
| place_bread_basket | 93 | 94 | 93.50% |
| place_bread_skillet | 89 | 87 | 88.00% |
| place_burger_fries | 97 | 98 | 97.50% |
| place_can_basket | 61 | 57 | 59.00% |
| place_cans_plasticbox | 99 | 96 | 97.50% |
| place_container_plate | 96 | 97 | 96.50% |
| place_dual_shoes | 91 | 89 | 90.00% |
| place_empty_cup | 100 | 100 | 100.00% |
| place_fan | 96 | 93 | 94.50% |
| place_mouse_pad | 96 | 95 | 95.50% |
| place_object_basket | 86 | 82 | 84.00% |
| place_object_scale | 98 | 97 | 97.50% |
| place_object_stand | 92 | 97 | 94.50% |
| place_phone_stand | 99 | 99 | 99.00% |
| place_shoe | 96 | 98 | 97.00% |
| press_stapler | 97 | 98 | 97.50% |
| put_bottles_dustbin | 90 | 89 | 89.50% |
| put_object_cabinet | 91 | 87 | 89.00% |
| rotate_qrcode | 82 | 80 | 81.00% |
| scan_object | 85 | 88 | 86.50% |
| shake_bottle | 100 | 100 | 100.00% |
| shake_bottle_horizontally | 100 | 100 | 100.00% |
| stack_blocks_three | 97 | 96 | 96.50% |
| stack_blocks_two | 100 | 100 | 100.00% |
| stack_bowls_three | 79 | 83 | 81.00% |
| stack_bowls_two | 95 | 96 | 95.50% |
| stamp_seal | 98 | 96 | 97.00% |
| turn_switch | 60 | 70 | 65.00% |

Controller elapsed time: 16.52 hours, excluding download and preflight. This is a controller-segment duration; any later manual resumes must be accounted for separately.
Interrupted attempts: 0; logged partial episodes excluded from scoring: 0.

Protocol: seed 42; clean/randomized; 100 accepted episodes per task/setting; NFE10; predict32/execute24; unseen instructions; native v0 4/4, epoch 10, update 474620. All smoke episodes and the eight-input graph gate are excluded. Actual scene seeds and instructions are in episodes.csv; unseeded Python instruction-template generation prevents assuming exact episode pairing across independent runs.

Full 50-task checkpoint. See [setup](setup_report.md), [protocol comparison](protocol_comparison.json), [final audit](final_audit.json), [task CSV](task_results.csv), and [episode CSV](episodes.csv).


## Comparison with epoch 5

| Setting | Epoch 5 | Epoch 10 | Difference (percentage points) |
| --- | ---: | ---: | ---: |
| demo_clean | 91.60% | 92.06% | +0.46 |
| demo_randomized | 91.86% | 91.68% | -0.18 |
| combined | 91.73% | 91.87% | +0.14 |

Epoch-5 results are historical as audited then; epoch 10 additionally requires clean renderer/resource logs for every accepted cell. Unseeded instruction draws prevent claiming exact paired trajectories.