# Elastic E0 teacher epoch10 / student 5ep — flat_k2_n2 — final full RoboTwin results

| Setting | Successes / episodes | SR |
| --- | ---: | ---: |
| demo_clean | 4529/5000 | 90.58% |
| demo_randomized | 4531/5000 | 90.62% |
| combined | 9060/10000 | 90.60% |

| Task | Clean / 100 | Randomized / 100 | Combined SR |
| --- | ---: | ---: | ---: |
| adjust_bottle | 100 | 98 | 99.00% |
| beat_block_hammer | 97 | 93 | 95.00% |
| blocks_ranking_rgb | 100 | 100 | 100.00% |
| blocks_ranking_size | 94 | 95 | 94.50% |
| click_alarmclock | 100 | 100 | 100.00% |
| click_bell | 100 | 100 | 100.00% |
| dump_bin_bigbin | 99 | 94 | 96.50% |
| grab_roller | 100 | 100 | 100.00% |
| handover_block | 91 | 95 | 93.00% |
| handover_mic | 96 | 99 | 97.50% |
| hanging_mug | 51 | 40 | 45.50% |
| lift_pot | 100 | 100 | 100.00% |
| move_can_pot | 94 | 96 | 95.00% |
| move_pillbottle_pad | 97 | 100 | 98.50% |
| move_playingcard_away | 100 | 100 | 100.00% |
| move_stapler_pad | 90 | 80 | 85.00% |
| open_laptop | 91 | 97 | 94.00% |
| open_microwave | 17 | 15 | 16.00% |
| pick_diverse_bottles | 80 | 84 | 82.00% |
| pick_dual_bottles | 100 | 99 | 99.50% |
| place_a2b_left | 96 | 98 | 97.00% |
| place_a2b_right | 98 | 97 | 97.50% |
| place_bread_basket | 88 | 93 | 90.50% |
| place_bread_skillet | 91 | 89 | 90.00% |
| place_burger_fries | 96 | 97 | 96.50% |
| place_can_basket | 70 | 64 | 67.00% |
| place_cans_plasticbox | 100 | 96 | 98.00% |
| place_container_plate | 98 | 99 | 98.50% |
| place_dual_shoes | 84 | 86 | 85.00% |
| place_empty_cup | 100 | 100 | 100.00% |
| place_fan | 93 | 89 | 91.00% |
| place_mouse_pad | 95 | 94 | 94.50% |
| place_object_basket | 83 | 85 | 84.00% |
| place_object_scale | 98 | 96 | 97.00% |
| place_object_stand | 95 | 95 | 95.00% |
| place_phone_stand | 97 | 100 | 98.50% |
| place_shoe | 91 | 98 | 94.50% |
| press_stapler | 97 | 97 | 97.00% |
| put_bottles_dustbin | 92 | 94 | 93.00% |
| put_object_cabinet | 92 | 94 | 93.00% |
| rotate_qrcode | 85 | 81 | 83.00% |
| scan_object | 80 | 79 | 79.50% |
| shake_bottle | 100 | 100 | 100.00% |
| shake_bottle_horizontally | 100 | 100 | 100.00% |
| stack_blocks_three | 90 | 94 | 92.00% |
| stack_blocks_two | 100 | 100 | 100.00% |
| stack_bowls_three | 81 | 79 | 80.00% |
| stack_bowls_two | 95 | 94 | 94.50% |
| stamp_seal | 92 | 93 | 92.50% |
| turn_switch | 55 | 65 | 60.00% |

Seed42; action depths[2, 2]; NFE2; four video loops; predict32/execute24. Smoke episodes excluded. All 10,000 videos, accepted episode records, scene seeds, instructions, action hashes and frozen runtime files verified.

Failed attempts: 0. Logged partial episodes excluded: 0.

See [campaign protocol](../report.md), [task results](task_results.csv), [episode records](episodes.csv), [audit](final_audit.json), and [manifest](manifest.json). Instruction-template generation is unseeded as in prior runs; these are not guaranteed identical paired rollouts across models.
