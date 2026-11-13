# Elastic E0 teacher epoch10 / student 5ep — flat_k4_n2 — final full RoboTwin results

| Setting | Successes / episodes | SR |
| --- | ---: | ---: |
| demo_clean | 4567/5000 | 91.34% |
| demo_randomized | 4582/5000 | 91.64% |
| combined | 9149/10000 | 91.49% |

| Task | Clean / 100 | Randomized / 100 | Combined SR |
| --- | ---: | ---: | ---: |
| adjust_bottle | 100 | 98 | 99.00% |
| beat_block_hammer | 99 | 93 | 96.00% |
| blocks_ranking_rgb | 99 | 99 | 99.00% |
| blocks_ranking_size | 97 | 93 | 95.00% |
| click_alarmclock | 98 | 100 | 99.00% |
| click_bell | 100 | 100 | 100.00% |
| dump_bin_bigbin | 97 | 97 | 97.00% |
| grab_roller | 100 | 100 | 100.00% |
| handover_block | 94 | 94 | 94.00% |
| handover_mic | 100 | 100 | 100.00% |
| hanging_mug | 58 | 52 | 55.00% |
| lift_pot | 100 | 100 | 100.00% |
| move_can_pot | 90 | 98 | 94.00% |
| move_pillbottle_pad | 99 | 100 | 99.50% |
| move_playingcard_away | 100 | 100 | 100.00% |
| move_stapler_pad | 87 | 83 | 85.00% |
| open_laptop | 93 | 94 | 93.50% |
| open_microwave | 26 | 36 | 31.00% |
| pick_diverse_bottles | 85 | 86 | 85.50% |
| pick_dual_bottles | 99 | 98 | 98.50% |
| place_a2b_left | 98 | 97 | 97.50% |
| place_a2b_right | 97 | 95 | 96.00% |
| place_bread_basket | 92 | 90 | 91.00% |
| place_bread_skillet | 93 | 93 | 93.00% |
| place_burger_fries | 95 | 100 | 97.50% |
| place_can_basket | 65 | 66 | 65.50% |
| place_cans_plasticbox | 100 | 95 | 97.50% |
| place_container_plate | 98 | 97 | 97.50% |
| place_dual_shoes | 88 | 91 | 89.50% |
| place_empty_cup | 100 | 100 | 100.00% |
| place_fan | 95 | 96 | 95.50% |
| place_mouse_pad | 94 | 96 | 95.00% |
| place_object_basket | 82 | 81 | 81.50% |
| place_object_scale | 98 | 98 | 98.00% |
| place_object_stand | 92 | 96 | 94.00% |
| place_phone_stand | 98 | 98 | 98.00% |
| place_shoe | 95 | 99 | 97.00% |
| press_stapler | 97 | 98 | 97.50% |
| put_bottles_dustbin | 91 | 86 | 88.50% |
| put_object_cabinet | 90 | 95 | 92.50% |
| rotate_qrcode | 80 | 78 | 79.00% |
| scan_object | 81 | 87 | 84.00% |
| shake_bottle | 100 | 100 | 100.00% |
| shake_bottle_horizontally | 100 | 100 | 100.00% |
| stack_blocks_three | 96 | 95 | 95.50% |
| stack_blocks_two | 100 | 100 | 100.00% |
| stack_bowls_three | 82 | 78 | 80.00% |
| stack_bowls_two | 99 | 92 | 95.50% |
| stamp_seal | 94 | 95 | 94.50% |
| turn_switch | 56 | 69 | 62.50% |

Seed42; action depths[4, 4]; NFE2; four video loops; predict32/execute24. Smoke episodes excluded. All 10,000 videos, accepted episode records, scene seeds, instructions, action hashes and frozen runtime files verified.

Failed attempts: 0. Logged partial episodes excluded: 0.

See [campaign protocol](../report.md), [task results](task_results.csv), [episode records](episodes.csv), [audit](final_audit.json), and [manifest](manifest.json). Instruction-template generation is unseeded as in prior runs; these are not guaranteed identical paired rollouts across models.
