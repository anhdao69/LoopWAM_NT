# Elastic E0 teacher epoch10 / student 5ep — flat_k1_n2 — final full RoboTwin results

| Setting | Successes / episodes | SR |
| --- | ---: | ---: |
| demo_clean | 4476/5000 | 89.52% |
| demo_randomized | 4452/5000 | 89.04% |
| combined | 8928/10000 | 89.28% |

| Task | Clean / 100 | Randomized / 100 | Combined SR |
| --- | ---: | ---: | ---: |
| adjust_bottle | 100 | 100 | 100.00% |
| beat_block_hammer | 96 | 93 | 94.50% |
| blocks_ranking_rgb | 100 | 99 | 99.50% |
| blocks_ranking_size | 95 | 97 | 96.00% |
| click_alarmclock | 100 | 100 | 100.00% |
| click_bell | 100 | 100 | 100.00% |
| dump_bin_bigbin | 96 | 94 | 95.00% |
| grab_roller | 100 | 100 | 100.00% |
| handover_block | 90 | 83 | 86.50% |
| handover_mic | 98 | 100 | 99.00% |
| hanging_mug | 50 | 39 | 44.50% |
| lift_pot | 99 | 100 | 99.50% |
| move_can_pot | 88 | 91 | 89.50% |
| move_pillbottle_pad | 94 | 96 | 95.00% |
| move_playingcard_away | 99 | 99 | 99.00% |
| move_stapler_pad | 83 | 77 | 80.00% |
| open_laptop | 93 | 96 | 94.50% |
| open_microwave | 17 | 12 | 14.50% |
| pick_diverse_bottles | 75 | 78 | 76.50% |
| pick_dual_bottles | 93 | 92 | 92.50% |
| place_a2b_left | 94 | 96 | 95.00% |
| place_a2b_right | 94 | 91 | 92.50% |
| place_bread_basket | 88 | 93 | 90.50% |
| place_bread_skillet | 91 | 87 | 89.00% |
| place_burger_fries | 96 | 97 | 96.50% |
| place_can_basket | 72 | 57 | 64.50% |
| place_cans_plasticbox | 98 | 98 | 98.00% |
| place_container_plate | 96 | 98 | 97.00% |
| place_dual_shoes | 85 | 81 | 83.00% |
| place_empty_cup | 100 | 100 | 100.00% |
| place_fan | 95 | 94 | 94.50% |
| place_mouse_pad | 90 | 89 | 89.50% |
| place_object_basket | 87 | 77 | 82.00% |
| place_object_scale | 93 | 98 | 95.50% |
| place_object_stand | 93 | 95 | 94.00% |
| place_phone_stand | 95 | 97 | 96.00% |
| place_shoe | 93 | 95 | 94.00% |
| press_stapler | 98 | 98 | 98.00% |
| put_bottles_dustbin | 93 | 97 | 95.00% |
| put_object_cabinet | 90 | 90 | 90.00% |
| rotate_qrcode | 85 | 81 | 83.00% |
| scan_object | 81 | 77 | 79.00% |
| shake_bottle | 100 | 100 | 100.00% |
| shake_bottle_horizontally | 100 | 100 | 100.00% |
| stack_blocks_three | 96 | 94 | 95.00% |
| stack_blocks_two | 100 | 100 | 100.00% |
| stack_bowls_three | 79 | 78 | 78.50% |
| stack_bowls_two | 93 | 96 | 94.50% |
| stamp_seal | 86 | 90 | 88.00% |
| turn_switch | 49 | 62 | 55.50% |

Seed42; action depths[1, 1]; NFE2; four video loops; predict32/execute24. Smoke episodes excluded. All 10,000 videos, accepted episode records, scene seeds, instructions, action hashes and frozen runtime files verified.

Failed attempts: 0. Logged partial episodes excluded: 0.

See [campaign protocol](../report.md), [task results](task_results.csv), [episode records](episodes.csv), [audit](final_audit.json), and [manifest](manifest.json). Instruction-template generation is unseeded as in prior runs; these are not guaranteed identical paired rollouts across models.
