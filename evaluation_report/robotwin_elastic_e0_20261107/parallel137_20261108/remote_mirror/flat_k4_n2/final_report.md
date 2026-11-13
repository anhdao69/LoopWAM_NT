# Elastic E0 — flat_k4_n2 — final full RoboTwin results

| Setting | Successes / episodes | SR |
| --- | ---: | ---: |
| demo_clean | 4512/5000 | 90.24% |
| demo_randomized | 4513/5000 | 90.26% |
| combined | 9025/10000 | 90.25% |

| Task | Clean / 100 | Randomized / 100 | Combined SR |
| --- | ---: | ---: | ---: |
| adjust_bottle | 100 | 100 | 100.00% |
| beat_block_hammer | 98 | 94 | 96.00% |
| blocks_ranking_rgb | 100 | 98 | 99.00% |
| blocks_ranking_size | 93 | 90 | 91.50% |
| click_alarmclock | 99 | 100 | 99.50% |
| click_bell | 100 | 100 | 100.00% |
| dump_bin_bigbin | 97 | 92 | 94.50% |
| grab_roller | 100 | 100 | 100.00% |
| handover_block | 92 | 93 | 92.50% |
| handover_mic | 98 | 99 | 98.50% |
| hanging_mug | 59 | 62 | 60.50% |
| lift_pot | 100 | 99 | 99.50% |
| move_can_pot | 89 | 97 | 93.00% |
| move_pillbottle_pad | 96 | 97 | 96.50% |
| move_playingcard_away | 100 | 100 | 100.00% |
| move_stapler_pad | 82 | 79 | 80.50% |
| open_laptop | 95 | 98 | 96.50% |
| open_microwave | 9 | 19 | 14.00% |
| pick_diverse_bottles | 90 | 85 | 87.50% |
| pick_dual_bottles | 94 | 96 | 95.00% |
| place_a2b_left | 94 | 94 | 94.00% |
| place_a2b_right | 93 | 93 | 93.00% |
| place_bread_basket | 88 | 92 | 90.00% |
| place_bread_skillet | 92 | 89 | 90.50% |
| place_burger_fries | 97 | 98 | 97.50% |
| place_can_basket | 73 | 64 | 68.50% |
| place_cans_plasticbox | 99 | 94 | 96.50% |
| place_container_plate | 95 | 98 | 96.50% |
| place_dual_shoes | 86 | 75 | 80.50% |
| place_empty_cup | 100 | 100 | 100.00% |
| place_fan | 96 | 94 | 95.00% |
| place_mouse_pad | 82 | 94 | 88.00% |
| place_object_basket | 86 | 83 | 84.50% |
| place_object_scale | 94 | 94 | 94.00% |
| place_object_stand | 91 | 92 | 91.50% |
| place_phone_stand | 99 | 99 | 99.00% |
| place_shoe | 94 | 99 | 96.50% |
| press_stapler | 97 | 97 | 97.00% |
| put_bottles_dustbin | 92 | 95 | 93.50% |
| put_object_cabinet | 93 | 89 | 91.00% |
| rotate_qrcode | 83 | 75 | 79.00% |
| scan_object | 86 | 82 | 84.00% |
| shake_bottle | 100 | 100 | 100.00% |
| shake_bottle_horizontally | 100 | 100 | 100.00% |
| stack_blocks_three | 93 | 97 | 95.00% |
| stack_blocks_two | 100 | 100 | 100.00% |
| stack_bowls_three | 81 | 78 | 79.50% |
| stack_bowls_two | 96 | 96 | 96.00% |
| stamp_seal | 86 | 89 | 87.50% |
| turn_switch | 55 | 65 | 60.00% |

Seed42; action depths[4, 4]; NFE2; four video loops; predict32/execute24. Smoke episodes excluded. All 10,000 videos, accepted episode records, scene seeds, instructions, action hashes and frozen runtime files verified.

Failed attempts: 0. Logged partial episodes excluded: 0.

See [campaign protocol](../report.md), [task results](task_results.csv), [episode records](episodes.csv), [audit](final_audit.json), and [manifest](manifest.json). Instruction-template generation is unseeded as in prior runs; these are not guaranteed identical paired rollouts across models.
