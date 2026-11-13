# Elastic E0 — flat_k4_n5 — final full RoboTwin results

| Setting | Successes / episodes | SR |
| --- | ---: | ---: |
| demo_clean | 4563/5000 | 91.26% |
| demo_randomized | 4533/5000 | 90.66% |
| combined | 9096/10000 | 90.96% |

| Task | Clean / 100 | Randomized / 100 | Combined SR |
| --- | ---: | ---: | ---: |
| adjust_bottle | 100 | 99 | 99.50% |
| beat_block_hammer | 97 | 93 | 95.00% |
| blocks_ranking_rgb | 100 | 98 | 99.00% |
| blocks_ranking_size | 92 | 97 | 94.50% |
| click_alarmclock | 99 | 100 | 99.50% |
| click_bell | 100 | 100 | 100.00% |
| dump_bin_bigbin | 97 | 97 | 97.00% |
| grab_roller | 100 | 100 | 100.00% |
| handover_block | 93 | 96 | 94.50% |
| handover_mic | 98 | 100 | 99.00% |
| hanging_mug | 61 | 56 | 58.50% |
| lift_pot | 100 | 100 | 100.00% |
| move_can_pot | 91 | 99 | 95.00% |
| move_pillbottle_pad | 99 | 100 | 99.50% |
| move_playingcard_away | 100 | 100 | 100.00% |
| move_stapler_pad | 90 | 81 | 85.50% |
| open_laptop | 93 | 97 | 95.00% |
| open_microwave | 20 | 23 | 21.50% |
| pick_diverse_bottles | 88 | 86 | 87.00% |
| pick_dual_bottles | 98 | 99 | 98.50% |
| place_a2b_left | 92 | 90 | 91.00% |
| place_a2b_right | 99 | 93 | 96.00% |
| place_bread_basket | 87 | 86 | 86.50% |
| place_bread_skillet | 92 | 89 | 90.50% |
| place_burger_fries | 97 | 99 | 98.00% |
| place_can_basket | 73 | 52 | 62.50% |
| place_cans_plasticbox | 98 | 96 | 97.00% |
| place_container_plate | 98 | 98 | 98.00% |
| place_dual_shoes | 82 | 75 | 78.50% |
| place_empty_cup | 100 | 100 | 100.00% |
| place_fan | 96 | 94 | 95.00% |
| place_mouse_pad | 92 | 94 | 93.00% |
| place_object_basket | 82 | 78 | 80.00% |
| place_object_scale | 98 | 95 | 96.50% |
| place_object_stand | 93 | 95 | 94.00% |
| place_phone_stand | 100 | 98 | 99.00% |
| place_shoe | 94 | 97 | 95.50% |
| press_stapler | 95 | 98 | 96.50% |
| put_bottles_dustbin | 91 | 92 | 91.50% |
| put_object_cabinet | 91 | 94 | 92.50% |
| rotate_qrcode | 85 | 81 | 83.00% |
| scan_object | 85 | 86 | 85.50% |
| shake_bottle | 100 | 100 | 100.00% |
| shake_bottle_horizontally | 100 | 100 | 100.00% |
| stack_blocks_three | 97 | 96 | 96.50% |
| stack_blocks_two | 99 | 100 | 99.50% |
| stack_bowls_three | 82 | 81 | 81.50% |
| stack_bowls_two | 95 | 98 | 96.50% |
| stamp_seal | 95 | 94 | 94.50% |
| turn_switch | 59 | 63 | 61.00% |

Seed42; action depths[4, 4, 4, 4, 4]; NFE5; four video loops; predict32/execute24. Smoke episodes excluded. All 10,000 videos, accepted episode records, scene seeds, instructions, action hashes and frozen runtime files verified.

Failed attempts: 0. Logged partial episodes excluded: 0.

See [campaign protocol](../report.md), [task results](task_results.csv), [episode records](episodes.csv), [audit](final_audit.json), and [manifest](manifest.json). Instruction-template generation is unseeded as in prior runs; these are not guaranteed identical paired rollouts across models.
