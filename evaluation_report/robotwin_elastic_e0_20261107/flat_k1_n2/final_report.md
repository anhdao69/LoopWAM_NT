# Elastic E0 — flat_k1_n2 — final full RoboTwin results

| Setting | Successes / episodes | SR |
| --- | ---: | ---: |
| demo_clean | 4348/5000 | 86.96% |
| demo_randomized | 4336/5000 | 86.72% |
| combined | 8684/10000 | 86.84% |

| Task | Clean / 100 | Randomized / 100 | Combined SR |
| --- | ---: | ---: | ---: |
| adjust_bottle | 100 | 99 | 99.50% |
| beat_block_hammer | 92 | 91 | 91.50% |
| blocks_ranking_rgb | 100 | 98 | 99.00% |
| blocks_ranking_size | 90 | 94 | 92.00% |
| click_alarmclock | 100 | 100 | 100.00% |
| click_bell | 100 | 100 | 100.00% |
| dump_bin_bigbin | 91 | 91 | 91.00% |
| grab_roller | 100 | 100 | 100.00% |
| handover_block | 86 | 80 | 83.00% |
| handover_mic | 96 | 99 | 97.50% |
| hanging_mug | 54 | 51 | 52.50% |
| lift_pot | 99 | 98 | 98.50% |
| move_can_pot | 74 | 85 | 79.50% |
| move_pillbottle_pad | 87 | 90 | 88.50% |
| move_playingcard_away | 99 | 100 | 99.50% |
| move_stapler_pad | 77 | 68 | 72.50% |
| open_laptop | 93 | 97 | 95.00% |
| open_microwave | 12 | 7 | 9.50% |
| pick_diverse_bottles | 71 | 79 | 75.00% |
| pick_dual_bottles | 82 | 84 | 83.00% |
| place_a2b_left | 96 | 95 | 95.50% |
| place_a2b_right | 95 | 96 | 95.50% |
| place_bread_basket | 91 | 92 | 91.50% |
| place_bread_skillet | 86 | 88 | 87.00% |
| place_burger_fries | 97 | 94 | 95.50% |
| place_can_basket | 60 | 61 | 60.50% |
| place_cans_plasticbox | 99 | 96 | 97.50% |
| place_container_plate | 97 | 97 | 97.00% |
| place_dual_shoes | 70 | 67 | 68.50% |
| place_empty_cup | 100 | 98 | 99.00% |
| place_fan | 83 | 84 | 83.50% |
| place_mouse_pad | 87 | 78 | 82.50% |
| place_object_basket | 90 | 83 | 86.50% |
| place_object_scale | 95 | 94 | 94.50% |
| place_object_stand | 95 | 94 | 94.50% |
| place_phone_stand | 94 | 93 | 93.50% |
| place_shoe | 89 | 96 | 92.50% |
| press_stapler | 98 | 97 | 97.50% |
| put_bottles_dustbin | 88 | 92 | 90.00% |
| put_object_cabinet | 91 | 90 | 90.50% |
| rotate_qrcode | 79 | 79 | 79.00% |
| scan_object | 74 | 78 | 76.00% |
| shake_bottle | 100 | 100 | 100.00% |
| shake_bottle_horizontally | 100 | 99 | 99.50% |
| stack_blocks_three | 93 | 77 | 85.00% |
| stack_blocks_two | 100 | 100 | 100.00% |
| stack_bowls_three | 75 | 78 | 76.50% |
| stack_bowls_two | 92 | 94 | 93.00% |
| stamp_seal | 86 | 81 | 83.50% |
| turn_switch | 45 | 54 | 49.50% |

Seed42; action depths[1, 1]; NFE2; four video loops; predict32/execute24. Smoke episodes excluded. All 10,000 videos, accepted episode records, scene seeds, instructions, action hashes and frozen runtime files verified.

Failed attempts: 0. Logged partial episodes excluded: 0.

See [campaign protocol](../report.md), [task results](task_results.csv), [episode records](episodes.csv), [audit](final_audit.json), and [manifest](manifest.json). Instruction-template generation is unseeded as in prior runs; these are not guaranteed identical paired rollouts across models.
