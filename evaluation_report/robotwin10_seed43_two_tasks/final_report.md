# RoboTwin-10 two-task seed comparison

Both models completed 400 seed-43 episodes each; all scored artifacts passed verification. Seed-42 results are the matching subset of the prior full evaluation.

| Task | v0 seed42 | v0 seed43 | Dense-s30 seed42 | Dense-s30 seed43 |
| --- | ---: | ---: | ---: | ---: |
| open_microwave | 31.50% | 46.50% | 55.50% | 64.00% |
| move_stapler_pad | 68.00% | 62.00% | 80.00% | 73.00% |
| Combined | 49.75% | 54.25% | 67.75% | 68.50% |

All rates combine clean and randomized equally. The selected two-task results are not full-ten-task benchmark scores. The original instruction sampler is not explicitly Python-seeded; instructions and accepted scene seeds are logged, but exact instruction pairing is not assumed. The new seed also changes the fixed-per-query CPU action noise.

[v0 detailed report](v0_4x4/final_report.md) · [Dense-s30 detailed report](dense_s30/final_report.md) · [Per-setting comparison CSV](comparison.csv) · [Setup](setup_report.md)
