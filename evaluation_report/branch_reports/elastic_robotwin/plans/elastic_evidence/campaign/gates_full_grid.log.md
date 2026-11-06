# CUDA-graph gates (graph vs eager actions on real simulator observations, epoch_10.pt)

| Gate | Run | Schedules | Bit-exact queries | Max abs diff | Checkpoint sha256 |
|---|---|---|---|---|---|
| E0 | E0 | flat_k1_n2, flat_k1_n10, flat_k4_n10 | 48/48 | 0.0 | 5569c47fa6c91e13 |
| E2 | E2 | flat_k1_n2, flat_k1_n10, flat_k4_n10 | 48/48 | 0.0 | 6d92ac13a36e8605 |
| E1 | E1 | flat_k1_n2, flat_k1_n10, flat_k4_n10 | 48/48 | 0.0 | be1cbe2cd1c9fa2b |
| SP11 | SP11 | flat_k1_n2 | 16/16 | 0.0 | ab974f357659d094 |
| E0__grid | E0 | flat_k2_n2, flat_k1_n5, flat_k4_n2, flat_k2_n5, flat_k4_n5, flat_k2_n10 | 96/96 | 0.0 | 5569c47fa6c91e13 |
| E1__grid | E1 | flat_k2_n2, flat_k1_n5, flat_k4_n2, flat_k2_n5, flat_k4_n5, flat_k2_n10 | 96/96 | 0.0 | be1cbe2cd1c9fa2b |
| E2__grid | E2 | flat_k2_n2, flat_k1_n5, flat_k4_n2, flat_k2_n5, flat_k4_n5, flat_k2_n10 | 96/96 | 0.0 | 6d92ac13a36e8605 |
