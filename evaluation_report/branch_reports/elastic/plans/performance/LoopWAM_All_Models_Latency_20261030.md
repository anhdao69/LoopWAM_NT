# LoopWAM: batch-one inference latency of every model, and an exact faster inference path (2026-10-30)

Branch `elastic`. Harness: `scripts/latency/`. Data: `plans/performance/latency_20261030/` (per-job result files,
`latency_results.json`/`.csv`, the job list, and checkpoint-hash records).

## 1. Summary

- **Scope:** every LIBERO model checkpoint in the repository was timed under one protocol on NVIDIA H100 80GB HBM3. That covers:
  - the released FastWAM baseline;
  - the all-suites v0 loop variants, Dense-S12/S30, the KV variants, the partition grid and the initialization variants;
  - the LIBERO-Long models;
  - the Residual-Transport runs at their evaluated schedules;
  - the Elastic campaign checkpoints at every evaluated operating point;
  - the ChronoLoop runs.
- **Same inputs everywhere:** every model replays the same 500 recorded raw observations. Every query is timed from the raw simulator observation to the denormalized 32-action chunk on the host.
- **New `fast` backend:** it is exact, meaning every compared action chunk is bit-identical to eager inference, and it lowers latency and peak memory for every model. It combines three changes:
  - BF16-stored Linear weights;
  - one CUDA graph for the whole query, including the VAE encode;
  - GPU-resident text embeddings.
- **Success rates:** each latency row carries the model's reported LIBERO success rate, labeled with its evaluation protocol.
- **Checkpoint identity (sha256).** Three checks tie each timed file to its records:
  - every locally stored timed checkpoint that has an evaluation record is byte-identical to the scored checkpoint (Section 10);
  - every Hugging Face download that has an export record matches it;
  - the HF all-suites and Long checkpoints with matched results are the exact files those evaluations scored.

### Results in brief

- **The exact `fast` backend is the fastest bit-exact path for every model.**
  - It reproduced eager's raw action chunk bit for bit on all 7,950 compared queries.
  - It was faster than both eager and the campaign CUDA graph, by mean and by p95, and it lowered peak memory, on every one of the 96 model × configuration rows.
  - Fast vs eager: median 2.66× (range 1.97–3.18×).
  - Fast vs the campaign graph: median 1.16× (range 1.05–1.31×). Most of the speedup over eager had already come from the campaign graph (median 2.28× over eager). Fast adds a further 5–31% on top.
  - Peak memory falls by 0.7–1.2 GB for the 3.1–3.6 GB LoopWAM/RT/Elastic/ChronoLoop models, because Linear weights are stored once in BF16. Dense-S30 falls from 6.4 to 3.8 GB, and FastWAM from 27.2 to 15.3 GB.
- **Fastest operating points (fast backend, mean).**
  - The [1,1] Elastic checkpoints run at about 23–25 ms (40–43 Hz). SP11 scores 93.77% and E0 94.33%.
  - RT-A2 S7 [4,2] runs at 31.5 ms with 94.93%.
  - F0-PB [4,4] runs at 37.0 ms with 95.88%, the highest four-suite success of any model under 50 ms.
  - The released FastWAM baseline scores 96.63% at 107.5 ms.
  - The success-latency front is in Section 8. Neighbouring points on it often differ by less than the latency noise (about 10%, see below) and by less than the success-rate uncertainty (about ±0.6 pp per 6,000-episode condition). So read it as three tiers rather than a strict order:
    - about 25 ms at 93.8–94.3%;
    - about 28–37 ms at 94.6–95.9%;
    - about 91–108 ms at 96.0–96.6% (ChronoLoop a4 and FastWAM).
- **Inductor `torch.compile` is faster but not exact.** It is 1.2× (FastWAM) to 1.6× (v0 4/4) faster than `fast`, but 0 of 100 compared action chunks matched eager (max abs difference 0.009–0.025 in normalized action units). It was therefore not adopted. Using it would require re-scoring success rates.
- **Measurement quality.** The 4-GPU sweep agrees with the GPU-alone control within ±3%, in both directions (Section 9).
  - Graph-replay latencies alternate between two levels about 7–11% apart, for a reason not identified (Section 11).
  - Treat mean differences under about 10% between models of equal compute as noise.

## 2. Protocol

- **Hardware.** Interactive job 1080499: node evc102, 4× H100 80GB HBM3, Intel Xeon Gold 6548Y+, PyTorch 2.7.1+cu128.
  - Four queue workers ran in parallel, one per GPU.
  - Each worker was pinned (`taskset`) to two physical cores (four hardware threads), with `torch.set_num_threads(4)`.
  - Section 9 reports a control in which the same job ran with the other three GPUs idle.
- **Observations.** `scripts/latency/record_observations.py` recorded 500 raw LIBERO observations once:
  - task 0 of each of the four suites, 125 consecutive control steps per suite, driven by random end-effector motions (numpy seed 0);
  - file sha256 `5815c6fdc22d2dfe0c241f89c687a6f2e83804f9bd298332867258eb7160e0f0`;
  - every model replays the same file, and the harness refuses any other.
- **One query** is everything between a raw simulator observation and the host action array:
  1. Two 224×224 camera images (agent view and wrist, antialiased resize), concatenated to 224×448.
  2. Host-to-device copy and online VAE encode.
  3. Visual prefill and the action-denoising steps of the configuration.
  4. Device-to-host copy and denormalization.

  The query never includes simulator stepping, text encoding (the T5 cache is used, as in every evaluation) or model loading. Every query uses seed-42 noise, the matched evaluation's fixed noise.
- **Timing.** `torch.cuda.synchronize()` runs before the clock starts. The clock (`perf_counter`) stops when the host action array exists.
- **Backends.**
  - `eager`: each model's native inference, and the reference for exactness.
  - `graph`: the CUDA-graph backend the matched evaluations used:
    - RT and Elastic: `TransportInference(compile_action_infer=True)`;
    - LoopWAM and FastWAM: the evaluator's captured prefill + denoising, with the VAE eager.
  - `fast`: Section 3.
  - `compile`: the upstream `torch.compile` option (Inductor, reduce-overhead). Measured for FastWAM and all-suites v0 4/4 only.
- **Samples per configuration.**
  - fast: all 500 observations, after 20 warmup queries.
  - graph and compile: all 500 observations, after 20 warmup queries.
  - eager: every 10th observation (50), after 5 warmup queries.
  - ChronoLoop is stateful, so eager and fast each run all 500 observations in order, with episode memory reset at each suite boundary. There is no graph row because no campaign graph backend exists for it.
  - Eager runs on every 10th observation so that every model fits into the allocation's remaining time. The job ended at 04:56, and the next allocation was not expected before 11-01. Uncontended eager latency varies by about 1%, so 50 queries pin its mean well.
- **Exactness.** On every observation that eager was timed on, the raw action chunk (before denormalization) of each other backend is compared bit for bit with eager.
- **Multi-schedule checkpoints** (RT, Elastic) are measured one operating point at a time, and captured graphs are released between points. Section 7 reports the memory with every point's graph resident at once (plan Section 12.4).
- **Elastic loader.** Every Elastic and RT checkpoint is timed through the same student-only `TransportInference` loader that the evaluation used, never a training wrapper with a teacher (plan Section 12.4).

## 3. The faster path (`fast`)

None of the three changes alters any arithmetic.

1. **BF16-stored Linear weights.**
   - Every model already runs under `torch.autocast(bfloat16)` with FP32 weights.
   - Autocast's weight cache only covers leaf tensors that require grad. So during inference, every frozen FP32 Linear weight is cast to BF16 again on every call: per loop iteration, per denoising step.
   - A kernel profile of an Elastic `flat_k4_n10` query showed about 25,000 kernels, many of them these casts.
   - Storing the weights in BF16 once performs the same round-to-nearest-even conversion a single time, so the matmul inputs are identical.
   - Only `nn.Linear` modules are converted. Norms, modulation tables, embeddings, convolutions and anything that runs in FP32 stay as they are.
2. **One CUDA graph per query.** VAE encode, prefill and denoising are captured together (`WholeQueryGraph`) and fed through static input buffers. The campaign graph left the VAE eager.
3. **GPU-resident text embeddings.** The cached T5 context is kept on the GPU instead of being copied on every query.

Implementation: `scripts/latency/fast_graph.py` (`bf16_linear`, `WholeQueryGraph`) and the per-family adapters in `scripts/latency/families.py`.
- Adoption in an evaluator takes one call to `bf16_linear(policy)` after loading, plus a graph wrapper around the query.
- The per-model exactness counts below confirm that the action chunks do not change.

Inductor `torch.compile` was measured as an alternative and **not** adopted. Section 6 shows it is not bit-exact: its fused kernels round differently, so its actions are not those of the evaluated models.

## 4. Headline numbers

- Measured: 52 models, 96 model x configuration rows, on NVIDIA H100 80GB HBM3 (evc102).
- Bit-exact vs eager: fast 7950/7950, graph 4450/4450, Inductor compile 0/100 compared queries.
- Fast vs eager (mean latency): median 2.66x, range 1.97x-3.18x over 96 rows.
- Fast vs the campaign CUDA-graph backend: median 1.16x, range 1.05x-1.31x over 89 rows.
- Peak allocated memory, eager -> fast: median 3.54 -> 2.43 GB.

## 5. All models

SR: four-suite matched success % (6,000 episodes) unless tagged L1500 (matched LIBERO-Long only, 1,500 episodes) or L100 (older single-seed Long evaluation, 100 episodes). Latency in ms, batch 1, mean (p50 / p95 for fast). Speedups are ratios of means.

| Group | Model | Architecture | Config | SR % | Eager mean | Graph mean | Fast mean (p50 / p95) | Fast Hz | Fast vs eager | Fast vs graph | Exact graph / fast | Peak GB eager -> fast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| elastic | elastic E0 | RT policy | flat_k1_n2 [1,1] | 94.33 | 48.8 | 30.0 | 24.7 (24.7 / 24.8) | 40.44 | 1.97x | 1.21x | 50/50 / 50/50 | 3.54 -> 2.58 |
| elastic | elastic E0 | RT policy | flat_k1_n5 [1]x5 | 94.53 | 81.4 | 40.1 | 33.3 (33.5 / 33.6) | 30.00 | 2.44x | 1.20x | 50/50 / 50/50 | 3.54 -> 2.62 |
| elastic | elastic E0 | RT policy | flat_k1_n10 [1]x10 | 93.68 | 113.8 | 52.9 | 47.9 (47.8 / 47.9) | 20.90 | 2.38x | 1.11x | 50/50 / 50/50 | 3.54 -> 2.65 |
| elastic | elastic E0 | RT policy | flat_k2_n2 [2,2] | 94.53 | 55.9 | 32.2 | 27.6 (27.6 / 27.7) | 36.18 | 2.02x | 1.17x | 50/50 / 50/50 | 3.54 -> 2.68 |
| elastic | elastic E0 | RT policy | flat_k2_n5 [2]x5 | 94.85 | 92.3 | 47.8 | 39.1 (40.3 / 40.8) | 25.57 | 2.36x | 1.22x | 50/50 / 50/50 | 3.54 -> 2.72 |
| elastic | elastic E0 | RT policy | flat_k2_n10 [2]x10 | 93.83 | 151.6 | 70.3 | 59.9 (61.9 / 62.0) | 16.70 | 2.53x | 1.17x | 50/50 / 50/50 | 3.54 -> 2.75 |
| elastic | elastic E0 | RT policy | flat_k4_n2 [4,4] | 94.97 | 70.7 | 39.6 | 33.2 (33.2 / 33.3) | 30.09 | 2.13x | 1.19x | 50/50 / 50/50 | 3.54 -> 2.78 |
| elastic | elastic E0 | RT policy | flat_k4_n5 [4]x5 | 95.45 | 131.6 | 62.9 | 54.5 (54.3 / 54.4) | 18.34 | 2.41x | 1.15x | 50/50 / 50/50 | 3.54 -> 2.82 |
| elastic | elastic E0 | RT policy | flat_k4_n10 [4]x10 | 94.87 | 227.5 | 98.3 | 86.2 (89.7 / 90.0) | 11.61 | 2.64x | 1.14x | 50/50 / 50/50 | 3.54 -> 2.85 |
| elastic | elastic E1 | RT policy | flat_k1_n2 [1,1] | 93.73 | 48.8 | 30.0 | 24.8 (24.8 / 24.9) | 40.35 | 1.97x | 1.21x | 50/50 / 50/50 | 3.54 -> 2.58 |
| elastic | elastic E1 | RT policy | flat_k1_n5 [1]x5 | 94.30 | 83.5 | 39.8 | 33.4 (33.7 / 33.8) | 29.90 | 2.50x | 1.19x | 50/50 / 50/50 | 3.54 -> 2.62 |
| elastic | elastic E1 | RT policy | flat_k1_n10 [1]x10 | 93.48 | 114.8 | 52.6 | 47.7 (47.7 / 47.8) | 20.95 | 2.41x | 1.10x | 50/50 / 50/50 | 3.54 -> 2.65 |
| elastic | elastic E1 | RT policy | flat_k2_n2 [2,2] | 94.58 | 56.4 | 32.0 | 27.6 (27.6 / 27.7) | 36.18 | 2.04x | 1.16x | 50/50 / 50/50 | 3.54 -> 2.68 |
| elastic | elastic E1 | RT policy | flat_k2_n5 [2]x5 | 94.52 | 92.7 | 47.5 | 39.1 (40.3 / 40.8) | 25.60 | 2.37x | 1.22x | 50/50 / 50/50 | 3.54 -> 2.72 |
| elastic | elastic E1 | RT policy | flat_k2_n10 [2]x10 | 94.00 | 153.1 | 70.6 | 59.7 (61.7 / 61.9) | 16.76 | 2.57x | 1.18x | 50/50 / 50/50 | 3.54 -> 2.75 |
| elastic | elastic E1 | RT policy | flat_k4_n2 [4,4] | 95.22 | 71.4 | 39.6 | 33.4 (33.4 / 33.5) | 29.95 | 2.14x | 1.19x | 50/50 / 50/50 | 3.54 -> 2.78 |
| elastic | elastic E1 | RT policy | flat_k4_n5 [4]x5 | 95.65 | 130.1 | 62.9 | 54.3 (54.2 / 54.4) | 18.42 | 2.40x | 1.16x | 50/50 / 50/50 | 3.54 -> 2.82 |
| elastic | elastic E1 | RT policy | flat_k4_n10 [4]x10 | 94.45 | 230.9 | 98.3 | 86.3 (90.1 / 90.2) | 11.59 | 2.68x | 1.14x | 50/50 / 50/50 | 3.54 -> 2.85 |
| elastic | elastic E2 | RT policy | flat_k1_n2 [1,1] | 94.07 | 49.0 | 29.8 | 24.8 (24.8 / 24.9) | 40.28 | 1.98x | 1.20x | 50/50 / 50/50 | 3.54 -> 2.58 |
| elastic | elastic E2 | RT policy | flat_k1_n5 [1]x5 | 94.25 | 84.3 | 40.4 | 33.6 (33.9 / 34.0) | 29.77 | 2.51x | 1.20x | 50/50 / 50/50 | 3.54 -> 2.62 |
| elastic | elastic E2 | RT policy | flat_k1_n10 [1]x10 | 94.00 | 118.3 | 53.0 | 48.4 (48.4 / 48.5) | 20.64 | 2.44x | 1.09x | 50/50 / 50/50 | 3.54 -> 2.65 |
| elastic | elastic E2 | RT policy | flat_k2_n2 [2,2] | 94.75 | 56.9 | 32.2 | 27.7 (27.6 / 27.8) | 36.07 | 2.05x | 1.16x | 50/50 / 50/50 | 3.54 -> 2.68 |
| elastic | elastic E2 | RT policy | flat_k2_n5 [2]x5 | 95.07 | 94.5 | 48.1 | 39.4 (40.8 / 41.1) | 25.36 | 2.40x | 1.22x | 50/50 / 50/50 | 3.54 -> 2.72 |
| elastic | elastic E2 | RT policy | flat_k2_n10 [2]x10 | 94.07 | 156.0 | 70.9 | 60.7 (62.4 / 62.6) | 16.49 | 2.57x | 1.17x | 50/50 / 50/50 | 3.54 -> 2.75 |
| elastic | elastic E2 | RT policy | flat_k4_n2 [4,4] | 95.27 | 71.1 | 39.5 | 33.4 (33.3 / 33.4) | 29.96 | 2.13x | 1.18x | 50/50 / 50/50 | 3.54 -> 2.78 |
| elastic | elastic E2 | RT policy | flat_k4_n5 [4]x5 | 95.28 | 130.5 | 62.8 | 54.2 (54.2 / 54.3) | 18.44 | 2.41x | 1.16x | 50/50 / 50/50 | 3.54 -> 2.82 |
| elastic | elastic E2 | RT policy | flat_k4_n10 [4]x10 | 94.70 | 233.1 | 98.0 | 86.0 (89.6 / 89.8) | 11.63 | 2.71x | 1.14x | 50/50 / 50/50 | 3.54 -> 2.85 |
| elastic | elastic F0-PB | RT policy | S8 [4,1] | 94.50 | 72.8 | 38.1 | 31.8 (31.8 / 31.9) | 31.47 | 2.29x | 1.20x | 50/50 / 50/50 | 3.55 -> 2.58 |
| elastic | elastic F0-PB | RT policy | S7 [4,2] | 94.70 | 85.4 | 37.7 | 31.3 (30.9 / 33.8) | 31.96 | 2.73x | 1.20x | 50/50 / 50/50 | 3.55 -> 2.62 |
| elastic | elastic F0-PB | RT policy | parent2 [4,4] | 95.88 | 90.3 | 43.6 | 37.0 (37.5 / 37.6) | 27.05 | 2.44x | 1.18x | 50/50 / 50/50 | 3.55 -> 2.65 |
| elastic | elastic F0-PB | RT policy | S6 [4,1,1,1,1] | 93.75 | 103.6 | 49.5 | 42.1 (42.1 / 42.2) | 23.78 | 2.46x | 1.18x | 50/50 / 50/50 | 3.55 -> 2.69 |
| elastic | elastic F0-PB | RT policy | S5 [4,2,2,2,2] | 94.25 | 128.2 | 57.8 | 45.1 (44.4 / 49.7) | 22.19 | 2.84x | 1.28x | 50/50 / 50/50 | 3.55 -> 2.72 |
| elastic | elastic F0-PB | RT policy | S2 [4]+[1]x9 | 92.05 | 155.3 | 67.4 | 54.5 (52.8 / 58.7) | 18.35 | 2.85x | 1.24x | 50/50 / 50/50 | 3.55 -> 2.75 |
| elastic | elastic F0-PB | RT policy | parent5 [4]x5 | 95.75 | 178.5 | 72.3 | 59.7 (57.8 / 64.5) | 16.75 | 2.99x | 1.21x | 50/50 / 50/50 | 3.55 -> 2.79 |
| elastic | elastic F0-PB | RT policy | S1 [4]+[2]x9 | 93.38 | 211.9 | 81.8 | 69.2 (67.7 / 75.9) | 14.46 | 3.06x | 1.18x | 50/50 / 50/50 | 3.55 -> 2.82 |
| elastic | elastic F0-PB | RT policy | parent10 [4]x10 | 94.68 | 321.3 | 120.7 | 103.8 (98.0 / 110.4) | 9.63 | 3.09x | 1.16x | 50/50 / 50/50 | 3.55 -> 2.85 |
| baseline | FastWAM (released, libero_uncond_2cam224) | FastWAM | native | 96.63 | 264.2 | 131.8 | 107.5 (108.5 / 108.9) | 9.30 | 2.46x | 1.23x | 50/50 / 50/50 | 27.18 -> 15.25 |
| residual-transport | RT-A2 | RT policy | S1 [4]+[2]x9 | 94.10 | 217.2 | 82.1 | 76.4 (76.4 / 76.6) | 13.08 | 2.84x | 1.07x | 50/50 / 50/50 | 3.55 -> 2.48 |
| residual-transport | RT-A2 | RT policy | S2 [4]+[1]x9 | 93.22 | 166.2 | 68.0 | 59.2 (59.2 / 59.3) | 16.88 | 2.81x | 1.15x | 50/50 / 50/50 | 3.55 -> 2.52 |
| residual-transport | RT-A2 | RT policy | S5 [4,2,2,2,2] | 94.98 | 136.5 | 56.6 | 47.7 (49.5 / 49.8) | 20.95 | 2.86x | 1.19x | 50/50 / 50/50 | 3.55 -> 2.55 |
| residual-transport | RT-A2 | RT policy | S6 [4,1,1,1,1] | 94.48 | 110.6 | 48.6 | 42.0 (42.0 / 42.2) | 23.82 | 2.63x | 1.16x | 50/50 / 50/50 | 3.55 -> 2.59 |
| residual-transport | RT-A2 | RT policy | S7 [4,2] | 94.93 | 80.1 | 40.6 | 31.5 (31.0 / 33.9) | 31.78 | 2.54x | 1.29x | 50/50 / 50/50 | 3.55 -> 2.62 |
| residual-transport | RT-A2 | RT policy | S8 [4,1] | 94.53 | 74.9 | 38.5 | 31.9 (31.9 / 32.0) | 31.36 | 2.35x | 1.21x | 50/50 / 50/50 | 3.55 -> 2.65 |
| residual-transport | RT-A | RT policy | S1 [4]+[2]x9 | 94.47 | 213.7 | 82.0 | 76.0 (75.9 / 76.1) | 13.16 | 2.81x | 1.08x | 50/50 / 50/50 | 3.55 -> 2.41 |
| residual-transport | RT-A | RT policy | S2 [4]+[1]x9 | 94.12 | 163.0 | 67.8 | 58.8 (58.8 / 59.0) | 17.00 | 2.77x | 1.15x | 50/50 / 50/50 | 3.55 -> 2.45 |
| residual-transport | RT-A | RT policy | S3 [4,3,2,2,2,1,1,1,1,1] | 94.05 | 196.4 | 76.3 | 68.3 (68.3 / 68.5) | 14.64 | 2.87x | 1.12x | 50/50 / 50/50 | 3.55 -> 2.48 |
| residual-transport | RT-A | RT policy | S4 [4,4,1,1,1,1,1,1,1,1] | 93.87 | 179.8 | 69.9 | 57.8 (57.8 / 58.1) | 17.29 | 3.11x | 1.21x | 50/50 / 50/50 | 3.55 -> 2.52 |
| all-suites | all-suites v0 4/4 | v0 v4/a4 | native | 94.92 | 266.7 | 104.2 | 99.3 (99.3 / 99.6) | 10.07 | 2.68x | 1.05x | 50/50 / 50/50 | 3.12 -> 2.09 |
| residual-transport | RT-B4 | RT policy | S1 [4]+[2]x9 | 94.32 | 212.6 | 87.2 | 68.9 (67.6 / 75.7) | 14.51 | 3.09x | 1.27x | 50/50 / 50/50 | 3.55 -> 2.34 |
| residual-transport | RT-B4 | RT policy | S2 [4]+[1]x9 | 92.77 | 158.4 | 69.9 | 56.4 (58.5 / 58.8) | 17.72 | 2.81x | 1.24x | 50/50 / 50/50 | 3.55 -> 2.38 |
| residual-transport | RT-Pa | RT policy | S1 [4]+[2]x9 | 93.95 | 166.3 | 74.4 | 61.5 (58.7 / 65.4) | 16.26 | 2.70x | 1.21x | 50/50 / 50/50 | 3.54 -> 2.34 |
| residual-transport | RT-Pa | RT policy | S2 [4]+[1]x9 | 92.92 | 133.4 | 61.8 | 52.5 (52.5 / 52.7) | 19.04 | 2.54x | 1.18x | 50/50 / 50/50 | 3.54 -> 2.38 |
| all-suites | all-suites v0 4/2 | v0 v4/a2 | native | not evaluated | 166.3 | 75.0 | 67.2 (67.1 / 67.3) | 14.89 | 2.48x | 1.12x | 50/50 / 50/50 | 3.12 -> 2.09 |
| all-suites | all-suites v0 4/1 | v0 v4/a1 | native | 94.00 | 123.3 | 60.9 | 48.3 (46.3 / 51.2) | 20.72 | 2.56x | 1.26x | 50/50 / 50/50 | 3.12 -> 2.09 |
| all-suites | all-suites v0 2/2 | v0 v2/a2 | native | 92.45 | 155.6 | 66.2 | 58.3 (56.6 / 63.0) | 17.16 | 2.67x | 1.14x | 50/50 / 50/50 | 3.12 -> 2.09 |
| all-suites | all-suites v0 1/4 | v0 v1/a4 | native | 91.68 | 235.7 | 98.8 | 86.4 (82.8 / 92.6) | 11.57 | 2.73x | 1.14x | 50/50 / 50/50 | 3.12 -> 2.09 |
| all-suites | all-suites Dense-S12 | dense_s12 v1/a1 | native | not evaluated | 111.6 | 47.8 | 40.8 (40.4 / 44.7) | 24.50 | 2.73x | 1.17x | 50/50 / 50/50 | 3.12 -> 2.09 |
| all-suites | all-suites Dense-S30 | dense_s30 v1/a1 | native | not evaluated | 253.6 | 107.7 | 90.5 (88.8 / 99.4) | 11.05 | 2.80x | 1.19x | 50/50 / 50/50 | 6.44 -> 3.79 |
| all-suites | all-suites v0 4/1 kv-mix | v0 v4/a1 kv-mix | native | 94.45 | 131.7 | 59.8 | 50.9 (50.2 / 54.9) | 19.66 | 2.59x | 1.18x | 50/50 / 50/50 | 3.12 -> 2.09 |
| partition-grid | partition p0c6s6_k4 | v0 v4/a4 | native | not evaluated | 255.2 | 104.3 | 91.4 (89.1 / 99.7) | 10.94 | 2.79x | 1.14x | 50/50 / 50/50 | 3.12 -> 2.09 |
| libero-long | long v1 4/4 | v1 v4/a4 | native | 82.00 L100 | 247.6 | 103.6 | 94.5 (89.3 / 99.6) | 10.59 | 2.62x | 1.10x | 50/50 / 50/50 | 3.12 -> 2.09 |
| libero-long | long v2 4/4 | v2 v4/a4 | native | 81.00 L100 | 272.9 | 111.6 | 98.5 (94.2 / 105.4) | 10.15 | 2.77x | 1.13x | 50/50 / 50/50 | 3.12 -> 2.09 |
| libero-long | long v0 4/1 kv-concat | v0 v4/a1 kv-concat | native | 74.53 L1500 | 125.2 | 64.3 | 54.0 (55.9 / 56.2) | 18.52 | 2.32x | 1.19x | 50/50 / 50/50 | 3.12 -> 2.09 |
| residual-transport | RT-B2a | RT policy | parent2 [4,4] | 94.57 | 71.6 | 36.8 | 33.8 (33.3 / 33.6) | 29.55 | 2.12x | 1.09x | 50/50 / 50/50 | 3.54 -> 2.31 |
| elastic | elastic SP11 | RT policy | flat_k1_n2 [1,1] | 93.77 | 49.1 | 30.6 | 23.4 (23.2 / 24.9) | 42.76 | 2.10x | 1.31x | 50/50 / 50/50 | 3.54 -> 2.31 |
| chronoloop | ChronoLoop CL-0_no-memory_a4 | ChronoLoop | native | 96.10 | 253.4 | n/a | 91.8 (89.0 / 99.6) | 10.89 | 2.76x |  | n/a / 500/500 | 3.12 -> 2.05 |
| chronoloop | ChronoLoop CL-A_mem16-learned-loopwrite_a4 | ChronoLoop | native | 96.18 | 283.8 | n/a | 92.1 (88.7 / 99.6) | 10.86 | 3.08x |  | n/a / 500/500 | 3.12 -> 2.05 |
| residual-transport | RT-T1ft | RT policy | S1 [4]+[2]x9 | 94.57 | 232.8 | 83.4 | 73.2 (75.4 / 75.6) | 13.66 | 3.18x | 1.14x | 50/50 / 50/50 | 3.55 -> 2.34 |
| residual-transport | RT-T1ft | RT policy | S2 [4]+[1]x9 | 94.18 | 154.2 | 62.6 | 56.1 (58.1 / 58.3) | 17.82 | 2.75x | 1.11x | 50/50 / 50/50 | 3.55 -> 2.38 |
| residual-transport | RT-TF | RT policy | S1 [4]+[2]x9 | 94.42 | 212.7 | 83.6 | 72.4 (76.0 / 76.2) | 13.82 | 2.94x | 1.16x | 50/50 / 50/50 | 3.55 -> 2.34 |
| residual-transport | RT-TF | RT policy | S2 [4]+[1]x9 | 94.13 | 158.3 | 68.7 | 54.1 (53.0 / 59.1) | 18.47 | 2.92x | 1.27x | 50/50 / 50/50 | 3.55 -> 2.38 |
| init-variants | init variant core16_21 | v0 v4/a4 | native | not evaluated | 251.5 | 105.5 | 91.2 (89.1 / 99.6) | 10.96 | 2.76x | 1.16x | 50/50 / 50/50 | 3.12 -> 2.09 |
| init-variants | init variant core_mean | v0 v4/a4 | native | 92.82 | 248.8 | 104.9 | 91.5 (89.0 / 99.3) | 10.93 | 2.72x | 1.15x | 50/50 / 50/50 | 3.12 -> 2.09 |
| init-variants | init variant core_random | v0 v4/a4 | native | 82.20 | 252.4 | 104.5 | 91.4 (88.8 / 99.3) | 10.95 | 2.76x | 1.14x | 50/50 / 50/50 | 3.12 -> 2.09 |
| init-variants | init variant ffn_magnitude | v0 v4/a4 | native | 92.12 | 249.2 | 107.7 | 93.5 (89.4 / 99.6) | 10.69 | 2.66x | 1.15x | 50/50 / 50/50 | 3.12 -> 2.09 |
| init-variants | init variant no_video_pretraining | v0 v4/a4 | native | 85.65 | 254.5 | 111.4 | 94.3 (99.1 / 99.3) | 10.60 | 2.70x | 1.18x | 50/50 / 50/50 | 3.12 -> 2.09 |
| init-variants | init variant stepwise | v0 v4/a4 | native | 91.47 | 249.6 | 105.5 | 91.5 (89.0 / 99.5) | 10.93 | 2.73x | 1.15x | 50/50 / 50/50 | 3.12 -> 2.09 |
| partition-grid | partition p1c6s5_k4 | v0 v4/a4 | native | not evaluated | 252.1 | 109.1 | 94.1 (98.8 / 99.1) | 10.62 | 2.68x | 1.16x | 50/50 / 50/50 | 3.12 -> 2.09 |
| partition-grid | partition p5c6s1_k4 | v0 v4/a4 | native | not evaluated | 252.4 | 104.9 | 91.2 (89.0 / 99.5) | 10.97 | 2.77x | 1.15x | 50/50 / 50/50 | 3.12 -> 2.09 |
| partition-grid | partition p6c6s0_k4 | v0 v4/a4 | native | not evaluated | 257.4 | 105.4 | 91.6 (88.8 / 99.4) | 10.92 | 2.81x | 1.15x | 50/50 / 50/50 | 3.12 -> 2.09 |
| partition-grid | partition p1c9s2_k3 | v0 v3/a3 | native | not evaluated | 257.0 | 106.0 | 91.6 (88.9 / 99.4) | 10.92 | 2.81x | 1.16x | 50/50 / 50/50 | 3.12 -> 2.09 |
| partition-grid | partition p4c3s5_k7 | v0 v7/a7 | native | not evaluated | 250.2 | 109.5 | 94.5 (99.1 / 99.3) | 10.59 | 2.65x | 1.16x | 50/50 / 50/50 | 3.12 -> 2.09 |
| libero-long | long v0 4/4 original | v0 v4/a4 | native | 77.73 L1500 | 250.5 | 109.7 | 94.7 (99.1 / 99.3) | 10.56 | 2.65x | 1.16x | 50/50 / 50/50 | 3.12 -> 2.09 |
| libero-long | long v0 4/4 repeat | v0 v4/a4 | native | 79.47 L1500 | 253.2 | 108.4 | 89.6 (88.8 / 99.1) | 11.17 | 2.83x | 1.21x | 50/50 / 50/50 | 3.12 -> 2.09 |
| libero-long | long v0 4/2 | v0 v4/a2 | native | 86.00 L100 | 168.4 | 72.9 | 66.1 (67.2 / 67.4) | 15.14 | 2.55x | 1.10x | 50/50 / 50/50 | 3.12 -> 2.09 |
| libero-long | long v0 4/1 | v0 v4/a1 | native | 74.00 L1500 | 122.7 | 56.9 | 48.4 (46.6 / 51.3) | 20.66 | 2.54x | 1.18x | 50/50 / 50/50 | 3.12 -> 2.09 |
| libero-long | long v0 2/2 | v0 v2/a2 | native | 81.00 L100 | 156.0 | 68.2 | 56.8 (56.8 / 57.2) | 17.59 | 2.74x | 1.20x | 50/50 / 50/50 | 3.12 -> 2.09 |
| libero-long | long v0 1/4 | v0 v1/a4 | native | 86.00 L100 | 238.8 | 96.7 | 87.3 (83.1 / 93.1) | 11.46 | 2.74x | 1.11x | 50/50 / 50/50 | 3.12 -> 2.09 |
| libero-long | long Dense-S12 | dense_s12 v1/a1 | native | 81.00 L100 | 109.7 | 47.8 | 41.2 (40.7 / 44.8) | 24.28 | 2.66x | 1.16x | 50/50 / 50/50 | 3.12 -> 2.09 |
| libero-long | long Dense-S30 | dense_s30 v1/a1 | native | 86.00 L100 | 252.6 | 103.9 | 89.5 (88.9 / 89.6) | 11.17 | 2.82x | 1.16x | 50/50 / 50/50 | 6.44 -> 3.79 |
| libero-long | long v0 4/1 kv-mix | v0 v4/a1 kv-mix | native | 75.53 L1500 | 133.8 | 60.0 | 52.9 (54.7 / 55.0) | 18.90 | 2.53x | 1.13x | 50/50 / 50/50 | 3.12 -> 2.09 |
| chronoloop | ChronoLoop CL-REG_mem16-reset-registers_a4 | ChronoLoop | native | 95.97 | 278.4 | n/a | 94.8 (99.6 / 99.9) | 10.55 | 2.94x |  | n/a / 500/500 | 3.12 -> 2.05 |
| chronoloop | ChronoLoop CL-W2_mem16-learned-external-updater_a4 | ChronoLoop | native | 96.02 | 284.2 | n/a | 91.0 (88.9 / 100.2) | 10.98 | 3.12x |  | n/a / 500/500 | 3.27 -> 2.13 |
| chronoloop | ChronoLoop CL-FRAME_history-frame-k3_a4 | ChronoLoop | native | 95.90 | 289.4 | n/a | 100.4 (100.3 / 100.5) | 9.96 | 2.88x |  | n/a / 500/500 | 3.13 -> 2.07 |
| chronoloop | ChronoLoop CL-0-at1_no-memory_a1 | ChronoLoop | native | 95.70 | 123.5 | n/a | 50.6 (51.3 / 51.5) | 19.77 | 2.44x |  | n/a / 500/500 | 3.12 -> 2.05 |
| chronoloop | ChronoLoop CL-A-at1_mem16-learned-loopwrite_a1 | ChronoLoop | native | 95.33 | 134.9 | n/a | 51.0 (52.1 / 52.4) | 19.62 | 2.65x |  | n/a / 500/500 | 3.12 -> 2.05 |

## 6. Upstream torch.compile (Inductor, reduce-overhead) vs fast

| Model | Eager mean | Compile mean (p95) | Exact compile | Max abs diff | Graph mean | Fast mean (p95) | Exact fast |
|---|---|---|---|---|---|---|---|
| FastWAM (released, libero_uncond_2cam224) | 264.2 | 87.3 (89.8) | 0/50 | 0.008941650390625 | 131.8 | 107.5 (108.9) | 50/50 |
| all-suites v0 4/4 | 266.7 | 62.4 (66.4) | 0/50 | 0.025390625 | 104.2 | 99.3 (99.6) | 50/50 |

## 7. Elastic checkpoints: success and latency per operating point

| Run | Schedule | Depths | Action-expert block passes | SR % | Fast mean (p95) | Graph mean | Eager mean |
|---|---|---|---|---|---|---|---|
| E0 | flat_k1_n2 | [1,1] | 2 | 94.33 | 24.7 (24.8) | 30.0 | 48.8 |
| E0 | flat_k1_n5 | [1]x5 | 5 | 94.53 | 33.3 (33.6) | 40.1 | 81.4 |
| E0 | flat_k1_n10 | [1]x10 | 10 | 93.68 | 47.9 (47.9) | 52.9 | 113.8 |
| E0 | flat_k2_n2 | [2,2] | 4 | 94.53 | 27.6 (27.7) | 32.2 | 55.9 |
| E0 | flat_k2_n5 | [2]x5 | 10 | 94.85 | 39.1 (40.8) | 47.8 | 92.3 |
| E0 | flat_k2_n10 | [2]x10 | 20 | 93.83 | 59.9 (62.0) | 70.3 | 151.6 |
| E0 | flat_k4_n2 | [4,4] | 8 | 94.97 | 33.2 (33.3) | 39.6 | 70.7 |
| E0 | flat_k4_n5 | [4]x5 | 20 | 95.45 | 54.5 (54.4) | 62.9 | 131.6 |
| E0 | flat_k4_n10 | [4]x10 | 40 | 94.87 | 86.2 (90.0) | 98.3 | 227.5 |
| E1 | flat_k1_n2 | [1,1] | 2 | 93.73 | 24.8 (24.9) | 30.0 | 48.8 |
| E1 | flat_k1_n5 | [1]x5 | 5 | 94.30 | 33.4 (33.8) | 39.8 | 83.5 |
| E1 | flat_k1_n10 | [1]x10 | 10 | 93.48 | 47.7 (47.8) | 52.6 | 114.8 |
| E1 | flat_k2_n2 | [2,2] | 4 | 94.58 | 27.6 (27.7) | 32.0 | 56.4 |
| E1 | flat_k2_n5 | [2]x5 | 10 | 94.52 | 39.1 (40.8) | 47.5 | 92.7 |
| E1 | flat_k2_n10 | [2]x10 | 20 | 94.00 | 59.7 (61.9) | 70.6 | 153.1 |
| E1 | flat_k4_n2 | [4,4] | 8 | 95.22 | 33.4 (33.5) | 39.6 | 71.4 |
| E1 | flat_k4_n5 | [4]x5 | 20 | 95.65 | 54.3 (54.4) | 62.9 | 130.1 |
| E1 | flat_k4_n10 | [4]x10 | 40 | 94.45 | 86.3 (90.2) | 98.3 | 230.9 |
| E2 | flat_k1_n2 | [1,1] | 2 | 94.07 | 24.8 (24.9) | 29.8 | 49.0 |
| E2 | flat_k1_n5 | [1]x5 | 5 | 94.25 | 33.6 (34.0) | 40.4 | 84.3 |
| E2 | flat_k1_n10 | [1]x10 | 10 | 94.00 | 48.4 (48.5) | 53.0 | 118.3 |
| E2 | flat_k2_n2 | [2,2] | 4 | 94.75 | 27.7 (27.8) | 32.2 | 56.9 |
| E2 | flat_k2_n5 | [2]x5 | 10 | 95.07 | 39.4 (41.1) | 48.1 | 94.5 |
| E2 | flat_k2_n10 | [2]x10 | 20 | 94.07 | 60.7 (62.6) | 70.9 | 156.0 |
| E2 | flat_k4_n2 | [4,4] | 8 | 95.27 | 33.4 (33.4) | 39.5 | 71.1 |
| E2 | flat_k4_n5 | [4]x5 | 20 | 95.28 | 54.2 (54.3) | 62.8 | 130.5 |
| E2 | flat_k4_n10 | [4]x10 | 40 | 94.70 | 86.0 (89.8) | 98.0 | 233.1 |
| F0-PB | S8 | [4,1] | 5 | 94.50 | 31.8 (31.9) | 38.1 | 72.8 |
| F0-PB | S7 | [4,2] | 6 | 94.70 | 31.3 (33.8) | 37.7 | 85.4 |
| F0-PB | parent2 | [4,4] | 8 | 95.88 | 37.0 (37.6) | 43.6 | 90.3 |
| F0-PB | S6 | [4,1,1,1,1] | 8 | 93.75 | 42.1 (42.2) | 49.5 | 103.6 |
| F0-PB | S5 | [4,2,2,2,2] | 12 | 94.25 | 45.1 (49.7) | 57.8 | 128.2 |
| F0-PB | S2 | [4]+[1]x9 | 13 | 92.05 | 54.5 (58.7) | 67.4 | 155.3 |
| F0-PB | parent5 | [4]x5 | 20 | 95.75 | 59.7 (64.5) | 72.3 | 178.5 |
| F0-PB | S1 | [4]+[2]x9 | 22 | 93.38 | 69.2 (75.9) | 81.8 | 211.9 |
| F0-PB | parent10 | [4]x10 | 40 | 94.68 | 103.8 (110.4) | 120.7 | 321.3 |
| SP11 | flat_k1_n2 | [1,1] | 2 | 93.77 | 23.4 (24.9) | 30.6 | 49.1 |

### 7.1 Memory with every operating point's fast graph resident

| Model | Captured graphs | Allocated GB | Peak GB | Reserved GB |
|---|---|---|---|---|
| elastic E0 | 9 | 2.69 | 3.18 | 8.36 |
| elastic E1 | 9 | 2.69 | 3.18 | 8.36 |
| elastic E2 | 9 | 2.69 | 3.18 | 8.36 |
| elastic F0-PB | 9 | 2.70 | 3.18 | 8.37 |
| RT-A2 | 6 | 2.38 | 2.87 | 6.29 |
| RT-A | 4 | 2.18 | 2.66 | 4.90 |
| RT-B4 | 2 | 1.97 | 2.45 | 3.49 |
| RT-Pa | 2 | 1.96 | 2.45 | 3.48 |
| RT-T1ft | 2 | 1.97 | 2.46 | 3.51 |
| RT-TF | 2 | 1.97 | 2.45 | 3.49 |

## 8. Success-latency Pareto front, matched four-suite models

Each row is faster than every row below it and more successful than every row above it.

| Model | Config | SR % | Fast mean ms | Fast p95 ms |
|---|---|---|---|---|
| elastic SP11 | flat_k1_n2 [1,1] | 93.77 | 23.4 | 24.9 |
| elastic E0 | flat_k1_n2 [1,1] | 94.33 | 24.7 | 24.8 |
| elastic E1 | flat_k2_n2 [2,2] | 94.58 | 27.6 | 27.7 |
| elastic E2 | flat_k2_n2 [2,2] | 94.75 | 27.7 | 27.8 |
| RT-A2 | S7 [4,2] | 94.93 | 31.5 | 33.9 |
| elastic E0 | flat_k4_n2 [4,4] | 94.97 | 33.2 | 33.3 |
| elastic E2 | flat_k4_n2 [4,4] | 95.27 | 33.4 | 33.4 |
| elastic F0-PB | parent2 [4,4] | 95.88 | 37.0 | 37.6 |
| ChronoLoop CL-W2_mem16-learned-external-updater_a4 | native  | 96.02 | 91.0 | 100.2 |
| ChronoLoop CL-0_no-memory_a4 | native  | 96.10 | 91.8 | 99.6 |
| ChronoLoop CL-A_mem16-learned-loopwrite_a4 | native  | 96.18 | 92.1 | 99.6 |
| FastWAM (released, libero_uncond_2cam224) | native  | 96.63 | 107.5 | 108.9 |

## 9. Interference control

The same job re-measured on one GPU with the node's other three GPUs idle, vs the 4-worker sweep.

| Model | Config | Backend | Sweep mean / p95 | Alone mean / p95 | Difference of means |
|---|---|---|---|---|---|
| all-suites v0 4/1 | native | eager | 123.3 / 124.9 | 121.5 / 122.7 | +1.5% |
| all-suites v0 4/1 | native | graph | 60.9 / 61.1 | 59.1 / 60.8 | +3.1% |
| all-suites v0 4/1 | native | fast | 48.3 / 51.2 | 49.7 / 51.3 | -2.9% |

## 10. Checkpoint identity

sha256 of each locally stored timed checkpoint vs the checkpoint its success rate was scored with (HF downloads are checked against their export records separately).

| Model | Timed file sha256 | Evaluated sha256 prefix | Match |
|---|---|---|---|
| elastic E0 | 5569c47fa6c91e13 | 5569c47fa6c9 | yes |
| elastic E1 | be1cbe2cd1c9fa2b | be1cbe2cd1c9 | yes |
| elastic E2 | 6d92ac13a36e8605 | 6d92ac13a36e | yes |
| elastic F0-PB | ff59791adcf2c43b | ff59791adcf2 | yes |
| FastWAM (released, libero_uncond_2cam224) | 1000437cfcf55c00 | 1000437cfcf5 | yes |
| RT-A2 | 25ad8ca4705e41c8 | 25ad8ca4705e | yes |
| RT-A | 15ba0de18c94dfd9 | 15ba0de18c94 | yes |
| RT-B4 | 7ec1dedc34446b27 | 7ec1dedc3444 | yes |
| RT-Pa | 17f089aed0a48756 | 17f089aed0a4 | yes |
| RT-B2a | 25175895948053e2 | 25175895948 | yes |
| elastic SP11 | ab974f357659d094 | ab974f357659 | yes |
| ChronoLoop CL-0_no-memory_a4 | 137f0a839b5a1e23 | 137f0a839b5a | yes |
| ChronoLoop CL-A_mem16-learned-loopwrite_a4 | 3b480febd1888712 | 3b480febd188 | yes |
| RT-T1ft | efb19b697478e4b8 | efb19b697478 | yes |
| RT-TF | c6cdba96a98a073f | c6cdba96a98a | yes |
| init variant stepwise | 0af0b9174fdd2cdc | - | n/a |
| ChronoLoop CL-REG_mem16-reset-registers_a4 | 6de1bbf80ab0566a | 6de1bbf80ab0 | yes |
| ChronoLoop CL-W2_mem16-learned-external-updater_a4 | d08096facd67779d | d08096facd67 | yes |
| ChronoLoop CL-FRAME_history-frame-k3_a4 | b322ae52c8166dbe | b322ae52c816 | yes |
| ChronoLoop CL-0-at1_no-memory_a1 | 77cbd77a521686e9 | 77cbd77a5216 | yes |
| ChronoLoop CL-A-at1_mem16-learned-loopwrite_a1 | 391d3b59bcd2dc6d | 391d3b59bcd2 | yes |

## 11. Caveats

- **Parallel measurement.** Four GPUs were measured at once, each with its own pinned cores. Cross-GPU interference (host memory bandwidth, PCIe, power) is quantified only by the control in Section 9.
- **Success rates come from earlier evaluations.**
  - Most use the matched protocol: 4 suites × 10 tasks × 50 states × seeds 42/43/44, 6,000 episodes.
  - Some RT rows were scored on RTX PRO 6000 GPUs in the remote campaign (`Residual_Transport_Results.csv`).
  - `L100` rows (single seed, 100 Long episodes) are not comparable with the matched rows.
  - Models without a matched evaluation are marked "not evaluated", among them the partition grid and Dense-S12/S30 on all suites.
- **Two latency levels.** The graph-replay backends (`graph`, `fast`) often alternate between two latency levels about 7–11% apart, even though the captured kernels are fixed:
  - v0 4/4-depth LoopWAM models sit at about 89 or about 99 ms; Elastic [4]x10 at about 80 or about 90 ms; [1,1] at about 23 or about 25 ms.
  - Some runs stay at one level for all 500 queries (all-suites v0 4/4: 99.3 ms, std 0.1 ms). Others switch within the run, which shows as a gap between p50 and p95.
  - It also happens with the GPU running alone (Section 9: the control's fast p5 is 46.3 ms and its p50 51.1 ms).
  - A spot check during the final Elastic re-run showed every GPU at its 1980 MHz SM clock, no throttle reasons, and 280–330 W of a 700 W limit. Clock throttling is therefore not the cause; the actual cause was not identified.
  - Means include both levels. Mean differences under about 10% between models of the same compute should not be read as real. One example: SP11 and E0 run the identical [1,1] graph at 23.4 vs 24.7 ms.
  - It also shows between repeated runs. The four Elastic jobs were measured twice. The first pass timed the graph backend on every 5th observation; it is kept in `latency_20261030/results_first_pass_elastic_graph_stride5/`.
    The fast mean of the same configuration differed between the passes by up to about 7%, for example E0 [4]x10 at 82.9 vs 86.2 ms. The report uses the second pass.
- **Peak memory in multi-schedule jobs.** Within one job, each later operating point's peak memory is about 34 MB higher than the one before, so some memory from earlier captures is not returned.
  - The figures for later operating points of RT and Elastic checkpoints therefore overstate their own peak by at most about 0.3 GB.
  - Single-configuration models are unaffected. The direction is conservative.
- **Latency is not evaluation throughput.** Evaluation runs many episodes per GPU concurrently. These numbers are single-stream, batch-one action-query latencies.
- **Out of scope: RoboTwin checkpoints.** The RoboTwin-trained ChronoLoop checkpoints (`robotwin5hard` CL-0 and CL-A, epoch 5; HF `anhdao69/ChronoLoop_Robotwin5`) take RoboTwin observations: other cameras and 14-D dual-arm actions. The LIBERO replay protocol does not apply to them. They share the CL-0/CL-A architectures whose LIBERO latency is reported here. No LoopWAM RoboTwin checkpoint exists on this cluster.
- **Initialization variant `stepwise`** is timed from its local training checkpoint `train/latest.pt`, which was not uploaded to Hugging Face.
- **Checkpoint provenance.** Hugging Face checkpoints were downloaded at revision `393310afabb5f84226d314da14b9222ac193193d`. All 25 with an export hash matched it. The 6 partition-grid exports carry no hash record.

## 12. Reproduce

```bash
source /groups/yshang/an221229/checkpoints/LoopWAM_NT/latency_20261030/latency_env.sh   # paths, threads, caches
LAT_HF=<HF snapshot dir> python scripts/latency/jobs.py > jobs.jsonl                      # job list (priority order)
srun --jobid=<job> --overlap --gres=gpu:4 ... bash scripts/latency/sweep.sh jobs.jsonl 0,1,2,3
python scripts/latency/build_report.py --results $LAT/results --jobs jobs.jsonl --out <dir> \
  --control $LAT/results_control --elastic-results <elastic results.json> --local-sha <hash json>
```

A single job runs as `run_jobs.sh <jobs.jsonl> <gpu> <cpus> <id>`.
