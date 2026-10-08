# Full LIBERO loop ablations — October 8, 2026

## Requested runs and matched controls

Four fresh LoopWAM v0 models: video/action loops **2/2, 4/1, 1/4, 3/3**.
Each has 584,536,135 policy parameters. Effective block applications are
18/18, 30/12, 12/30, and 24/24 respectively (3 prelude + 6 shared core per
loop + 3 coda). The action stream uses the existing late video-loop alignment.

All 1,712 available demonstrations from Spatial/Object/Goal/Long are training
data: 277,713 windows per epoch, 10 epochs, 21,700 optimizer updates,
2,777,130 real windows. Seed 42, global batch 128, canonical Wan initialization,
fresh AdamW and the original learning-rate schedule. FP32 policy/optimizer
states, BF16 compute, exact tail-batch normalization. No trained checkpoint is
used for initialization. The existing immutable VAE cache is reused only after
coverage and provenance validation.

The original full 4/4 model trained on four H100s. These ablations use two H100s
per model; global batch and optimization settings match, but random-number
consumption and reduction order can differ with distributed configuration.

## Evaluation protocol

Each model receives three complete evaluation rounds, **all using seed 42** as
requested to match the original full 4/4 run. Each round covers all four suites,
ten tasks per suite, ten fixed initial states per task: 400 episodes/round and
1,200 episodes/model, 4,800 final episodes overall. Report each round separately;
these are repeated same-seed evaluations, not three independent seed estimates.

Keep the original 700 policy-step cap, 30 settling steps, 32-action chunk,
replanning every ten actions, ten denoising steps, CFG 1, normalization,
observation preprocessing, gripper mapping and task/episode/replan seed offsets.
Videos, per-episode records and final checkpoint hashes remain on the server.

## Evaluation optimization

Persistent independent model/simulator processes share each allocated GPU.
A dynamic episode queue spans suites and rounds; Long starts first to reduce
stragglers. Each worker resets all episode RNGs and uses the original single-
episode inference function. This avoids changing inference batch semantics.
Each worker uses one Torch/OpenMP/MKL/OSMesa thread to avoid simulator CPU
oversubscription. The concurrency speedup includes additional CPU parallelism
as well as GPU overlap; it is not a measurement of faster individual kernels.

Preflight compares 1, 2, 4 and 8 workers per GPU on the same 16 episodes, with
100-step smoke horizons and saved action traces. Candidate workers must match
seeds, outcomes, step/replan counts, and every predicted action (absolute
threshold 1e-5) before being considered for throughput selection. These smoke
success rates are not final benchmark results.

## Slurm structure and safety checks

Two jobs, each two H100s, 32 CPU threads, 256 GiB RAM, 72-hour limit:

- **4768, worker-0:** 4/1 training → three-round evaluation → 1/4 training → evaluation.
- **4769, worker-1:** 3/3 training → three-round evaluation → 2/2 training → evaluation.

Production source: `d166e8f7b89a4fe9fe807c1d1896e447a7963a88`.
Output parent: `/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt`.
Output directories: `full_loop_job4768`, `full_loop_job4769`.
Slurm scripts: `scripts/submit_full_loop_worker-0.sbatch` and
`scripts/submit_full_loop_worker-1.sbatch`.

Each job first measures evaluation concurrency and full-data DDP/ZeRO training
throughput, then runs native fresh training and all-suite inference smokes for
both assigned models. Production waits for an audit release matching the pinned
source revision and exact experiment pair. Every stage fails closed; evaluation
requires the complete final training contract. The next model starts only after
all 1,200 preceding evaluation episodes and videos pass completeness checks.
The batch allocation is released naturally when its pipeline exits.

## Verification and measurements

Server preflight tests: 32 passed, including 3/3 cached inference, gradients,
future-video isolation and checkpoint reconstruction. Additional local queue
failure/completion tests: 13 passed together with episode-grid tests.

GPU benchmark results, selected configurations, production checks and estimated
completion times will be appended after the preflight audit.

## Production verification and subsequent user-directed reorder

Production was released at 22:28:20 UTC after all four native smokes passed.
Every data-manifest field matched the original full run except the destination
`normalization_path`; initialization/VAE hashes also matched. Fresh optimizers,
VAE anchors, finite gradients and all-suite checkpoint reloads were verified.
At 22:30:37 UTC, 4/1 had 19 finite updates (3.3925 s/update), and 3/3 had 33
(3.0036 s/update). Both H100s in each allocation showed 100% utilization.

The user then requested concat/mix experiments **before** these full-suite runs,
followed by Dense-S12/S30 full-suite controls, four training runs per replacement
job. The short production runs in jobs 4768/4769 are therefore being superseded;
their directories and evidence are preserved. They are not completed results.
The new campaign will use evaluation seeds 42/43/44, superseding this document's
original repeated-seed-42 plan. Implementation proceeds on `KV_concat` while
`LoopWAM_NT` remains untouched.

Launch evidence: [Full_LIBERO_Loop_Ablations_20261008_evidence.json](Full_LIBERO_Loop_Ablations_20261008_evidence.json).
