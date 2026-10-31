> **Queue changed on 27 October 2026:** Only Ka=1 and Ka=2 are retained. Ka=3,5,6 were removed at user request; the active Ka=2 runner continues and no further evaluation will launch. The original plan below is historical. See [queue change record](stop_after_ka2.json).

# v0 action-depth sweep: setup and reproducibility

Latest main branch pulled successfully: `56e33285a82c6abe075565b3d60c26a5ba7a148f`. The only upstream changes since the previous local head were evaluation reports. Local report edits were preserved in a filesystem backup and git stash before fast-forwarding, then restored. No report changes were discarded.

Isolated evaluation branch: `eval/v0-ka-sweep-20261027`, local commit `fa42937f35ba45ffa936f209d28d787ee2247a4b` at `/storage/anhdh35/LoopWAM-v0-ka-eval`. This branch has not been pushed. [Implementation patch](implementation.patch) and [source archive](evaluator_source.tar.gz) preserve the inference-only extension and orchestration.

## Requested conditions

Same original full-suite v0 checkpoint, trained video/action loops 4/4, ten complete epochs and 21,700 training updates. Policy SHA-256 `1c659a7c2f6970c3ea5972a8fba4abd9db35857c83a67a053c0ee35555fb602b`. It is never replaced by separately trained asymmetric checkpoints. Video depth stays four; action depth runs **1 → 2 → 3 → 5 → 6**; denoising NFE stays ten. Each condition runs 4 suites × 10 tasks × 3 seeds × 50 initial states = **6,000 scored episodes**, total **30,000**.

Seeds: **42,43,44**, once per task/environment group; every query uses its group's seed; no per-episode reseeding. Initial states are indices 0–49. Suite caps: Long700, Spatial/Object/Goal400 policy steps plus30 settling. Predict32, execute10, CFG1, original normalization, FP32 model and BF16 autocast/VAE. Native denormalization followed by `sign(-(2*g-1))`, including neutral zero at g=.5. Every episode saves video, timings, state/checkpoint identity and per-query action hashes.

## Depth extrapolation definition

Only eval-mode v0, Kv4, trained depth4 can use the new bounded action-depth setter. Constructor/training depth validation remains unchanged. Twelve unique action blocks are traversed as three pre blocks + six shared core blocks repeated Ka times + three post blocks. Effective action depths are 12,18,24,30,36,42 for Ka1..6. Physical weights do not change. Native late alignment maps core iteration r to video core max(1,4−Ka+r): Ka1=[4], Ka2=[3,4], Ka3=[2,3,4], Ka4=[1,2,3,4], Ka5=[1,1,2,3,4], Ka6=[1,1,1,2,3,4]. This is an inference-only extrapolation of the existing cached-KV rule.

Historical Ka4 result: 5,695/6,000 =94.9167%. It remains a prior reference; it is not another requested scored job. A fresh eight-observation native Ka4 comparison against the previous evaluator installation, across all four suites and seeds42/43/44, must match bit-for-bit before the queue proceeds. This finite check does not prove identical full trajectories.

## Speed selection and resumability

Per-depth real-observation CUDA-graph checks compare complete finite32×7 predictions exactly against eager output, including changed observations, prompts and seeds. Eligible graphs then run full calibration episodes. Each depth has16 unscored calibration episodes: all four suites, tasks0 and8, seed42, states0 and1. Native4 workers is the reference; graph4 and graph8 are tested if the graph gate and VRAM allowance pass. Only profiles with identical outcome, policy-step count and complete per-query action-hash traces can win. Selection uses wall time after all workers are ready, including rollout/logging overhead. Other GPU workloads are left running. Maximum global inference speed is not claimed beyond these measured candidates.

The serial controller runs in tmux `v0_ka_sweep`. Locking rejects duplicate controllers/runners. Complete task/seed groups are committed atomically; interrupted scored groups replay from episode0. Scored resumes retain the previously selected backend/workers. Interrupted calibration is archived and rerun from scratch; partial-work timing is excluded. Checkpoint, normalization, VAE, text, task assets and current/legacy source identities are frozen and checked. Unexpected scored errors stop the queue and retain logs rather than silently counting missing episodes as failures. A manual restart uses `bash /storage/anhdh35/LoopWAM-v0-ka-eval/scripts/launch_v0_ka_queue.sh` after checking no old runner remains. No OS reboot service was installed.

## Validation and limitations

Nineteen new regression cases pass, including numerical Ka4 identity, action/KV schedules, bounds/training-mode restrictions, exact6,000-episode coverage, wrong-depth/checkpoint/duplicate/missing rejection, calibration parity/timing, and actual video encode/decode. The complete project test directory has319 passes,7 CUDA skips and one failure: `test_dense_s30_eval.py::test_complete_dense30_and_partial_smoke`. That Dense-S30 metadata fixture also fails on unmodified upstream and does not exercise v0. Running bare repository-wide pytest additionally reaches unrelated RoboTwin examples requiring unavailable `openai` and `sapien` packages. All results are recorded; the suite is not described as fully green.

Independent review found and resolved a video helper API mismatch, missing normalization/VAE rechecks, backend/worker changes on resume, and biased partial calibration timing. The initial unscored staging attempt failed on the video API before any scored episodes; its artifacts remain under `/data14t/anhdh35/loopwam_v0_ka_sweep_20261027/staging_before_review_fixes`. Final launch regenerates provenance and gates after fixes.

## Files

- [Live report](report.md)
- [Live queue state](queue_status.json)
- [Frozen protocol and source hashes](protocol.json)
- [Preflight result](preflight.json)
- [Setup audit](setup_audit.json)
- Bulk outputs: `/data14t/anhdh35/loopwam_v0_ka_sweep_20261027`
- Full project tests: `/data14t/anhdh35/loopwam_v0_ka_sweep_20261027/final_project_tests.log`
- Upstream fixture reproduction: `/data14t/anhdh35/loopwam_v0_ka_sweep_20261027/upstream_dense30_test.log`

This campaign evaluates standard LIBERO, not Plus or Pro. Calibration results never enter scored SR. Scores and timing remain pending until actual execution; startup and partial progress should not be presented as completion of the30,000 episodes.
