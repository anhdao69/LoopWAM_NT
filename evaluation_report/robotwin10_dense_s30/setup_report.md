# RoboTwin-10 dense-s30 — evaluation setup

The source blocker is resolved. Branch `robotwin10_training_20261101` at `3080145` was pulled into the isolated evaluator; all nine training-source SHA256 values match the release. The merged local evaluator is `7a2181d0ecc8e30d64806e94535b347bc9e1da2e`. Prior v0/epoch6 evaluation worktrees and results remain unchanged.

## Model and release

HF revision `cdf71976208ade41a93465f6b8166b5e6b7170ca`; checkpoint `robotwin-10/dense-s30/policy.pt`; SHA256 `cc549b65bb849198ab1d11fd78e2a14bd0d1682927665b24c5f82b3ee8b78985`. All ten release files are verified against API metadata and release SHA256SUMS. The native checkpoint is epoch 5, update 45,445, trained on 5,500 RoboTwin-10 demonstrations and 5,816,755 window presentations. It has 1,416,145,998 policy parameters, 30 physical block pairs, and one video/action pass. No v0 loop-depth override is used. The dataset-statistics file is the release's supplied z-score statistics.

Strict CPU loading passed with the exact published variant loader, finite weights, 30 layers per expert, one loop, and the expected training cursor. VAE and normalization hashes match the training contract. The initial payload-only check is historical evidence; cpu_validation.json records the later strict model load.

## Protocol

Ten tasks: adjust_bottle, grab_roller, hanging_mug, move_stapler_pad, open_microwave, press_stapler, scan_object, stack_blocks_two, stamp_seal, turn_switch.

Each task uses clean and randomized settings with 100 accepted episodes per setting: 1,000 clean + 1,000 randomized = 2,000 scored episodes. Seed 42; candidate scene seeds start at 4,300,000; expert planning and success filtering are unchanged. Unseen instructions, NFE 10, CFG 1, fixed CPU noise seed 42 per query, prediction horizon 32 and execution horizon 24. Absolute 14D qpos commands. FP32 policy weights, BF16 inference/VAE. Same PIL uint8/BF16 three-camera packing and RT32/depth8/OIDN2.3.3 renderer as the prior local RoboTwin runs.

The inherited FastWAM harness does not explicitly seed Python instruction-template selection. Equal seed labels therefore do not guarantee identical instruction strings across runs; actual instructions, scene seeds and action hashes are logged. Candidate expert/setup exceptions follow the original scene-skip logic. Policy rollout interruptions retry a whole task/setting pair, retaining and excluding partial attempts. Completion requires all 20 pairs and 2,000 scored episodes with validated videos and traces.

## Gates and speed

38 targeted tests passed after adding the newly published training-script directory to PYTHONPATH. The initial test attempt failed one import-path test; both logs are retained. No model code was changed to resolve that path issue.

Native and CUDA-graph inference matched exactly on eight captured real observations, forward and reverse order, with varied noise seeds. Native mean query time: 0.1794 s; CUDA graph: 0.1322 s, about 26.3% lower latency in that isolated gate. This is not end-to-end episode throughput or proof of equality for every possible input.

The preflight runs eight unscored episodes with one native worker, then eight with two graph workers, on adjust_bottle and grab_roller in both settings. It validates every video/trace and records total GPU memory, including the separate process already using approximately 35 GiB. The profiles differ in both backend and concurrency and are not an isolated worker-count benchmark. Two workers are selected only if the second gate completes with peak total GPU memory below 88,000 MiB; otherwise one worker is selected. The other GPU workload is left untouched. A shared T5 server and observation reuse between replans remain enabled. NFE, renderer quality and scored episode counts are not reduced for speed.

## Queue and evidence

Bulk outputs: `/data14t/anhdh35/loopwam_robotwin10_dense_s30_20261102`. The persistent preflight pipeline launches the full queue only after the simulator gates pass. The full controller checks the checkpoint hash, graph/CPU gates, runtime assets and smoke completion records. It uses at most three attempts per pair with a 1,800-second no-progress timeout. A protocol-manifest mismatch blocks resume; completed pairs are revalidated before reuse.

See report.md / progress.json for actual scored progress, setup_status.json for the current preflight phase, calibration.json for measured smoke profiles and concurrency selection, source_check.json for matching training-code hashes, cpu_validation.json for strict loading, graph_gate.json for native/graph comparison, and tests.log for the 38 tests. Smoke outcomes are excluded from scored SR. Training uses only these ten tasks, whereas the preceding v0 models were trained on all 50; matching evaluation settings does not equal matching training data.


## Automatic final reporting

A separate post-run reporter waits for successful controller completion, then revalidates the checkpoint, frozen source/runtime hashes, all 20 completed task/setting records and all 2,000 videos and episode traces. It writes final_report.md, final_results.json, task_results.csv, episodes.csv and final_audit.json. Its state is recorded in finalizer_status.json. It does not publish final scores after an evaluator error. The post-run reporter is operational reporting code outside the immutable policy/runtime contract; its source and hash are archived here.
