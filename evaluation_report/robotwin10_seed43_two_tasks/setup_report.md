# RoboTwin-10 two-task rerun, seed 43

The user requested a different-seed rerun of the two tasks highlighted in the v0/dense-s30 comparison: **open_microwave** and **move_stapler_pad**. Both models are included. The checkpoints are unchanged epoch-5 RoboTwin-10 releases at update 45,445. Queue order is v0 4/4, then dense-s30.

## Scope and scoring

| Item | Setting |
| --- | --- |
| Models | v0-video4-action4 (584,567,886 parameters); dense-s30 (1,416,145,998) |
| Tasks | open_microwave, move_stapler_pad |
| Settings | demo_clean, demo_randomized |
| Scored episodes | 100 per model/task/setting; 400 per model; 800 total |
| Evaluation seed | 43 only, replacing baseline seed42 |
| Candidate scene-seed start | 4,400,000; original expert acceptance can skip candidates |
| Action noise | fixed-per-query CPU noise, seed43 |
| NFE / CFG | 10 / 1 |
| Action chunk | predict32, execute24 |
| Precision | FP32 policy weights, BF16 compute |
| Renderer | RT32/depth8/OIDN2.3.3, unchanged |
| Inputs | released FastWAM PIL uint8 3-camera packing at 384×320; checkpoint z-score normalization |
| Instructions | unseen; pool size100 for scored cells; original unseeded Python template sampling |
| Scene acceptance / horizons | original FastWAM/RoboTwin expert plan and success acceptance; original task horizons |
| Attempts | up to3 per scored cell; 1,800-second no-progress timeout |

The change of evaluator seed changes both candidate scenes and the policy action noise, as implemented by the frozen evaluator. It does not independently reseed Python instruction sampling. Actual instructions, scene seeds and action query hashes are recorded. Treat seed43 as another evaluation draw, not an exactly instruction-paired test.

Seed42 results are preserved separately. The new selected-task scores must not be reported as full-ten-task benchmark performance. Calibration episodes and interrupted-attempt partial episodes are excluded from final scores.

## Speed selection

Reuse exact-output-validated CUDA graphs and one shared T5 server per model run. The existing real-input graph checks covered seeds42/43/44; they are reused after checking checkpoint and code/runtime hashes. Native strict-load evidence is also reused for the identical checkpoint. All 38 existing evaluator/model tests passed again during setup.

At setup, only an unrelated VLLM process used GPU memory, leaving about61GB free. The queue measures current-checkpoint speed on the requested tasks, two episodes per task/setting (8 unscored episodes per profile): v0 profiles2 and4 workers; dense profiles2 and3 workers. Higher profiles run only if the admission memory threshold is met. One worker is available when there is insufficient room for two. Pick the fastest successful profile with at least8,000MiB observed free-memory margin. Full scoring also requires the selected profile's admission threshold or measured added peak plus10,000MiB, whichever is larger. No other process is stopped.

Dense-s30 has a higher per-worker memory footprint, so its planned concurrency cap is3. Neither model uses the previously problematic eight-worker configuration. Actual throughput depends on task failures and competing GPU activity; short calibration is not a guaranteed runtime estimate.

The longer open_microwave cells are scheduled first so additional workers can evaluate the shorter move_stapler_pad cells concurrently. Concurrency is fixed for each scored run after calibration.

## Code, assets and provenance

Frozen evaluator: `/storage/anhdh35/LoopWAM-robotwin10-dense-eval`, commit `7a2181d0ecc8e30d64806e94535b347bc9e1da2e`. No model, simulator or scoring source changes were made. Only new queue/report copies configure seed43 and this task subset. Runtime asset/source hashes are checked before evaluation and at final audit.

- v0 pinned HF revision: `df3d575d3f3fde3cf1b8d5f609fab93dc6a95649`; checkpoint SHA256 `bb1563f16bcd102b3f4ad4a1a14377ee49de7e461d862b7bbc5dba497f036127`.
- Dense-s30 pinned HF revision: `cdf71976208ade41a93465f6b8166b5e6b7170ca`; checkpoint SHA256 `cc549b65bb849198ab1d11fd78e2a14bd0d1682927665b24c5f82b3ee8b78985`.

Existing checkpoints, environment and assets are reused through explicit paths. Original release manifests, strict-load evidence and graph-gate evidence retain their original timestamps. New preflight.json files identify this evidence as reused and record fresh hash verification; they do not claim a new native/graph test ran.

## Queue and outputs

Persistent tmux session: `robotwin10_seed43_two_tasks`.

Bulk directory: `/data14t/anhdh35/loopwam_robotwin10_seed43_two_tasks_20261103`.

Report directory: `/storage/anhdh35/LoopWAM_NT/evaluation_report/robotwin10_seed43_two_tasks`.

[Queue status](queue_status.json) · [v0 live report](v0_4x4/report.md) · [dense-s30 live report](dense_s30/report.md).

Each model has separate calibration results, scored results, per-episode videos, instructions, action hashes, commands, progress JSON and logs. After400 scored episodes, its auditor revalidates videos, traces, counts, checkpoint and runtime hashes and writes final_report.md, final_results.json, episodes.csv, task_results.csv and final_audit.json. After both complete, the root final_report.md and comparison.csv compare the same task/settings across seed42 and43 and both models.

Tmux survives client disconnection but not a host reboot. Validation or unrecoverable scorer errors stop the queue and are logged; no protocol fallback occurs silently.
