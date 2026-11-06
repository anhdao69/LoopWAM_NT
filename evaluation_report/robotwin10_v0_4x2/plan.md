# RoboTwin-10 v0 4/2 evaluation plan

User request: download robotwin-10/v0-video4-action2 and carefully evaluate. Use the established full-ten-task seed42 protocol for fair comparison with v0 4/4 and dense-s30: clean/randomized, 100 episodes per task/setting, NFE10, 2,000 total.

- [x] Pin HF revision and inspect release metadata; verify all nine recorded code hashes against the frozen evaluator.
- [x] Prepare isolated queue, strict CPU validator (4 video / 2 action loops), graph validator, live report and final artifact auditor using the previous validated workflow.
- [x] Verify queue scripts and 38 existing tests; launch persistent queue.
- [x] Download and verify all ten release files, strictly load the exact native checkpoint, check stats/VAE/training recipe.
- [x] Require exact native/graph actions on eight real observations, seeds42/43/44, forward/reverse order.
- [x] Run unscored native and graph simulator gates; all32 episodes succeeded; four graph workers selected (102.77seconds/8 calibration episodes).
- [ ] Score 20 task/setting pairs and audit every episode artifact; write final seed42-only three-model comparison.

No model/simulator changes, no NFE reduction, no seed43 substitutions. Existing checkpoints/results stay unchanged. The queue waits for memory rather than stopping other GPU jobs. Partial worker attempts and calibration episodes are excluded from scored results. Same unseeded Python instruction-template generation as previous evaluations; actual instructions are logged.
