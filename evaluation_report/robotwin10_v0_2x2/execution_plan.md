# RoboTwin-10 v0 2/2 evaluation plan

User request: download the robotwin-10/v0-video2-action2 release and carefully evaluate under the established comparison protocol.

- [x] Pin the Hugging Face revision and verify all nine recorded training source hashes against the frozen evaluator.
- [x] Create an isolated run directory and adapt the proven 4/2 launch configuration to native 2/2; reuse the established environment and assets.
- [x] Download and verify all ten release files using Hugging Face hashes and SHA256SUMS.
- [x] Run the 38 existing evaluator/model tests; strictly load the checkpoint and verify loops, dimensions, finite weights, parameter count, training contract, normalization and VAE.
- [x] Verify bound source/runtime assets and exact native/CUDA-graph action equality on eight real observations, with seeds 42/43/44 and forward/reverse order.
- [x] Run unscored native and 1/2/4-worker CUDA-graph calibration; choose the fastest passing profile with measured GPU memory margin.
- [x] Start the persistent full evaluation and verify scored progress: seed42, NFE10, ten tasks, 100 episodes per task/setting, clean plus randomized, 2,000 total.
- [ ] Automatically audit all 2,000 episode artifacts and write final Markdown/CSV/JSON comparisons against dense-s30, original 4/4, and 4/2.

The model and benchmark source are reused unchanged at evaluator commit 7a2181d0ecc8e30d64806e94535b347bc9e1da2e. Model-specific run paths, release hashes, labels and native-loop assertions are the only launcher configuration changes. No development branch or model implementation changes are required.
