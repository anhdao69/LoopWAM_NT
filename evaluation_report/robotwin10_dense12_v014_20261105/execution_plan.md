# Dense-s12 and v0 1/4 RoboTwin-10 evaluation plan

**Goal:** Download both requested releases and launch careful, independently audited evaluations on idle machines.

**Architecture:** Reuse the frozen evaluator and previously validated host environments. Create isolated run directories, fresh checkpoint/source/runtime checks, and persistent queues. Dense-s12 runs locally; v0 1/4 runs on 10.148.182.137. Both 96 GB GPUs were idle at admission. Keep the idle 48 GB server available.

**Protocol:** Seed 42, NFE 10, 10 release tasks, clean plus randomized, 100 accepted episodes per task/setting, 2,000 scored episodes per model. Preserve original image preprocessing, unseen instruction generation, normalization, horizons, RT rendering and acceptance criteria. Calibrate native 4 and graph 4/8/12/10 workers independently; exclude calibration from SR. No score-based model/seed selection.

- [x] Inspect three machines and pin both releases to HF revision dbb527e8f83fbf00fcbbc017f886122d5f4ec4df.
- [x] Download every release file; verify HF LFS/Git blob hashes and SHA256SUMS; compare training source and task/data contract.
- [x] Configure isolated queues and reports using previous local/remote templates. Verify strict native dense-s12 and v0 1/4 architectures, checkpoint step/epoch, VAE, statistics and finite weights.
- [x] Run the existing 38 tests freshly on each host; verify graph outputs exactly against native inference on real inputs. Runtime hashes are checked again after the reviewed calibration changes.
- [ ] Calibrate native 4 and graph 4/8/12/10 workers across all ten tasks with measured memory admission. Launch both scored evaluations and verify their first progress and manifests.
- [ ] Keep remote reports synchronized locally; automatically audit all episode artifacts and publish final comparisons after completion.

**Review focus:** Correct model architecture (dense-s12 is 12 blocks without recurrent looping); distinct paths and provenance; unchanged benchmark protocol; GPU headroom under changing load; correct final-report labels and portable remote paths.

**Operational ruling:** This configures established evaluation scripts rather than modifying model or benchmark behavior. Existing isolated frozen checkout is retained. The user already authorized downloads and execution; no approval pause or repository commit is needed.

**Speed steering:** User requested maximizing speed during initial preflight. Expand the sweep to all-task 20-episode profiles with up to 12 workers; retain the original scored protocol and archive superseded calibration. Eleven memory/selection unit tests passed on both hosts; review findings resolved.
