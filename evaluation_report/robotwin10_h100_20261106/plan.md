# H100 retrains RoboTwin-10 evaluation plan

Goal: Evaluate the requested v0-video4-action4 and v0-video4-action2 robotwin-10-h100 final releases, separately from prior checkpoints, using the established seed-42 RoboTwin-10 protocol.

Operational template reuse only: do not change the frozen model, simulator, renderer or scorer. Store each run in a new bulk directory, keep the existing code and runtime, and bind its hashes. No Git merge or deployment is required.

- [x] Inspect server availability and pin HF revision bd03a4e2fbe0b4c2d653977a8f13a62b18bda234. Local 96GB free, remote189 48GB free, remote137 busy. Assign 4/4 local four workers, 4/2 remote189 two workers.
- [x] Download all release artifacts; verify LFS/Git hashes and supplied SHA256SUMS; preserve original epoch_05.pt and use a policy.pt symlink.
- [x] Prepare separate queues from tested templates, matching ten tasks, clean/randomized, 100 episodes/task/setting, seed42, NFE10, predict32/execute24, RT32/depth8/OIDN2.3.3, shared T5, BF16 compute. Validate release contract and strict native payload; do not synthesize a release config.
- [x] Verify all nine training-source hashes and inherited runtime; verify old/new normalization JSON equality despite different raw encoding. Preserve and disclose donor-artifact/data-manifest hash differences.
- [x] Run evaluator tests on each host, native/graph exact gate on each checkpoint, four simulator smoke episodes per checkpoint, renderer guard. Smoke excluded.
- [x] Review full prepared launch/finalization artifacts. Persistent queues launch only after gates. Confirm real scored progress and no renderer/resource errors.
- [x] Mirror remote reports automatically. Finalizers validate every episode/video/trace and render log before publishing SR; compare against the respective original RoboTwin-10 variant.

Review focus: new release has no config.json and uses epoch_05.pt; raw normalization hashes differ while JSON values match; remote48GB supports two workers; final SR must not include smoke/interrupted attempts; remote reports must remain observable after this session.
