# vmo_aLinh v0 4/4 evaluation deployment

Run the original RoboTwin-10 v0 4/4 epoch5 checkpoint, seed42, NFE10, ten tasks, clean/randomized,100 episodes per setting:2,000 total. Use the frozen evaluation code and copy the established PyTorch2.7.1+cu128 environment; the existing host PyTorch2.11 environment is not used for scoring.

- [x] Verify SSH, free48GB GPU, CUDA/BF16/graph/Vulkan,2.5TB writable secondary disk, and authenticated internet access.
- [x] Transfer exact weights/assets/environment/code; verify environment archive and frozen git commit; perform documented path-only relocation. Full runtime hash verification is a separate queue gate.
- [ ] Run38 existing tests, strict native loading and fresh native/graph equality gate.
- [ ] Run unscored simulator gates and choose the fastest safe1/2-worker graph profile.
- [ ] Launch2,000 scored episodes and mirror reports locally.
- [ ] Once scoring is verified on189, stop the waiting duplicate on137, as explicitly requested by the user. Preserve its files and record cancellation; do not stop the local v0 4/2 job.
- [ ] Automatically audit final artifacts and write original-local-versus-new-host comparison.

Do not change model computation, NFE, image processing, normalization, renderer or task acceptance. Original Python instruction sampling is not explicitly seeded, so exact cross-run instruction pairing is not assumed. This host has a different Blackwell GPU model/driver; record the hardware and do not claim exact cross-host episode reproducibility.
