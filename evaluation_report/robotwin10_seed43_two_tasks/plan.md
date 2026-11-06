# RoboTwin-10 two-task seed-43 rerun plan

User asked to rerun the two tasks highlighted in the comparison with a different seed and maximize inference speed. The scope is open_microwave and move_stapler_pad, both v0 4/4 and dense-s30, clean/randomized, 100 episodes per pair: 400 per model, 800 total. Seed 43 is used for scene generation and fixed-per-query CPU action noise. NFE remains 10.

- [x] Check available GPU resources and select a safe calibration range: up to 4 v0 workers and 3 dense workers.
- [x] Prepare isolated output directories and seed/task-specific controller, live report and auditor copies. Preserve all seed-42 results.
- [x] Verify both checkpoint hashes, reused native/graph validation evidence (graph gate includes seed43), 21,947 unique runtime/source paths, and baseline protocol.
- [ ] Benchmark current-checkpoint graph throughput on the requested tasks with 2 and maximum safe workers; require artifact validation and memory margin. Reuse known-safe lower concurrency only if necessary.
- [x] Start persistent queue: v0 then dense-s30; retain all smoke and scored logs separately.
- [ ] Audit 400 episodes per model and generate seed42/seed43 and cross-model comparison tables.

No model, renderer, acceptance criteria, task horizon, instruction policy or NFE changes are authorized or needed. CUDA-graph and shared T5 acceleration are reused after verifying their bound inputs. Python instruction-template RNG remains as in the original harness; actual instructions are recorded, so exact instruction pairing is not assumed. Task order starts with the longer open_microwave cells to overlap remaining short cells. Report these selected-task results separately from the full ten-task benchmark.
