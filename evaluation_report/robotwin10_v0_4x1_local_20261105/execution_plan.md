# Local RoboTwin-10 v0 4/1 rerun

- [x] Confirm local GPU memory, storage and pinned checkpoint; prepare isolated run directory.
- [x] Verify all nine training-source hashes and configure temporary/cache storage on /data14t because the root disk is full; Unix-socket preflight passed.
- [x] Reverify all ten release files, pass38 fresh evaluator tests, strict native4/1 loading, runtime integrity and exact native/graph action equality.
- [x] Calibrate native1 and graph1/2/4 workers under current GPU load; select fastest passing profile with memory headroom.
- [x] Start and verify scored seed42/NFE10 evaluation, all ten tasks, clean/randomized,100episodes/task/setting,2,000total.
- [ ] Automatically audit2,000episode artifacts and publish final comparisons, including the completed remote4/1 run.
