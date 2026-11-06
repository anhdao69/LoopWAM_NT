# RoboTwin-10 v0 4/1 evaluation plan

- [x] Pin release revision and verify native4/1 configuration and all nine training-source hashes.
- [x] Prepare isolated run directory, reusing the frozen evaluator and validated server189 environment.
- [x] Download and verify all ten release files and SHA256SUMS.
- [x] Pass 38 fresh evaluator/model tests, strict checkpoint load, VAE/stats checks, runtime integrity and exact native/CUDA-graph action equality.
- [x] Run native1 and graph1/graph2 simulator calibration; select fastest passing profile with measured memory headroom.
- [x] Start and verify scored seed42/NFE10 evaluation on ten tasks, clean/randomized, 100 episodes/task/setting (2,000 total).
- [ ] Automatically verify all episode artifacts and publish final Markdown/CSV/JSON comparisons.

The local GPU is busy and local root filesystem is full. Downloads and staging use /data14t; evaluation uses /media/newdisk on10.148.183.189, which has2.4TB available and no compute workloads at inspection. Prior runs and other jobs remain separate.
