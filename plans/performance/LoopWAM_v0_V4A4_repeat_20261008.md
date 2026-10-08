# Fresh LoopWAM v0 4/4 repeat — October 8, 2026

Verified at 2026-10-08T12:44:09.234529+00:00. Production is running in Slurm **4728.20** on worker-1, using two H100 80 GB GPUs. Source revision: `8a29ffdce1537409e2b7e975e838d990ebbe068a`, identical to the 4/2, 2/2 and 1/4 experiments.

## Matched setup

- Video/action core repetitions: **4/4**, 30 block applications per expert.
- Fresh canonical Wan donor initialization and fresh optimizer; **no resume** and no smoke checkpoint reused for production.
- LIBERO-Long, original 344-training / 44-validation demonstration split; seed 42.
- Global batch 128; 10 epochs; 7,250 updates; 926,780 real training windows.
- DDP, microbatch 8 per GPU, accumulation 8, eight data workers per rank; FP32 policy and optimizer states, BF16 compute.
- Same AdamW hyperparameters, schedule, clipping, data normalization and asset hashes as the previous variants. Frozen VAE cache reused for speed.

## Verification and speed

DDP measured 3.672 s/update, ZeRO-1 3.737 s/update, and ZeRO-2 3.854 s/update. Microbatch 16 exceeded GPU memory and was rejected. The selected DDP configuration passed a twenty-update benchmark confirmation, fresh cold/warm ten-update native training runs, and two simulator smoke episodes with videos. Data and initialization hashes were checked against the original Long v0 run.

At the production audit, **20 updates** had completed with finite losses and gradients. Mean production time: **3.681 s/update**. Estimated full training time: **7.47 hours**, plus approximately 45–90 minutes for final evaluation. Expected full pipeline completion is approximately **17:00–17:45 EDT October 8**, subject to runtime variation.

## Automatic final evaluation

After verifying all 7,250 training updates, the pipeline evaluates the final checkpoint on 100 LIBERO-Long episodes (10 tasks × 10 episodes), with the same seed, horizons, action chunk, replanning interval and denoising settings as the prior three variants. It audits checkpoint identity, unique episode coverage and nonempty videos. Videos remain on the server. The interactive allocation is not automatically cancelled.

Server output: `/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/loop_v4a4_repeat_job4728_20261008/v4a4`.

Machine-readable evidence: [LoopWAM_v0_V4A4_repeat_20261008_evidence.json](LoopWAM_v0_V4A4_repeat_20261008_evidence.json).
