# LoopWAM_NT implementation and H100 run report

## Scope and status

Implements the supplied LoopWAM-S v0/v1/v2 design in FastWAM, based on `7faa71108368fbb3b6885649f112af607427a2d4`. Existing environment edits are preserved. The compact policy has **584,536,135 trainable parameters**, excluding the frozen Wan2.1 VAE. Neither a robot teacher nor a robot-trained checkpoint is used.

v0 uses final-exit flow matching. v1 supervises both experts at all four shared-coda exits with weights 1/6,1/6,1/6,1/2. v2 adds parameter-free FP32 XSA only to shared core attention. Both experts share depth schedules but retain independent parameters. Observation caches have separate entries for each virtual layer. FP32 policy/master weights and AdamW states use BF16 autocast compute.

## Hardware and data

SSH: `ssh -i /Users/hoanganh692004/.ssh/id_ed25519_vinmotion anhdh35@10.254.152.73`.
Remote checkout: `/mnt/data/vmo-ai-task/anhdh35/FastWAM`.

Interactive job 4689 on worker-2 has two free H100 80GB GPUs. Interactive job 4659 on worker-3 has four H100s occupied by SimpleMemVLN; the user explicitly requested leaving that work running. Job 4657 is unrelated batch work. Only job 4689 is used here.

Available LIBERO-Long data contains 388 demonstrations, not 500. Per-task deterministic 90/10 episode splits produce 344 training and 44 validation episodes, with 92,678 training and 11,602 validation windows. Training-only action/state min/max statistics are shared with validation. Original text padding masks are preserved. Video offsets 0,4,...,32; action offsets 0,...,31; two 224px views concatenate to 224×448.

Global 128 = 2 GPUs × microbatch 2 × 32 accumulation. Each epoch has 724 full updates and one 6-window tail update, with gradients normalized by the actual valid count. No tail windows are dropped or counted twice. Ten epochs = 7,250 optimizer updates and 926,780 real windows.

## Production launch

The full v0 run is running in **Slurm step 4689.49**, on worker-2's two H100s, from source commit `ca8df93a59a527d2cdd660e1a437e53c84d86ac2` on `LoopWAM_NT`. It starts from the original canonical Wan initialization, not a smoke checkpoint. The first full global-128 update completed with finite losses and gradients.

Output: `/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/v0_long_bs128_20261005/`.

```bash
tail -f runs/loopwam_nt/v0_long_bs128_20261005/launcher.log
cat runs/loopwam_nt/v0_long_bs128_20261005/timing.json
```

The job runs under `nohup srun` and continues after SSH disconnects while the interactive allocation remains alive. The user must keep allocation 4689 alive. SimpleMemVLN job 4659 remains running.

## Verification

**48/48 tests passed on H100**, including the two BF16 CUDA equivalence cases. Two-GPU resumed and uninterrupted update 2 produced identical reported action/video losses, window counts, and learning rate. Full-size checkpoint reconstruction without donors and held-out 10-step inference also passed. Independent review of integration and recurrence found no remaining material issue after inference/provenance fixes.

- Native 825-tensor source coverage, paired 6144-neuron FFN selection, native-source action resizing, donor selection, strict loading, parameter count and separate expert storage.
- K1/dense and K4/expanded equivalence, summed shared gradients, all-exit/truncation equivalence, coda isolation, no future/action leakage, virtual cache separation.
- Joint/cache output and gradient equivalence; independent headwise XSA reference; checkpointed forward/backward equivalence.
- Masked losses, scheduler direction, strict save/load across K1–4, fixed-noise miniature overfit, deterministic episode split and train-only normalization, distributed tail weighting.
- Real VAE output `[B,16,3,28,56]`; anchor equality with standalone current-image encoding is exact in tested BF16 runs.
- Both-GPU smoke training checks all intended gradients and finite global norms.

These establish tested implementation contracts. They do not establish closed-loop manipulation success, superiority over Dense-S12, or absence of every possible bug. Such claims require completed training and controlled rollout evaluation.

## Measured smoke timings

Each row starts from the same canonical initialization, processes 384 real windows in 3 optimizer updates, and uses two H100s/global 128/microbatch 2. No compilation or activation checkpointing. Online frozen VAE encoding is included. Steady update time is the mean of updates 2–3. Peak memory below is the reported rank0 allocated memory from these smoke runs; the final runner reports the maximum across ranks.

| Version | Steady seconds/update | Peak allocated GB | Smoke training wall seconds including final save | Projected 10 epochs |
|---|---:|---:|---:|---:|
| v0 |9.8733|25.9143|36.7150|19.8838 hours|
| v1 |11.9563|29.4490|42.4051|24.0786 hours|
| v2 |11.9635|30.5234|45.1887|24.0932 hours|

Projected durations multiply measured update time by 7,250; they are not completed ten-epoch timings and exclude setup, checkpoint/validation overhead. Larger production samples provide a better estimate.

Remote evidence: `runs/loopwam_nt/smoke_v{0,1,2}_bs128/{manifest.json,metrics.jsonl,timing.json}` and corresponding logs. Each final training run records code SHA256s, canonical initialization and VAE hashes, dataset parquet/video hashes, split metadata, text cache hashes and normalization hash.

## Reproduce

```bash
cd /mnt/data/vmo-ai-task/anhdh35/FastWAM
source scripts/h100_env.sh
# CPU acceptance suite
OMP_NUM_THREADS=2 python -m pytest tests -q
# CUDA equivalence checks, inside an idle allocation
python -m pytest tests/test_loop_mot.py -k cuda -q
# v0 training on the already available 2-GPU interactive allocation
bash scripts/launch_loopwam_slurm.sh 4689 runs/loopwam_nt/v0_long_bs128_20261005
# Other versions, with the same initialization and budget
# torchrun --standalone --nproc_per_node=2 scripts/train_loopwam.py \
#   --config configs/loopwam_s_v1.yaml --output-dir runs/your_v1_run
```

`latest.pt` includes strict policy weights, optimizer, per-rank RNG states, epoch/microbatch cursor and batch/data contract. Resume using the same run configuration and `--resume RUN/latest.pt`; the script preserves the split and skips completed windows. Weights are saved atomically every 100 updates and at epoch boundaries. Fixed held-out 10-step action-sampling MSE is recorded each epoch (8 observations); it is not a rollout success metric.

Standalone checkpoint reconstruction is available through `create_loopwam(checkpoint_path=..., vae_path=..., device=...)` without the donor artifact. The VAE remains a separate frozen asset. Inference requires current proprioception, predicts 32 actions, defaults to 10 denoiser evaluations and uses CFG 1. Use the dataset adapter's `denormalize_action` for the training normalization convention.

## Decisions and limits

- Execute the supplied detailed design directly on requested `LoopWAM_NT` branch; preserve pre-existing environment edits.
- Use only the two idle GPUs per the user's clarification.
- Replicated DDP AdamW replaces the proposed ZeRO-1 starting point because measured memory fits comfortably. FP32 optimizer/master state is explicit; no activation checkpointing needed.
- Use the actual 388-demo data and record the per-task split; no 500-demo result is implied.
- Existing T5 caches lack embedded generation provenance. Both official Wan families and the local downloaded encoder share the verified same SHA256; cache files are individually hashed. See `LoopWAM_Data_Audit.md`.
- Frozen VAE encoding remains online. Latent caching is a future throughput optimization requiring its own anchoring/cache checks.
- D12/D30 are initialization/equivalence controls, not trained baselines in this task. No scientific performance comparison is claimed.

Local timing evidence is preserved in `plans/evidence/loopwam_nt/`. Full source/data manifests and training logs remain in the remote run directories. Final ten-epoch elapsed time and control quality remain pending because training is running.

Production snapshot: update 8/7250, 1,024 windows, 9.741 seconds/update, projected 19.62 training hours. This snapshot is historical; read remote timing.json for current progress.
