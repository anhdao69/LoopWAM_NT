# LoopWAM LIBERO-Long data audit

Audited 2026-10-05 on `anhdh35@10.254.152.73`, repository
`/mnt/data/vmo-ai-task/anhdh35/FastWAM`. CPU inspection and decoding only;
no GPU workloads were started or existing jobs stopped. Dataset source files
and existing training artifacts were not edited.

## Available data and the missing demonstrations

Both available layouts contain **388 episodes, 104,280 frames, ten tasks,
20 FPS**. The v3 conversion did not drop the missing 112 demonstrations:
the original v2 metadata already has the same counts.

- Original v2.1: `data/libero_mujoco3.3.2/libero_10_no_noops_lerobot`.
- Converted v3.0: `data/lerobot_v30/libero_10_no_noops_lerobot`.
- Cached instructions: `data/text_embeds_cache/libero`, 40 files across all suites.

| Task | Available | Train | Validation |
|---|---:|---:|---:|
| Turn on stove, place moka pot | 43 | 38 | 5 |
| Black bowl in bottom drawer, close | 35 | 31 | 4 |
| Yellow/white mug in microwave, close | 32 | 28 | 4 |
| Both moka pots on stove | 33 | 29 | 4 |
| Alphabet soup and cream cheese in basket | 45 | 40 | 5 |
| Alphabet soup and tomato sauce in basket | 38 | 34 | 4 |
| Cream cheese and butter in basket | 46 | 41 | 5 |
| White mug left plate, yellow/white mug right | 37 | 33 | 4 |
| White mug on plate, pudding right of plate | 38 | 34 | 4 |
| Book in back compartment of caddy | 41 | 36 | 5 |
| Total | 388 | 344 | 44 |

`build_long_datasets(dataset_dir, text_cache_dir, output_dir, seed=42)` in
`src/fastwam/datasets/loopwam_long.py` sorts each task's episodes by
`SHA256(seed:task:episode_id)` and takes `floor(0.9*n)` for training.
This produces exactly 45/5 if a task has 50 demonstrations. With the present
source and seed 42 it produces **92,678 train windows and 11,602 validation
windows**. Every frame is a window anchor; episode tails remain included.
The manifest explicitly marks `complete_500_demo_set=false`.

Use v3 for faster initialization and nine-frame image queries. Both readers
were exercised successfully. A strict 450/50 experiment requires acquiring
a complete 500-demo source, independently checking preprocessing/action
conventions, and rebuilding the split and normalization. Do not duplicate
missing demonstrations or describe the present set as 500 demos.

## Alignment, padding and normalization

At anchor `t`, image queries are `t+[0,4,8,12,16,20,24,28,32]`, actions
are `t+[0,...,31]`, and state queries are `t+[0,...,32]`, with the final state
removed before returning proprioception. Episode zero's sampled timestamps
were `[0,.2,.4,.6,.8,1,1.2,1.4,1.6]`; stored action zero and timestamp zero
share the same parquet row. The reader adds no action shift. This verifies
stored alignment; establishing that the original simulator action caused
the subsequent observation still requires the original conversion/rollout
provenance.

Both 512x512 camera streams are resized to 224x224 with bilinear antialiasing,
then concatenated image-left/wrist-right, scaled to [-1,1], and returned as
`[3,9,224,448]`. Other returned tensors are actions `[32,7]`, proprioception
`[32,8]`, text `[128,4096]`, and native text mask `[128]`.

The final anchor of episode zero is frame 271, timestamp 13.55 seconds.
Its action padding mask is `[false,true x31]`; image mask is
`[false,true x8]`. Image/state queries repeat the last row. Padded delta pose
dimensions 0:6 become zero before normalization; the absolute gripper retains
its final value. Returned `proprio_is_pad` has length 32 and aligns with
returned proprioception. Decoder errors propagate instead of replacing
failed examples with random episodes.

Only the 344 training episodes contribute normalization. The new adapter
uses the stock `SingleFieldLinearNormalizer` min/max transform and clipping;
global extrema are computed from actual rows before padding. The resulting
`dataset_stats.json` SHA256 is
`78058a689c41382a2759541d5590cb3ada8c576e736285aaeebc344874366f1b` for both layouts.
The action gripper is stored in [0,1] and normalizes to [-1,1].
`dataset.denormalize_action(tensor)` is available for evaluation.

The previous remote `runs/dataset_stats.json` describes **1,712 episodes /
277,713 transitions across suites**. It is unsuitable for the held-out Long
screen. Stock splitting shuffles all suite episodes together instead of
stratifying by task. Stock `RobotVideoDataset` also writes stats to its global
work directory, even when loading a provided normalization file. The new
adapter writes exclusively under the requested output directory and refuses
to overwrite a conflicting manifest.

## Text encoder compatibility and provenance

The existing preprocessing log says Wan2.2-TI2V-5B encoder, Wan2.1-T2V-1.3B
tokenizer, BF16, context length 128. The filename suffix `wan22ti2v5b` alone
does not establish incompatibility. The official Hugging Face model APIs
identify the **same 11,361,920,418-byte UMT5 encoder** in both models:

- [Wan2.1 model metadata](https://huggingface.co/api/models/Wan-AI/Wan2.1-T2V-1.3B?blobs=true), commit `37ec512624d61f7aa208f7ea8140a131f93afc9a`.
- [Wan2.2 model metadata](https://huggingface.co/api/models/Wan-AI/Wan2.2-TI2V-5B?blobs=true), commit `921dbaf3f1674a56f47e83fb80a34bac8a8f203e`.

Both official LFS hashes, local download metadata, and a full local
`sha256sum` of `checkpoints/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth`
equal `7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d`.

Cache payloads contain only `context` and `mask`, so an existing cache file
does not cryptographically bind its generation to a particular encoder or
tokenizer. The new manifest records each used cache file's SHA256 and valid
token count, the expected shared encoder hash, and this limitation. It
preserves the cached mask; the stock wrapper replaces that mask with all
ones after zeroing padded embeddings. The first Long instruction has thirty
valid tokens. Changing the mask is an explicit correction to the report's
required text-mask contract and should be applied to matched baselines too.

## Training budget and precision audit

Stock `Wan22Trainer` estimates `epochs * ceil(ceil(N/(world*micro))/accum)`
optimizer updates, then runs a step-count loop. A supplied `max_steps`
overrides this estimate. Its epoch sampler delegates distributed padding and
end-of-loader accumulation behavior to Accelerate; exact unique sample
exposure is not recorded. It sets BF16 autocast, but this does not by itself
guarantee FP32 master parameters or Adam states; inspect actual tensors.
Stock optimizer settings match AdamW `(0.9,0.95)`, epsilon default 1e-8,
task-config weight decay .01, 5% warmup, cosine floor .01, clip 1.0.

The new `scripts/train_loopwam.py` was reviewed: `ExactDistributedBatches`
covers each real window once per epoch, appends marked dummy entries for
equal rank lengths, and gives their all-pad branch losses zero weight.
Multiplying each microbatch-mean loss by `world*microbatch/valid_global`
correctly offsets DDP's gradient averaging and the final short group.
With 92,678 windows and global batch 128, each epoch is **724 full updates
and one six-window update: 725 updates; ten passes = 7,250 updates and
926,780 real windows**. FP32 policy tensors + FP32 Adam states + BF16
autocast are declared explicitly. The runner constructs validation data;
open-loop validation evaluation still needs its own implementation.

## Verification performed

- New data tests first failed because the adapter did not exist, then passed:
  **4/4** on the remote CPU environment.
- Real v3 first train, final episode-zero, and first validation samples decoded
  and satisfied shape/timestamp/padding/mask contracts.
- Real v2 first training sample decoded with the identical split sizes and
  normalization hash.
- New initial audit artifacts reside in remote
  `runs/loopwam_data_audit_20261005/` and `runs/loopwam_data_audit_v2_20261005/`.
- Final adapter tests passed again (4/4); the final manifest including all ten
  cache hashes was built in `runs/loopwam_data_audit_provenance_20261005/`.
- A repository-wide `pytest -q` attempt stopped during unrelated collection:
  the in-progress `runs/loopwam_setup/test_loopwam_policy.py` lacked remote
  `fastwam.models.wan22.loopwam`; RoboTwin `code_gen/test_gen_code.py` required
  absent `openai`; RoboTwin `script/test_render.py` required absent `sapien`.
  This is not a claim that the complete repository test suite passes.

No VAE latent cache was found or created by this audit. Online native Wan2.1
VAE encoding and anchor equivalence remain the model runner's responsibility.
