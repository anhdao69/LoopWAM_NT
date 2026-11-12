# RoboCasa-GR1 elastic student E0 (T0), 5 epochs, distilled from the LoopWAM v0 GR1 epoch-9 teacher

Run `GR1-E0`: the elastic E0 recipe (T0 transport, FLAT9 schedules) distilled for five epochs of the RoboCasa-GR1 tabletop
data from the frozen teacher [`Robocasa-GR1/epoch_09.pt`](https://huggingface.co/anhdao69/LoopWAM_NT/tree/main/Robocasa-GR1) (continuation epoch 9, step 211,644, sha256 `694a679b4c07934b57995dc3a2ab62e2f65357cea29108b109b30049531e2748`;
official eval 52.17%, 626/1200; chosen over epoch 10, 620/1200).
Training complete 2026-11-11 02:58:39 UTC; export verified.

One checkpoint serves every schedule. A schedule gives the number of core loops used at each denoising step: `[2, 2]` is
2 steps with 2 loops each; the teacher runs 10 steps with 4 loops each (`[4]x10`). Action-block passes per query are
steps x (3 entry + 6 x loops + 3 exit).

## Official evaluation of this checkpoint

Protocol (the teacher's): starVLA `examples/simBenchmarks/Robocasa_tabletop`, 24 tasks x 50 episodes, n_envs 1,
max_episode_steps 720, n_action_steps 12, state sent, env seed 0, fixed noise seed 42 per query, EGL rendering, H100.

| schedule | steps x loops | action-block passes | success |
|---|---|---|---|
| `flat_k1_n2` [1, 1] | 2 x 1 | 24 | 44.42% (533/1200) |
| teacher `epoch_09.pt`, [4]x10 | 10 x 4 | 300 | 52.17% (626/1200) |

The other FLAT9 schedules (`flat_k2_n2`, `flat_k4_n2`, `flat_k4_n5`, `flat_k1_n5`, `flat_k2_n5`, `flat_k1_n10`, `flat_k2_n10`, `flat_k4_n10`) were still being evaluated at upload time (2026-11-11 04:26 UTC).

Earlier epochs of this run under the same protocol (their exports are not published): `[1, 1]` at the end of epochs 1-4 =
39.50% (474/1200), 44.42% (533/1200), 42.08% (505/1200), 43.17% (518/1200); epoch 4 at `[2, 2]` = 51.58% (619/1200), `[4, 4]` = 52.42% (629/1200), `[4]x5` = 53.50% (642/1200).

## Files

| file | content |
|---|---|
| `epoch_05.pt` | final export, update 235,160 (end of epoch 5), format `loopwam-s-gr1-rt-v1`, sha256 `9429592750578a84896b7d6049b230d7e8cedc8c389286149cbd5c211aaf11f9` |
| `training_config.json` | run config: recipe, teacher / VAE / init hashes, arguments, code revision |
| `dataset_stats.json`, `data_manifest.json` | 29-D normalization (required at inference) and data manifest of the training data |
| `schedule_sequence.json` | the loop schedule drawn for every update |
| `training_metrics.jsonl.gz`, `training_status.json`, `launches.jsonl` | per-update metrics and run records |
| `export_verification.json` | export check: format, frozen tensors bit-equal to the FP32 teacher, student == training state, bit-equal rollouts, eval server bit-equal for all nine schedules |
| `teacher_selection.json`, `teacher_provenance.json` | teacher choice (epoch 9 vs 10) and the Hugging Face LFS hash of the teacher |
| `eval/e05_<schedule>_validation.json` | official eval validation of this checkpoint, per-task successes |
| `SHA256SUMS` | hashes of the files in this folder |

The export is self-contained: the teacher's FP32 frozen video expert, proprio encoder, architecture and loop metadata,
plus the trained student action expert and step condition (105,701,021 parameters).
339 tensors bit-equal to the FP32 teacher; 338/338 student tensors trained. Load it with
`fastwam.models.wan22.transport_inference_gr1.load_gr1_transport(path, vae_path)` (Wan2.1 VAE); the starVLA evaluation is served by
`experiments/robocasa_gr1/elastic_policy_server.py --schedule <name>`. Code: branch `elastic_gr1` of LoopWAM_NT at revision `7c5a5db`
(not on GitHub at upload time).

Recipe: T0 (no transport), scope `pc` (whole student action expert trainable), step conditioning, anchor, regime `fu`, schedules FLAT9
(flat_k{1,2,4}_n{2,5,10}: 1, 2 or 4 core loops at each of 2, 5 or 10 denoising steps) with the balanced sampler, init `common_v1`, seed 42.
GR1 adaptations of the LIBERO recipe: endpoint loss over the 12 executed actions, no binarized gripper loss.
Data: RoboCasa-GR1 tabletop (same data and normalization as the teacher), 6,020,058 windows, 5 epochs (235,160 updates, 47,032 per epoch).
Global batch 128 (4 x H100, microbatch 32). AdamW betas (0.9, 0.95), eps 1e-8; peak lr 3e-5 for the student action expert
(weight decay 0.01) and 1e-3 for the step condition (weight decay 0); 3% warmup (7,055 updates), cosine to 1% at update 235,160; grad clip 1.0.
Elapsed 63.1 h from the first launch to the last update (6 launches, each resumed from `latest.pt`); this includes the
official evaluation run with training paused after each of epochs 1-4 and a 4-hour stop in epoch 5 at update 191,359 for
diagnostic evaluations. Last 2,000 updates: total loss 0.00270 (end 0.00135, local 0.00063, anchor 0.00279),
grad norm 0.0590.
