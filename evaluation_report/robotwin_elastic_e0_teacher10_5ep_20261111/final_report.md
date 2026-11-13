# Final RT-50 Elastic E0 comparison: [1,1], [2,2], [4,4]

Updated **2026-11-13T22:39:50+07:00** (Vietnam time, UTC+7). All three requested schedules are **complete and audited**. This report covers **E0-epoch10-teacher-5ep**, a five-epoch student distilled from the epoch-10 teacher. The earlier epoch-1 E0 appears only in the explicitly labeled historical comparison.

**[4,4] has the highest observed aggregate SR: 91.49%, versus 90.60% for [2,2] and 89.28% for [1,1].** Every score uses all 50 tasks, both clean and randomized settings, and 100 accepted episodes per task/setting: **10,000 scored episodes per schedule, 30,000 across the three separate evaluations**. All use seed 42 and NFE 2. These totals do not include smoke episodes or mix checkpoints, schedules, or seeds.

This update rechecks the records and updates reporting only. It does not launch or resume evaluation.

## Final aggregate scores

| Action schedule | NFE | Clean successes / 5,000 | Clean SR | Randomized successes / 5,000 | Randomized SR | All successes / 10,000 | Overall SR | Final audit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [1,1] | 2 | 4476 | 89.52% | 4452 | 89.04% | 8928 | 89.28% | Passed; 100/100 cells |
| [2,2] | 2 | 4529 | 90.58% | 4531 | 90.62% | 9060 | 90.60% | Passed; 100/100 cells |
| [4,4] | 2 | 4567 | 91.34% | 4582 | 91.64% | 9149 | 91.49% | Passed; 100/100 cells |

SR = successful accepted episodes / all accepted scored episodes. Each task contributes 200 episodes and each setting contributes 5,000, so the overall episode-weighted SR equals the average of the 50 per-task SRs and the mean of clean/randomized SR. A pooled SR across schedules is not a model result and is not reported.

## Changes between schedules

| Comparison | Clean change | Randomized change | Overall change | Additional successes / 10,000 | Tasks improved / tied / worse |
| --- | --- | --- | --- | --- | --- |
| [2,2] minus [1,1] | +1.06 pp | +1.58 pp | +1.32 pp | +132 | 30 / 12 / 8 |
| [4,4] minus [2,2] | +0.76 pp | +1.02 pp | +0.89 pp | +89 | 24 / 11 / 15 |
| [4,4] minus [1,1] | +1.82 pp | +2.60 pp | +2.21 pp | +221 | 33 / 7 / 10 |

Task comparisons use the clean/randomized mean for each task. The extra action loops improve the aggregate in this evaluation, but do not improve every task. These are descriptive results from one evaluation seed; there are no repeat-run error bars or claims of statistical significance.

Largest task-level changes from [1,1] to [4,4]:

| Task | [1,1] SR | [4,4] SR | Change |
| --- | --- | --- | --- |
| open_microwave | 14.50% | 31.00% | +16.50 pp |
| hanging_mug | 44.50% | 55.00% | +10.50 pp |
| pick_diverse_bottles | 76.50% | 85.50% | +9.00 pp |
| handover_block | 86.50% | 94.00% | +7.50 pp |
| turn_switch | 55.50% | 62.50% | +7.00 pp |
| put_bottles_dustbin | 95.00% | 88.50% | -6.50 pp |
| rotate_qrcode | 83.00% | 79.00% | -4.00 pp |
| adjust_bottle | 100.00% | 99.00% | -1.00 pp |
| blocks_ranking_size | 96.00% | 95.00% | -1.00 pp |
| click_alarmclock | 100.00% | 99.00% | -1.00 pp |

## All 50 tasks

Each clean/randomized entry is **successes out of 100**, numerically equal to that setting's percentage. Each task SR is successes out of 200, expressed as a percentage. No task is omitted or replaced by a better result from another run.

| Task | [1,1] C | [1,1] R | [1,1] SR | [2,2] C | [2,2] R | [2,2] SR | [4,4] C | [4,4] R | [4,4] SR | [4,4] − [1,1] pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adjust_bottle | 100 | 100 | 100.00% | 100 | 98 | 99.00% | 100 | 98 | 99.00% | -1.00 |
| beat_block_hammer | 96 | 93 | 94.50% | 97 | 93 | 95.00% | 99 | 93 | 96.00% | +1.50 |
| blocks_ranking_rgb | 100 | 99 | 99.50% | 100 | 100 | 100.00% | 99 | 99 | 99.00% | -0.50 |
| blocks_ranking_size | 95 | 97 | 96.00% | 94 | 95 | 94.50% | 97 | 93 | 95.00% | -1.00 |
| click_alarmclock | 100 | 100 | 100.00% | 100 | 100 | 100.00% | 98 | 100 | 99.00% | -1.00 |
| click_bell | 100 | 100 | 100.00% | 100 | 100 | 100.00% | 100 | 100 | 100.00% | +0.00 |
| dump_bin_bigbin | 96 | 94 | 95.00% | 99 | 94 | 96.50% | 97 | 97 | 97.00% | +2.00 |
| grab_roller | 100 | 100 | 100.00% | 100 | 100 | 100.00% | 100 | 100 | 100.00% | +0.00 |
| handover_block | 90 | 83 | 86.50% | 91 | 95 | 93.00% | 94 | 94 | 94.00% | +7.50 |
| handover_mic | 98 | 100 | 99.00% | 96 | 99 | 97.50% | 100 | 100 | 100.00% | +1.00 |
| hanging_mug | 50 | 39 | 44.50% | 51 | 40 | 45.50% | 58 | 52 | 55.00% | +10.50 |
| lift_pot | 99 | 100 | 99.50% | 100 | 100 | 100.00% | 100 | 100 | 100.00% | +0.50 |
| move_can_pot | 88 | 91 | 89.50% | 94 | 96 | 95.00% | 90 | 98 | 94.00% | +4.50 |
| move_pillbottle_pad | 94 | 96 | 95.00% | 97 | 100 | 98.50% | 99 | 100 | 99.50% | +4.50 |
| move_playingcard_away | 99 | 99 | 99.00% | 100 | 100 | 100.00% | 100 | 100 | 100.00% | +1.00 |
| move_stapler_pad | 83 | 77 | 80.00% | 90 | 80 | 85.00% | 87 | 83 | 85.00% | +5.00 |
| open_laptop | 93 | 96 | 94.50% | 91 | 97 | 94.00% | 93 | 94 | 93.50% | -1.00 |
| open_microwave | 17 | 12 | 14.50% | 17 | 15 | 16.00% | 26 | 36 | 31.00% | +16.50 |
| pick_diverse_bottles | 75 | 78 | 76.50% | 80 | 84 | 82.00% | 85 | 86 | 85.50% | +9.00 |
| pick_dual_bottles | 93 | 92 | 92.50% | 100 | 99 | 99.50% | 99 | 98 | 98.50% | +6.00 |
| place_a2b_left | 94 | 96 | 95.00% | 96 | 98 | 97.00% | 98 | 97 | 97.50% | +2.50 |
| place_a2b_right | 94 | 91 | 92.50% | 98 | 97 | 97.50% | 97 | 95 | 96.00% | +3.50 |
| place_bread_basket | 88 | 93 | 90.50% | 88 | 93 | 90.50% | 92 | 90 | 91.00% | +0.50 |
| place_bread_skillet | 91 | 87 | 89.00% | 91 | 89 | 90.00% | 93 | 93 | 93.00% | +4.00 |
| place_burger_fries | 96 | 97 | 96.50% | 96 | 97 | 96.50% | 95 | 100 | 97.50% | +1.00 |
| place_can_basket | 72 | 57 | 64.50% | 70 | 64 | 67.00% | 65 | 66 | 65.50% | +1.00 |
| place_cans_plasticbox | 98 | 98 | 98.00% | 100 | 96 | 98.00% | 100 | 95 | 97.50% | -0.50 |
| place_container_plate | 96 | 98 | 97.00% | 98 | 99 | 98.50% | 98 | 97 | 97.50% | +0.50 |
| place_dual_shoes | 85 | 81 | 83.00% | 84 | 86 | 85.00% | 88 | 91 | 89.50% | +6.50 |
| place_empty_cup | 100 | 100 | 100.00% | 100 | 100 | 100.00% | 100 | 100 | 100.00% | +0.00 |
| place_fan | 95 | 94 | 94.50% | 93 | 89 | 91.00% | 95 | 96 | 95.50% | +1.00 |
| place_mouse_pad | 90 | 89 | 89.50% | 95 | 94 | 94.50% | 94 | 96 | 95.00% | +5.50 |
| place_object_basket | 87 | 77 | 82.00% | 83 | 85 | 84.00% | 82 | 81 | 81.50% | -0.50 |
| place_object_scale | 93 | 98 | 95.50% | 98 | 96 | 97.00% | 98 | 98 | 98.00% | +2.50 |
| place_object_stand | 93 | 95 | 94.00% | 95 | 95 | 95.00% | 92 | 96 | 94.00% | +0.00 |
| place_phone_stand | 95 | 97 | 96.00% | 97 | 100 | 98.50% | 98 | 98 | 98.00% | +2.00 |
| place_shoe | 93 | 95 | 94.00% | 91 | 98 | 94.50% | 95 | 99 | 97.00% | +3.00 |
| press_stapler | 98 | 98 | 98.00% | 97 | 97 | 97.00% | 97 | 98 | 97.50% | -0.50 |
| put_bottles_dustbin | 93 | 97 | 95.00% | 92 | 94 | 93.00% | 91 | 86 | 88.50% | -6.50 |
| put_object_cabinet | 90 | 90 | 90.00% | 92 | 94 | 93.00% | 90 | 95 | 92.50% | +2.50 |
| rotate_qrcode | 85 | 81 | 83.00% | 85 | 81 | 83.00% | 80 | 78 | 79.00% | -4.00 |
| scan_object | 81 | 77 | 79.00% | 80 | 79 | 79.50% | 81 | 87 | 84.00% | +5.00 |
| shake_bottle | 100 | 100 | 100.00% | 100 | 100 | 100.00% | 100 | 100 | 100.00% | +0.00 |
| shake_bottle_horizontally | 100 | 100 | 100.00% | 100 | 100 | 100.00% | 100 | 100 | 100.00% | +0.00 |
| stack_blocks_three | 96 | 94 | 95.00% | 90 | 94 | 92.00% | 96 | 95 | 95.50% | +0.50 |
| stack_blocks_two | 100 | 100 | 100.00% | 100 | 100 | 100.00% | 100 | 100 | 100.00% | +0.00 |
| stack_bowls_three | 79 | 78 | 78.50% | 81 | 79 | 80.00% | 82 | 78 | 80.00% | +1.50 |
| stack_bowls_two | 93 | 96 | 94.50% | 95 | 94 | 94.50% | 99 | 92 | 95.50% | +1.00 |
| stamp_seal | 86 | 90 | 88.00% | 92 | 93 | 92.50% | 94 | 95 | 94.50% | +6.50 |
| turn_switch | 49 | 62 | 55.50% | 55 | 65 | 60.00% | 56 | 69 | 62.50% | +7.00 |

The complete numerical table, including historical per-task E0 comparisons, is available as [CSV](task_comparison.csv) and [JSON](final_comparison.json).

## Comparison with the earlier epoch-1 E0

The earlier E0 was student **epoch 1 / update 47,462**, distilled from the **epoch-5 / update 237,310** v0 teacher. The new release is student **epoch 5 / update 237,310**, distilled from the **epoch-10 / update 474,620** teacher. Both teacher quality and student training duration change; these differences cannot be attributed solely to training longer or to loop depth.

The two historical NFE-2 results below were freshly recomputed from their saved 20,000 episode records, checked against their CSV/JSON exports, and linked to their passed final audits through the COMPLETE.json SHA256. The historical media were not re-audited in this update.

| Schedule / NFE 2 | Old E0 clean | Old E0 randomized | Old E0 overall | New E0 overall | Overall change |
| --- | --- | --- | --- | --- | --- |
| [1,1] | 86.96% | 86.72% | 86.84% | 89.28% | +2.44 pp |
| [2,2] | Not evaluated | — | — | 90.60% | No matched baseline |
| [4,4] | 90.24% | 90.26% | 90.25% | 91.49% | +1.24 pp |

Historical sources: [E0 epoch-1 [1,1]](../robotwin_elastic_e0_20261107/flat_k1_n2/final_report.md) and [E0 epoch-1 [4,4]](../robotwin_elastic_e0_20261107/parallel137_20261108/remote_mirror/flat_k4_n2/final_report.md). There is no completed epoch-1 [2,2] baseline in this campaign; it is not a zero score or a pending result. The old `[4]×5` result uses NFE 5 and is excluded from this matched NFE-2 comparison. E1 and E2 are separate checkpoint families and are outside this report's main comparison.

## Exact checkpoint and source identity

The evaluated release is [robotwin/Elastic/E0-epoch10-teacher-5ep at the pinned revision](https://huggingface.co/anhdao69/LoopWAM_NT/tree/b72f355505d98c473fb91e45d10e16d4cc7ce536/robotwin/Elastic/E0-epoch10-teacher-5ep).

| Field | Value |
| --- | --- |
| Repository | `anhdao69/LoopWAM_NT` |
| Pinned HF revision | `b72f355505d98c473fb91e45d10e16d4cc7ce536` |
| Student file | `epoch_05.pt`, 2,340,331,885 bytes |
| Student epoch / update | 5 / 237,310 |
| Student SHA256 | `07d24ff77410e516ac90d24d76d651e2102b09b0458d9d5d86ea18dff39e823e` |
| Teacher epoch / update | 10 / 474,620 |
| Teacher SHA256 | `70cf5af73db5fe582c50c5f9fb68296d5271a9c0fea51509b3cdb5f9696be1ea` |
| Native training-source revision | `2c78ff6f186c1d9073068b8eee5b328d19dba826` |
| Native export / recipe | `loopwam-s-robot-rt-v1`; T0, no learned transport; scope `pc`, step conditioning enabled |
| Architecture metadata | v0, 3/6/3, four video loops; 14-D state/action |
| Dataset statistics file SHA256 | `7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095` |
| Older E0 student SHA256 | `abc3c0075a6e1f3ce380be8d6c7b6b9d8bb697ac21cf3f2d40a93efb821b539a` |

The release folder's “epoch10” refers to the teacher. Calling this an epoch-10 student would be incorrect. All three schedules use identical student bytes. All 13 release files were verified at download; the [download receipt](download_receipt.json) and [HF manifest](hf_manifest.json) retain their digests. The [CPU validation](cpu_validation.json) records direct equality of 339 frozen video tensors and 2 proprioception tensors to the teacher, finite FP32 student tensors, export/state agreement, and matching normalization. This update rehashed the deployed checkpoint on both hosts; it did not repeat the tensor-loading gate.

The [source provenance](source_provenance.json) records adapter changes to expose `[2,2]` and validate the new release identity. All 19 native model source files match the training revision. The frozen source manifest has 984 entries and was freshly rehashed on all three deployments. The scored manifests also agree on evaluator, policy, normalization and VAE hashes. These reports describe the code actually executed; a later checkout revision does not retroactively change it.

## Evaluation protocol and meaning of the schedules

| Setting | Applied value |
| --- | --- |
| Benchmark | Full RoboTwin 50-task roster shown above; not the RT-10 subset |
| Settings | `demo_clean` and `demo_randomized` |
| Accepted scored episodes | 100 per task/setting, 200 per task, 10,000 per schedule |
| Evaluation seed | 42 only |
| Candidate scene seed origin | `100000 × (1 + 42) = 4,300,000`; increments after attempted scenes |
| Scene admission | Original expert planning/success filter; rejected scenes do not count in the 100 accepted episodes |
| Task horizons and success | Original task-specific limits and environment success checks |
| Language | Unseen instructions; instruction pool 100; actual strings saved |
| Video loops | 4 for every schedule |
| Action loop schedules | `[1,1]`, `[2,2]`, `[4,4]` across the two action-denoising steps |
| NFE and time grid | 2; native velocity calls at 1000 and 500, integrating to 0 |
| Step conditioning | 0.5 at both steps |
| Predicted / executed action chunk | Predict 32; execute 24 before replanning |
| Action representation | Original 14-D absolute joint-position output with training z-score normalization |
| Action noise | FP32 CPU generator manually seeded to 42 afresh for every policy query |
| Image input | Released FastWAM PIL uint8 resize and head/wrist packing, tensor `[1,3,384,320]`; original BF16 image arithmetic |
| Policy precision | FP32 policy weights with BF16 autocast; native VAE behavior |
| CFG / compilation | CFG 1; `torch.compile` disabled |
| Rendering / evidence | RT shader, 32 samples/pixel, depth 8; validated OIDN compatibility; video and query logging enabled |

`[4,4]` means action-core depth 4 at each of two denoising steps. It does not mean NFE 4, four video and four action denoising steps, or an action chunk length of 4. Total action-core loop iterations across the schedule are 2, 4 and 8 respectively. The native rollout reuses video conditioning, applies the requested loop depth, and preserves the native integration rule.

The evaluation command explicitly overrides the data-collection config's episode count to **100 scored episodes**. Clean settings disable background/clutter/light/height randomization, but still evaluate seeded task scenes. Randomized settings enable background, clutter and light variation, table-height parameter 0.03, clean-background rate 0.02 and crazy-light rate 0.02. Both keep head-camera distance randomization at 0 and use the same embodiment and cameras.

### Reproducibility and actual episode matching

NumPy and Torch scene RNGs are seeded by the environment. The inherited Python `random.seed(...)` call is commented out, and instruction-template generation remains unseeded; the final instruction is selected from generated templates. The action-noise seed is a separate fixed policy seed. Thus “seed 42” does not imply that independently executed schedules receive identical scene/instruction trajectories.

Fresh comparisons of the saved records, matching task, setting and episode index, found:

| Pair | Same scene seed / 10,000 positions | Same seed and instruction / 10,000 positions | Cells with identical 100-seed sequence / 100 |
| --- | --- | --- | --- |
| [1,1] vs [2,2] | 4553 | 174 | 21 |
| [1,1] vs [4,4] | 4369 | 174 | 19 |
| [2,2] vs [4,4] | 4496 | 187 | 22 |

These counts describe matching positions, not the intersection of all seeds or a causal explanation for differences. The protocol and checkpoint are matched; individual rollouts are not perfectly paired. No paired statistical test or exact deterministic replay claim is justified by these runs. To measure variability, a separate experiment with fully controlled instruction/scene assignment and repeated seeds would be needed; none was started for this report.

## Hardware, scheduling, and measured runtime

`[1,1]` ran locally on `VMO-ANHDH35-D`; `[2,2]` and `[4,4]` ran on server137 (`10.148.182.137`, hostname `VMO-VINHLD8-D`). Both GPUs were RTX PRO 6000 Blackwell Max-Q, approximately 96 GB. Each evaluation used four simulator workers on its assigned GPU, one shared T5 service and validated CUDA graphs. `[1,1]` and `[2,2]` ran concurrently. `[4,4]` began after both passed their final audits and after the user requested server137.

The setup receipts record Python 3.10.22, PyTorch 2.7.1+cu128, NumPy 1.26.4, Pillow 12.0.0, SAPIEN 3.0.0b1, transforms3d 0.4.2, Transformers 4.49.0, Gymnasium 0.29.1 and NVIDIA driver 580.178.04 on both hosts. The score manifests leave `gpu_pci` empty; GPU identity is supported by host/setup evidence rather than a per-run PCI field. No host or package upgrade is implied by this report update.

The initially waiting local `[4,4]` supervisor was stopped before any scored episode and migrated. Its retired status records a SIGTERM/KeyboardInterrupt as an error; it is a scheduling stop, not a failed scored attempt. The accepted `[4,4]` results all come from server137. Earlier GPU waiting and migration time are separate from scored evaluation runtime.

Timestamps below are the recorded host Unix times displayed in UTC+7. The saved November 11 clock measurement was **local minus server137 = −27.915 seconds** (server137 approximately 28 seconds ahead). This table leaves the original timestamps unmodified; it does not assume that offset stayed constant until November 13. Per-run durations are calculated on one host, so a constant cross-host offset cancels out.

| Schedule | Host | Evaluator started (ICT) | Evaluator finished (ICT) | Final audit completed (ICT) | Evaluator wall time | Post-process audit/finalization | Episodes/hour |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [1,1] | local | 2026-11-11 12:19:46 | 2026-11-12 03:53:53 | 2026-11-12 03:54:25 | 15h 34m 06s | 0h 00m 33s | 642.3 |
| [2,2] | 137 | 2026-11-11 12:20:14 | 2026-11-12 03:58:53 | 2026-11-12 04:01:04 | 15h 38m 40s | 0h 02m 11s | 639.2 |
| [4,4] | 137 | 2026-11-12 10:10:53 | 2026-11-13 01:51:48 | 2026-11-13 01:56:44 | 15h 40m 56s | 0h 04m 55s | 637.7 |

Evaluator wall time includes its startup and per-cell validation, simulation, policy queries, rendering and evidence recording. The separate finalization interval is evaluator exit to final audit timestamp. Cumulative cell runtimes overlap across four workers and must not be presented as elapsed wall time. Download, source/setup gates, GPU waiting, smoke checks and the gap before `[4,4]` are outside the evaluator interval.

| Schedule | Queue launch → final audit | Sum of worker-cell durations (overlapping) | Recorded policy queries |
| --- | --- | --- | --- |
| [1,1] | 15h 38m 00s | 61h 38m 57s | 110581 |
| [2,2] | 15h 44m 11s | 61h 47m 03s | 108929 |
| [4,4] | 15h 49m 11s | 62h 01m 34s | 106218 |

The sum of the three evaluator wall intervals is **46h 53m 42s of assigned-GPU wall time**, not measured GPU kernel busy time. The actual uncorrected calendar span from the first scoring launch to the final `[4,4]` audit is **37h 36m 57s**, because the first two ran in parallel and there was a later scheduling gap. There is no remaining runtime estimate for these three completed runs.

### Speed measures and their limits

The queue admitted work with at least 60,000 MiB free, used shared T5 and four simulator workers, and enabled CUDA graphs only after exact native-output checks. Higher concurrency had previously caused renderer/memory failures, so it was not increased for these runs. Video logging, the renderer, scene admission, horizons and arithmetic remained unchanged for speed. This was the validated configuration, not proof of a globally optimal worker count.

Pre-scoring mean policy-query timings on eight saved simulator inputs, using the gate on the host assigned to each schedule:

| Schedule | Native query | CUDA graph query | Gate native/graph ratio | Max absolute output difference |
| --- | --- | --- | --- | --- |
| [1,1] | 44.25 ms | 38.34 ms | 1.15× | 0 |
| [2,2] | 49.88 ms | 39.86 ms | 1.25× | 0 |
| [4,4] | 60.67 ms | 45.15 ms | 1.34× | 0 |

These short gate timings include the inference adapter with cached T5 context and exclude simulation/planning/rendering, initial text encoding and graph capture. They were measured at different times and can be affected by contention; they are not controlled end-to-end speed benchmarks. The completed wall times are the authoritative observed throughput. More loops can also change episode success and duration, so wall-time changes cannot be interpreted as pure model-latency changes.

## Validation, exclusions, and scope of this update

Before scoring, the retained evidence records strict checkpoint/export loading, the CPU teacher/student checks, source/runtime freezes, schedule checks, native-versus-graph comparisons on eight real inputs in forward and reverse order, and smoke checks at the full four-worker concurrency. Native calls confirm depths 1/2/4 at times 1000/500 with step conditioning 0.5.

For `[1,1]` and `[2,2]`, all 16 saved cross-host action output pairs matched exactly. The `[4,4]` native-versus-graph gate passed on server137; there is **no separate `[4,4]` cross-host parity gate**, and the inherited cross-host receipt is not presented as one. There were eight unscored smoke episodes per schedule, 24 in total, excluded from the 30,000 scored episodes.

The historical test receipts record 96 scoped inference/adapter/queue/report tests on each initial host and 55 passed / 7 training-only deselected tests for the server137 `[4,4]` migration. Broader selections exposed absent training-controller helpers in this inference snapshot; this is not a claim that the complete repository training test suite passed. Review findings and their fixes remain in the setup/review records. No GPU numerical gates or simulations were rerun during this report update.

Each original finalizer rechecked the checkpoint and bound source/runtime files, reconstructed success records from logs, checked scene-seed uniqueness within a cell, validated instructions and action-query sequences, checked video existence, SHA256 and ffprobe readability, and required all 100 cells / 10,000 videos before marking a schedule complete. Every accepted scored cell is `attempt_01`; all three final audits report **0 failed attempts and 0 excluded partial scored episodes**. Smoke and the retired waiting local supervisor are separate from those scored-attempt counts.

This report's fresh, CPU-only reconciliation performed:

1. Recalculated all three new results from 30,000 episode success flags and checked all task/episode CSVs and final JSON totals. Also reconciled 20,000 historical epoch-1 baseline records.
2. Rehashed each COMPLETE.json against its saved passed audit and checked checkpoint identity, all 100 task/setting cells, unique seeds within cells, schedule/NFE/seed/chunk settings and equal denominators.
3. On the original local/server137 deployments, verified each accepted `result.json`, success log, seed, instruction, query count and action-trace digest, and confirmed all 30,000 video files remain present and nonempty. Confirmed original completion/manifest/result/audit files match the local report mirrors byte-for-byte.
4. Rehashed the deployed checkpoint and all 984 frozen source files on each of the three deployments. Checked launch, successful process exit and smoke receipts; calculated runtimes from recorded start/end times.
5. Compared all task/setting/episode positions to quantify actual scene/instruction pairing, and generated the tables from the reconciled data.

Video content hashes and ffprobe checks were **not rerun in this update**; those guarantees refer to the saved original final audits whose completion hashes were freshly verified. The current recheck also does not rehash every inherited simulator asset/package from the full runtime manifest. The distinction matters: record reconciliation, current file retention, original full artifact auditing, and deterministic replay are different claims.

## Evidence and reproduction

| Schedule | Final scored source | Protocol | Saved audit |
| --- | --- | --- | --- |
| [1,1] | [Final results](local/flat_k1_n2/final_results.json), [episodes](local/flat_k1_n2/episodes.csv) | [Manifest](local/flat_k1_n2/manifest.json) | [Audit](local/flat_k1_n2/final_audit.json) |
| [2,2] | [Final results](remote137/flat_k2_n2/final_results.json), [episodes](remote137/flat_k2_n2/episodes.csv) | [Manifest](remote137/flat_k2_n2/manifest.json) | [Audit](remote137/flat_k2_n2/final_audit.json) |
| [4,4] | [Final results](../robotwin_elastic_e0_teacher10_5ep_44_20261112/flat_k4_n2/final_results.json), [episodes](../robotwin_elastic_e0_teacher10_5ep_44_20261112/flat_k4_n2/episodes.csv) | [Manifest](../robotwin_elastic_e0_teacher10_5ep_44_20261112/flat_k4_n2/manifest.json) | [Audit](../robotwin_elastic_e0_teacher10_5ep_44_20261112/flat_k4_n2/final_audit.json) |

Fresh evidence: [local original-host recheck](final_evidence/local_recheck.json), [server137 original-host recheck](final_evidence/remote137_recheck.json), [derived full comparison and input hashes](final_comparison.json), [50-task CSV](task_comparison.csv). These contain the original launch commands/timestamps and exact paths. The record checker is [reconcile.py](final_evidence/reconcile.py); the read-only original-host checker is [recheck_host.py](final_evidence/recheck_host.py); the report renderer is [build_report.py](final_evidence/build_report.py).

The checkpoint/source/runtime, gate and review receipts are linked in the [initial setup report](setup_report.md) and the [[4,4] continuation setup](../robotwin_elastic_e0_teacher10_5ep_44_20261112/setup_report.md). The [clock-offset receipt](sync_config.json) documents the timestamp limitation.

Original bulk deployments, including per-cell logs and videos:

- Local `[1,1]`: `/data14t/anhdh35/loopwam_robotwin_elastic_e0_teacher10_5ep_20261111`.
- Server137 `[2,2]`: `/media/vmo-perception/disk_2/anhdh35/loopwam_robotwin_elastic_e0_teacher10_5ep_20261111`.
- Server137 `[4,4]`: `/media/vmo-perception/disk_2/anhdh35/loopwam_robotwin_elastic_e0_teacher10_5ep_44_20261112`.

The historical raw results, audits and frozen evaluator files were preserved. `current_results.json` in the original two-run campaign remains its historical two-schedule snapshot; use this final report and `final_comparison.json` for the combined three-schedule view.
