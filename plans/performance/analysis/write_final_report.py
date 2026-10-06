"""Render the audited final comparison from archived JSON/CSV records."""
from pathlib import Path
import datetime as dt
import json
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[3]
E=ROOT/'plans/evidence/final_results_20261006'
read=lambda p:json.loads(p.read_text())
a=read(E/'analysis.json');audit=read(E/'remote_audit.json');models=a['models']
R=E/'raw';link='../evidence/final_results_20261006'

def hms(seconds):
 n=round(seconds);return f'{n//3600}h {(n%3600)//60:02d}m {n%60:02d}s'

def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def stages(path):
 result={}
 for line in path.read_text().splitlines():
  try:d=json.loads(line)
  except ValueError:continue
  if 'stage' in d:result[d['stage']]=d
 return result

v1=stages(R/'v1_slurm_4691.log');v0=stages(R/'v0_auto_eval_job4689/watcher.log')
dense=read(R/'dense_v2_job4659_20261006/pipeline_status.json')
train_command={'v0':30681,'v1':v1['inference']['elapsed_seconds']-v1['training']['elapsed_seconds'],
 'dense_s12':dense['stage_durations_seconds']['dense_s12_training'],'v2':dense['stage_durations_seconds']['v2_training']}
eval_time={'v0':v0['evaluation_verified']['elapsed_seconds']-v0['evaluating']['elapsed_seconds'],
 'v1':v1['complete']['elapsed_seconds']-v1['inference']['elapsed_seconds'],
 'dense_s12':dense['stage_durations_seconds']['dense_s12_evaluation'],'v2':dense['stage_durations_seconds']['v2_evaluation']}
for v in models:models[v].update(training_command_seconds=train_command[v],evaluation_wall_seconds=eval_time[v])
(E/'runtime_summary.json').write_text(json.dumps(models,indent=2))
summary=[];runtime=[];config=[];loss=[];offline=[];checkpoints=[];timeline=[]
for v,r in models.items():
 ci=r['wilson_95'];name=r['label']
 summary.append([name,f"**{r['successes']}/100 ({r['success_rate']:.0%})**",f'{ci[0]:.1%}–{ci[1]:.1%}',r['loops'],r['gpu_count']])
 runtime.append([name,hms(r['train_seconds']),hms(train_command[v]),hms(eval_time[v]),f"{r['training_gpu_hours']:.2f}"])
 config.append([name,{'ddp':'DDP','zero1':'ZeRO-1'}.get(r['backend'],r['backend']),r['microbatch'],r['accumulation'],f"{r['warm_full_update_mean_seconds']:.3f}",f"{r['peak_allocated_decimal_gb']:.2f} / {r['peak_reserved_decimal_gb']:.2f}"])
 loss.append([name,f"{r['first100_video']:.4f} → {r['last100_video']:.4f}",f"{r['first100_action']:.4f} → {r['last100_action']:.4f}",f"{r['final_exit_video']:.4f} / {r['final_exit_action']:.4f}"])
 offline.append([name,f"{r['first_validation_mse']:.5f} → {r['final_validation_mse']:.5f}",f"{r['mean_episode_seconds']:.2f}",f"{r['mean_episode_policy_steps']:.2f}",r['failures_timeout']])
 checkpoints.append([name,r['training_revision'][:12],r['evaluation_revision'][:12],f"`{r['checkpoint_sha256']}`"])
 x=audit['runs'][v];finish=dt.datetime.fromisoformat(x['evaluation_summary_mtime_utc']);trainfinish=dt.datetime.fromisoformat(x['training_timing_mtime_utc'])
 timeline.append([name,trainfinish.strftime('%H:%M:%S'),finish.strftime('%H:%M:%S'),finish.astimezone(ZoneInfo('America/New_York')).strftime('%I:%M:%S %p')])

tasks=[[t['task_id'],t['task'],t['v0'],t['v1'],t['dense_s12'],t['v2']] for t in a['tasks']]
paired=[[p['model_a'],p['model_b'],p['both_success'],p['only_a_success'],p['only_b_success'],p['both_fail']] for p in a['paired_outcomes']]

text=f'''# LoopWAM final results audit — LIBERO-Long, October 6, 2026

Audit timestamp: **{audit['audited_at_utc']}**. All four production runs and their final evaluations are complete. This report supersedes the earlier runtime estimates.

## 1. Main results

{table(['Model','Simulator success','Approx. 95% Wilson interval','Core traversals K','H100 GPUs'],summary)}

**v0 is the strongest result in this experiment:** +15 percentage points over Dense-S12 and +14 over v1. v2 equals Dense-S12 at81% and trails v1 by one episode. The present run provides evidence that recurrence with final-exit supervision can help WAM at the same parameter count. It does **not** show an improvement from adding all-exit supervision or core XSA under these settings.

All models contain **584,536,135 trainable policy parameters**. Dense-S12 executes3 pre +6 middle +3 post block pairs once, for12 effective pairs. v0/v1/v2 reuse the same six middle pairs four times, for30 effective pairs. v0 uses final-exit supervision; v1 adds joint all-exit supervision; v2 additionally applies XSA in both core experts. For v1/v2, the final exit receives weight0.5 and each earlier exit receives1/6.

![Results and actual runtime]({link}/figures/results_and_runtime.png)

The Wilson intervals summarize100 rollout outcomes per checkpoint. There is only **one training seed (42)**, with10 fixed initial states per task. Episodes share task structure; these approximate intervals do not measure variability across training seeds, and are not a claim of definitive statistical superiority across all settings.

## 2. Actual runtime and GPU usage

{table(['Model','Training loop wall time','Whole training command','100-rollout evaluation','Training GPU-hours'],runtime)}

Definitions:

- Training loop wall time comes from final `timing.json`: timed training execution including epoch validation and checkpoint work after the training timer starts. It excludes initial model/setup work.
- Whole training command includes initialization and process teardown. v0 uses Slurm step accounting; v1 uses parent stage timestamps; dense/v2 use recorded pipeline stage durations.
- Evaluation wall time is the complete parallel evaluation stage, including setup and video output. v0 includes its brief result verification before cancellation.
- GPU-hours = training loop hours × allocated training GPUs. This is a resource-use accounting quantity, not a hardware-independent FLOP measurement or monetary cost. Benchmarking, earlier abandoned runs and evaluation GPU time are excluded from that column.

**Dense → evaluation → v2 → evaluation finished in {hms(dense['elapsed_seconds'])}.** v1's entire Slurm job, including benchmark/preflight/training/evaluation, took11h28m. The final fresh v0 training start through evaluation and cancellation took9h15m37s; its older allocation lifetime also includes setup, benchmarking and an earlier stopped run.

{table(['Model','Backend','Batch/GPU','Accumulation','Warm full update mean (s)','Peak allocated / reserved GB per GPU'],config)}

Every run uses global batch128 and FP32 policy/master weights and optimizer moments with BF16 compute. Warm update averages use epochs2–10 and full128-window groups. Memory is framework-reported decimal GB; H10080GB capacity naming uses a different convention, so reserved82.71 decimal GB is approximately77.03GiB and fits the device.

Dense's warm update median was {models['dense_s12']['warm_full_update_median_seconds']:.3f}s, below its {models['dense_s12']['warm_full_update_mean_seconds']:.3f}s mean, showing substantial timing variability. Its short winning benchmark of0.893s/update was optimistic relative to the full run. v2's full warm rate of2.581s/update closely matched the2.561s benchmark.

v0/v1 used two GPUs; dense/v2 used four. v2 also reused the fully populated **preprocessing cache** from dense while initializing its policy and optimizer afresh. The other runs populated their caches during epoch1. Consequently, raw wall-time differences must not be attributed solely to model architecture. v1 took {(models['v1']['train_seconds']/models['v0']['train_seconds']-1)*100:.1f}% longer than v0 on the same two-GPU configuration. v2's shorter wall time than v1 came with approximately the same training GPU-hours.

### Completion times

All dates below are **October6,2026**. File timestamp completion markers can precede process teardown by a few seconds.

{table(['Model','Training finished UTC','Evaluation finished UTC','Evaluation finished EDT'],timeline)}

## 3. Per-task simulator results

Each entry is successful episodes out of10. Task IDs follow the installed `libero_10` suite ordering.

{table(['ID','Task','v0','v1','Dense-S12','v2'],tasks)}

![Per-task success]({link}/figures/per_task_success.png)

Notable differences:

- v0 achieved10/10 on seven tasks; its four failures were task0 once, task4 once and task8 twice.
- v1's largest weakness was task6 (5/10), followed by task7 (6/10).
- v2 recovered task6 to9/10 compared with v1, but dropped task8 to6/10 and task9 to7/10.
- Every model achieved10/10 on task5 (book into the caddy).
- Dense-S12 and v2 have the same overall score but different episode outcomes and task strengths.

### Matched initial-state outcomes

The four models used the same task/episode pairs, initial-state indices and evaluation seed formula. This permits descriptive pairing of rollout outcomes:

{table(['A','B','Both succeed','Only A succeeds','Only B succeeds','Both fail'],paired)}

For v0 versus dense, v0 uniquely solved16 episodes and dense uniquely solved1. For v1 versus v2, the14-versus-13 discordant outcomes underline how small the aggregate difference is. These are checkpoint comparisons, not repeated-training-seed experiments.

## 4. Loss trends and held-out diagnostics

The numbers below are **real-window-weighted means over the first/last100 updates**, avoiding conclusions based on one noisy batch. Main objective losses for v1/v2 combine exits; the final column reports each model's final exit separately.

{table(['Model','Video objective: first 100 → last100','Action objective: first 100 → last100','Final-exit last100 video / action'],loss)}

All28,000+ production updates were checked; precisely **29,000 updates total** are present, ordered without missing or duplicate update indices. Video/action losses and gradient norms are finite throughout all four runs. Both objectives decrease substantially for every model, although individual updates fluctuate with diffusion noise, timesteps and sample difficulty.

![Learning curves]({link}/figures/learning_curves.png)

{table(['Model','Held-out action MSE: epoch1 → epoch10','Mean rollout duration (s)','Mean policy steps','Failed rollouts hitting timeout'],offline)}

Held-out MSE evaluates only **eight fixed windows** from the held-out split, using ten denoising steps and normalized actions. It is a small open-loop diagnostic. v2 obtains the lowest final MSE here but only81% simulator success, while v0 reaches96%. Therefore this diagnostic is not a reliable standalone model-selection metric for closed-loop performance.

Episode duration includes policy calls and simulator/rendering work, but excludes video encoding and some setup. Policies also terminate at different steps. These means are **not** isolated neural-network inference latency measurements.

Every unsuccessful rollout reached the700-policy-step cap. There were no successes during settling alone, so the scores were not inflated by already-complete initial states. The episode logs do not identify why a manipulation failed; attributing failures to grasping, planning or perception requires reviewing the corresponding videos. All60 failed episodes are indexed in [failed_episodes.csv]({link}/failed_episodes.csv).

### Tail-batch logging caveat

Each epoch contains724 full128-window updates and a final **6-window** update. The training loss and gradient normalization account for real windows correctly. However, hidden-state RMS diagnostics are averaged across physical microbatches and subsequently scaled by the real-window denominator, which inflates them on padded tail batches. For example, dense's final logged RMS is much larger than its preceding full-batch RMS (video0.951/action0.682). This is a diagnostic aggregation artifact; the report does not treat it as evidence of activation explosion. A future logging cleanup should mask or separately aggregate RMS diagnostics.

## 5. Data, initialization and evaluation protocol

- Dataset: local LIBERO-Long `libero_10_no_noops_lerobot` export, **388 available demonstrations**, split into344 training and44 validation episodes by the recorded deterministic task-stratified procedure. This is not the complete500-demo distribution.
- Each epoch:92,678 training windows; validation split:11,602 windows. All runs completed10 epochs,7,250 optimizer updates and926,780 real training windows.
- Fresh initialization: same canonical compact donor artifact from pretrained `Wan-AI/Wan2.1-T2V-1.3B`; no resumed optimizer or smoke checkpoint. Random new action/proprioception heads use seed42.
- Same normalization hash, VAE hash, dataset content/split and text-cache assets across all four models. Pre-existing text caches do not embed their encoder hash; the earlier source provenance audit is recorded in the manifests, but that cache metadata limitation remains.
- Evaluation:100 episodes/model,10/task, distinct initial-state indices0–9; seed formula42 + task_id×100000 + episode_index×1000 + replan_index.
- Protocol:700 policy steps maximum,30 settling steps,32-action predictions, execute10 before replanning,10 denoising steps, CFG1, full trained depth (K1 dense; K4 looped), the training normalizer, and saved videos.
- Evaluator revisions differ for v1 versus the later generalized evaluator. The reviewed diff changes architecture/version handling and validation metadata; the simulator episode loop, observation/action conversion and termination logic remain the same.

There is no separately trained Dense-S30 or original published FastWAM baseline in this comparison. The current scores cannot establish compression-versus-quality claims against those unmeasured controls or be equated directly with published500-demo benchmark results. Although the scalar seed is shared, changing GPU count/microbatch changes random-number consumption and noise assignment; the v1/v2 comparison does not isolate XSA as strictly as a matched-layout repeated-seed study would.

## 6. Integrity and operational audit

Verified directly on the server, then checked against the archived results:

1. Complete fresh training manifests and final checkpoint contracts for all four models.
2. Recomputed SHA-256 of each final checkpoint matches the checkpoint evaluated.
3. Exactly100 unique task/episode pairs/model, ten per task, matching summary, rank and individual episode files; successes independently recounted.
4. All 400 videos are present and nonempty. `ffprobe` reports readable video streams and frame counts equal to policy steps + settling steps + initial frame. SHA-256 inventories are archived.
5. Slurm production steps/jobs completed with exit0:0. v0 allocation4689 was deliberately cancelled **after** successful evaluation by the authorized watcher; its interactive shell's cancellation signal is expected.

{table(['Model','Training source','Evaluation source','Final checkpoint SHA-256'],checkpoints)}

**v1 status-file artifact:** `job_status.json` retains `exit_code:1` from its deliberately rejected microbatch16 CUDA-OOM benchmark. The parent log traces that value through later stages without clearing it. Final `stage:complete`, Slurm exit0:0 and the verified complete results establish successful production. This is stale status metadata, not a failed final run; source evidence is preserved unedited.

**Current resources at audit:** v0 allocation4689 has been released; v1 batch4691 ended normally; dense/v2 step4659.17 completed. The parent interactive allocation4659 still exists even though this pipeline is finished. `smv-next`4690 remains pending for resources. This audit does not cancel allocation4659.

## 7. Interpretation and next experiments

1. **Use v0 as the current best trained checkpoint** for this LIBERO-Long setup. It has the highest measured success and lower training GPU-hours than v1/v2.
2. The main positive finding for LoopWAM is v0's96% versus dense's81% at the same584.5M parameters, with additional recurrent computation. Reproduce this across multiple seeds and a matched GPU/backend setup before making a broad claim.
3. All-exit supervision did not improve the final K4 score here. Evaluate the trained intermediate exits in a separate evaluation protocol to determine whether v1/v2 offer useful accuracy/latency trade-offs; no such early-exit results are claimed in this report.
4. The plan's video-only/action-only XSA ablations would help locate v2's regressions. The one-episode v1/v2 aggregate difference is too small to support a mechanism-level conclusion from one run.
5. Add a Dense-S30 control if the intended claim includes matching a larger untied model. Keep dataset size, initialization, normalization, simulator protocol and seed comparisons explicit.

## 8. Local and GitHub deliverables

The report, figures, derived CSV tables, source analysis scripts, raw production metrics/manifests/logs, all 400 episode records, and video SHA-256 inventories are archived under [final_results_20261006]({link}). **All 400 videos remain on the H100 server, as requested by the user; they are not copied into Git.** Episode CSVs contain their absolute server paths. The original multi-gigabyte training checkpoints and optimizer shards also remain on the H100 filesystem at the run paths recorded in [remote_audit.json]({link}/remote_audit.json); they are not duplicated into Git.

Useful files:

- [Model summary]({link}/models.csv), [runtime summary]({link}/runtime_summary.json), [epoch metrics]({link}/epochs.csv).
- [Per-task scores]({link}/tasks.csv), [all episodes and server video paths]({link}/episodes.csv), [failed episodes]({link}/failed_episodes.csv), [matched outcomes]({link}/paired_outcomes.csv).
- [Server integrity audit and video hashes]({link}/remote_audit.json).
- Reproduce tables/plots with `python plans/performance/analysis/summarize_final_results.py`; render this report with `python plans/performance/analysis/write_final_report.py` after the archived files are present.
'''
# Keep human-facing numeric prose readable without changing paths or tables.
for old,new in [('at81%','at 81%'),('executes3','executes 3'),('+6','+ 6'),('+3','+ 3'),('for12','for 12'),('for30','for 30'),('weight0.5','weight 0.5'),('receives1/6','receives 1/6'),('summarize100','summarize 100'),('with10','with 10'),('took11','took 11'),('took9','took 9'),('batch128','batch 128'),('epochs2','epochs 2'),('full128','full 128'),('H10080GB','H100 80GB'),('reserved82.71','reserved 82.71'),('approximately77.03GiB','approximately 77.03 GiB'),('of0.893','of 0.893'),('of2.581','of 2.581'),('the2.561','the 2.561'),('epoch1','epoch 1'),('October6,2026','October 6, 2026'),('of10','of 10'),('achieved10','achieved 10'),('task0','task 0'),('task4','task 4'),('task8','task 8'),('task6','task 6'),('task7','task 7'),('task9','task 9'),('task5','task 5'),('to9/10','to 9/10'),('to6/10','to 6/10'),('to7/10','to 7/10'),('solved16','solved 16'),('solved1','solved 1'),('the14','the 14'),('last100','last 100'),('All28,000+ production updates were checked; precisely','All production updates were checked; precisely'),('only81%','only 81%'),('reaches96%','reaches 96%'),('the700','the 700'),('All60','All 60'),('contains724','contains 724'),('full128','full 128'),('video0.951/action0.682','video 0.951/action 0.682'),('into344','into 344'),('and44','and 44'),('complete500','complete 500'),('epoch:92','epoch: 92'),('split:11','split: 11'),('completed10','completed 10'),('epochs,7','epochs, 7'),('and926','and 926'),('seed42','seed 42'),('Evaluation:100','Evaluation: 100'),('model,10','model, 10'),('indices0','indices 0'),('formula42','formula 42'),('Protocol:700','Protocol: 700'),('maximum,30','maximum, 30'),('steps,32','steps, 32'),('execute10','execute 10'),('replanning,10','replanning, 10'),('steps, CFG1','steps, CFG 1'),('(K1 dense; K4','(K=1 dense; K=4'),('published500','published 500'),('Exactly100','Exactly 100'),('All 400','All 400'),('exit0:0','exit 0:0'),('allocation4689','allocation 4689'),('microbatch16','microbatch 16'),('batch4691','batch 4691'),('step4659.17','step 4659.17'),('allocation4659','allocation 4659'),('`smv-next`4690','`smv-next` 4690'),("v0's96%","v0's 96%"),("dense's81%","dense's 81%"),('same584.5M','same 584.5M'),('final K4','final K=4')]:text=text.replace(old,new)
(ROOT/'plans/performance/LoopWAM_Final_Results_20261006.md').write_text(text)
print('Report written')
