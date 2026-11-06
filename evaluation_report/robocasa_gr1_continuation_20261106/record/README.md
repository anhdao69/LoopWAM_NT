# RoboCasa-GR1 LoopWAM v0: epochs 6-10 continuation + official eval of each epoch (automatic)

Set up 2026-11-06 (user: after the RoboTwin-10 v0 4/2 run finishes, continue GR1 for 5 more epochs and evaluate every
epoch 6-10, strictly on 4 H100s; training pauses at each epoch end for the eval; lr cosine 3e-5 -> 3e-7, no warmup).
Runs unattended in job 5758 (worker-2) on GPUs 4-7 + their NUMA-1 CPUs only. Nothing is published to HF.

- Orchestrator: tools/orchestrate.sh (2-CPU step, all steps named training_vln); idempotent, decides every phase from
  files. Keeper: tools/keeper.sh on the login node relaunches it if its heartbeat stops (max 5).
- Flow: wait for RoboTwin-10 v0-video4-action2 COMPLETE -> stage shared/latents_backup to /tmp/gr1_cont_latents_20261106
  (stage_cache.py, chunk-sha256 verified) -> gates (20-update continuation + 10-update resume, gate_check.py) ->
  for E in 6..10: train to the end of epoch E (pause with --max-updates E*23516; epoch 10 to COMPLETE; 4 attempts,
  resume from latest.pt) -> eval_epoch.sh E (official protocol, 24 tasks x 50, GPUs 4-7, EGL) -> gr1_validate.py ->
  eval/official_results_continuation.md.
- Code: /mnt/data/vmo-ai-task/anhdh35/LoopWAM_gr1cont_20261106 (branch gr1_cont_run @b748d92 = production 45ee0c07 +
  continuation mode; tests/test_gr1_continuation.py). Parent: train/production/latest.pt sha256 e9fa7008... (update 117,580).
  Eval code: LoopWAM_gr1_eval_code @d937167 (same as the epoch 4/5 evals).
- Outputs: train/continuation_e06_10/{checkpoints/epoch_06..10.pt, latest.pt, metrics.jsonl, timing.json, COMPLETE};
  eval/official_epoch06..10/ + official_epochNN_validation.json; eval/official_results_continuation.md.
- Checks before launch (gates/, logs/): 32 unit tests passed; CPU dry run (gates/dryrun.json): continuation contract ==
  parent contract except the schedule fields, first lr 3e-5, last 3e-7; EGL id k = physical GPU k for 4-7;
  eval smoke on GPUs 4-7 (gates/eval_smoke); synthetic self-tests of stage/gate/report tools (tools/selftest_tools.sh).

Status:   cat STATUS.json; tail orchestrator.log; cat ../train/continuation_e06_10/timing.json; cat keeper.log
Stop:     on the login host running keeper.sh: pkill -f continuation_20261106/tools/keeper.sh; then scancel the
          training_vln steps of job 5758 that belong to this pipeline (orchestrator first). Training resumes later from
          latest.pt by relaunching the orchestrator:
Relaunch: cd /tmp; setsid nohup srun --jobid=5758 --overlap -n1 -c2 --gres=none -J training_vln bash tools/orchestrate.sh >> logs/orchestrator.stdout 2>&1 < /dev/null &
