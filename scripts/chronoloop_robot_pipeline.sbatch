#!/usr/bin/env bash
#SBATCH --job-name=chrono-pipeline
#SBATCH --partition=hpg-b200
#SBATCH --account=yu954774.ucf
#SBATCH --qos=yu954774.ucf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:USR1@900
# ChronoLoop robot sweep in one 2-GPU job on UF HiPerGator (group QOS: 2 GPUs, 16 CPUs, 125 GB in total).
# For each run in CHRONO_ORDER: train CHRONO_EPOCHS epochs (resumes from <run>/latest.pt), then evaluate the final
# checkpoint with the RoboTwin 2.0 protocol (scripts/evaluate_chronoloop_robotwin.py: every task of the subset,
# demo_clean + demo_randomized, 50 expert-checked scenes each, seed 0, unseen instructions, replan 24; every run
# after the first is evaluated on exactly the first evaluated run's scenes, see SCENE_REFERENCE), then refresh
# the comparison report. Every step is idempotent (COMPLETE / EVAL_COMPLETE markers), so a job that reaches its
# walltime stops cleanly, resubmits itself and the next job continues where it stopped.
#
#   bash scripts/chronoloop_robot_pipeline_submit.sh      (freezes a code snapshot and submits this script from it)
# Env: CHRONO_CODE (required: the snapshot), CHRONO_DATASET (robotwin5cov), CHRONO_EPOCHS (5),
#   CHRONO_ORDER ("CL-0 CL-REG CL-A ..."), CHRONO_EVAL_PER_GPU (5), CHRONO_NO_RESUBMIT=1.
#   Tests only: CHRONO_TAG (suffix for the runs/eval directories), CHRONO_EVAL_EPISODES (50),
#   CHRONO_TEST_SKIP_LATENTS=1 (skip the latent-cache step).
set -uo pipefail
CODE="${CHRONO_CODE:?set CHRONO_CODE to the frozen code snapshot (chronoloop_robot_pipeline_submit.sh does)}"
cd "$CODE" && source scripts/chrono_env_hpg.sh
DATASET=${CHRONO_DATASET:-robotwin5cov}
EPOCHS=${CHRONO_EPOCHS:-5}
read -r -a ORDER <<< "${CHRONO_ORDER:-CL-0 CL-REG CL-A CL-W2 CL-0@1 CL-A@1 CL-FRAME}"
SHARED=$CHRONO_ROBOT_SHARED/$DATASET
RUNS=$CHRONO_ROBOT_SHARED/runs/$DATASET${CHRONO_TAG:-}
EVALS=$CHRONO_ROBOT_SHARED/eval/$DATASET${CHRONO_TAG:-}
PIPE=$RUNS/pipeline
mkdir -p "$PIPE" "$EVALS"
LOG=$PIPE/job_${SLURM_JOB_ID}.log
exec >>"$LOG" 2>&1

# RoboTwin evaluation environment (code/RoboTwin_eval/SETUP_LOG.md).
EVAL_ROOT=${CHRONO_ROBOTWIN_ROOT:-$CHRONO_BLUE/code/RoboTwin_eval/RoboTwin}
EVAL_OVERLAY=${CHRONO_ROBOTWIN_OVERLAY:-$CHRONO_BLUE/envs/robotwin_overlay}
EVAL_NP1=${CHRONO_ROBOTWIN_NP1:-$CHRONO_BLUE/envs/robotwin_overlay_np1}
EVAL_GCC_LIB=${CHRONO_ROBOTWIN_GCC_LIB:-/apps/compilers/gcc/12.2.0/lib64}
EVAL_PER_GPU=${CHRONO_EVAL_PER_GPU:-5}
EVAL_EPISODES=${CHRONO_EVAL_EPISODES:-50}
# SAPIEN's Vulkan loader needs the NVIDIA ICD; HiPerGator B200 nodes ship it as nvidia_icd.x86_64.json only.
for ICD in /usr/share/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json /etc/vulkan/icd.d/nvidia_icd.json; do
  [[ -f $ICD ]] && { export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-$ICD}; break; }
done

# name | trainer flags | run directory | uncheckpointed virtual layers (largest that fits a B200 with margin)
declare -A FLAGS DIRS UNCKPT
add() { FLAGS[$1]="$2"; DIRS[$1]="$3"; UNCKPT[$1]="$4"; }
add CL-0     "--memory-tokens 0 --mem-source none --mem-write none --action-loops 4 --history-frame 0"      CL-0_no-memory_a4                       10
add CL-REG   "--memory-tokens 16 --mem-source reset --mem-write loop --action-loops 4 --history-frame 0"    CL-REG_mem16-reset-registers_a4         12
add CL-A     "--memory-tokens 16 --mem-source learned --mem-write loop --action-loops 4 --history-frame 0"  CL-A_mem16-learned-loopwrite_a4         12
add CL-W2    "--memory-tokens 16 --mem-source learned --mem-write external --action-loops 4 --history-frame 0" CL-W2_mem16-learned-external-updater_a4 12
add CL-0@1   "--memory-tokens 0 --mem-source none --mem-write none --action-loops 1 --history-frame 0"      CL-0-at1_no-memory_a1                   12
add CL-A@1   "--memory-tokens 16 --mem-source learned --mem-write loop --action-loops 1 --history-frame 0"  CL-A-at1_mem16-learned-loopwrite_a1     12
add CL-FRAME "--memory-tokens 0 --mem-source none --mem-write none --action-loops 4 --history-frame 3"      CL-FRAME_history-frame-k3_a4            8

# This job's own end time from Slurm (an exported SLURM_JOB_END_TIME may belong to the submitting allocation).
END=""; for _ in 1 2 3; do END=$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null) && [[ $END == 20* ]] && break; END=""; sleep 20; done
DEADLINE=$([[ -n $END ]] && date -d "$END" +%s || echo "${SLURM_JOB_END_TIME:-0}")
if (( DEADLINE < $(date +%s) + 3600 )); then echo "[$(date -Is)] bad job end time $DEADLINE; exiting"; exit 1; fi
STOP_AT=$((DEADLINE - 900))
STAGED=/tmp/chronoloop_${DATASET}_latents_${SLURM_JOB_ID}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_DEBUG=WARN
export TRITON_CACHE_DIR=/tmp/chrono_${SLURM_JOB_ID}_triton
TASKS=$(python -c "from fastwam.datasets.chronoloop_robot_tasks import ROBOTWIN_SUBSETS as S; print(','.join(S['$DATASET'][0]))")
say() { echo "[$(date -Is)] $*"; }
say "pipeline job $SLURM_JOB_ID on $(hostname): dataset=$DATASET epochs=$EPOCHS order=${ORDER[*]} tasks=$TASKS"
say "code=$CODE ($(cat REVISION 2>/dev/null || git rev-parse HEAD)); allocation ends $(date -d "@$DEADLINE" -Is)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

LOCK=$PIPE/.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  OWNER=$(cat "$LOCK/job" 2>/dev/null || echo none)
  if [[ "$OWNER" != "$SLURM_JOB_ID" ]] && squeue -h -j "$OWNER" -t R,CG 2>/dev/null | grep -q .; then
    say "pipeline locked by running job $OWNER; exiting"; exit 0
  fi
  say "taking over stale lock from $OWNER"; rm -rf "$LOCK"; mkdir "$LOCK" || exit 0
fi
echo "$SLURM_JOB_ID" > "$LOCK/job"
trap 'rm -rf "$STAGED"; [[ "$(cat "$LOCK/job" 2>/dev/null)" == "$SLURM_JOB_ID" ]] && rm -rf "$LOCK"' EXIT
GOT_USR1=0
trap 'say "USR1: walltime near"; GOT_USR1=1' USR1

# run CMD... in the background and wait for it; a trapped USR1 interrupts wait (status > 128), so wait again.
run_bg() {
  "$@" & local pid=$! st
  while :; do wait $pid; st=$?; if [[ $GOT_USR1 -eq 1 && $st -gt 128 ]] && kill -0 $pid 2>/dev/null; then continue; fi; break; done
  return $st
}
fail() {  # record a failed step; stop the pipeline (no resubmission) after two failures of the same step
  echo "$SLURM_JOB_ID $(date -Is)" >> "$PIPE/failures_$1"
  if [[ $(wc -l < "$PIPE/failures_$1") -ge 2 ]]; then say "step $1 failed twice; stopping the pipeline"; exit 1; fi
}
resubmit() {
  [[ -n "${CHRONO_NO_RESUBMIT:-}" ]] && { say "not resubmitting (CHRONO_NO_RESUBMIT)"; exit 0; }
  local pending; pending=$(squeue -h -u "$USER" -n "$SLURM_JOB_NAME" -t PD -o %i | head -1)
  if [[ -n "$pending" ]]; then say "continuation already pending: $pending"; exit 0; fi
  local new; new=$(env $(compgen -e | grep '^SLURM_' | sed 's/^/-u /') sbatch --parsable --job-name="$SLURM_JOB_NAME" --output="$PIPE/slurm_%j.out" --export=ALL,CHRONO_CODE="$CODE" \
    "$CODE/scripts/chronoloop_robot_pipeline.sbatch")
  say "continuation submitted: $new"; echo "$new" >> "$PIPE/continuations.txt"; exit 0
}
time_left() { echo $((STOP_AT - $(date +%s))); }

# ---------------------------------------------------------------- latent cache (built once, shared by every run)
VALID=$(python -I -c "
import numpy as np, os
p = '$SHARED/latents/valid.uint8'
print('missing' if not os.path.exists(p) else ('complete' if (np.memmap(p, mode='r', dtype=np.uint8) == 1).all() else 'partial'))")
if [[ "$VALID" != complete && -z "${CHRONO_TEST_SKIP_LATENTS:-}" ]]; then
  say "latent cache $VALID: building the missing windows"
  run_bg torchrun --standalone --nproc_per_node=2 scripts/chronoloop_robot_precompute.py --dataset "$DATASET" \
    --shared-dir "$SHARED" --workers 4 --decode-threads 1 --batch 32 || { fail latents; resubmit; }
fi

for RUN in "${ORDER[@]}"; do
  [[ -n "${FLAGS[$RUN]:-}" ]] || { say "unknown run $RUN"; exit 1; }
  OUT=$RUNS/${DIRS[$RUN]}
  CKPT=$OUT/checkpoints/epoch_$(printf %02d "$EPOCHS").pt
  EV=$EVALS/${DIRS[$RUN]}_epoch$(printf %02d "$EPOCHS")
  mkdir -p "$OUT"
  # ---------------------------------------------------------------- train
  if [[ ! -f "$OUT/COMPLETE" ]]; then
    [[ $(time_left) -lt 1800 ]] && resubmit
    say "$RUN: training (uncheckpointed layers ${UNCKPT[$RUN]})"
    run_bg torchrun --standalone --nproc_per_node=2 scripts/train_chronoloop_robot.py --dataset "$DATASET" \
      --run-name "$RUN" ${FLAGS[$RUN]} --epochs "$EPOCHS" --save-epochs "$(seq -s, 1 "$EPOCHS")" --checkpoint-every 500 \
      --text-threads 4 --uncheckpointed-layers "${UNCKPT[$RUN]}" --shared-dir "$SHARED" --stream-micro 16 \
      --stage-latents "$STAGED" --output-dir "$OUT" --deadline-unix "$DEADLINE" >> "$OUT/train_${SLURM_JOB_ID}.log" 2>&1
    ST=$?
    say "$RUN: trainer exit $ST"
    if [[ -f "$OUT/COMPLETE" ]]; then rm -f "$PIPE/failures_train_$RUN"
    elif [[ $ST -eq 0 ]]; then resubmit                  # stopped for the walltime after a checkpoint
    else fail "train_$RUN"; resubmit; fi
  fi
  [[ -f "$CKPT" ]] || { say "$RUN: $CKPT missing after COMPLETE"; exit 1; }
  # ---------------------------------------------------------------- evaluate
  if [[ ! -f "$EV/EVAL_COMPLETE" ]]; then
    [[ $(time_left) -lt 7200 ]] && resubmit             # a full evaluation takes ~1.5-3 h; start it in a fresh job
    [[ -d "$EV" ]] && mv "$EV" "${EV}.partial_$(date +%s)"   # an interrupted evaluation restarts from scratch
    ln -sfn "$CODE/experiments/robotwin/chronoloop_policy" "$EVAL_ROOT/policy/chronoloop_policy"
    cp "$CODE/third_party/RoboTwin/script/eval_policy.py" "$EVAL_ROOT/script/eval_policy.py"   # the snapshot's harness
    cp "$CODE/third_party/RoboTwin/envs/_base_task.py" "$EVAL_ROOT/envs/_base_task.py"
    # Same scenes for every run: the first completed evaluation fixes each task/setting's 50 accepted seeds and
    # every later run is evaluated on exactly those (the expert check alone can accept slightly different seeds).
    SCENES=()
    [[ -f "$EVALS/SCENE_REFERENCE" ]] && SCENES=(--seeds-from "$(cat "$EVALS/SCENE_REFERENCE")")
    say "$RUN: evaluating $CKPT ${SCENES[*]}"
    run_bg python scripts/evaluate_chronoloop_robotwin.py --checkpoint "$CKPT" --output-dir "$EV" \
      --dataset "$DATASET" --data-dir "$SHARED/data" --tasks "$TASKS" --configs demo_clean,demo_randomized \
      --episodes "$EVAL_EPISODES" --seed 0 --instruction-type unseen --replan-steps 24 --gpus 0,1 --per-gpu "$EVAL_PER_GPU" \
      --t5-gpu 0 --nice 0 --stop-unix "$STOP_AT" --robotwin-root "$EVAL_ROOT" --overlay "$EVAL_OVERLAY" \
      --numpy-overlay "$EVAL_NP1" --gcc-lib "$EVAL_GCC_LIB" "${SCENES[@]}" >> "$PIPE/eval_${SLURM_JOB_ID}.log" 2>&1
    ST=$?
    DONE=$(python -I -c "
import json
s = json.load(open('$EV/summary.json'))['finished']
n = len('$TASKS'.split(',')) * 2
print('yes' if len(s) == n and all(v['complete'] for v in s.values()) else 'no')" 2>/dev/null || echo no)
    say "$RUN: evaluation exit $ST, complete=$DONE"
    if [[ $DONE == yes ]]; then
      date -Is > "$EV/EVAL_COMPLETE"; rm -f "$PIPE/failures_eval_$RUN"
      [[ -f "$EVALS/SCENE_REFERENCE" ]] || { echo "$EV" > "$EVALS/SCENE_REFERENCE"; say "scene reference: $EV"; }
    elif [[ $(time_left) -lt 600 ]]; then resubmit
    else fail "eval_$RUN"; resubmit; fi
  fi
  # ---------------------------------------------------------------- report over every evaluated run so far
  ARGS=()
  for R in "${ORDER[@]}"; do
    E=$EVALS/${DIRS[$R]}_epoch$(printf %02d "$EPOCHS"); [[ -f "$E/EVAL_COMPLETE" ]] && ARGS+=("$R=$E")
  done
  python scripts/chronoloop_robotwin_eval_report.py "${ARGS[@]}" --tasks "$TASKS" \
    --json "$EVALS/results_epoch$(printf %02d "$EPOCHS").json" > "$EVALS/report_epoch$(printf %02d "$EPOCHS").md" \
    && say "report updated: ${ARGS[*]}"
done
say "pipeline complete"
