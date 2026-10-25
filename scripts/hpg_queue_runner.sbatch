#!/usr/bin/env bash
#SBATCH --job-name=loopwam-queue
#SBATCH --partition=hpg-b200
#SBATCH --account=yu954774.ucf
#SBATCH --qos=yu954774.ucf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=4-00:00:00
# One long 2-GPU allocation that runs queued stage scripts in order, so setup, preflight and production all happen
# inside the same job.  sbatch --output=QUEUE/slurm_%j.out --export=ALL,QUEUE=DIR scripts/hpg_queue_runner.sbatch
#   QUEUE/NN_name.sh   item, run with bash in name order; output to NN_name.log, exit code to NN_name.done/.failed
#   QUEUE/HALT         written after a failed item; nothing else runs until it is removed
#   QUEUE/STOP         ends the job once the running item finishes
# Items are picked up when present; the job idles (polling) otherwise. The cluster cancels jobs whose GPUs sit idle
# for 60 minutes, so queue the next GPU item promptly.
set -uo pipefail
: "${QUEUE:?set QUEUE}"
mkdir -p "$QUEUE"
say() { echo "[$(date -Is)] $*" | tee -a "$QUEUE/runner.log"; }
say "job $SLURM_JOB_ID on $(hostname), gpus ${CUDA_VISIBLE_DEVICES:-?}"
echo "$SLURM_JOB_ID" > "$QUEUE/JOB"
while [[ ! -e "$QUEUE/STOP" ]]; do
  item=""
  if [[ ! -e "$QUEUE/HALT" ]]; then
    for f in $(ls "$QUEUE"/[0-9]*.sh 2>/dev/null | sort); do
      b=${f%.sh}; [[ -e $b.done || -e $b.failed ]] || { item=$f; break; }
    done
  fi
  if [[ -z "$item" ]]; then sleep 15; continue; fi
  b=${item%.sh}
  say "start $(basename "$item")"
  bash "$item" > "$b.log" 2>&1; st=$?
  if [[ $st -eq 0 ]]; then echo 0 > "$b.done"; say "done $(basename "$item")"
  else echo $st > "$b.failed"; touch "$QUEUE/HALT"; say "FAILED $(basename "$item") exit $st; halted"; fi
done
say "STOP"
