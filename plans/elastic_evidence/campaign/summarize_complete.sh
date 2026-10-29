#!/usr/bin/env bash
# Writes summary.json for every condition of EVAL_CONDITIONS whose 120 task/seed groups are all scored but which has no
# summary yet (the lanes' runners summarize only when they exit, i.e. after the whole priority list). It invokes the
# same runner (scripts/evaluate_rt_matched.py) with only those conditions: with nothing pending it starts no workers,
# re-validates each checkpoint, sha256 and manifest, checks every group's identity and episode list, and writes the
# summary. CPU only (no GPU is touched). Run inside a step of an eval job, or in the job's interactive shell.
set -uo pipefail
source /groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_work/eval_env.sh
cd "$ELASTIC_ROOT"
conds=""; shas=""; declare -A seen=()
for c in $EVAL_CONDITIONS; do
  run=${c%%:*}; rest=${c#*:}; s=${rest%%:*}
  for d in "$EVAL_OUT"/final_rollout/elastic__${run}__${s}__*; do
    [ -d "$d" ] && [ ! -f "$d/summary.json" ] || continue
    [ "$(ls "$d/groups" 2>/dev/null | grep -c '\.json$')" = 120 ] || continue
    backend=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['backend'])" "$d/manifest.json")
    conds="$conds${conds:+,}$run:$s:$backend"
    if [ -z "${seen[$run]+x}" ]; then
      seen[$run]=1
      shas="$shas${shas:+,}$run=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['sha256'])" "$EVAL_RUNS/$run/epoch_10.pt.uploaded.json")"
    fi
  done
done
[ -z "$conds" ] && { echo "[$(date -Is)] nothing to summarize"; exit 0; }
echo "[$(date -Is)] summarizing $conds"
CUDA_VISIBLE_DEVICES="" nice -n 10 python scripts/evaluate_rt_matched.py --tag final_rollout \
  --key-prefix elastic --conditions "$conds" --expected-sha "$shas" --output-root "$EVAL_OUT" \
  --weights-root "$EVAL_RUNS" --vae-path "$ELASTIC_VAE" --text-cache-dir "$EVAL_TEXT" --verify-text-hash \
  --gpus 0 --workers-per-gpu 1 --no-videos 2>&1 | grep -E '"event": "(plan|summary)"|Error|Traceback|refused'
