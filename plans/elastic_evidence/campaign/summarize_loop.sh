#!/usr/bin/env bash
# Every 15 min: summarize newly complete conditions (summarize_complete.sh) until every condition of EVAL_CONDITIONS
# (eval_env.sh, re-read each round: 30 since v3.4 added F0-PB S8/S2) has a summary.
W=/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_work
E=/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_eval
want() { bash -c "source $W/eval_env.sh; echo \$EVAL_CONDITIONS" | wc -w; }
while [ "$(ls $E/final_rollout/*/summary.json 2>/dev/null | wc -l)" -lt "$(want)" ]; do
  timeout 900 bash $W/summarize_complete.sh | grep -v '"event": "plan"'
  sleep 900
done
echo "[$(date -Is)] all $(want) conditions summarized"
