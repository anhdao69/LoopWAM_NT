#!/usr/bin/env bash
# Exit (to wake the monitoring session) on a notable campaign event, or after MAX seconds.
WK=/groups/yshang/an221229/checkpoints/LoopWAM_NT/robotwin10_work; R=/groups/yshang/an221229/checkpoints/LoopWAM_NT/robotwin10
MAX=${1:-7200}
snap(){ { squeue -h -u an221229 -o '%i %T' 2>/dev/null | sort; ls $R/*/train/COMPLETE $R/*/PUBLISHED $R/INDEX_PUBLISHED 2>/dev/null;
          grep -h -c "CRASHED\|killed\|error\|staging failed" $WK/control/dispatcher_events.log $WK/control/agent_*.log 2>/dev/null;
          grep -h "CRASHED\|killed\|cooldown\|started on\|exited\|task" $WK/control/agent_*.status.json 2>/dev/null | wc -l; } | md5sum; }
s0=$(snap); start=$SECONDS
while [ $((SECONDS-start)) -lt $MAX ]; do sleep 60; [ "$(snap)" != "$s0" ] && { echo "EVENT $(date)"; break; }; done
date; cat $WK/control/STATUS.txt; tail -5 $WK/control/dispatcher_events.log
