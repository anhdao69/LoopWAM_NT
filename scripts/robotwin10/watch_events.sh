#!/usr/bin/env bash
# Exit (to wake the monitoring session) on a notable campaign event, or after MAX seconds.
WK=/groups/yshang/an221229/checkpoints/LoopWAM_NT/robotwin10_work; R=/groups/yshang/an221229/checkpoints/LoopWAM_NT/robotwin10
MAX=${1:-7200}
snap(){ { squeue -h -u an221229 -o '%i %T' 2>/dev/null | sort; ls $R/*/train/COMPLETE $R/*/PUBLISHED $R/INDEX_PUBLISHED 2>/dev/null;
          grep -h -c "CRASHED\|killed\|cooldown\|staging not possible\|task .* exited" $WK/control/agent_*.srun.log 2>/dev/null;
          grep -c "error\|plan for" $WK/control/dispatcher_events.log 2>/dev/null; } | md5sum; }
s0=$(snap); start=$SECONDS
while [ $((SECONDS-start)) -lt $MAX ]; do sleep 60; [ "$(snap)" != "$s0" ] && { echo "EVENT $(date)"; break; }; done
date; cat $WK/control/STATUS.txt; tail -5 $WK/control/dispatcher_events.log
