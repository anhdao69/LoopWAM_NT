#!/usr/bin/env bash
# Owns only the RT Slurm step; the user's allocation and shell remain intact.
set -euo pipefail
: "${RT_SOURCE:?source snapshot}"
: "${RT_BACKUP_JOB_ID:?dependent recovery job}"
code=0
bash "$RT_SOURCE/scripts/run_transport_job.sh" || code=$?
source "$RT_SOURCE/scripts/transport_env.sh"
if python - "$RT_OUTPUT" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1])
assert json.loads((out/"status.json").read_text())["status"]=="complete"
assert all((out/f"epoch_{e:02d}.pt.uploaded.json").exists() for e in (8,9,10))
PY
then
    scancel "$RT_BACKUP_JOB_ID"
else
    # The wrapper and its children have ended; release the queued recovery.
    scontrol update JobId="$RT_BACKUP_JOB_ID" Dependency=0
fi
exit "$code"
