"""Durable child-stage status, with an exclusive lock surviving controller restarts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def run(command_file,status_file):
    command_file,status_file=Path(command_file),Path(status_file)
    lock=status_file.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    command=json.loads(command_file.read_text())
    status=dict(state='running',pid=os.getpid(),step_id=os.environ.get('SLURM_STEP_ID'),command=command,started=time.time())
    write(status_file,status)
    code=subprocess.run(command).returncode
    status.update(state='finished',exit_code=code,finished=time.time());write(status_file,status)
    return code

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--command',required=True);p.add_argument('--status',required=True);a=p.parse_args()
    raise SystemExit(run(a.command,a.status))
