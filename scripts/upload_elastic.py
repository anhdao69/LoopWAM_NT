#!/usr/bin/env python3
"""Idempotent, verified publication of elastic checkpoints to anhdao69/LoopWAM_NT.

Files go to libero-all-suites/Elastic/<RUN>/. Never uploads latest.pt (optimizer/RNG), recovery
copies or credentials. Each checkpoint gets a local receipt after the remote size and LFS
SHA-256 match the local file.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from huggingface_hub import HfApi

REPO="anhdao69/LoopWAM_NT"
PREFIX="libero-all-suites/Elastic"
LABELS=("step_00002000","epoch_08","epoch_09","epoch_10")
METADATA=["config.json","schedule_sequence.json","dataset_stats.json","data/data_manifest.json","data/dataset_stats.json"]

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): digest.update(b)
    return digest.hexdigest()

def remote_file(api,path):
    info=api.model_info(REPO,files_metadata=True)
    return info,next((x for x in info.siblings if x.rfilename==path),None)

def upload_once(directory,run,api):
    directory=Path(directory)
    for label in LABELS:
        path=directory/(label+".pt")
        marker=directory/(path.name+".uploaded.json")
        if not path.exists() or marker.exists(): continue
        metadata=[m for m in METADATA if (directory/m).is_file()]
        if not all((directory/m).is_file() for m in ("config.json","dataset_stats.json","data/data_manifest.json")):
            raise ValueError("Incomplete export metadata")
        digest=sha256(path)
        api.upload_folder(repo_id=REPO,folder_path=str(directory),path_in_repo=f"{PREFIX}/{run}",
            allow_patterns=[path.name,*metadata],commit_message=f"Elastic {run}: {label} and configuration")
        info,remote=remote_file(api,f"{PREFIX}/{run}/{path.name}")
        if remote is None or remote.size!=path.stat().st_size: raise ValueError("Uploaded checkpoint size mismatch")
        if remote.lfs is None or remote.lfs.sha256!=digest: raise ValueError("Uploaded checkpoint checksum mismatch")
        record=dict(repo=REPO,path=f"{PREFIX}/{run}/{path.name}",sha256=digest,size=path.stat().st_size,
                    revision=info.sha,verified_utc=time.time())
        temporary=marker.with_suffix(".tmp")
        temporary.write_text(json.dumps(record,indent=2)); temporary.replace(marker)
        (directory/(path.name+".upload_pending")).unlink(missing_ok=True)
        print(json.dumps(record),flush=True)

def upload_extra(directory,run,api,files):
    """Final run records (manifest, metrics, receipts) after training completes."""
    directory=Path(directory)
    present=[f for f in files if (directory/f).is_file()]
    api.upload_folder(repo_id=REPO,folder_path=str(directory),path_in_repo=f"{PREFIX}/{run}",
        allow_patterns=present,commit_message=f"Elastic {run}: run manifest and training records")
    for f in present:
        _,remote=remote_file(api,f"{PREFIX}/{run}/{f}")
        if remote is None or remote.size!=(directory/f).stat().st_size: raise ValueError(f"Remote {f} differs")
    print(json.dumps(dict(event="records_uploaded",run=run,files=present)),flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",required=True); p.add_argument("--run-name",required=True)
    p.add_argument("--token-file",required=True); p.add_argument("--watch",action="store_true")
    p.add_argument("--attempts",type=int,default=12); p.add_argument("--extra",nargs="*",default=[])
    a=p.parse_args(); failures=0
    api=HfApi(token=Path(a.token_file).read_text().strip())
    while True:
        try:
            upload_once(a.output_dir,a.run_name,api)
            if a.extra: upload_extra(a.output_dir,a.run_name,api,a.extra)
            failures=0
            if not a.watch: return
            if all((Path(a.output_dir)/f"{label}.pt.uploaded.json").exists() for label in LABELS): return
            time.sleep(120)
        except Exception as e:
            failures+=1
            # Type only: HTTP exceptions may contain sensitive request details.
            print(json.dumps(dict(event="upload_retry",error_type=type(e).__name__,attempt=failures)),flush=True)
            if failures>=a.attempts: raise RuntimeError("Upload retries exhausted") from None
            time.sleep(min(300,5*2**min(failures,6)))

if __name__=="__main__": main()
