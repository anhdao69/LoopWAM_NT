#!/usr/bin/env python3
"""Idempotent retryable epoch publication; never uploads optimizer/RNG/credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from huggingface_hub import HfApi

def upload_once(directory,run,api=None):
    directory=Path(directory); api=api or HfApi()
    repo="anhdao69/ResidualTrans"
    if api.model_info(repo).private: raise ValueError("ResidualTrans must be public")
    for label in ("step_00002000", "epoch_08", "epoch_09", "epoch_10"):
        path=directory/(label+".pt")
        marker=directory/(path.name+".uploaded.json")
        if not path.exists() or marker.exists(): continue
        required=[path,directory/"config.json",directory/"dataset_stats.json",directory/"data/data_manifest.json",directory/"data/dataset_stats.json"]
        if not all(p.is_file() for p in required): raise ValueError("Incomplete epoch export")
        digest=hashlib.sha256()
        with path.open("rb") as f:
            for b in iter(lambda:f.read(8*1024*1024),b""): digest.update(b)
        api.upload_folder(repo_id=repo,folder_path=str(directory),path_in_repo=run,
            allow_patterns=[path.name,"config.json","dataset_stats.json","data/data_manifest.json","data/dataset_stats.json"],
            commit_message=f"{run}: {label} and configuration")
        info=api.model_info(repo,files_metadata=True)
        remote=next(x for x in info.siblings if x.rfilename==f"{run}/{path.name}")
        if remote.size!=path.stat().st_size: raise ValueError("Uploaded checkpoint size mismatch")
        if remote.lfs and remote.lfs.sha256!=digest.hexdigest():
            raise ValueError("Uploaded checkpoint checksum mismatch")
        record=dict(repo=repo,path=f"{run}/{path.name}",sha256=digest.hexdigest(),
                    size=path.stat().st_size,revision=info.sha,verified_utc=time.time())
        temporary=marker.with_suffix(".tmp")
        temporary.write_text(json.dumps(record,indent=2)); temporary.replace(marker)
        pending=directory/(path.name+".upload_pending")
        pending.unlink(missing_ok=True)
        print(json.dumps(record),flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",required=True); p.add_argument("--run-name",required=True)
    p.add_argument("--watch",action="store_true"); p.add_argument("--attempts",type=int,default=12)
    a=p.parse_args(); failures=0
    while True:
        try:
            upload_once(a.output_dir,a.run_name)
            failures=0
            if not a.watch: return
            if all((Path(a.output_dir)/f"epoch_{e:02d}.pt.uploaded.json").exists() for e in (8,9,10)): return
            time.sleep(60)
        except Exception as e:
            failures+=1
            # Type only: HTTP exceptions may contain sensitive request details.
            print(json.dumps(dict(event="upload_retry",error_type=type(e).__name__,attempt=failures)),flush=True)
            if failures>=a.attempts: raise RuntimeError("Upload retries exhausted") from None
            time.sleep(min(300,5*2**min(failures,6)))
if __name__=="__main__": main()
