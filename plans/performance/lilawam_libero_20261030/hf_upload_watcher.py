#!/usr/bin/env python
"""Upload each LiLa-WAM LIBERO suite to HF anhdao69/LoopWAM_NT under baselines/ as soon as train_job.sh reports
'<suite> stage2 rc=0 ckpt=...', then verify the HF LFS sha256 of every checkpoint against the local file.
Runs on the login node (evuser2), so it does not depend on the Slurm job or a Claude session.
After job 1094056 ends, a suite that never finished gets its latest checkpoint uploaded as INCOMPLETE_<...>.pt.
Writes uploaded_<suite>.json (local verification record) and manifest.json (also uploaded)."""
import glob, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path
from huggingface_hub import CommitOperationAdd, HfApi

RUN = Path("/groups/yshang/an221229/checkpoints/wam_baselines/lilawam/libero_ours_paper_recipe")
CODE = Path("/groups/yshang/an221229/code/lilawam_libero")
TASKCOND = Path("/groups/yshang/an221229/data/LiLaWAM_libero_ours_taskcond")
STATS = "/groups/yshang/an221229/data/LiLaWAM_libero_ours_stats_{}.json"
REPO, P, JOB = "anhdao69/LoopWAM_NT", "baselines/LiLa-WAM_LIBERO_ours", "1094056"
SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
api = HfApi(token=open("/groups/yshang/an221229/cache/huggingface/token").read().strip())


def log(*a):
    print(time.strftime("[%F %T]"), *a, flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def ckpt_info(p):
    """epoch / global_step / epoch loss of .../<suite>/<stage>/sft_*/checkpoint_epoch_<k>.pt, read from that run's
    training log (torch cannot be imported on the login node: libnccl mmap fails under its memory limit).
    train.py saves global_step = k * batches_per_epoch (grad_accum 1) and loss = epoch k's mean loss."""
    parts = Path(p).parts
    suite, stage, k = parts[-4], parts[-3], int(re.search(r"checkpoint_epoch_(\d+)\.pt", p).group(1))
    t = (RUN / "logs" / f"{suite}_{stage}.log").read_text(errors="ignore")
    batches = int(re.search(r"Batches per Epoch: (\d+)", t).group(1))
    losses = [float(x) for x in re.findall(r"Epoch \d+ Completed\. Avg Loss: ([0-9.]+)", t)]
    return {"stage": stage, "epoch": k, "global_step": k * batches, "epoch_avg_loss": losses[k - 1]}


def epoch_losses(logfile):
    return [float(x) for x in re.findall(r"Epoch \d+ Completed\. Avg Loss: ([0-9.]+)", Path(logfile).read_text())]


def commit(ops, msg):
    for attempt in range(5):
        try:
            return api.create_commit(REPO, operations=ops, commit_message=msg)
        except Exception as e:  # network hiccups: retry
            log("commit failed", attempt, repr(e)[:300]); time.sleep(60 * (attempt + 1))
    raise RuntimeError("commit failed 5 times")


def verify(paths_local):
    """paths_local: {path_in_repo: local path}; compare HF LFS sha256 with local sha256."""
    infos = {i.path: i for i in api.get_paths_info(REPO, list(paths_local))}
    out = {}
    for rp, lp in paths_local.items():
        i = infos.get(rp)
        remote = i.lfs.sha256 if i is not None and i.lfs else None
        out[rp] = {"local_sha256": sha256(lp), "hf_lfs_sha256": remote, "size": os.path.getsize(lp)}
        out[rp]["match"] = out[rp]["local_sha256"] == remote
    return out


def manifest_load():
    m = RUN / "manifest.json"
    return json.loads(m.read_text()) if m.exists() else {"repo": REPO, "folder": P, "suites": {}}


def upload_suite(suite, c1, c2, incomplete=None):
    d = RUN / suite
    ops, ck = [], {}
    if incomplete:
        ck[f"{P}/{suite}/INCOMPLETE_{incomplete['name']}.pt"] = incomplete["path"]
    else:
        ck[f"{P}/{suite}/stage2_epoch2_final.pt"] = c2
    if c1:
        ck[f"{P}/{suite}/stage1_epoch5.pt"] = c1
    for rp, lp in ck.items():
        ops.append(CommitOperationAdd(rp, lp))
    for stage in ("stage1", "stage2"):
        cfgs = sorted(glob.glob(f"{d}/{stage}/sft_*/config.yaml"))
        if cfgs:
            ops.append(CommitOperationAdd(f"{P}/{suite}/config_{stage}.yaml", cfgs[-1]))
        lg = RUN / "logs" / f"{suite}_{stage}.log"
        if lg.exists():
            ops.append(CommitOperationAdd(f"{P}/{suite}/logs/train_{stage}.log", str(lg)))
    for csv in sorted(glob.glob(f"{d}/log/loss/*.csv")):
        ops.append(CommitOperationAdd(f"{P}/{suite}/logs/{Path(csv).name}", csv))
    ops.append(CommitOperationAdd(f"{P}/{suite}/norm_stats.json", STATS.format(suite)))
    for npy in sorted(glob.glob(f"{TASKCOND}/{suite}/*/task_cond.npy")):
        ops.append(CommitOperationAdd(f"{P}/task_cond/{suite}/{Path(npy).parent.name}/task_cond.npy", npy))
    if not (RUN / "uploaded_common.json").exists():
        for f in sorted(glob.glob(f"{CODE}/*.py")) + [f"{CODE}/libero_ours.yaml", str(RUN / "train_job.sh"),
                                                       "/groups/yshang/an221229/checkpoints/wam_baselines/lilawam/convert_ours_to_lilawam_hdf5.py"]:
            ops.append(CommitOperationAdd(f"{P}/code/{Path(f).name}", f))
        ops.append(CommitOperationAdd(f"{P}/README.md", str(RUN / "hf_staging" / "README.md")))
        ops.append(CommitOperationAdd(f"{P}/task_cond/meta.json", str(TASKCOND / "meta.json")))
    m = manifest_load()
    entry = {"status": "INCOMPLETE" if incomplete else "complete", "checkpoints": {}}
    for rp, lp in ck.items():
        entry["checkpoints"][rp] = {"local_path": lp, "size": os.path.getsize(lp), "sha256": sha256(lp), **ckpt_info(lp)}
    for stage in ("stage1", "stage2"):
        lg = RUN / "logs" / f"{suite}_{stage}.log"
        if lg.exists():
            entry[f"{stage}_epoch_avg_losses"] = epoch_losses(lg)
    m["suites"][suite] = entry
    (RUN / "manifest.json").write_text(json.dumps(m, indent=1))
    ops.append(CommitOperationAdd(f"{P}/manifest.json", str(RUN / "manifest.json")))
    log(suite, "committing", len(ops), "files")
    info = commit(ops, f"LiLa-WAM LIBERO baseline (our data, paper recipe): {suite}" + (" [INCOMPLETE]" if incomplete else ""))
    ver = verify(ck)
    rec = {"suite": suite, "commit": getattr(info, "oid", None), "time": time.strftime("%F %T"), "verify": ver,
           "all_match": all(v["match"] for v in ver.values()), "files": len(ops)}
    (RUN / f"uploaded_{suite}.json").write_text(json.dumps(rec, indent=1))
    (RUN / "uploaded_common.json").write_text(json.dumps({"commit": rec["commit"]}))
    log(suite, "uploaded commit", rec["commit"], "sha256 all match:", rec["all_match"])


def job_running():
    r = subprocess.run(["squeue", "-h", "-j", JOB, "-o", "%T"], capture_output=True, text=True)
    return "RUNNING" in r.stdout


def main():
    log("watcher start")
    while True:
        text = (RUN / "train_job.log").read_text() if (RUN / "train_job.log").exists() else ""
        for suite in SUITES:
            if (RUN / f"uploaded_{suite}.json").exists():
                continue
            m2 = re.search(rf"\] {suite} stage2 rc=(\d+) ckpt=(\S*)", text)
            m1 = re.search(rf"\] {suite} stage1 rc=(\d+) ckpt=(\S*)", text)
            if m2 and m2.group(1) == "0" and m2.group(2) and os.path.exists(m2.group(2)):
                upload_suite(suite, m1.group(2) if m1 else None, m2.group(2))
        done = all((RUN / f"uploaded_{s}.json").exists() for s in SUITES)
        if done:
            log("all suites uploaded"); return
        if not job_running():
            time.sleep(120)  # let the final checkpoint write settle
            text = (RUN / "train_job.log").read_text()
            for suite in SUITES:
                if (RUN / f"uploaded_{suite}.json").exists():
                    continue
                cks = sorted(glob.glob(f"{RUN}/{suite}/stage*/sft_*/checkpoint_epoch_*.pt"), key=os.path.getmtime)
                if not cks:
                    log(suite, "no checkpoint at all; nothing uploaded"); continue
                last = cks[-1]
                name = f"{Path(last).parent.parent.name}_{Path(last).stem.replace('checkpoint_', '')}"
                m1 = re.search(rf"\] {suite} stage1 rc=(\d+) ckpt=(\S+)", text)
                upload_suite(suite, m1.group(2) if m1 else None, None, incomplete={"name": name, "path": last})
            log("job ended; watcher exit"); return
        time.sleep(60)


if __name__ == "__main__":
    main()
