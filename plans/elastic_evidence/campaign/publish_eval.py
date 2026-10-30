#!/usr/bin/env python3
"""Publish the elastic matched-evaluation results to HF anhdao69/LoopWAM_NT under libero-all-suites/Elastic/eval/.

Uploads (one commit): the report results JSON + Markdown table, one episodes CSV per condition, every condition's
manifest.json + summary.json, and the CUDA-graph gate files. Refuses unless every listed condition has a complete
summary (6,000 episodes) whose checkpoint sha256 matches the run's epoch_10.pt.uploaded.json record.
Usage: publish_eval.py <report dir with results.json, results.md, csv/> [--dry-run]
"""
import json, sys, glob, os
from pathlib import Path

EVAL = Path("/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_eval")
RUNS = Path("/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic")
REPO, PREFIX = "anhdao69/LoopWAM_NT", "libero-all-suites/Elastic/eval"


def main():
    report, dry = Path(sys.argv[1]), "--dry-run" in sys.argv
    conds = [c.split(":")[:2] for c in os.environ["EVAL_CONDITIONS"].split()]
    files = {}
    for run, sched in conds:
        sha = json.loads((RUNS / run / "epoch_10.pt.uploaded.json").read_text())["sha256"]
        d = EVAL / "final_rollout" / f"elastic__{run}__{sched}__{sha}"
        summary = json.loads((d / "summary.json").read_text())
        # The runner writes summary.json only once all 120 groups exist; also check size and checkpoint identity.
        if summary.get("episodes") != 6000 or summary.get("checkpoint_sha256") != sha:
            raise SystemExit(f"{run} {sched}: summary incomplete or checkpoint mismatch")
        for name in ("manifest.json", "summary.json"):
            files[f"{PREFIX}/conditions/{run}__{sched}/{name}"] = d / name
        csv = report / "csv" / f"{run}__{sched}__episodes.csv"
        if not csv.is_file():
            raise SystemExit(f"missing {csv}")
        files[f"{PREFIX}/conditions/{run}__{sched}/episodes.csv"] = csv
    for name in ("results.json", "results.md"):
        files[f"{PREFIX}/{name}"] = report / name
    for g in sorted(glob.glob(str(EVAL / "gates" / "*.json"))):
        files[f"{PREFIX}/gates/{Path(g).name}"] = Path(g)
    print(f"{len(conds)} conditions, {len(files)} files")
    if dry:
        for k in list(files)[:8]:
            print(" ", k, "<-", files[k])
        return
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()
    ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v)) for k, v in files.items()]
    info = api.create_commit(REPO, operations=ops, commit_message="Elastic: matched LIBERO evaluation results "
                             f"({len(conds)} conditions x 6,000 episodes), episodes CSVs, manifests and CUDA-graph gates")
    remote = set(api.list_repo_files(REPO, revision=info.oid))
    missing = [k for k in files if k not in remote]
    print("commit", info.oid, "missing", missing)
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
