#!/usr/bin/env python3
"""Read-only: show the HF Robocasa-GR1/ folder, its README, and the metadata of continuation epochs 9 and 10."""
import json, os, sys
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download

REPO, PREFIX = "anhdao69/LoopWAM_NT", "Robocasa-GR1"
RUN = Path("/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robocasa_gr1_v0_20261101")
CONT = RUN / "train" / "continuation_e06_10"
PROD = RUN / "train" / "production"

api = HfApi(token=os.environ["HF_TOKEN"])
head = api.list_repo_commits(REPO)[0]
print("HEAD", head.commit_id, head.created_at, head.title)
files = [f for f in api.list_repo_tree(REPO, path_in_repo=PREFIX, recursive=True, expand=True)]
for f in files:
    lfs = getattr(f, "lfs", None)
    print("REMOTE", f.path, getattr(f, "size", None), lfs.sha256 if lfs else "")
p = hf_hub_download(REPO, f"{PREFIX}/README.md", revision=head.commit_id)
print("=== README ===")
print(Path(p).read_text())
print("=== /README ===")

cm = json.loads((CONT / "manifest.json").read_text())
pm = json.loads((PROD / "manifest.json").read_text())
print("CONT manifest keys", sorted(cm))
print("PROD manifest keys", sorted(pm))
diff = {k: (pm["contract"].get(k), cm["contract"].get(k)) for k in set(pm["contract"]) | set(cm["contract"])
        if pm["contract"].get(k) != cm["contract"].get(k)}
print("contract diff prod->cont", json.dumps(diff, default=str)[:3000])
for e in (9, 10):
    ck = CONT / "checkpoints" / f"epoch_{e:02d}.pt"
    h = torch.load(ck, map_location="cpu", mmap=True, weights_only=False)
    ts = h.get("training_state", {})
    print(f"EPOCH {e}", "keys", sorted(h), "step", h.get("step"),
          "ts", {k: ts.get(k) for k in ("update", "next_epoch", "epoch", "status")},
          "contract==cont", ts.get("contract") == cm["contract"], "contract==prod", ts.get("contract") == pm["contract"])
    del h
print("COMPLETE", (CONT / "COMPLETE").read_text())
for e in (9, 10):
    v = json.loads((RUN / "eval" / f"official_epoch{e:02d}_validation.json").read_text())
    print(f"VALIDATION {e}", json.dumps({k: v.get(k) for k in ("valid", "step", "tasks", "successes", "episodes", "mean_success", "problems")}))
print("data dir", sorted(p.name for p in (CONT / "data").iterdir()))
