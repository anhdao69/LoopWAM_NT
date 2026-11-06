#!/usr/bin/env python3
"""Fetch the BASE10 elastic reference (E0/E1/E2 run metadata + E0 epoch_10.pt, and every E0/E1/E2 condition's
published H100 episode records) from anhdao69/LoopWAM_NT into <out>/; verify the checkpoint's LFS sha256."""
import hashlib, json, sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO = "anhdao69/LoopWAM_NT"; BASE = "libero-all-suites/Elastic"
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
api = HfApi()
files = {f.path: f for f in api.list_repo_tree(REPO, path_in_repo=BASE, recursive=True) if hasattr(f, "size")}
want = []
for run in ("E0", "E1", "E2"):
    for name in ("config.json", "dataset_stats.json", "data/data_manifest.json", "data/dataset_stats.json",
                 "epoch_10.pt.uploaded.json", "manifest.json", "status.json"):
        want.append(f"{BASE}/{run}/{name}")
want.append(f"{BASE}/E0/epoch_10.pt")
want += [p for p in files if p.startswith(f"{BASE}/eval/") and (p.endswith(".json") or p.endswith(".csv") or p.endswith(".md"))
         and any(f"/{r}__flat" in p or f"/{r}.json" in p or f"/{r}__grid" in p for r in ("E0", "E1", "E2"))
         or p in (f"{BASE}/eval/results.json", f"{BASE}/eval/results.md")]
for p in want:
    if p not in files:
        print("missing on hub:", p); continue
    local = hf_hub_download(REPO, p, local_dir=str(out))
    f = files[p]
    if getattr(f, "lfs", None):
        h = hashlib.sha256()
        with open(local, "rb") as s:
            for chunk in iter(lambda: s.read(1 << 24), b""): h.update(chunk)
        ok = h.hexdigest() == f.lfs.sha256
        print(p, f.size, "sha256", h.hexdigest(), "OK" if ok else "MISMATCH"); assert ok
print("fetched", len(want), "files into", out)
