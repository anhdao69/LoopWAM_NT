#!/usr/bin/env python3
"""Upload RoboCasa-GR1 LoopWAM v0 continuation epochs 9 and 10 to anhdao69/LoopWAM_NT under Robocasa-GR1/ and verify them.

User, 2026-11-08: "Push the GR1 epoch 9, 10 to hf for me".

    env -u HF_HUB_OFFLINE python -I hf_upload_gr1_cont.py [--dry-run]   (login node; HF token from the local cache)

Checkpoints come from the continuation run train/continuation_e06_10 (epochs 6-10 continued from the end-of-epoch-5
training state). Refuses to upload unless, for each epoch E in (9, 10): the stored step == E x 23,516 ==
training_state.update, training_state.next_epoch == E, its contract equals the continuation manifest's contract, the
official eval validation of that step is VALID with problems [] (24 tasks); epoch 9's sha256 is the one recorded by the
GR1-E0 teacher selection; the run is COMPLETE at 235,160; the continuation normalization equals the one already on HF;
neither epoch exists on HF yet. One commit (parented on the HF head read here) adds epoch_09.pt, epoch_10.pt,
continuation_manifest.json, the two eval validations and the README rows; existing files stay as they are. Afterwards
every remote file must match the local bytes; receipts epoch_NN.pt.uploaded.json are then committed and kept locally.
"""
import hashlib, json, sys, time
from pathlib import Path

import torch
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

REPO, PREFIX = "anhdao69/LoopWAM_NT", "Robocasa-GR1"
RUN = Path("/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robocasa_gr1_v0_20261101")
CONT = RUN / "train" / "continuation_e06_10"
PROD = RUN / "train" / "production"
EVAL = RUN / "eval"
PER = 23516
EPOCHS = (9, 10)
EPOCH9_SHA = "694a679b4c07934b57995dc3a2ab62e2f65357cea29108b109b30049531e2748"   # elastic_gr1/teacher_selection.json
SUCCESSES = {9: 626, 10: 620}                                                      # orchestrator.log, official validation
DRY = "--dry-run" in sys.argv


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def local_checks():
    manifest = json.loads((CONT / "manifest.json").read_text())
    c = manifest["contract"]
    assert (c["updates_per_epoch"], c["planned_updates"], c["epochs"], c["phase_start_update"]) == (PER, 10 * PER, 10, 5 * PER), c
    done = json.loads((CONT / "COMPLETE").read_text())
    assert (done["status"], done["update"], done["planned_updates"]) == ("complete", 10 * PER, 10 * PER), done
    stats = CONT / "data" / "dataset_stats.json"
    assert stats.read_bytes() == (PROD / "data" / "dataset_stats.json").read_bytes() \
        == (RUN / "shared" / "data" / "dataset_stats_29d.json").read_bytes()
    assert (CONT / "data" / "data_manifest.json").read_bytes() == (PROD / "data" / "data_manifest.json").read_bytes()
    out = {}
    for e in EPOCHS:
        ck = CONT / "checkpoints" / f"epoch_{e:02d}.pt"
        st0 = ck.stat()
        h = torch.load(ck, map_location="cpu", mmap=True, weights_only=False)
        ts, step = h["training_state"], h["step"]
        assert step == ts["update"] == e * PER and ts["next_epoch"] == e, (e, step, ts["update"], ts["next_epoch"])
        assert ts["contract"] == c, f"epoch {e}: contract differs from the continuation manifest"
        del h
        v = json.loads((EVAL / f"official_epoch{e:02d}_validation.json").read_text())
        succ = sum(t["successes"] for t in v["per_task"].values())
        assert v["valid"] is True and v["problems"] == [] and v["step"] == step and len(v["per_task"]) == 24, (e, v["valid"], v["problems"])
        assert succ == SUCCESSES[e] and abs(v["mean_success"] - succ / 1200) < 1e-12, (e, succ, v["mean_success"])
        sha = sha256(ck)
        if e == 9:
            assert sha == EPOCH9_SHA, sha
        assert ck.stat().st_mtime == st0.st_mtime and ck.stat().st_size == st0.st_size, f"{ck} changed while hashing"
        out[e] = dict(ck=ck, sha=sha, size=st0.st_size, mtime=st0.st_mtime, step=step, successes=succ,
                      validation=(EVAL / f"official_epoch{e:02d}_validation.json").read_bytes())
        print(json.dumps(dict(epoch=e, file=str(ck), step=step, sha256=sha, size=st0.st_size, successes=succ)), flush=True)
    return manifest, out


ROW = "| `epoch_{e:02d}.pt` | continuation epoch {e} (update {step:,}); official eval {pct:.2f}% ({succ}/1200){best} |\n"
CONT_TEXT = """
## Continuation: epochs 6-10

`epoch_09.pt` and `epoch_10.pt` come from a continuation of the same run for five more epochs (6-10), started from the
end-of-epoch-5 training state `latest.pt` (update 117,580, sha256 `{parent}`) with a new learning-rate phase:
AdamW lr 3e-5 (backbone and robot I/O), no warmup, cosine to 1% (3e-7) at update 235,160; otherwise the same contract
(same data and normalization, global batch 256, 23,516 updates per epoch, seed 42, 4 x H100). The full continuation
manifest and contract are in `continuation_manifest.json`.

Official evaluation (protocol above, 24 tasks x 50 episodes, H100, EGL rendering, env seed 0): epoch 9 = **{p9:.2f}%
({s9}/1200)**, epoch 10 = {p10:.2f}% ({s10}/1200). Per-task results: `eval/official_epochNN_validation.json`.
"""


def main():
    manifest, L = local_checks()
    api = HfApi()
    assert api.whoami()["name"] == "anhdao69"
    info0 = api.model_info(REPO)
    files = api.list_repo_files(REPO, revision=info0.sha)
    under = [f for f in files if f.startswith(PREFIX + "/")]
    print(json.dumps(dict(repo=REPO, private=info0.private, head=info0.sha, existing=under)), flush=True)
    for e in EPOCHS:
        assert f"{PREFIX}/epoch_{e:02d}.pt" not in files, f"epoch_{e:02d}.pt already on HF"
    new_paths = ["continuation_manifest.json"] + [f"eval/official_epoch{e:02d}_validation.json" for e in EPOCHS]
    assert not [p for p in new_paths if f"{PREFIX}/{p}" in files], "continuation files already on HF"
    remote_stats = api.get_paths_info(REPO, [f"{PREFIX}/dataset_stats.json"], expand=True, revision=info0.sha)[0]
    local_stats = (CONT / "data" / "dataset_stats.json").read_bytes()
    ok = (remote_stats.lfs.sha256 == hashlib.sha256(local_stats).hexdigest()) if remote_stats.lfs else (remote_stats.blob_id == git_blob(local_stats))
    assert ok and remote_stats.size == len(local_stats), "HF dataset_stats.json differs from the continuation normalization"

    readme = Path(hf_hub_download(REPO, f"{PREFIX}/README.md", revision=info0.sha)).read_text()
    assert "epoch_09.pt" not in readme and "## Continuation" not in readme, "README already mentions the continuation"
    anchor = "| `epoch_05.pt` | end of epoch 5 = final (update 117,580) |\n"
    assert readme.count(anchor) == 1, "epoch_05 row not found"
    rows = "".join(ROW.format(e=e, step=L[e]["step"], pct=100 * L[e]["successes"] / 1200, succ=L[e]["successes"],
                              best=" (best evaluated epoch)" if e == 9 else "") for e in EPOCHS)
    readme = readme.replace(anchor, anchor.replace("= final ", "") + rows, 1)
    old = "| `epoch_0N.pt.uploaded.json` | upload receipt (sha256, size, HF revision) |\n"
    assert readme.count(old) == 1
    readme = readme.replace(old, "| `epoch_NN.pt.uploaded.json` | upload receipt (sha256, size, HF revision) |\n", 1)
    old = "| `manifest.json` | training run manifest and contract |\n"
    assert readme.count(old) == 1
    readme = readme.replace(old, "| `manifest.json` | training run manifest and contract (epochs 1-5) |\n"
                                 "| `continuation_manifest.json` | continuation run manifest and contract (epochs 6-10) |\n"
                                 "| `eval/official_epochNN_validation.json` | official eval validation, per-task successes |\n", 1)
    old = "max_episode_steps 720, n_action_steps 12, state sent); results will be added here.\n"
    assert readme.count(old) == 1
    readme = readme.replace(old, "max_episode_steps 720, n_action_steps 12, state sent).\n", 1)
    parent = manifest["contract"]["parent_checkpoint_sha256"]
    assert parent == manifest["expected_parent_sha256"] == "e9fa70080709d251ce8e946808ddcfea9df66b007f7389f34d18e4b37a6a1a58", parent
    readme += CONT_TEXT.format(parent=parent, p9=100 * L[9]["successes"] / 1200, s9=L[9]["successes"],
                               p10=100 * L[10]["successes"] / 1200, s10=L[10]["successes"])
    extra = {"continuation_manifest.json": (CONT / "manifest.json").read_bytes(),
             "README.md": readme.encode()}
    for e in EPOCHS:
        extra[f"eval/official_epoch{e:02d}_validation.json"] = L[e]["validation"]
    ops = [CommitOperationAdd(f"{PREFIX}/epoch_{e:02d}.pt", str(L[e]["ck"])) for e in EPOCHS]
    ops += [CommitOperationAdd(f"{PREFIX}/{n}", b) for n, b in extra.items()]
    expect = {f"{PREFIX}/epoch_{e:02d}.pt": ("sha256", L[e]["sha"], L[e]["size"]) for e in EPOCHS}
    expect.update({f"{PREFIX}/{n}": ("bytes", b, len(b)) for n, b in extra.items()})
    print("PLAN " + json.dumps({p: v[2] for p, v in expect.items()}), flush=True)
    out = EVAL / "hf_receipts"
    if DRY:
        (out / "dry_run_continuation_README.md").write_text(readme)
        print("DRY RUN: nothing uploaded; README draft at " + str(out / "dry_run_continuation_README.md"), flush=True)
        return
    t = time.time()
    info = api.create_commit(REPO, operations=ops, parent_commit=info0.sha,
                             commit_message="Robocasa-GR1: LoopWAM v0 continuation epoch_09.pt (update 211644) and epoch_10.pt (update 235160)")
    print(json.dumps(dict(commit=info.oid, seconds=round(time.time() - t))), flush=True)
    for e in EPOCHS:
        st1 = L[e]["ck"].stat()
        assert (st1.st_size, st1.st_mtime) == (L[e]["size"], L[e]["mtime"]), f"epoch_{e:02d}.pt changed during upload"
    remote = {r.path: r for r in api.get_paths_info(REPO, list(expect), expand=True, revision=info.oid)}
    checked = {}
    for path, (kind, val, size) in expect.items():
        r = remote[path]
        if kind == "sha256":
            ok = r.lfs is not None and r.lfs.sha256 == val and r.size == size
        else:
            ok = (r.lfs.sha256 == hashlib.sha256(val).hexdigest()) if r.lfs is not None else (r.blob_id == git_blob(val))
            ok = ok and r.size == size
        assert ok, f"remote verification failed for {path}: {r}"
        checked[path] = size
    receipts = []
    for e in EPOCHS:
        name = f"epoch_{e:02d}.pt"
        rec = dict(repo=REPO, path=f"{PREFIX}/{name}", sha256=L[e]["sha"], size=L[e]["size"], step=L[e]["step"],
                   source=str(L[e]["ck"]), official_successes=L[e]["successes"], revision=info.oid,
                   files_verified=checked, verified_utc=time.time())
        (out / f"{name}.uploaded.json").write_text(json.dumps(rec, indent=2))
        receipts.append(CommitOperationAdd(f"{PREFIX}/{name}.uploaded.json", json.dumps(rec, indent=2).encode()))
        print("VERIFIED " + json.dumps({k: rec[k] for k in ("path", "sha256", "size", "step", "revision")}), flush=True)
    api.create_commit(REPO, operations=receipts, commit_message="Robocasa-GR1: receipts for epoch_09.pt and epoch_10.pt")
    print("RECEIPTS COMMITTED", flush=True)


if __name__ == "__main__":
    main()
