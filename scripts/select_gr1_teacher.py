#!/usr/bin/env python3
"""Choose the GR1-E0 teacher by the registry rule (transport_experiments.GR1_PARENT) and write the selection record.

Usage: select_gr1_teacher.py --eval-dir EVAL --out RECORD.json
Evidence: EVAL/official_epoch09_validation.json and EVAL/official_epoch10_validation.json (tools/gr1_validate.py of the
GR1 continuation). Epoch 9 must be the registry's candidate (valid, step 211,644, 626/1200 successes, file sha256).
Epoch 10 is chosen only when its evaluation is valid, at step 235,160, and its success count is strictly greater than
epoch 9's; otherwise epoch 9 (the reason is recorded). The chosen checkpoint header must be a loopwam-s-gr1-v1 file at
the exact epoch boundary. The record is written once; a rerun re-derives it and requires the same content.
`parent_from_selection` (used by the trainer, the init builder, preflight, verification and the policy server) turns a
record into the recipe's config["parent"] after checking it against the registry and the teacher file.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

FORMAT = "gr1-teacher-selection-v1"
UPDATES_PER_EPOCH = 23516          # GR1 v0 continuation at global batch 256


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""): h.update(chunk)
    return h.hexdigest()


def evidence(path, step):
    path = Path(path)
    if not path.exists():
        return dict(validation=str(path), present=False)
    d = json.loads(path.read_text())
    successes = sum(int(v["successes"]) for v in d.get("per_task", {}).values())
    return dict(validation=str(path), present=True, validation_sha256=sha256_file(path), valid=d.get("valid") is True,
                problems=d.get("problems"), step=d.get("step"), expected_step=step, tasks=len(d.get("per_task", {})),
                successes=successes, episodes=50 * len(d.get("per_task", {})), mean_success=d.get("mean_success"))


def header_ok(path, epoch, step):
    import torch
    x = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    st = x.get("training_state", {})
    return (x.get("format_version") == "loopwam-s-gr1-v1" and x.get("step") == step == st.get("update")
            and st.get("next_epoch") == epoch and st.get("next_batch") == 0 and step == epoch * UPDATES_PER_EPOCH)


def select(eval_dir):
    from transport_experiments import GR1_PARENT
    c = GR1_PARENT["candidates"]
    ev = {e: evidence(Path(eval_dir) / f"official_epoch{int(e):02d}_validation.json", c[e]["step"]) for e in ("9", "10")}
    e9 = ev["9"]
    if not (e9["present"] and e9["valid"] and e9["step"] == c["9"]["step"] and e9["tasks"] == 24
            and e9["successes"] == c["9"]["official_successes"]):
        raise ValueError(f"Epoch 9 evidence is not the registry's: {e9}")
    e10 = ev["10"]
    if not e10["present"]: chosen, why = "9", "no epoch 10 validation"
    elif not e10["valid"]: chosen, why = "9", f"epoch 10 evaluation not valid: {e10['problems']}"
    elif e10["step"] != c["10"]["step"] or e10["tasks"] != 24: chosen, why = "9", "epoch 10 evaluation at an unexpected step/size"
    elif e10["successes"] > e9["successes"]:
        chosen, why = "10", f"epoch 10 {e10['successes']}/1200 > epoch 9 {e9['successes']}/1200"
    else: chosen, why = "9", f"epoch 10 {e10['successes']}/1200 <= epoch 9 {e9['successes']}/1200"
    teacher = c[chosen]["file"]
    if not header_ok(teacher, int(chosen), c[chosen]["step"]):
        raise ValueError(f"{teacher} is not the epoch-{chosen} loopwam-s-gr1-v1 boundary checkpoint")
    sha = sha256_file(teacher)
    if chosen == "9" and sha != c["9"]["sha256"]:
        raise ValueError("Epoch 9 checkpoint bytes differ from the registry sha256")
    return dict(format=FORMAT, rule=GR1_PARENT["rule"], chosen_epoch=int(chosen), reason=why, teacher=teacher,
                teacher_sha256=sha, step=c[chosen]["step"], evidence=ev)


def parent_from_selection(record_path, teacher_path=None, teacher_sha256=None):
    """config["parent"] of GR1 recipes from a selection record (checked against the registry and, when given, the
    teacher path / its sha256)."""
    from transport_experiments import GR1_PARENT
    raw = Path(record_path).read_bytes(); r = json.loads(raw)
    c = GR1_PARENT["candidates"].get(str(r.get("chosen_epoch")))
    if r.get("format") != FORMAT or c is None or r["rule"] != GR1_PARENT["rule"] or r["teacher"] != c["file"] or r["step"] != c["step"]:
        raise ValueError("Teacher selection record does not match the GR1 registry")
    if "sha256" in c and r["teacher_sha256"] != c["sha256"]:
        raise ValueError("Selected teacher sha256 differs from the registry candidate")
    if teacher_path is not None and os.path.realpath(teacher_path) != os.path.realpath(r["teacher"]):
        raise ValueError("--teacher is not the selected GR1 teacher")
    if teacher_sha256 is not None and teacher_sha256 != r["teacher_sha256"]:
        raise ValueError("Teacher file sha256 differs from the selection record")
    keep = {k: v for k, v in GR1_PARENT.items() if k != "candidates"}
    return dict(keep, epoch=r["chosen_epoch"], file=r["teacher"], step=r["step"], sha256=r["teacher_sha256"],
                selection_sha256=hashlib.sha256(raw).hexdigest(), selection_reason=r["reason"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", required=True); p.add_argument("--out", required=True)
    a = p.parse_args()
    record = select(a.eval_dir)
    out = Path(a.out)
    if out.exists():
        old = json.loads(out.read_text())
        if {k: v for k, v in old.items() if k != "created_utc"} != record:
            raise SystemExit(f"{out} exists with a different selection; refusing to change it")
        print(json.dumps(dict(event="unchanged", **old), indent=1)); return
    record["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(record, indent=1)); os.replace(tmp, out)
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
