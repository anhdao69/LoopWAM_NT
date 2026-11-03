#!/usr/bin/env python3
"""Re-wrap the memoryless ChronoLoop CL-0 checkpoint as a LoopWAM v0 student checkpoint (plan section 16.1).

CL-0 (anhdao69/ChronoLoop, CL-0_no-memory_a4) has 0 memory tokens and no history frame. For that configuration
ChronoLoopWAM.configure_chrono() sets mot.memory=None and every forward takes the parent LoopWAM v0 path, so its
`chronoloop-v1` payload holds exactly the LoopMoT v0 tensors. This script copies those tensors unchanged into the
`loopwam-s-v1` format that create_loopwam / train_transport.py load as the frozen teacher and student initializer.

Non-weight metadata (architecture, exit_weight_scale, loop contract) is taken from CL-0's own parent, the BASE10
export it was fine-tuned from (`--template`, sha256 must equal CL-0's recorded parent_sha256), and checked against
CL-0's own fields. Every output tensor is verified bit-equal to CL-0 after reloading the written file.

    python scripts/convert_chronoloop_parent.py --chronoloop CL-0/policy.pt --template BASE10/policy.pt --out CL-0_loopwam.pt
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch

NO_MEMORY = dict(memory_tokens=0, mem_source="none", mem_write="none", history_frame=0, action_loops=4, video_loops=4)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def same_layout(a, b, what):
    if set(a) != set(b):
        raise ValueError(f"{what}: key sets differ (only in CL-0: {sorted(set(a) - set(b))[:5]}, "
                         f"only in template: {sorted(set(b) - set(a))[:5]})")
    for k in a:
        if a[k].shape != b[k].shape or a[k].dtype != b[k].dtype:
            raise ValueError(f"{what}.{k}: {a[k].shape}/{a[k].dtype} vs {b[k].shape}/{b[k].dtype}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chronoloop", required=True, help="CL-0 epoch_10/policy.pt (format chronoloop-v1)")
    p.add_argument("--template", required=True, help="BASE10 export v0-video4-action4/policy.pt (CL-0's parent)")
    p.add_argument("--expected-sha256", default="137f0a839b5a1e2363621420ab599cd36a5693000f784a04a11b9bea2bea2cae")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    source_sha, template_sha = sha256_file(a.chronoloop), sha256_file(a.template)
    if source_sha != a.expected_sha256:
        raise ValueError(f"CL-0 checkpoint sha256 {source_sha} != pinned {a.expected_sha256}")
    cl0 = torch.load(a.chronoloop, map_location="cpu", weights_only=False, mmap=True)
    base = torch.load(a.template, map_location="cpu", weights_only=False, mmap=True)
    if cl0.get("format_version") != "chronoloop-v1":
        raise ValueError("Expected a chronoloop-v1 checkpoint")
    if base.get("format_version") != "loopwam-s-v1" or base.get("version") != "v0":
        raise ValueError("Template must be a LoopWAM v0 student checkpoint")
    if any(cl0["chrono"].get(k) != v for k, v in NO_MEMORY.items()):
        raise ValueError(f"Not the memoryless 4/4 configuration: {cl0['chrono']}")
    if cl0.get("parent_sha256") != template_sha:
        raise ValueError(f"CL-0 parent {cl0.get('parent_sha256')} is not the template {template_sha}")
    for key in ("version", "architecture"):
        if cl0.get(key) != base.get(key):
            raise ValueError(f"CL-0 {key}={cl0.get(key)!r} differs from its parent {base.get(key)!r}")
    # The BASE10 export predates explicit loop fields; it declares 4/4 through inference/trained_max_loops.
    if (base.get("inference_loops"), base.get("trained_max_loops")) != (4, 4) or base.get("exit_weight_scale") != 1.0:
        raise ValueError("Template is not the v0 4/4 export")
    loop_contract = dict(video_loops=4, action_loops=4, loop_alignment="late")
    if any(cl0.get(k) != v for k, v in loop_contract.items()):
        raise ValueError(f"CL-0 loop fields differ from v0 4/4 late: {[cl0.get(k) for k in loop_contract]}")
    if any(k.startswith("memory.") for k in cl0["mot"]):
        raise ValueError("Memoryless CL-0 must not carry memory tensors")
    same_layout(cl0["mot"], base["mot"], "mot")
    same_layout(cl0["proprio_encoder"], base["proprio_encoder"], "proprio_encoder")
    changed = sum(not torch.equal(cl0["mot"][k], base["mot"][k]) for k in base["mot"])
    payload = {k: v for k, v in base.items() if k not in ("mot", "proprio_encoder", "step", "training_state")}
    payload.update(**loop_contract)  # CL-0's own explicit (symmetric) loop contract
    payload.update(
        mot={k: v.clone() for k, v in cl0["mot"].items()},
        proprio_encoder={k: v.clone() for k, v in cl0["proprio_encoder"].items()},
        step=cl0.get("step"), training_state=None,
        converted_from=dict(format="chronoloop-v1", sha256=source_sha, chrono=cl0["chrono"],
                            chrono_flags_sha256=cl0.get("chrono_flags_sha256"), parent_sha256=cl0["parent_sha256"],
                            step=cl0.get("step"), template_sha256=template_sha,
                            template_metadata_keys=sorted(payload)))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)
    written = torch.load(out, map_location="cpu", weights_only=False, mmap=True)
    for family in ("mot", "proprio_encoder"):
        if set(written[family]) != set(cl0[family]) or not all(
                torch.equal(written[family][k], cl0[family][k]) for k in cl0[family]):
            raise ValueError(f"{family} tensors differ after writing")
    record = dict(out=str(out), sha256=sha256_file(out), source=str(a.chronoloop), source_sha256=source_sha,
                  template=str(a.template), template_sha256=template_sha, mot_tensors=len(cl0["mot"]),
                  mot_tensors_changed_vs_parent=changed, proprio_tensors=len(cl0["proprio_encoder"]),
                  metadata={k: payload[k] for k in payload if k not in ("mot", "proprio_encoder", "architecture")},
                  tensors_bit_equal_to_source=True)
    out.with_suffix(".json").write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps(record, indent=2, default=str))


if __name__ == "__main__":
    main()
