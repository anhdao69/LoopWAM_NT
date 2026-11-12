#!/usr/bin/env python3
"""CPU self-test of the GR1-E0 port (no GPU; run while GPUs 4-7 train the GR1 continuation). Writes into TEST_DIR only.
Checks: recipe registry; teacher selection on the current evidence (epoch 10 not evaluated yet -> epoch 9) and
parent_from_selection; TransportGR1Dataset vs the GR1 v0 trainer's GR1Windows fields (64 windows + dummy) and its
latent-cache provenance digest vs the teacher contract; elastic_init --gr1 (epoch 9) with step-condition/T4 digests equal
to the RoboTwin init's; export_gr1 + load_gr1_transport round trip on CPU (tensor equality, metadata); trainer argument
guard for --teacher-selection. Usage: cpu_test_gr1e.py TEST_DIR"""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

T = Path(sys.argv[1]); T.mkdir(parents=True, exist_ok=True)
fails = []
def ok(name, cond, **kw):
    print(json.dumps(dict(check=name, passed=bool(cond), **kw), default=str), flush=True)
    if not cond: fails.append(name)

from transport_experiments import ELASTIC, GR1_PARENT
c = ELASTIC["GR1-E0"]
ok("recipe", c["kind"] == "T0" and c["scope"] == "pc" and c["dataset"] == "robocasa_gr1" and c["end_prefix"] == 12
   and c["gripper_loss"] is False and c["init"] == "common_v1" and c["sampler"] == "balanced"
   and {k: v for k, v in c.items() if k not in ("parent", "dataset", "end_prefix", "gripper_loss")}
       == {k: v for k, v in ELASTIC["RT50-E0"].items() if k not in ("parent", "dataset", "end_prefix", "gripper_loss")})

import select_gr1_teacher as sel
rec = sel.select(os.environ["GR1_EVAL"])
ok("selection_epoch9_now", rec["chosen_epoch"] == 9 and rec["teacher_sha256"] == GR1_PARENT["candidates"]["9"]["sha256"],
   reason=rec["reason"], evidence10=rec["evidence"]["10"].get("present"))
sp = T / "selection_dryrun.json"; sp.write_text(json.dumps(dict(rec, created_utc="test"), indent=1))
parent = sel.parent_from_selection(sp, rec["teacher"], rec["teacher_sha256"])
ok("parent_from_selection", parent["epoch"] == 9 and parent["sha256"] == rec["teacher_sha256"] and "candidates" not in parent)
try:
    sel.parent_from_selection(sp, rec["teacher"], "0" * 64); ok("parent_rejects_wrong_sha", False)
except ValueError:
    ok("parent_rejects_wrong_sha", True)

from fastwam.datasets.transport_gr1 import TransportGR1Dataset, WINDOWS
from fastwam.datasets.robocasa_gr1 import GR1Windows, gr1_spec
from fastwam.models.wan22.loopwam_init import sha256_file
vae_sha = sha256_file(os.environ["ELASTIC_VAE"])
t0 = time.time()
data = TransportGR1Dataset(os.environ["ELASTIC_MANIFEST"], os.environ["ELASTIC_LATENTS"], vae_sha)
header = torch.load(rec["teacher"], map_location="cpu", mmap=True, weights_only=False)
tc = header["training_state"]["contract"]
ok("dataset_identity", len(data) == WINDOWS and data.latent_cache_provenance_sha256 == tc["latent_cache_provenance_sha256"]
   and data.data_manifest_sha256 == tc["data_manifest_sha256"] and data.manifest["normalization_sha256"] == tc["normalization_sha256"]
   and vae_sha == tc["vae_sha256"], seconds=time.time() - t0)
shared = Path(os.environ["ELASTIC_MANIFEST"]).parent
ref = GR1Windows(gr1_spec(stats_path=shared / "dataset_stats_29d.json"), cache_dir=shared, text_threads=2)
rng = np.random.default_rng(3); bad = []
idx = rng.integers(0, len(data), 48).tolist() + [e[2] for e in ref.episodes[:8]] + [e[2] + e[3] - 1 for e in ref.episodes[-8:]]
for i in idx:
    ref.warm_text([ref.prompt(i)]); s, f = data[i], ref.fields(i)
    same = all(torch.equal(s[k], f[k]) for k in ("action", "context", "context_mask", "action_is_pad")) and torch.equal(s["proprio"], f["proprio"][:1]) \
        and s["first_frame_latents"].shape == (16, 1, 32, 32) and torch.isfinite(s["first_frame_latents"]).all()
    if not same: bad.append(i)
d = data[-1]
ok("dataset_fields", not bad and d["sample_valid"] is False and d["training_index"] == -1, windows=len(idx), bad=bad[:5],
   padded=sum(bool(ref.fields(i)["action_is_pad"].any()) for i in idx))
loader = torch.utils.data.DataLoader(data, batch_size=8, num_workers=2, shuffle=False)
it = iter(loader); b = next(it)
ok("dataloader_workers", b["first_frame_latents"].shape == (8, 16, 1, 32, 32) and b["context"].dtype == torch.bfloat16
   and b["proprio"].shape == (8, 1, 29))
del it, loader

init = T / "init_ep9.pt"
r = subprocess.run([sys.executable, "scripts/elastic_init.py", "common_v1", "--gr1", "--teacher", rec["teacher"],
                    "--vae-path", os.environ["ELASTIC_VAE"], "--out", str(init)], capture_output=True, text=True)
ref_digest = json.loads(Path(os.environ["ELASTIC_REFERENCE_INIT_JSON"]).read_text())["tensor_sha256"]
got = json.loads(init.with_suffix(".json").read_text()) if r.returncode == 0 else {}
ok("elastic_init_gr1", r.returncode == 0 and got["teacher_sha256"] == rec["teacher_sha256"]
   and got["tensor_sha256"]["step_condition"] == ref_digest["step_condition"]
   and got["tensor_sha256"]["transport_T4"] == ref_digest["transport_T4"], stderr=r.stderr[-500:])

from fastwam.models.wan22.loopwam_gr1 import create_loopwam_gr1
from fastwam.models.wan22.transport_policy import TransportPolicy
from fastwam.models.wan22.transport_inference_gr1 import load_gr1_transport, EXPORT_FORMAT
from train_transport import load_elastic_init, export_epoch, GR1_METADATA
p, _ = create_loopwam_gr1(os.environ["ELASTIC_VAE"], checkpoint_path=rec["teacher"], device="cpu", checkpoint_blocks=False)
p.requires_grad_(False).eval()
cfg = dict(ELASTIC["GR1-E0"], parent=parent)
m = TransportPolicy(p, **{k: cfg[k] for k in ("kind", "scope", "step_cond", "regime", "anchor")}, checkpoint_blocks=False,
                    reuse_teacher_prefix=True, collect_diagnostics=True, end_prefix=12, gripper_loss=False)
parts = load_elastic_init(m, cfg, str(init), rec["teacher_sha256"])
with torch.no_grad():                                     # pretend training moved the student
    for q in m.student.mixtures["action"].parameters(): q.add_(1e-3)
    for q in m.step_condition.parameters(): q.add_(1e-3)
contract = dict(teacher=str(Path(rec["teacher"]).resolve()), asset_sha256=dict(teacher=rec["teacher_sha256"]), dataset="robocasa_gr1",
                global_batch=128, windows=WINDOWS)
path = export_epoch(m, T, 1, cfg, 47032, contract)
x = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
pol, payload = load_gr1_transport(path, os.environ["ELASTIC_VAE"], device="cpu")
sa = {k: v for k, v in m.student.mixtures["action"].state_dict().items()}
same_student = all(torch.equal(pol.student.mixtures["action"].state_dict()[k], v) for k, v in sa.items())
same_parent_action = all(torch.equal(pol.parent.action_expert.state_dict()[k], v) for k, v in sa.items())
same_step = all(torch.equal(pol.step_condition.state_dict()[k], v) for k, v in m.step_condition.state_dict().items())
frozen = all(torch.equal(v, header["mot"][k]) for k, v in x["mot"].items() if not k.startswith("mixtures.action."))
ok("export_roundtrip_cpu", x["format_version"] == EXPORT_FORMAT and x["base_format_version"] == "loopwam-s-gr1-v1"
   and all(x[k] == header[k] for k in GR1_METADATA if k != "architecture") and x["architecture"] == header["architecture"]
   and same_student and same_parent_action and same_step and frozen and x["training_state"]["windows_seen"] == WINDOWS
   and x["step"] == 47032 and x["transport_config"]["parent"]["epoch"] == 9, parts=parts)
os.remove(path)

r = subprocess.run([sys.executable, "scripts/train_transport.py", "--run-name", "GR1-E0", "--output-dir", str(T / "x"),
                    "--init-artifact", str(init)], capture_output=True, text=True)
ok("trainer_requires_selection", r.returncode == 2 and "--teacher-selection" in r.stderr, tail=r.stderr[-200:])
print(json.dumps(dict(event="cpu_selftest_done", fails=fails)), flush=True)
sys.exit(1 if fails else 0)
