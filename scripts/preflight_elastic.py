#!/usr/bin/env python3
"""Real-model elastic preflight on one GPU (plan Tasks B, C, E, F). Writes one JSON report.

Checks: parent/VAE/normalization identity and tensor equality with the RT teacher; deduplicated
parameter counts; common-initialization equality; bit-identical teacher-prefix reuse and adaptive
checkpointing (losses and every gradient); finite forward/backward on all nine flat paths and the
six first-full paths with intended transport participation; export -> TransportInference reload
-> eager/CUDA-graph rollout parity with the in-memory student.
"""
import argparse, copy, json, os, time
from pathlib import Path
import torch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--common-init", required=True); p.add_argument("--legacy-init", required=True)
    p.add_argument("--rt-a2", required=True, help="Downloaded RT-A2 epoch_10.pt (frozen-tensor reference)")
    p.add_argument("--microbatch", type=int, default=8); p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.use_deterministic_algorithms(True)
    dev = torch.device("cuda", 0); torch.cuda.set_device(dev)
    from fastwam.datasets.transport_cached import TransportCachedDataset
    from fastwam.models.wan22.loopwam import create_loopwam
    from fastwam.models.wan22.transport_policy import TransportPolicy
    from fastwam.models.wan22.transport_inference import TransportInference
    from fastwam.models.wan22.residual_transport import SCHEDULES, FLAT9
    from fastwam.models.wan22.loopwam_init import sha256_file
    from train_loopwam import ExactDistributedBatches
    from train_transport import sample_noise, load_elastic_init, export_epoch
    from transport_experiments import ELASTIC
    report = dict(started=time.time(), checks={})
    def check(name, ok, **detail):
        report["checks"][name] = dict(passed=bool(ok), **detail)
        print(json.dumps({name: report["checks"][name]}, default=str), flush=True)
        if not ok: raise SystemExit(f"preflight failed: {name}")

    teacher, vae = os.environ["ELASTIC_TEACHER"], os.environ["ELASTIC_VAE"]
    export = json.loads((Path(teacher).parent / "export.json").read_text())
    teacher_sha = sha256_file(teacher)
    check("asset_identity", export["source_sha256"] == "4ffe1b14f35484ea582189ba0f06c5a41ac199e1fa733b491719727105e5e5cc"
          and export["export_sha256"] == teacher_sha and export["tensor_equality_verified"] is True
          and sha256_file(vae) == "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981",
          teacher_export_sha256=teacher_sha, source_sha256=export["source_sha256"])
    data = TransportCachedDataset(os.environ["ELASTIC_MANIFEST"], os.environ["ELASTIC_LATENTS"])
    check("data_identity", len(data) == 277713 and data.manifest["normalization_sha256"] ==
          "d5ad351eb3272f416d34082e8af5483ba1118ceca355a37d9f268fdb269fb3e4",
          windows=len(data), staged=json.loads((Path(os.environ["ELASTIC_LATENTS"]) / "STAGED.json").read_text())["chunk_sha256_digest"])

    parent = create_loopwam(checkpoint_path=teacher, vae_path=vae, device=str(dev)).requires_grad_(False).eval()
    # RT trained against source checkpoint 4ffe1b14; its exports carry the frozen video expert,
    # proprio encoder and (P-b) the frozen base action weights, so compare them bit for bit.
    rt = torch.load(a.rt_a2, map_location="cpu", weights_only=False, mmap=True)
    mot = parent.mot.state_dict()
    video_equal = all(torch.equal(v.cpu(), rt["mot"][k]) for k, v in mot.items() if k.startswith("mixtures.video."))
    proprio_equal = all(torch.equal(v.cpu(), rt["proprio_encoder"][k]) for k, v in parent.proprio_encoder.state_dict().items())
    base = {k.replace(".base.", "."): v for k, v in rt["student_action"].items() if not k.endswith((".A.weight", ".B.weight"))}
    action_equal = all(torch.equal(v.cpu(), base[k]) for k, v in parent.action_expert.state_dict().items())
    check("teacher_tensors_equal_rt_teacher", video_equal and proprio_equal and action_equal,
          video=video_equal, proprio=proprio_equal, action=action_equal)
    del rt

    count = lambda module: sum(p.numel() for p in module.parameters())
    policy_parameters = count(parent.mot) + count(parent.proprio_encoder)
    check("parameter_counts", policy_parameters == 584_536_135 and count(parent.action_expert) == 105_277_959,
          base_policy=policy_parameters, action_expert=count(parent.action_expert))

    order = next(iter(ExactDistributedBatches(len(data), a.microbatch, 0, 1, 42)))
    batch = torch.utils.data.default_collate([data[i] for i in order])
    noise, anoise, atau = (x.to(dev) for x in sample_noise(batch["training_index"].tolist(), 0, 42))
    sample = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
    sample.update(action_scale=data.transform.normalizer["action"].scale.to(dev),
                  action_offset=data.transform.normalizer["action"].offset.to(dev))

    def build(name):
        config = ELASTIC[name]
        model = TransportPolicy(parent, **{k: config[k] for k in ("kind", "scope", "step_cond", "regime", "anchor")},
                                checkpoint_blocks=False, reuse_teacher_prefix=True, collect_diagnostics=True).to(dev)
        load_elastic_init(model, config, a.legacy_init if config["init"] == "rt_a2_legacy_v1" else a.common_init, teacher_sha)
        return model.train()
    def step(model, schedule):
        model.zero_grad(set_to_none=True); model.teacher_passes = 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, logs = model(sample, SCHEDULES[schedule], noise, anoise, atau)
        loss.backward()
        grads = {n: None if q.grad is None else q.grad.detach().clone() for n, q in model.named_parameters() if q.requires_grad}
        return loss.detach(), {k: v.detach() for k, v in logs.items()}, grads, model.teacher_passes
    def same(x, y):
        loss_a, logs_a, grads_a, _ = x; loss_b, logs_b, grads_b, _ = y
        return (torch.equal(loss_a, loss_b) and logs_a.keys() == logs_b.keys() and all(torch.equal(logs_a[k], logs_b[k]) for k in logs_a)
                and all((grads_a[k] is None and grads_b[k] is None) or torch.equal(grads_a[k], grads_b[k]) for k in grads_a))

    models = {name: build(name) for name in ("E0", "E1", "E2", "SP11")}
    reference = models["E0"].initial_state()
    common_ok = all(all(torch.equal(m.initial_state()[part][k], reference[part][k]) for k in reference[part])
                    for m in models.values() for part in ("student_action", "step_condition"))
    parent_action = parent.action_expert.state_dict()
    common_ok &= all(torch.equal(reference["student_action"][k], v.cpu()) for k, v in parent_action.items())
    trainable = {n: sum(q.numel() for q in m.parameters() if q.requires_grad) for n, m in models.items()}
    check("common_initialization_and_trainable_counts", common_ok and trainable == dict(E0=105_678_471, E1=105_678_471,
          E2=105_754_919, SP11=105_678_471), trainable=trainable,
          deployed_parameters=dict(E0=policy_parameters + 400_512, E2=policy_parameters + 400_512 + 76_448))

    e2 = models["E2"]; results = {}
    for name in FLAT9:
        e2.reuse_teacher_prefix = False; plain = step(e2, name)
        e2.reuse_teacher_prefix = True; reused = step(e2, name)
        finite = torch.isfinite(reused[0]).item() and all(g is None or torch.isfinite(g).all().item() for g in reused[2].values())
        transport_grads = [reused[2][n] for n in reused[2] if n.startswith("transport.")]
        k = SCHEDULES[name][0]
        participation = (all(g is None for g in transport_grads) if k == 4 else all(g is not None for g in transport_grads))
        others = all(g is not None for n, g in reused[2].items() if not n.startswith("transport."))
        results[name] = dict(reuse_bit_identical=same(plain, reused), finite=finite, transport_participation=participation,
                             student_grads_present=others, teacher_passes=(plain[3], reused[3]),
                             loss=float(reused[0]) / a.microbatch, diag=sorted(k_ for k_ in reused[1] if k_.startswith("diag_")))
        print(json.dumps({name: results[name]}), flush=True)
    check("flat9_paths_E2", all(r["reuse_bit_identical"] and r["finite"] and r["transport_participation"] and r["student_grads_present"]
                                for r in results.values()), paths=results)

    e2.checkpoint_blocks = True; ck = step(e2, "flat_k1_n5"); e2.checkpoint_blocks = False; nock = step(e2, "flat_k1_n5")
    e0 = models["E0"]; e0.checkpoint_blocks = True; ck0 = step(e0, "flat_k2_n10"); e0.checkpoint_blocks = False; nock0 = step(e0, "flat_k2_n10")
    check("adaptive_checkpointing_bit_identical", same(ck, nock) and same(ck0, nock0))

    for name in ("E0", "E1", "E2", "SP11"): models[name].cpu()
    del models, e0, e2; torch.cuda.empty_cache()
    f0 = build("F0-PB"); front = {}
    for name in ("S1", "S2", "S5", "S6", "S7", "S8"):
        f0.reuse_teacher_prefix = False; plain = step(f0, name)
        f0.reuse_teacher_prefix = True; reused = step(f0, name)
        lora = [g for n, g in reused[2].items() if n.endswith((".A.weight", ".B.weight"))]
        front[name] = dict(reuse_bit_identical=same(plain, reused), finite=torch.isfinite(reused[0]).item(),
                           lora_grads=len(lora) == 120 and all(g is not None for g in lora))
    trainable_f0 = sum(q.numel() for q in f0.parameters() if q.requires_grad)
    check("front6_paths_F0PB", all(all(r.values()) for r in front.values()) and trainable_f0 == 2_064_384 + 400_512,
          paths=front, trainable=trainable_f0)
    f0.cpu(); del f0; torch.cuda.empty_cache()

    # Export the initialized E2 student, reload through the deployment adapter, compare rollouts.
    e2 = build("E2")
    with torch.no_grad():  # move away from the zero-initialized output layers so the test is informative
        for module in (e2.step_condition.net[-1], e2.transport.B):
            module.weight.normal_(0, 1e-3, generator=torch.Generator(device=dev).manual_seed(5))
    out = Path(a.out).parent / "export_probe"; out.mkdir(parents=True, exist_ok=True)
    export_epoch(e2, out, 10, ELASTIC["E2"], 0, dict(global_batch=128, windows=277713))
    loaded = TransportInference(str(out / "epoch_10.pt"), vae, device=str(dev))
    one = {k: (v[:1] if torch.is_tensor(v) and v.ndim and v.shape[0] == len(noise) else v) for k, v in sample.items()}
    rollouts = {}
    e2.eval()
    for name in ("flat_k1_n2", "flat_k2_n5", "flat_k4_n10"):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            expected = e2.rollout(one, SCHEDULES[name], noise[:1])
        eager = loaded.actions(one, noise[:1], name, use_graph=False)
        graph = loaded.actions(one, noise[:1], name, use_graph=True)
        graph2 = loaded.actions(one, noise[:1], name, use_graph=True)
        rollouts[name] = dict(eager_equal=torch.equal(eager, expected), graph_equal=torch.equal(graph, eager) and torch.equal(graph2, eager))
    t4_bypass = torch.equal(loaded.policy.transport(torch.zeros(1, 32, 512, device=dev), None, torch.ones(1, device=dev)),
                            torch.zeros(1, 32, 512, device=dev))
    payload = torch.load(out / "epoch_10.pt", map_location="cpu", weights_only=False, mmap=True)
    check("export_reload_rollout", all(all(r.values()) for r in rollouts.values()) and t4_bypass
          and payload["transport_config"]["schedules"] == list(FLAT9) and payload["step_condition"] is not None,
          rollouts=rollouts)
    for f in out.iterdir(): f.unlink()
    out.rmdir()
    report.update(finished=time.time(), passed=True)
    Path(a.out).write_text(json.dumps(report, indent=2, default=str))

if __name__ == "__main__":
    main()
