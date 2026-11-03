#!/usr/bin/env python3
"""Real-model preflight of the CL-0 elastic campaign on one GPU. Writes one JSON report.

Checks: converted CL-0 teacher identity (re-wrap record, pinned hashes, VAE); data identity (277,713 windows,
normalization d5ad351e); deduplicated parameter counts; common_v1 initialization = CL-0 action expert and the
E0/E1/E2 trainable counts; then, for each speed option the trainer offers (--bf16-frozen, --teacher-graph, both),
bit equality of the loss, every logged component and every gradient against the eager path, over the flat paths
named in --schedules and over two different micro-batches (the second exercises the graph's copy-in path).
"""
import argparse, json, os, time
from pathlib import Path
import torch

CL0_SHA = "137f0a839b5a1e2363621420ab599cd36a5693000f784a04a11b9bea2bea2cae"
BASE10_SHA = "1c659a7c2f6970c3ea5972a8fba4abd9db35857c83a67a053c0ee35555fb602b"
VAE_SHA = "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
NORM_SHA = "d5ad351eb3272f416d34082e8af5483ba1118ceca355a37d9f268fdb269fb3e4"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--common-init", required=True); p.add_argument("--microbatch", type=int, default=8)
    p.add_argument("--runs", default="CL0-E0,CL0-E2")
    p.add_argument("--schedules", default="flat_k1_n2,flat_k2_n5,flat_k4_n10,flat_k1_n10")
    p.add_argument("--options", default="bf16_frozen,teacher_graph,no_fill,all")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.use_deterministic_algorithms(True)
    dev = torch.device("cuda", 0); torch.cuda.set_device(dev)
    from fastwam.datasets.transport_cached import TransportCachedDataset
    from fastwam.models.wan22.loopwam import create_loopwam
    from fastwam.models.wan22.transport_policy import TransportPolicy
    from fastwam.models.wan22.residual_transport import SCHEDULES
    from fastwam.models.wan22.loopwam_init import sha256_file
    from train_loopwam import ExactDistributedBatches
    from train_transport import sample_noise, load_elastic_init, bf16_frozen
    from transport_experiments import ELASTIC
    report = dict(started=time.time(), checks={})
    def check(name, ok, **detail):
        report["checks"][name] = dict(passed=bool(ok), **detail)
        print(json.dumps({name: report["checks"][name]}, default=str), flush=True)
        Path(a.out).write_text(json.dumps(report, indent=2, default=str))
        if not ok: raise SystemExit(f"preflight failed: {name}")

    teacher, vae = os.environ["ELASTIC_TEACHER"], os.environ["ELASTIC_VAE"]
    record = json.loads(Path(teacher).with_suffix(".json").read_text())
    teacher_sha = sha256_file(teacher)
    check("asset_identity", record["sha256"] == teacher_sha and record["source_sha256"] == CL0_SHA
          and record["template_sha256"] == BASE10_SHA and record["tensors_bit_equal_to_source"] is True
          and sha256_file(vae) == VAE_SHA, teacher_sha256=teacher_sha, source_sha256=record["source_sha256"],
          mot_tensors_changed_vs_base10=record["mot_tensors_changed_vs_parent"])
    data = TransportCachedDataset(os.environ["ELASTIC_MANIFEST"], os.environ["ELASTIC_LATENTS"])
    staged = Path(os.environ["ELASTIC_LATENTS"]) / "STAGED.json"
    check("data_identity", len(data) == 277713 and data.manifest["normalization_sha256"] == NORM_SHA,
          windows=len(data), staged=json.loads(staged.read_text())["chunk_sha256_digest"] if staged.exists() else None)

    def parent_model():
        return create_loopwam(checkpoint_path=teacher, vae_path=vae, device=str(dev)).requires_grad_(False).eval()
    eager_parent = parent_model()
    count = lambda module: sum(q.numel() for q in module.parameters())
    policy_parameters = count(eager_parent.mot) + count(eager_parent.proprio_encoder)
    check("parameter_counts", policy_parameters == 584_536_135 and count(eager_parent.action_expert) == 105_277_959,
          base_policy=policy_parameters, action_expert=count(eager_parent.action_expert))

    def batch(index):
        batches_ = iter(ExactDistributedBatches(len(data), a.microbatch, 0, 1, 42))
        for _ in range(index + 1): order = next(batches_)
        b = torch.utils.data.default_collate([data[i] for i in order])
        noise, anoise, atau = (x.to(dev) for x in sample_noise(b["training_index"].tolist(), 0, 42))
        sample = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()}
        sample.update(action_scale=data.transform.normalizer["action"].scale.to(dev),
                      action_offset=data.transform.normalizer["action"].offset.to(dev))
        return sample, noise, anoise, atau
    batches = [batch(0), batch(1)]

    def build(parent, name, **kw):
        config = ELASTIC[name]
        model = TransportPolicy(parent, **{k: config[k] for k in ("kind", "scope", "step_cond", "regime", "anchor")},
                                checkpoint_blocks=False, reuse_teacher_prefix=True, collect_diagnostics=True, **kw).to(dev)
        load_elastic_init(model, config, a.common_init, teacher_sha)
        return model.train()
    def step(model, schedule, data_):
        sample, noise, anoise, atau = data_
        model.zero_grad(set_to_none=True); model.teacher_passes = 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, logs = model(sample, SCHEDULES[schedule], noise, anoise, atau)
        loss.backward()
        grads = {n: None if q.grad is None else q.grad.detach().clone() for n, q in model.named_parameters() if q.requires_grad}
        return loss.detach(), {k: v.detach() for k, v in logs.items()}, grads, model.teacher_passes
    def compare(x, y):
        loss_a, logs_a, grads_a, passes_a = x; loss_b, logs_b, grads_b, passes_b = y
        diff = max([(g.float() - grads_b[k].float()).abs().max().item() for k, g in grads_a.items()
                    if g is not None and grads_b[k] is not None] or [0.])
        equal = (torch.equal(loss_a, loss_b) and logs_a.keys() == logs_b.keys() and all(torch.equal(logs_a[k], logs_b[k]) for k in logs_a)
                 and all((grads_a[k] is None and grads_b[k] is None) or torch.equal(grads_a[k], grads_b[k]) for k in grads_a))
        return dict(bit_equal=equal, passes=(passes_a, passes_b), loss=(float(loss_a), float(loss_b)), max_grad_abs_diff=diff)

    reference = build(eager_parent, "CL0-E0").initial_state()
    parent_action = eager_parent.action_expert.state_dict()
    init_ok = all(torch.equal(reference["student_action"][k], v.cpu()) for k, v in parent_action.items())
    trainable = {}
    for name in ("CL0-E0", "CL0-E1", "CL0-E2"):
        m = build(eager_parent, name); trainable[name] = sum(q.numel() for q in m.parameters() if q.requires_grad)
        state = m.initial_state()
        init_ok &= all(torch.equal(state[part][k], reference[part][k]) for part in ("student_action", "step_condition") for k in reference[part])
        del m
    check("common_initialization_and_trainable_counts", init_ok and trainable == {"CL0-E0": 105_678_471,
          "CL0-E1": 105_678_471, "CL0-E2": 105_754_919}, trainable=trainable)
    torch.cuda.empty_cache()

    schedules = a.schedules.split(",")
    options = dict(bf16_frozen=dict(bf16=True, graph=False, nofill=False), teacher_graph=dict(bf16=False, graph=True, nofill=False),
                   no_fill=dict(bf16=False, graph=False, nofill=True), all=dict(bf16=True, graph=True, nofill=True))
    options = {k: v for k, v in options.items() if k in a.options.split(",")}
    for run in a.runs.split(","):
        eager = build(eager_parent, run)
        expected = {(s, i): step(eager, s, batches[i]) for s in schedules for i in (0, 1)}
        del eager; torch.cuda.empty_cache()
        for option, flags in options.items():
            torch.utils.deterministic.fill_uninitialized_memory = not flags["nofill"]
            fast_parent = parent_model()
            fast = build(fast_parent, run, teacher_graph=flags["graph"])
            converted = bf16_frozen(fast) if flags["bf16"] else 0
            rows = {}
            for s in schedules:
                for i in (0, 1):
                    rows[f"{s}/batch{i}"] = compare(expected[(s, i)], step(fast, s, batches[i]))
            graph = getattr(fast, "_teacher_graph", None)
            check(f"speed_{option}_{run}", all(r["bit_equal"] and r["passes"][0] == r["passes"][1] for r in rows.values()),
                  converted_linear=converted, graph_captures=None if graph is None else graph.captures, paths=rows)
            del fast, fast_parent, graph; torch.cuda.empty_cache()
            torch.utils.deterministic.fill_uninitialized_memory = True
        del expected
    report.update(finished=time.time(), passed=True)
    Path(a.out).write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
