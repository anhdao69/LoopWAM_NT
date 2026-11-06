"""Scientific experiment matrix and explicit, evidence-backed submission gates."""
from fastwam.models.wan22.residual_transport import SCHEDULES

def spec(kind="T4",scope="pb",regime="fu",schedules=("S1","S2"),step_cond=False,anchor=True,**kw):
    return dict(kind=kind,scope=scope,regime=regime,schedules=list(schedules),
                step_cond=step_cond,anchor=anchor,seed=42,**kw)
EXPERIMENTS={
    "RT-A":spec(),
    "RT-B4":spec(kind="T0"),
    "RT-T1ft":spec(kind="T1",gate="round2"),
    "RT-Pa":spec(scope="pa",anchor=False,gate="round2"),
    "RT-TF":spec(regime="tf",gate="round3"),
    "RT-A2":spec(schedules=("S1","S2","S5","S6","S7","S8"),step_cond=True,gate="phase2"),
    "RT-B2a":spec(kind="T0",scope="pc",schedules=("parent2",),step_cond=True,gate="phase2"),
    "RT+B2":spec(schedules=("S8","S7"),step_cond=True,gate="stack",init_run="RT-B2a"),
    "RT-B4-2":spec(kind="T0",schedules=("S1","S2","S5","S6","S7","S8"),step_cond=True,gate="stack"),
}
# Explicitly defined deferred axes are implemented and represented, but gated.
for kind in ("T3","T5","T6"):
    EXPERIMENTS["RT-"+kind]=spec(kind=kind,gate="both_phases_read")
EXPERIMENTS["RT-Pc"]=spec(scope="pc",gate="both_phases_read")
EXPERIMENTS["RT-NFE1"]=spec(kind="T0",scope="pc",schedules=("parent1",),step_cond=True,gate="both_phases_read")
for term in ("local","end","grip","anchor"):
    weights=dict(local=1.,end=.5,grip=.1,anchor=.5); weights[term]=0.
    EXPERIMENTS["RT-no-"+term]=spec(loss_weights=weights,gate="both_phases_read")
for seed in (43,44):
    EXPERIMENTS[f"RT-A-seed{seed}"]={**spec(gate="both_phases_read"),"seed":seed}

# Elastic campaign (plans/LoopWAM_elastic.md, section 8). A separate registry, so the RT
# gates above and their recorded evidence are unchanged; no elastic run waits on an RT gate.
# The teacher-prefix reuse is bit-identical to recomputing the first local target, and the
# diagnostics are detached, so neither changes the optimized objective.
def elastic(kind,scope="pc",schedules=None,sampler="balanced",init="common_v1",**kw):
    from fastwam.models.wan22.residual_transport import FLAT9
    return dict(kind=kind,scope=scope,regime="fu",schedules=list(schedules or FLAT9),
                step_cond=True,anchor=True,seed=42,sampler=sampler,init=init,
                reuse_teacher_prefix=True,collect_diagnostics=True,campaign="elastic-v1",**kw)
ELASTIC={
    "E0":elastic("T0"),
    "E1":elastic("T1"),
    # T4 is bypassed on full-depth updates, so DDP must tolerate globally unused parameters.
    "E2":elastic("T4",find_unused_parameters=True),
    "SP11":elastic("T0",schedules=("flat_k1_n2",),sampler="fixed"),
    # Historical RT-A2 recipe with transport off: legacy selector and reconstructed RT-A2 init.
    "F0-PB":elastic("T0",scope="pb",schedules=("S1","S2","S5","S6","S7","S8"),
                    sampler="legacy_choice",init="rt_a2_legacy_v1"),
    # Verification only: RT-A2 itself (T4) from the reconstructed init, to replay its first updates.
    "RT-A2-replay":elastic("T4",scope="pb",schedules=("S1","S2","S5","S6","S7","S8"),
                           sampler="legacy_choice",init="rt_a2_legacy_v1",verification_only=True),
}
# CL-0 campaign (plan section 16.1): the E0/E1/E2 recipes unchanged, with the memoryless ChronoLoop CL-0
# (anhdao69/ChronoLoop CL-0_no-memory_a4: LoopWAM v0 4/4, BASE10 fine-tuned ten more LIBERO epochs) as both the
# frozen teacher and the student initializer. The teacher file is its loopwam-s-v1 re-wrap
# (scripts/convert_chronoloop_parent.py); the campaign tag stays "elastic-v1" so evaluation applies the same rules.
CL0_PARENT=dict(name="CL-0_no-memory_a4",repository="anhdao69/ChronoLoop",file="CL-0_no-memory_a4/epoch_10/policy.pt",
                chronoloop_sha256="137f0a839b5a1e2363621420ab599cd36a5693000f784a04a11b9bea2bea2cae",
                grandparent_sha256="1c659a7c2f6970c3ea5972a8fba4abd9db35857c83a67a053c0ee35555fb602b")
for _name in ("E0","E1","E2"):
    ELASTIC[f"CL0-{_name}"]=dict(ELASTIC[_name],parent=CL0_PARENT)
# RoboTwin-50 port (2026-11-06): the E0/E1/E2 recipes on the RoboTwin-50 v0 training data, with the full-RoboTwin
# LoopWAM v0 4/4 epoch-10 checkpoint (loopwam-s-robot-v1, = HF robotwin/v0-video4-action4/epoch_10/policy.pt) as the
# frozen teacher and the student initializer. Two embodiment adaptations of the endpoint loss: RoboTwin executes 24
# actions per query (end over 24, LIBERO 10), and its grippers are continuous qpos targets, so LIBERO's binarized
# gripper hinge does not apply (grip off). Everything else (losses, schedules, sampler, LRs, init rule) is unchanged.
RT50_PARENT=dict(name="LoopWAM-v0-full-RoboTwin epoch 10",repository="anhdao69/LoopWAM_NT",
                 file="robotwin/v0-video4-action4/epoch_10/policy.pt",format="loopwam-s-robot-v1",step=474620,
                 sha256="70cf5af73db5fe582c50c5f9fb68296d5271a9c0fea51509b3cdb5f9696be1ea")
# RT50-ep5-*: the same recipes distilled from the same run's epoch 5 (identical data/normalization/VAE contract) =
# HF robotwin/v0-video4-action4/policy.pt (its checkpoint_provenance.json: epoch 5, step 237310) = the cluster's
# runs/robotwin_full_v0_20261028/release_v2/production/checkpoints/epoch_05.pt (same bytes).
RT50_EP5_PARENT=dict(name="LoopWAM-v0-full-RoboTwin epoch 5",repository="anhdao69/LoopWAM_NT",
                     file="robotwin/v0-video4-action4/policy.pt",format="loopwam-s-robot-v1",
                     step=237310,sha256="38975220e9441303878d2dfb9956d57e6416c6a906c99c43d9bca015b092fc51")
for _name in ("E0","E1","E2"):
    ELASTIC[f"RT50-{_name}"]=dict(ELASTIC[_name],parent=RT50_PARENT,dataset="robotwin50",end_prefix=24,gripper_loss=False)
    ELASTIC[f"RT50-ep5-{_name}"]=dict(ELASTIC[f"RT50-{_name}"],parent=RT50_EP5_PARENT)
del _name

def eligible(name,evidence):
    gate=EXPERIMENTS[name].get("gate")
    if gate is None: return True
    base=evidence.get("gate0") is True and evidence.get("round1_complete") is True
    if gate=="round2": return base
    if gate=="round3": return base and evidence.get("round2_complete") is True
    if gate=="phase2": return base and evidence.get("gate1") is True
    if gate=="stack": return base and evidence.get("gate1") is True and evidence.get("RT-B2a_complete") is True
    if gate=="both_phases_read": return evidence.get("both_phases_read") is True
    return False
