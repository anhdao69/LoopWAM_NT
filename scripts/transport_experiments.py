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
