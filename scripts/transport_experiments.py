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
