#!/usr/bin/env python3
"""Submit only scientifically eligible runs, with durable job IDs and deduplication."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
from transport_experiments import EXPERIMENTS,eligible

def planned_runs(evidence, independent_phase1=False, independent_phase2_pair=False):
    if independent_phase2_pair:
        return ["RT-A2","RT-B2a"]
    if independent_phase1:
        return ["RT-A","RT-B4","RT-T1ft","RT-Pa","RT-TF"]
    return [name for name in EXPERIMENTS if eligible(name,evidence)]

def retryable_state(state):
    return state.split("+")[0] in {"FAILED","TIMEOUT","NODE_FAIL","OUT_OF_MEMORY","PREEMPTED","BOOT_FAIL"}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--output-root",required=True)
    p.add_argument("--gates")
    p.add_argument("--submit",action="store_true")
    p.add_argument("--independent-phase1",action="store_true",
        help="Explicit user override: queue all five phase-1 runs without evaluation waits or job dependencies")
    p.add_argument("--independent-phase2-pair",action="store_true",
        help="Explicit user override: select RT-A2/RT-B2a without Gate 1 or dependencies")
    p.add_argument("--gpus",type=int,choices=(1,2),default=2)
    p.add_argument("--microbatch",type=int)
    p.add_argument("--checkpoint-blocks",action=argparse.BooleanOptionalAction,default=None)
    p.add_argument("--interactive-gpus",type=int,choices=(1,2),default=2)
    p.add_argument("--retry-failed",action="store_true")
    p.add_argument("--interactive-run",help="Run already owned by an interactive Slurm step")
    p.add_argument("--interactive-job",type=int)
    a=p.parse_args()
    source=Path(a.source).resolve(); out=Path(a.output_root).resolve(); out.mkdir(parents=True,exist_ok=True)
    evidence=json.loads(Path(a.gates).read_text()) if a.gates else {}
    if a.independent_phase1 and a.independent_phase2_pair: p.error("Choose one explicit override group")
    plans=planned_runs(evidence,a.independent_phase1,a.independent_phase2_pair)
    if not a.submit:
        print(json.dumps(dict(eligible=plans,gated=[r for r in EXPERIMENTS if r not in plans]),indent=2))
        return
    verification=json.loads((source/"VERIFIED.json").read_text())
    if not all(verification.get(k) is True for k in ("native_variants","resume","data","cuda_graphs")):
        raise ValueError("Source has not passed required preflight")
    ledger=out/"submissions.json"
    with (out/"submission.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        records=json.loads(ledger.read_text()) if ledger.exists() else {}
        queue=subprocess.check_output(["squeue","-u",os.environ["USER"],"-h","-o","%i|%j|%k"],text=True)
        for name in plans:
            previous=None
            if name in records:
                if not a.retry_failed: continue
                previous=records[name]
                status=subprocess.check_output(["sacct","-X","-n","-P","-j",previous["job_id"],
                                                "--format=State"],text=True).strip().splitlines()
                if not status or not retryable_state(status[0].strip("|")): continue
                if previous.get("launch")=="interactive_step":
                    raise ValueError("Confirm the interactive training step ended before batch recovery")
                # Preserve history while resuming the same output from latest.pt.
                records.pop(name)

            comment="residual-transport:"+name+":"+str(out)
            existing=[line.split("|")[0] for line in queue.splitlines() if line.endswith("|"+comment)]
            if existing:
                records[name]=dict(job_id=existing[0],recovered_from_queue=True)
            elif name==a.interactive_run:
                if not a.interactive_job: raise ValueError("Interactive allocation ID required")
                subprocess.check_call(["scontrol","show","job",str(a.interactive_job)],stdout=subprocess.DEVNULL)
                records[name]=dict(job_id=str(a.interactive_job),launch="interactive_step",
                                   output=str(out/name),source=str(source),gpus=a.interactive_gpus)
            else:
                env=os.environ.copy()
                if a.independent_phase1 or a.independent_phase2_pair: env.pop("SBATCH_DEPENDENCY",None)
                env.update(RT_SOURCE=str(source),RT_RUN=name,RT_OUTPUT=str(out/name),
                    RT_WORLD_SIZE=str(a.gpus),
                    RT_MICROBATCH=str(a.microbatch or (8 if EXPERIMENTS[name]["scope"]=="pc" else 64)),
                    RT_CHECKPOINT_BLOCKS=str(int(a.checkpoint_blocks if a.checkpoint_blocks is not None else EXPERIMENTS[name]["scope"]=="pc")))
                if EXPERIMENTS[name].get("init_run"):
                    env["RT_INIT_CHECKPOINT"]=str(out/EXPERIMENTS[name]["init_run"]/"epoch_10.pt")
                command=["sbatch","--parsable","--job-name=rt_"+name,"--comment="+comment,
                         f"--gres=gpu:nvidia_h100_80gb_hbm3:{a.gpus}",
                         "--cpus-per-task="+str(15*a.gpus),"--mem="+str(128*a.gpus)+"G",
                         "--output="+str(out/(name+"_%j.slurm.log")),str(source/"scripts/submit_transport.sbatch")]
                job=subprocess.check_output(command,env=env,text=True).strip().split(";")[0]
                if not job.isdigit(): raise RuntimeError("Unexpected sbatch response")
                records[name]=dict(job_id=job,launch="batch",output=str(out/name),source=str(source),gpus=a.gpus)
            if a.independent_phase1:
                records[name]["submission_policy"]="user_requested_independent_phase1"
                records[name]["dependencies"]=[]
            if a.independent_phase2_pair:
                records[name]["submission_policy"]="user_requested_independent_phase2_pair"
                records[name]["dependencies"]=[]
            if previous is not None: records[name]["previous_attempt"]=previous
            temp=ledger.with_suffix(".tmp"); temp.write_text(json.dumps(records,indent=2)); temp.replace(ledger)
            print(json.dumps({name:records[name]}),flush=True)
        for name in EXPERIMENTS:
            if name not in plans and name not in records:
                print(json.dumps(dict(run=name,status="gated",gate=EXPERIMENTS[name].get("gate"))))
if __name__=="__main__": main()
