#!/usr/bin/env bash
# All single-GPU checks of the port at once (one 8-GPU step): preflight_rt50 parts on GPUs 0-3, bench_rt50 probes on 4-7.
set -uo pipefail
source "$(dirname "$0")/env.sh"
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
V=$ERT_WORK/validation; mkdir -p "$V"; cd "$ELASTIC_ROOT"
# 15 disjoint NUMA-local CPUs per GPU (job cpuset 0-29,32-61,64-93,96-125; GPUs 0-3 on NUMA 0, 4-7 on NUMA 1)
cpus() { case $1 in 0) echo 0-7,64-70;; 1) echo 8-14,71-78;; 2) echo 15-22,79-85;; 3) echo 23-29,86-93;;
                    4) echo 32-39,96-102;; 5) echo 40-46,103-110;; 6) echo 47-54,111-117;; 7) echo 55-61,118-125;; esac; }
run() { local gpu=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu taskset -c "$(cpus "$gpu")" python -u "$@" > "$V/$name.log" 2>&1; echo "$name rc=$?" >> "$V/gpu_checks.rc"; }
: > "$V/gpu_checks.rc"
run 0 preflight_core scripts/preflight_rt50.py --common-init "$ELASTIC_INIT" --checks identity,dataset,teacher,init --out "$V/preflight_core.json" &
run 1 preflight_speed_E0 scripts/preflight_rt50.py --common-init "$ELASTIC_INIT" --checks "" --speed-runs RT50-E0 --out "$V/preflight_speed_E0.json" &
run 2 preflight_speed_E1 scripts/preflight_rt50.py --common-init "$ELASTIC_INIT" --checks "" --speed-runs RT50-E1 --out "$V/preflight_speed_E1.json" &
run 3 preflight_speed_E2 scripts/preflight_rt50.py --common-init "$ELASTIC_INIT" --checks "" --speed-runs RT50-E2 --out "$V/preflight_speed_E2.json" &
run 4 bench_E0_mb16 scripts/bench_rt50.py --run RT50-E0 --common-init "$ELASTIC_INIT" --microbatch 16 --options bf16+graph+nofill --out "$V/bench_E0_mb16.json" &
run 5 bench_E0_mb32 scripts/bench_rt50.py --run RT50-E0 --common-init "$ELASTIC_INIT" --microbatch 32 --options bf16+graph+nofill --out "$V/bench_E0_mb32.json" &
run 6 bench_E2_mb16 scripts/bench_rt50.py --run RT50-E2 --common-init "$ELASTIC_INIT" --microbatch 16 --options bf16+graph+nofill --out "$V/bench_E2_mb16.json" &
run 7 bench_E2_mb32 scripts/bench_rt50.py --run RT50-E2 --common-init "$ELASTIC_INIT" --microbatch 32 --options bf16+graph+nofill --out "$V/bench_E2_mb32.json" &
wait
echo GPU_CHECKS_DONE >> "$V/gpu_checks.rc"
