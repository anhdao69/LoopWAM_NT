#!/usr/bin/env bash
# MPS trial (inside an overlapping step of the eval job): starts a private MPS control daemon (node-local pipe/log
# directories, so only processes given CUDA_MPS_PIPE_DIRECTORY connect to it), runs the RT-A2 parity check on GPU $1
# under it, records where the MPS server and its clients sit, then shuts the daemon down.
set -uo pipefail
GPU=$1
M=/dev/shm/mps_test_${SLURM_JOB_ID:-x}
rm -rf "$M"; mkdir -p "$M/pipe" "$M/log"
export CUDA_MPS_PIPE_DIRECTORY=$M/pipe CUDA_MPS_LOG_DIRECTORY=$M/log
( unset CUDA_VISIBLE_DEVICES; nvidia-cuda-mps-control -d ) && echo "mps daemon started"
( for i in 1 2 3 4 5 6; do sleep 45; echo "--- nvidia-smi $(date +%T)"; nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader | grep -v -E "^$" ; nvidia-smi | grep -E "M\+C|mps" | head -n 8; done ) &
mon=$!
PARITY_OUT=/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_work/parity_rt_mps bash /groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_work/parity_rt.sh "$GPU"
kill $mon 2>/dev/null
echo quit | nvidia-cuda-mps-control; sleep 3
echo "--- mps control log"; tail -n 15 "$M/log/control.log" 2>/dev/null
echo "--- mps server log"; tail -n 8 "$M/log/server.log" 2>/dev/null
rm -rf "$M"
