#!/usr/bin/env bash
#SBATCH --partition=highgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --time=1-00:00:00
#SBATCH --job-name=rt-missing-eval
# Continuation of the resumable RT missing-condition evaluation (group claims prevent duplicate work).
WPG=8 bash /lustre/fs1/home/an221229/code/LoopWAM_rt/scripts/run_rt_missing_eval.sh
