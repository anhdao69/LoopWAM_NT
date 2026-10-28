#!/usr/bin/env bash
# Inside an srun step: matched 6,000-episode evaluation of HF-downloaded init-variant exports (RUNS, default
# no_video_pretraining then core_mean) on GPUS with WPG workers per GPU, after the GPUs are free and the
# CUDA-graph gate passes. Exports are hash-checked against their export.json before use.
set -uo pipefail
E=/groups/yshang/an221229/checkpoints/LoopWAM_NT/v0_init_variants_work/eval
source $E/eval_env.sh || exit 1
cd $E
HF=$E/hf_models/libero-all-suites/init_variants
GPUS=${GPUS:?}; WPG=${WPG:-4}; RUNS=${RUNS:-"no_video_pretraining core_mean"}
echo "$(date) hf eval step job=$SLURM_JOB_ID step=$SLURM_STEP_ID host=$(hostname) GPUS=$GPUS WPG=$WPG RUNS=$RUNS code=$(git -C $LOOPWAM_ROOT rev-parse HEAD)"
gpus_free() {
    local g; for g in ${GPUS//,/ }; do
        [[ $(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits) -lt 2000 ]] || return 1
    done
}
gate_passed() {
    python -c "import json,sys; sys.exit(0 if json.load(open('$E/logs/gate_$1.json'))['all_bit_exact'] is True else 1)" 2>/dev/null
}
for run in $RUNS; do
    expected=$(python -c "import json; print(json.load(open('$HF/$run/export.json'))['export_sha256'])")
    until gpus_free; do echo "$(date) waiting for GPUs $GPUS to be free"; sleep 60; done
    common=(--run $run --checkpoint $HF/$run/policy.pt --expected-sha $expected --vae-path $EVAL_VAE
            --text-cache-dir $EVAL_TEXT --output-dir $EVAL_RUNS/$run/eval_matched)
    if ! gate_passed $run; then
        first=${GPUS%%,*}
        CUDA_VISIBLE_DEVICES=$first MUJOCO_EGL_DEVICE_ID=$first python evaluate_v0_matched.py "${common[@]}" \
            --gate --gate-queries 8 --gate-output $E/logs/gate_$run.json > $E/logs/gate_${run}_${SLURM_JOB_ID}_$first.log 2>&1
        gate_passed $run || { echo "$(date) CUDA graph gate FAILED for $run; not evaluating it"; continue; }
    fi
    for attempt in 1 2 3; do
        echo "$(date) $run runner attempt $attempt"
        python evaluate_v0_matched.py "${common[@]}" --gpus $GPUS --workers-per-gpu $WPG --no-videos && break
        echo "$(date) $run runner attempt $attempt failed"; sleep 60
    done
done
echo "$(date) hf eval step finished"
