#!/usr/bin/env bash
# Inside an srun step on 990547: after the ffn_magnitude evaluation is done and all four GPUs are free, run the
# reference-policy harness parity check (parity_reference.py, LIBERO-Long seed 42, 500 episodes).
set -uo pipefail
E=/groups/yshang/an221229/checkpoints/LoopWAM_NT/v0_init_variants_work/eval
source $E/eval_env.sh || exit 1
cd $E
echo "$(date) parity step job=$SLURM_JOB_ID step=$SLURM_STEP_ID"
done_ffn() {
    [[ -f $EVAL_RUNS/ffn_magnitude/eval_matched/summary.json ]] || \
        { grep -q "eval step finished" logs/eval_step_990547_gpu01.log && grep -q "eval step finished" logs/eval_step_990547_gpu23.log; }
}
gpus_free() {
    local g; for g in 0 1 2 3; do
        [[ $(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits) -lt 2000 ]] || return 1
    done
}
until done_ffn && gpus_free; do sleep 60; done
echo "$(date) starting parity"
python parity_reference.py --output-dir $EVAL_RUNS/parity_reference_v0_video4_action4 --vae-path $EVAL_VAE \
    --text-cache-dir $EVAL_TEXT --gpus 0,1,2,3
echo "$(date) parity step finished: exit $?"
