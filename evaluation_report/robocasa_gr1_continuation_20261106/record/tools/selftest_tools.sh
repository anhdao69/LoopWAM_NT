#!/bin/bash
# Synthetic self-tests of stage_cache.py, gate_check.py and the results report (2026-11-06). Scratch: /tmp/gr1cont_selftest.
set -u
CR=/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robocasa_gr1_v0_20261101/continuation_20261106
source $CR/env.sh > /dev/null 2>&1
X=/tmp/gr1cont_selftest; rm -rf $X; mkdir -p $X/src
python - <<'PY'
import json, numpy as np
X = '/tmp/gr1cont_selftest'
size, shape = 3000, [16, 3, 32, 32]
rng = np.random.default_rng(0)
rng.integers(0, 65535, size=(size, *shape), dtype=np.uint16).tofile(f'{X}/src/latents.uint16')
np.ones(size, np.uint8).tofile(f'{X}/src/valid.uint8')
open(f'{X}/src/metadata.json', 'w').write(json.dumps(dict(size=size, latent_shape=shape, format='t'), sort_keys=True) + '\n')
PY
sed "s#^ALLOWED_DST = .*#ALLOWED_DST = '$X/dst/latents'#" $CR/tools/stage_cache.py > $X/stage_test.py
python $X/stage_test.py $X/src $X/dst/latents --threads 4 | tail -1 | cut -c1-200
cmp $X/src/latents.uint16 $X/dst/latents/latents.uint16 && cmp $X/src/valid.uint8 $X/dst/latents/valid.uint8 && cmp $X/src/metadata.json $X/dst/latents/metadata.json && echo 'STAGE: byte-identical'
python $X/stage_test.py $X/src $X/dst/latents | cut -c1-60
rm $X/dst/latents/STAGED.json; python $X/stage_test.py $X/src $X/dst/latents | grep -oE 'removing_partial|"event": "done"' | tr '\n' ' '; echo
python $CR/tools/stage_cache.py $X/src $X/dst/latents 2>&1 | tail -1 | cut -c1-100
# gate_check on synthetic gate dirs
python - <<'PY'
import json, torch, train_loopwam_gr1 as T
X = '/tmp/gr1cont_selftest'; P = T.PARENT_UPDATES
import os
for name, last, resume in (('g1', P + 20, None), ('g2', P + 30, P + 20), ('bad', P + 20, None)):
    d = f'{X}/{name}'; os.makedirs(d)
    with open(f'{d}/metrics.jsonl', 'w') as f:
        for u in range(P + 1, last + 1):
            lr = 3e-5 * T.continuation_factor(u - 1) * (1.5 if name == 'bad' and u == P + 7 else 1)
            f.write(json.dumps(dict(update=u, loss=.03, loss_video=.027, loss_action=.003, grad_norm=.05, lr_backbone=lr, valid_windows=256,
                                    seconds=1.24, data_wait_seconds=0., peak_allocated_gb=55.5)) + '\n')
    json.dump(dict(status='max_updates', update=last), open(f'{d}/timing.json', 'w'))
    json.dump(dict(all_gradients_present_and_finite=True), open(f'{d}/precision.json', 'w'))
    torch.save(dict(step=last, training_state=dict(update=last, contract=dict(epochs=10, lr_backbone=3e-5, phase_start_update=P))), f'{d}/latest.pt')
    if resume:
        json.dump(dict(passed=True, next_lr=3e-5 * T.continuation_factor(resume), continuation=True), open(f'{d}/resume_audit_u{resume}.json', 'w'))
PY
P=117580
python $CR/tools/gate_check.py $X/g1 $((P + 20)) > /dev/null && echo 'GATE g1: pass (expected pass)' || echo 'GATE g1: FAIL (unexpected)'
python $CR/tools/gate_check.py $X/g2 $((P + 30)) --resume-from $((P + 20)) > /dev/null && echo 'GATE g2: pass (expected pass)' || echo 'GATE g2: FAIL (unexpected)'
python $CR/tools/gate_check.py $X/bad $((P + 20)) > $X/bad.out; echo "GATE bad: rc=$? (expected 1): $(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["problems"])' $X/bad.out)"
# report with the existing epoch 4/5 validations
EV=/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robocasa_gr1_v0_20261101/eval
python3 /mnt/data/vmo-ai-task/anhdh35/gr1_eval/tools/gr1_report.py $X/report.md "H100 ep4=$EV/official_epoch04_validation.json" "H100 ep5=$EV/official_epoch05_validation.json" > /dev/null && grep -E 'Average' $X/report.md
rm -rf $X
