#!/usr/bin/env python3
"""Check a bounded GR1 continuation gate run (trainer output dir) and print/write a JSON verdict; exit 1 on failure.

python gate_check.py OUT_DIR LAST_UPDATE [--resume-from U] [--report R.json]

Requires: metrics rows exactly 117581..LAST (after resume reconciliation), every loss/grad finite, every row 256 windows,
lr == 3e-5 * continuation_factor(update - 1) exactly as planned, losses close to the parent's final epoch (parent
last-2000 mean 0.0275, max 0.0347: here mean <= 0.05 and max <= 0.08), grad norm <= 1 (parent max 0.17),
timing.json status max_updates at LAST, latest.pt at LAST with the continuation contract, precision.json present,
and with --resume-from U a passed resume audit at U with next_lr = 3e-5 * factor(U).
"""
import argparse, json, math, statistics as st, sys
from pathlib import Path

import torch

import train_loopwam_gr1 as T


def main():
    p = argparse.ArgumentParser(); p.add_argument('out'); p.add_argument('last', type=int)
    p.add_argument('--resume-from', type=int); p.add_argument('--report')
    a = p.parse_args(); out = Path(a.out); problems = []
    rows = [json.loads(l) for l in (out / 'metrics.jsonl').read_text().splitlines()]
    ups = [r['update'] for r in rows]
    if ups != list(range(T.PARENT_UPDATES + 1, a.last + 1)):
        problems.append(f'metrics updates {ups[:3]}..{ups[-3:]} (n={len(ups)}) != {T.PARENT_UPDATES + 1}..{a.last}')
    for r in rows:
        vals = [r['loss'], r['loss_video'], r['loss_action'], r['grad_norm'], r['lr_backbone']]
        if not all(math.isfinite(v) for v in vals):
            problems.append(f'non-finite values at update {r["update"]}')
        if r['valid_windows'] != 256:
            problems.append(f'update {r["update"]}: {r["valid_windows"]} windows')
        want = 3e-5 * T.continuation_factor(r['update'] - 1)
        if abs(r['lr_backbone'] - want) > 1e-12 * want:
            problems.append(f'update {r["update"]}: lr {r["lr_backbone"]} != planned {want}')
    losses = [r['loss'] for r in rows]; grads = [r['grad_norm'] for r in rows]
    if rows and (st.mean(losses) > .05 or max(losses) > .08):
        problems.append(f'loss mean {st.mean(losses):.4f} / max {max(losses):.4f} above the parent-based bound (0.05 / 0.08)')
    if rows and max(grads) > 1.:
        problems.append(f'grad norm max {max(grads):.3f} > 1')
    timing = json.loads((out / 'timing.json').read_text())
    if (timing.get('status'), timing.get('update')) != ('max_updates', a.last):
        problems.append(f'timing.json {timing}')
    if not json.loads((out / 'precision.json').read_text()).get('all_gradients_present_and_finite'):
        problems.append('precision.json does not report finite gradients')
    ck = torch.load(out / 'latest.pt', map_location='cpu', mmap=True, weights_only=False)
    state = ck['training_state']; c = state['contract']
    if (ck['step'], state['update']) != (a.last, a.last) or (c.get('epochs'), c.get('lr_backbone'), c.get('phase_start_update')) != (10, 3e-5, T.PARENT_UPDATES):
        problems.append(f'latest.pt step {ck["step"]} update {state["update"]} contract epochs/lr/phase {c.get("epochs")}/{c.get("lr_backbone")}/{c.get("phase_start_update")}')
    del ck
    audit = None
    if a.resume_from is not None:
        audit = json.loads((out / f'resume_audit_u{a.resume_from}.json').read_text())
        want = 3e-5 * T.continuation_factor(a.resume_from)
        if not audit.get('passed') or abs(audit['next_lr'] - want) > 1e-12 * want or not audit.get('continuation'):
            problems.append(f'resume audit {audit}')
    secs = [r['seconds'] for r in rows[5:]] or [0.]
    rep = dict(out=str(out), last=a.last, passed=not problems, problems=problems[:20], rows=len(rows),
               loss_mean=st.mean(losses) if rows else None, loss_max=max(losses) if rows else None,
               loss_video_mean=st.mean(r['loss_video'] for r in rows) if rows else None,
               loss_action_mean=st.mean(r['loss_action'] for r in rows) if rows else None,
               grad_max=max(grads) if rows else None, first_lr=rows[0]['lr_backbone'] if rows else None,
               median_seconds_after5=st.median(secs), data_wait_max=max(r['data_wait_seconds'] for r in rows) if rows else None,
               peak_allocated_gb=max(r['peak_allocated_gb'] for r in rows) if rows else None, resume_audit=audit)
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep), flush=True)
    sys.exit(0 if not problems else 1)


if __name__ == '__main__':
    main()
