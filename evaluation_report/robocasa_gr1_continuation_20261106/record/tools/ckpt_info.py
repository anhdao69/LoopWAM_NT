#!/usr/bin/env python3
"""Print 'step update next_epoch next_batch epochs phase_start' of a LoopWAM GR1 checkpoint (exit 1 if unreadable)."""
import sys

import torch

try:
    p = torch.load(sys.argv[1], map_location='cpu', mmap=True, weights_only=False)
    s = p['training_state']; c = s['contract']
    print(p['step'], s['update'], s['next_epoch'], s['next_batch'], c['epochs'], c.get('phase_start_update', -1))
except Exception as e:  # noqa: BLE001
    print(f'unreadable: {e!r}'); sys.exit(1)
