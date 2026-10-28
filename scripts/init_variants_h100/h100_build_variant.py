#!/usr/bin/env python3
"""Run the branch's scripts/build_init_variant.py unchanged, except for the canonical donor artifact hash it pins.

The variant builder pins the canonical artifact of the earlier runs (a3bf2097...). That file does not exist on this
cluster; its rebuild here (a990a583..., the init of the v0 4/2 run) has identical video tensors and interpolated
action tensors within about one FP32 ulp (v0_video4_action2_work/evidence/fairness_evidence.json). The builder still
re-derives and checks every student tensor against the canonical artifact it was given and records that hash.
    python h100_build_variant.py --variant core_random --output .../init/core_random.pt [--threads N]
"""
import sys

import build_init_variant as builder

H100_CANONICAL_SHA256 = 'a990a5831c63e88ae5143689adad2f2aeb02cb1cd151d61ae0dd992990ac3296'

if __name__ == '__main__':
    builder.CANONICAL_SHA256 = H100_CANONICAL_SHA256
    sys.argv[0] = 'build_init_variant.py'
    builder.main()
