#!/usr/bin/env python3
"""Run scripts/gr1_precompute_latents.py with the Wan2.1 VAE in channels_last_3d memory format.

Speed A/B only: same weights, same bf16 compute and the same cache provenance (the provenance hashes the
VAE source and preprocessing code, not the memory layout). Conv algorithm choice changes, so latent bits can
differ at bf16 rounding level, as they already do between cudnn.benchmark picks for different batch sizes.
Same arguments as the script.
"""
import runpy
import sys
from pathlib import Path

import torch

import fastwam.models.wan22.loopwam_init as loopwam_init

_load = loopwam_init.load_wan21_vae


def load_channels_last(*args, **kwargs):
    vae = _load(*args, **kwargs)
    vae.model.to(memory_format=torch.channels_last_3d)
    return vae


loopwam_init.load_wan21_vae = load_channels_last
script = Path(__file__).resolve().parents[2] / 'scripts' / 'gr1_precompute_latents.py'
sys.argv = [str(script)] + sys.argv[1:]
runpy.run_path(str(script), run_name='__main__')
