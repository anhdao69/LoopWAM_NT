#!/usr/bin/env python3
"""T5 text cache for a ChronoLoop robot dataset: every prompt its training episodes use.

torchrun --standalone --nproc_per_node=2 scripts/chronoloop_robot_text_cache.py --dataset robotwin10b [--batch 32]

The encoder, tokenizer and prompt template are the RoboTwin policy's own (deploy_policy.TextEncoder: Wan2.2-TI2V-5B
umt5-xxl, 128 tokens, DEFAULT_PROMPT), so training and evaluation contexts come from the same code. Files are named
sha256(prompt).<text_suffix> in the spec's text_cache_dir and stored packed (only the unmasked rows; the readers zero
the rest anyway). Existing files are skipped. Rank 0 first checks that batched encoding equals batch-1 encoding,
the evaluation path, on a few prompts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/robotwin/chronoloop_policy'))
from deploy_policy import TextEncoder  # noqa: E402

from fastwam.datasets.chronoloop_robot import robot_spec, _read_episode  # noqa: E402
from fastwam.datasets.loopwam_long import DEFAULT_PROMPT  # noqa: E402


def episode_prompts(spec, threads=8):
    """Distinct prompts of the spec's episodes, in first-use order."""
    from concurrent.futures import ThreadPoolExecutor
    root = Path(spec.root)
    info = json.loads((root / 'meta/info.json').read_text())
    tasks = {}
    with (root / 'meta/tasks.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            tasks[int(row['task_index'])] = row['task']
    eps = [e for _, e in spec.episode_list()]
    with ThreadPoolExecutor(threads) as pool:
        idx = list(pool.map(lambda ep: np.unique(_read_episode(root, ep, int(info['chunks_size']))[2]), eps))
    used = dict.fromkeys(int(t) for part in idx for t in part)
    return [DEFAULT_PROMPT.format(task=tasks[t]) for t in used]


def encode(enc, prompts):
    ids, mask = enc.tokenizer(prompts, return_mask=True, add_special_tokens=True)
    mask = mask.to(enc.device, torch.bool)
    with torch.no_grad():
        context = enc.encoder(ids.to(enc.device), mask).to('cpu', torch.bfloat16)
    return context, mask.cpu()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--batch', type=int, default=32)
    a = p.parse_args()
    rank, world, local = (int(os.environ.get(k, d)) for k, d in (('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)))
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('nccl', device_id=torch.device('cuda', local))
    spec = robot_spec(a.dataset)
    out = Path(spec.text_cache_dir); out.mkdir(parents=True, exist_ok=True)
    prompts = episode_prompts(spec)
    name = lambda pr: out / f'{hashlib.sha256(pr.encode("utf-8")).hexdigest()}.{spec.text_suffix}'
    todo = [pr for pr in prompts if not name(pr).exists()]
    mine = todo[rank::world]
    if rank == 0:
        print(json.dumps(dict(event='prompts', total=len(prompts), todo=len(todo), cache_dir=str(out))), flush=True)
    enc = TextEncoder(out, spec.text_suffix, f'cuda:{local}')
    enc._load_encoder()
    if rank == 0 and prompts:
        probe = prompts[:min(len(prompts), a.batch)]
        batched, bmask = encode(enc, probe)
        diffs = []
        for i, pr in enumerate(probe[:4]):
            single, smask = encode(enc, [pr])
            m = smask[0]
            if not torch.equal(m, bmask[i]):
                raise AssertionError('batched and single masks differ')
            diffs.append(float((single[0][m].float() - batched[i][m].float()).abs().max()))
        print(json.dumps(dict(event='batch_check', max_abs_diff=diffs, batch=len(probe))), flush=True)
    t0 = time.time()
    for lo in range(0, len(mine), a.batch):
        chunk = mine[lo:lo + a.batch]
        context, mask = encode(enc, chunk)
        for i, pr in enumerate(chunk):
            m = mask[i]
            payload = dict(context=context[i][m].contiguous(), mask=m.contiguous(), packed_context=True)
            tmp = out / f'.{name(pr).name}.tmp.{uuid.uuid4().hex}'
            torch.save(payload, tmp)
            os.replace(tmp, name(pr))
        if (lo // a.batch) % 50 == 0:
            print(json.dumps(dict(rank=rank, done=lo + len(chunk), of=len(mine),
                                  per_s=round((lo + len(chunk)) / (time.time() - t0), 1))), flush=True)
    if world > 1:
        dist.barrier()
    if rank == 0:
        missing = [pr for pr in prompts if not name(pr).exists()]
        print(json.dumps(dict(event='done', prompts=len(prompts), missing=len(missing),
                              seconds=round(time.time() - t0, 1))), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
