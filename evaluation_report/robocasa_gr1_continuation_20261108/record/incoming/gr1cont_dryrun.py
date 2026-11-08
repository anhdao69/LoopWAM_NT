#!/usr/bin/env python3
"""CPU dry run of the GR1 continuation launch: the trainer's own parser, recipe checks and contract builder on the exact
production argument list (gr1cont_args.py), then the trainer's parent validation and contract-equality assertion
against the real parent checkpoint. No GPU and no write outside --report.

python gr1cont_dryrun.py --latent-metadata DIR_WITH_metadata.json --report OUT.json
"""
import argparse, hashlib, json, sys
from pathlib import Path

import torch

import train_loopwam_gr1 as T
from gr1cont_args import continuation_args, PARENT, PARENT_SHA256, RUN, OVERRIDES
from fastwam.datasets.robocasa_gr1 import GR1Windows, gr1_spec, LATENT_SHAPE, latent_provenance_gr1
from fastwam.datasets.loopwam_latent_cache import LoopWAMLatentCache
from fastwam.datasets.loopwam_robot_batches import training_budget, epoch_rank_batches
from fastwam.models.wan22.loopwam_init import sha256_file


class CacheId:   # the provenance hash LoopWAMLatentCache computes, without opening the payload memmaps
    def __init__(self, directory, size, provenance):
        meta = dict(format=LoopWAMLatentCache.FORMAT, size=size, latent_shape=list(LATENT_SHAPE),
                    storage_dtype='uint16_bfloat16_bits', output_dtype='float32', provenance=provenance)
        text = json.dumps(meta, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
        on_disk = (Path(directory) / 'metadata.json').read_text()
        if on_disk != text:
            raise ValueError(f'{directory}/metadata.json differs from the provenance computed by this code')
        self.provenance_sha256 = hashlib.sha256(text.encode()).hexdigest()


def main():
    q = argparse.ArgumentParser(); q.add_argument('--latent-metadata', required=True); q.add_argument('--report', required=True)
    o = q.parse_args()
    out_dir = RUN / 'train/continuation_e06_10'
    report = {}
    for mode, kw in (('first_launch', dict(max_updates=6 * T.UPDATES_PER_EPOCH)),
                     ('resume_launch', dict(resume=True, max_updates=7 * T.UPDATES_PER_EPOCH)),
                     ('final_launch', dict(resume=True))):
        argv = continuation_args(out_dir, '/tmp/x/latents', **kw)
        p = T.build_parser(); a = p.parse_args(argv); cont = T.check_recipe(p, a)
        assert cont and a.resume == kw.get('resume', False) and a.max_updates == kw.get('max_updates')
        report[mode] = argv
    parent = json.loads((RUN / 'train/production/manifest.json').read_text())
    argv = continuation_args(out_dir, '/tmp/x/latents', max_updates=6 * T.UPDATES_PER_EPOCH)
    p = T.build_parser(); a = p.parse_args(argv); T.check_recipe(p, a); v = vars(a)
    diffs = {k: (parent[k], v[k]) for k in parent if k in v and parent[k] != v[k]}
    allowed = set(OVERRIDES) | {'output_dir', 'latent_dir', 'max_updates'}
    assert set(diffs) <= allowed, {k: d for k, d in diffs.items() if k not in allowed}
    report['argument_differences_vs_parent'] = diffs
    shared = Path(a.shared_dir)
    w = GR1Windows(gr1_spec(stats_path=shared / 'data/dataset_stats_29d.json'), cache_dir=shared / 'data', text_threads=a.text_threads)
    m = w.manifest(); assert len(w.groups) == 24 and len(w.episodes) == 24000
    assert json.loads((shared / 'data/data_manifest.json').read_text()) == m, 'Data manifest changed'
    hashes = dict(init=sha256_file(a.init_artifact), vae=sha256_file(a.vae_path), parent=sha256_file(a.continuation_from))
    assert hashes['parent'] == a.expected_parent_sha256 == PARENT_SHA256
    cache = CacheId(o.latent_metadata, len(w), latent_provenance_gr1(w, hashes['vae']))
    per_epoch, total, tail = training_budget(len(w), T.GLOBAL_BATCH, a.epochs)
    assert len(w) == T.WINDOWS and per_epoch == T.UPDATES_PER_EPOCH and total == T.UPDATES_PER_EPOCH * a.epochs
    world = 4; accumulation = T.GLOBAL_BATCH // world // a.microbatch
    contract = T.make_contract(a, w, m, hashes, cache, world, accumulation, per_epoch, total, tail)
    payload = torch.load(a.continuation_from, map_location='cpu', mmap=True, weights_only=False)
    T.validate_parent_state(payload)
    expected = T.continuation_contract(payload['training_state']['contract'], hashes['parent'])
    mism = {k: (expected.get(k), contract.get(k)) for k in set(expected) | set(contract) if expected.get(k) != contract.get(k)}
    assert not mism, f'Continuation contract mismatch: {mism}'
    # a later --resume launch rebuilds the same contract (resume / max-updates are not part of it)
    a2 = p.parse_args(continuation_args(out_dir, '/tmp/x/latents', resume=True, max_updates=7 * T.UPDATES_PER_EPOCH))
    assert T.make_contract(a2, w, m, hashes, cache, world, accumulation, per_epoch, total, tail) == contract
    state = payload['training_state']
    ids = next(epoch_rank_batches(len(w), T.GLOBAL_BATCH, world, 0, a.seed + state['next_epoch'], state['next_batch']))
    report.update(passed=True, contract=contract, hashes=hashes, parent_step=payload['step'],
                  parent_cursor=[state['update'], state['windows_seen'], state['next_epoch'], state['next_batch']],
                  first_lr=a.lr_backbone * T.continuation_factor(state['update']),
                  epoch_start_lrs={e: a.lr_backbone * T.continuation_factor((e - 1) * per_epoch) for e in range(6, 11)},
                  last_lr=a.lr_backbone * T.continuation_factor(total - 1), first_rank0_window_ids=ids[:8],
                  epoch_targets={e: e * per_epoch for e in range(6, 11)})
    Path(o.report).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(dict(passed=True, first_lr=report['first_lr'], last_lr=report['last_lr'], diffs=diffs)), flush=True)


if __name__ == '__main__':
    main()
