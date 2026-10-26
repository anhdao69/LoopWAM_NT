#!/usr/bin/env python3
"""Build a LoopWAM-S student initialization variant from the canonical donor artifact and check it tensor by tensor.

  python scripts/build_init_variant.py --variant core_mean --output checkpoints/LoopWAM/init_variants/core_mean.pt
Variants (0-based Wan block indices; prelude 0-2 and coda 27-29 always come from Wan):
  default        core 9..14 (v0; must reproduce the canonical selection bit for bit)
  core16_21      core 15..20
  core_mean      core i = mean of donor blocks {3+i, 9+i, 15+i, 21+i} after FFN pruning / action resizing
  core_random    core blocks are fresh default DiTBlocks (seed 42), both experts
  ffn_magnitude  video FFN keeps the top-6144 neurons per block by ||W1[k,:]|| * ||W2[:,k]|| (needs --native-dir)
  stepwise       core = Wan blocks 4, 8, 12, 16, 20, 24
"""
import argparse
import json
from pathlib import Path

import torch

from fastwam.models.wan22.loopwam_init import (
    CORE_SLOTS, INIT_VARIANTS, NativeSafetensors, build_init_variant, build_target_experts, canonical_ffn_indices,
    load_any_init_artifact, load_init_artifact, magnitude_ffn_indices, random_core_blocks, select_donor_layers,
    sha256_file, variant_block_plan,
)

CANONICAL_SHA256 = 'a3bf2097bafddac278148c1cb0321163f68cfe2b954e206436da0e3f0bd3c535'
FFN_KEYS = ('ffn.0.weight', 'ffn.0.bias', 'ffn.2.weight')


def check(variant, art, canonical, native):
    """Independent re-derivation of every student tensor; returns a summary for the manifest."""
    plan = variant_block_plan(variant)
    summary = dict(tensors=0, copied=0, mean=0, random=0, magnitude_ffn=0)
    random_blocks = random_core_blocks(art['metadata']['variant_seed']) if variant == 'core_random' else None
    for family in ('video', 'action'):
        donor, student = canonical[f'{family}_state_dict'], art[f'{family}_state_dict']
        default = select_donor_layers(donor)
        assert student.keys() == default.keys(), f'{family}: key set differs from the v0 student'
        for key, value in student.items():
            summary['tensors'] += 1
            assert value.shape == default[key].shape and value.dtype == torch.float32, key
            if not key.startswith('blocks.'):
                assert torch.equal(value, donor[key]), key; summary['copied'] += 1; continue
            _, slot, suffix = key.split('.', 2)
            kind, arg = plan[int(slot)]
            if family == 'video' and variant == 'ffn_magnitude' and suffix in FFN_KEYS:
                w1 = native[f'blocks.{arg}.ffn.0.weight'].float()
                w2 = native[f'blocks.{arg}.ffn.2.weight'].float()
                idx = magnitude_ffn_indices(w1, w2)
                assert idx.tolist() == art['metadata']['student_ffn_indices'][slot], key
                axis = 1 if suffix == 'ffn.2.weight' else 0
                expected = native[f'blocks.{arg}.{suffix}'].float().index_select(axis, idx)
                assert torch.equal(value, expected), key; summary['magnitude_ffn'] += 1
            elif kind == 'wan':
                assert torch.equal(value, donor[f'blocks.{arg}.{suffix}']), key; summary['copied'] += 1
            elif kind == 'mean':
                expected = sum(donor[f'blocks.{i}.{suffix}'].double() for i in arg) / len(arg)
                assert torch.equal(value, expected.float()), key; summary['mean'] += 1
            elif kind == 'random':
                assert torch.equal(value, random_blocks[family][arg][suffix]), key; summary['random'] += 1
            else:
                raise AssertionError(kind)
        if variant == 'default':
            assert all(torch.equal(student[k], default[k]) for k in default), f'{family}: default differs from v0'
    for key, value in art['proprio_state_dict'].items():
        assert torch.equal(value, canonical['proprio_state_dict'][key]), key
    if variant == 'ffn_magnitude':
        canon = set(canonical_ffn_indices().tolist())
        summary['ffn_overlap_with_canonical'] = {
            slot: len(canon & set(chosen)) for slot, chosen in art['metadata']['student_ffn_indices'].items()}
    if variant == 'core_random':
        for family in ('video', 'action'):
            donor = canonical[f'{family}_state_dict']
            q = art[f'{family}_state_dict']['blocks.3.self_attn.q.weight']
            assert not any(torch.equal(q, donor[f'blocks.{i}.self_attn.q.weight']) for i in range(30)), family
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--variant', choices=INIT_VARIANTS, required=True)
    p.add_argument('--canonical', type=Path, default=Path('checkpoints/LoopWAM/wan21_compact_donors.pt'))
    p.add_argument('--native-dir', type=Path, default=Path('checkpoints/Wan-AI/Wan2.1-T2V-1.3B'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    canonical_sha = sha256_file(a.canonical)
    if canonical_sha != CANONICAL_SHA256:
        raise SystemExit(f'canonical artifact sha256 {canonical_sha} != {CANONICAL_SHA256}')
    canonical = load_init_artifact(a.canonical)
    native = None
    if a.variant == 'ffn_magnitude':
        expected = canonical['metadata']['source_hashes']
        paths = sorted(a.native_dir.glob('diffusion_pytorch_model*.safetensors'))
        got = {path.name: sha256_file(path) for path in [a.native_dir / 'config.json', *paths]}
        if got != expected:
            raise SystemExit(f'native checkpoint hashes differ from the canonical artifact source: {got} != {expected}')
        native = NativeSafetensors(paths)
    art = build_init_variant(canonical, a.variant, canonical_sha256=canonical_sha, native=native, seed=a.seed)
    summary = check(a.variant, art, canonical, native)
    # The trainer's construction path must materialize exactly these tensors at the declared size.
    video, action, proprio = build_target_experts(art)
    count = sum(t.numel() for m in (video, action, proprio) for t in m.parameters())
    assert count == 584536135, count
    for module, family in ((video, 'video'), (action, 'action'), (proprio, 'proprio')):
        state = module.state_dict()
        assert state.keys() == art[f'{family}_state_dict'].keys(), family
        assert all(torch.equal(state[k], art[f'{family}_state_dict'][k]) for k in state), family
    del video, action, proprio
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_suffix(a.output.suffix + '.tmp')
    torch.save(art, tmp)
    reloaded = load_any_init_artifact(tmp)
    for family in ('video', 'action', 'proprio'):
        assert all(torch.equal(reloaded[f'{family}_state_dict'][k], v) for k, v in art[f'{family}_state_dict'].items())
    tmp.replace(a.output)
    if native is not None:
        native.close()
    meta = art['metadata']
    manifest = dict(variant=a.variant, artifact=str(a.output), artifact_sha256=sha256_file(a.output),
                    canonical_artifact=str(a.canonical), canonical_artifact_sha256=canonical_sha,
                    student_block_plan=meta['student_block_plan'], donor_indices=meta['donor_indices'],
                    ffn_rule=meta['ffn_rule'], variant_seed=meta['variant_seed'], policy_parameters=count,
                    checks=summary, core_slots=list(CORE_SLOTS))
    a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'student_block_plan'}), flush=True)


if __name__ == '__main__':
    main()
