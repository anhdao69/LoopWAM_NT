#!/usr/bin/env python3
"""Check a v0 initialization-variant run against the four-suite v0 release (libero-all-suites/v0-video4-action4).

smoke --train DIR --reference REF --init canonical|VARIANT   10-update run: contract, data and init identity; prints
                                                              losses, s/update and peak memory
run   --train DIR --reference REF --init VARIANT             finished run: completion + fairness, writes RUN/fairness.json
Everything must equal the release except the initialization artifact, which must be the declared variant's.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dense_libero_hf_export import SUITES, TRAINING_FIELDS, UPDATES, WINDOWS, PATH_FIELDS, read, strip_paths, write  # noqa: E402

PARAMETERS = 584536135
CANONICAL_SHA256 = 'a3bf2097bafddac278148c1cb0321163f68cfe2b954e206436da0e3f0bd3c535'
SAME_FIELDS = [f for f in TRAINING_FIELDS if f not in ('asset_sha256', 'initialization_mode')]


def expected_init(init):
    if init == 'canonical':
        return CANONICAL_SHA256, 'canonical_wan_artifact_fresh_optimizer', None
    built = read(Path('checkpoints/LoopWAM/init_variants') / f'{init}.manifest.json')
    assert built['variant'] == init and built['canonical_artifact_sha256'] == CANONICAL_SHA256, built
    return built['artifact_sha256'], f'wan_init_variant_{init}_fresh_optimizer', built['student_block_plan']


def contract(train, reference, init, need):
    m = read(train / 'manifest.json')
    ref = read(reference / 'training_manifest.json')
    world = m['world_size']
    need(m['version'] == 'v0' and m['loops'] == 4 and m['action_core_loops'] == 4, 'v0 video 4 / action 4')
    need(m['policy_parameters'] == PARAMETERS, 'policy parameters')
    need(m['resume'] is None and not m['checkpoint_blocks'], 'fresh run')
    need(m['backend'] == 'ddp' and m['microbatch'] * world * m['gradient_accumulation'] == 128, 'batch contract')
    need(m['val_windows'] == 0 and m['validation_samples'] == 0, 'no validation split')
    need(m['dataset_scope'] == 'full_libero' and m['data']['split'] == 'all_train' and m['data']['suites'] == SUITES,
         'four-suite scope')
    need(m['train_windows'] == WINDOWS and m['planned_updates'] == UPDATES, 'schedule')
    differ = [f for f in SAME_FIELDS if m[f] != ref[f]]
    need(not differ, f'training fields differ from the v0 release: {differ}')
    sha, mode, plan = expected_init(init)
    need(m['asset_sha256']['vae'] == ref['asset_sha256']['vae'], 'VAE differs from the v0 release')
    need(m['asset_sha256']['initialization'] == sha, f'initialization artifact is not {init}')
    need(m['initialization_mode'] == mode, f'initialization mode {m["initialization_mode"]} != {mode}')
    meta = m['initialization_metadata']
    need(meta.get('init_variant') == (None if init == 'canonical' else init)
         and meta.get('student_block_plan') == plan, 'initialization metadata names a different variant')
    ref_data, data = strip_paths(read(reference / 'data_manifest.json')), strip_paths(read(train / 'data/data_manifest.json'))
    data_differ = sorted(k for k in set(ref_data) | set(data) if ref_data.get(k) != data.get(k))
    need(not data_differ, f'data manifest differs from the v0 release: {data_differ}')
    return m, ref_data


def rows(train):
    return [json.loads(l) for l in (train / 'metrics.jsonl').read_text().splitlines() if l.startswith('{"update"')]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=['smoke', 'run'])
    p.add_argument('--train', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--init', required=True)
    a = p.parse_args()
    problems = []
    def need(ok, what):
        if not ok:
            problems.append(what)
    m, ref_data = contract(a.train, a.reference, a.init, need)
    t = read(a.train / 'timing.json')
    r = rows(a.train)
    need(all(math.isfinite(x['loss_video'] + x['loss_action'] + x['grad_norm']) for x in r), 'non-finite loss or gradient')
    if a.mode == 'smoke':
        need(t['status'] == 'smoke_complete' and t['completed_updates'] == 10 and t['windows_seen'] == 1280, 'smoke length')
        out = dict(init=a.init, problems=problems, mean_s=t['measured_mean_update_seconds'],
                   last5=sum(x['seconds'] for x in r[-5:]) / 5, peak_gb=max(x['peak_allocated_gb'] for x in r),
                   misses=sum(x['latent_cache_rank_local']['misses'] for x in r),
                   loss_video=[round(x['loss_video'], 5) for x in r], loss_action=[round(x['loss_action'], 5) for x in r])
        print(json.dumps(out), flush=True)
        if problems:
            raise SystemExit('smoke check failed: ' + '; '.join(problems))
        return
    s = read(a.train / 'trainer_state.json')
    world = m['world_size']
    need(m['max_updates'] is None, 'full run')
    need(t['status'] == 'complete' and t['completed_updates'] == UPDATES and t['windows_seen'] == 10 * WINDOWS,
         'timing complete')
    need(len(r) == UPDATES and [x['update'] for x in r] == list(range(1, UPDATES + 1)), 'one metrics row per update')
    need(s['update'] == UPDATES and s['epoch'] == 9 and s['windows_seen'] == t['windows_seen']
         and s['next_micro'] == math.ceil(WINDOWS / (world * m['microbatch'])), 'trainer state at the end of epoch 10')
    need((a.train / 'latest.pt').stat().st_size > 0, 'final checkpoint')
    resumes = sorted(x.name for x in a.train.glob('resume_manifest*.json'))
    report = dict(passed=not problems, problems=problems, version='v0', init_variant=a.init,
                  initialization_sha256=m['asset_sha256']['initialization'], reference=str(a.reference),
                  compared_training_fields=SAME_FIELDS, data_fields=sorted(ref_data), ignored_path_fields=sorted(PATH_FIELDS),
                  world_size=world, microbatch=m['microbatch'], gradient_accumulation=m['gradient_accumulation'],
                  gpu=m['gpu'], git_revision=m['git_revision'], resumed=bool(resumes), resume_manifests=resumes,
                  normalization_sha256=m['data']['normalization_sha256'],
                  final_epoch_loss_video=sum(x['loss_video'] for x in r if x['epoch'] == 10) / max(1, sum(x['epoch'] == 10 for x in r)),
                  final_epoch_loss_action=sum(x['loss_action'] for x in r if x['epoch'] == 10) / max(1, sum(x['epoch'] == 10 for x in r)),
                  training_hours=t['elapsed_training_seconds'] / 3600)
    write(a.train.parent / 'fairness.json', report)
    print(json.dumps(report), flush=True)
    if problems:
        raise SystemExit('verification failed: ' + '; '.join(problems))


if __name__ == '__main__':
    main()
