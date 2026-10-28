#!/usr/bin/env python3
"""Verify, export and publish an H100 v0 4/4 init-variant run (core_random, ffn_magnitude) with its matched
6,000-episode LIBERO evaluation to anhdao69/LoopWAM_NT at libero-all-suites/init_variants/<variant>.

Run from the clean v0-init-variants worktree (LoopWAM_initvar at 4ebf38a3) with v0_init_variants_work/env.sh sourced,
so the exported policy is loaded back by the code that trained it:
    python publish_initvar.py --variant core_random [--dry-run]
The HF token file is read, never printed or deleted. Upload and index steps are idempotent.
"""
import argparse, datetime, hashlib, json, math, re, shutil, subprocess, tempfile, time
from pathlib import Path

import numpy as np
import torch

RUNS = Path('/groups/yshang/an221229/checkpoints/LoopWAM_NT/v0_init_variants')
WORK = Path('/groups/yshang/an221229/checkpoints/LoopWAM_NT/v0_init_variants_work')
REFERENCE = WORK / 'reference/libero-all-suites/v0-video4-action4'
REPO = 'anhdao69/LoopWAM_NT'
TOKEN = Path('/groups/yshang/an221229/cache/huggingface/token')
REVISION = '4ebf38a3c65aa5cc9d6c07d94c52f91c51551b57'
NORMALIZATION = 'd5ad351eb3272f416d34082e8af5483ba1118ceca355a37d9f268fdb269fb3e4'
VAE_SHA = '38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981'
CANONICAL_HERE, CANONICAL_BRANCH = ('a990a5831c63e88ae5143689adad2f2aeb02cb1cd151d61ae0dd992990ac3296',
                                    'a3bf2097bafddac278148c1cb0321163f68cfe2b954e206436da0e3f0bd3c535')
TRAIN_WINDOWS, PLANNED, PARAMETERS = 277713, 21700, 584536135
MICROSTEPS = math.ceil(TRAIN_WINDOWS / 16)
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
SUITE_NAMES = dict(libero_spatial='Spatial', libero_object='Object', libero_goal='Goal', libero_10='Long')
REFERENCE_EVAL = dict(success_rate=0.9491666666666667, successes=5695, per_suite=dict(
    libero_spatial=0.9693333333333334, libero_object=0.9873333333333333, libero_goal=0.9613333333333334,
    libero_10=0.8786666666666667))
DESCRIPTION = dict(
    core_random=('the six shared core blocks of both experts start as freshly initialized default DiT blocks (PyTorch '
                 'default initialization, variant seed 42) instead of Wan blocks 9-14. Prelude (Wan blocks 0-2) and coda '
                 '(Wan blocks 27-29) are as in v0.'),
    ffn_magnitude=('the student takes the same Wan blocks as v0 (0-2, 9-14, 27-29), but every video-expert block keeps '
                   'the 6,144 FFN hidden neurons with the largest ||W1[k,:]|| x ||W2[:,k]|| in the native Wan2.1 checkpoint '
                   "instead of v0's evenly spaced canonical selection (4,165-4,237 of the 6,144 neurons per block overlap "
                   'the canonical choice). Everything else is as in v0.'))
NVP_ROW = ('| [libero-all-suites/init_variants/no_video_pretraining](libero-all-suites/init_variants/no_video_pretraining) '
           '| v0 init `no_video_pretraining` (no Wan video pretraining) | 4/4 | 584,536,135 | Training only; evaluation not run |\n')
SECTION_START = 'v0 (video 4 / action 4) retrained on all four suites'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def git_blob_sha1(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def same(a, b, where='payload'):
    """Exact recursive equality for checkpoint payloads (tensors, numpy, containers, scalars)."""
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), where
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b), where
    elif isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), where
        for key in a:
            same(a[key], b[key], f'{where}/{key}')
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f'{where}[{i}]')
    else:
        assert a == b or (isinstance(a, float) and math.isnan(a) and math.isnan(b)), where


class Publisher:
    def __init__(self, variant):
        self.variant = variant
        self.run = RUNS / variant
        self.train = self.run / 'train'
        self.repo_path = f'libero-all-suites/init_variants/{variant}'
        self.init = read(WORK / 'init' / f'{variant}.manifest.json')
        self.init_sha = self.init['artifact_sha256']
        self.init_mode = f'wan_init_variant_{variant}_fresh_optimizer'

    def status(self, stage, **kw):
        record = dict(stage=stage, variant=self.variant, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **kw)
        write(self.run / 'publish_status.json', record)
        print(json.dumps(record), flush=True)

    def verify_training(self):
        train = self.train
        manifest, timing, state = (read(train / n) for n in ('manifest.json', 'timing.json', 'trainer_state.json'))
        required = dict(version='v0', video_loops=4, action_loops=4, loops=4, action_core_loops=4, loop_alignment='late',
                        backend='ddp', world_size=2, microbatch=8, gradient_accumulation=8, global_batch=128, epochs=10,
                        seed=42, train_windows=TRAIN_WINDOWS, planned_updates=PLANNED, planned_windows=TRAIN_WINDOWS * 10,
                        policy_parameters=PARAMETERS, git_revision=REVISION, dataset_scope='full_libero',
                        validation_samples=0, initialization_mode=self.init_mode, resume=None, max_updates=None,
                        fused_optimizer=True, bucket_views=True, structured_attention=True, smoke=True,
                        skip_cached_video_decode=True, policy_dtype='float32', optimizer_state_dtype='float32',
                        compute_dtype='bfloat16', init_artifact=f'checkpoints/LoopWAM/init_variants/{self.variant}.pt')
        bad = {k: manifest.get(k) for k, v in required.items() if manifest.get(k) != v}
        assert not bad, f'Training manifest contract mismatch: {bad}'
        assert manifest['asset_sha256'] == dict(vae=VAE_SHA, initialization=self.init_sha), 'Unexpected asset hashes'
        assert self.init['canonical_artifact_sha256'] == CANONICAL_HERE and self.init['variant'] == self.variant
        assert sha256(WORK / 'init' / f'{self.variant}.pt') == self.init_sha, 'Initialization artifact changed'
        assert not list((self.run / 'segments').glob('*')) and not (train / 'resume_manifest.json').exists(), \
            'Expected one uninterrupted segment'
        rows = [json.loads(line) for line in (train / 'metrics.jsonl').read_text().splitlines()]
        updates = [r for r in rows if 'loss_video' in r]
        assert [r['update'] for r in updates] == list(range(1, PLANNED + 1)), 'Metric rows are not one per update'
        assert all(math.isfinite(r[k]) for r in updates for k in ('loss_video', 'loss_action', 'grad_norm'))
        windows = sum(r['windows'] for r in updates)
        assert timing['status'] == 'complete' and timing['completed_updates'] == PLANNED
        assert timing['windows_seen'] == windows == updates[-1]['windows_seen'] == TRAIN_WINDOWS * 10
        assert state['update'] == PLANNED and state['epoch'] == 9 and state['next_micro'] == MICROSTEPS
        coverage = [r for r in rows if r.get('event') == 'latent_cache_coverage']
        assert all(c['valid_windows'] == c['total_windows'] == TRAIN_WINDOWS for c in coverage)
        return manifest, timing, updates

    def fairness(self):
        """The branch verifier's comparison with the v0 4/4 release plus the H100 decision (fairness_h100.json)."""
        record = read(self.run / 'fairness_h100.json')
        h100 = record.pop('h100')
        assert h100['h100_passed'] is True and h100['residual_problems'] == []
        assert h100['canonical_artifact_sha256'] == CANONICAL_HERE
        assert record['init_variant'] == self.variant and record['initialization_sha256'] == self.init_sha
        branch = dict(passed=record.pop('passed'), problems=record.pop('problems'), message=h100['branch_verifier_message'])
        return dict(passed=True, problems=[], documented_exceptions=h100['known_cluster_exceptions'] + [dict(
                        problem='canonical donor artifact differs from the branch pin',
                        accepted_because=(f'the branch pins {CANONICAL_BRANCH[:12]}... (built on HiPerGator); this cluster '
                            f'built {CANONICAL_HERE[:12]}... from the same Wan2.1 source and code (also used by '
                            'libero-all-suites/v0-video4-action2, whose fairness.json documents the <=1 FP32 ulp '
                            'difference); the variant artifact was built from it by the branch builder'))],
                    branch_verifier=branch, canonical_artifact_sha256=CANONICAL_HERE,
                    branch_canonical_artifact_sha256=CANONICAL_BRANCH, **record)

    def evaluation(self, source_sha):
        out = self.run / 'eval_matched'
        summary, manifest = read(out / 'summary.json'), read(out / 'manifest.json')
        gate = read(WORK / 'eval' / 'logs' / f'gate_{self.variant}.json')
        parity = read(RUNS / 'parity_reference_v0_video4_action4' / 'parity.json')
        assert summary['episodes'] == 6000 and summary['checkpoint_sha256'] == source_sha == manifest['sha256'] == gate['checkpoint_sha256']
        assert summary['checkpoint_step'] == PLANNED and gate['all_bit_exact'] is True
        assert sorted(summary['per_suite']) == sorted(SUITES) and all(v['episodes'] == 1500 for v in summary['per_suite'].values())
        assert parity['runner_sha256'] == manifest['runner_sha256']
        groups = list((out / 'groups').glob('*.json'))
        assert len(groups) == 120
        runs = [read(p) for p in sorted((out / 'runs').glob('*.json'))]
        assert all(r['errors'] == 0 for r in runs)
        started, finished = min(r['started'] for r in runs), max(r['finished'] for r in runs)
        return dict(summary, protocol=manifest['protocol'], runner_sha256=manifest['runner_sha256'],
                    adapter_source_sha256=manifest['adapter_source_sha256'], code_revision=REVISION,
                    vae_sha256=manifest['vae_sha256'], torch=manifest['torch'], gpu=gate['gpu'],
                    evaluated_checkpoint='training checkpoint latest.pt (source_sha256 in export.json; tensors equal policy.pt)',
                    cuda_graph_gate={k: v for k, v in gate.items() if k != 'details'},
                    runner_parity=dict({k: parity[k] for k in ('export_sha256', 'suite', 'seed', 'episodes', 'h100_successes',
                        'campaign_successes', 'outcome_agreement')}, note='released v0-video4-action4 through this runner '
                        'vs the campaign record (evaluation_report/final/episodes.csv)'),
                    reference_release_same_protocol=dict(repo_path='libero-all-suites/v0-video4-action4', **REFERENCE_EVAL),
                    runner_sessions=runs, wall_seconds=finished - started,
                    wall_note='first runner start to last runner end; 990547, 4x H100, 16 CPUs; videos not saved')

    def export(self, dst):
        source = self.train / 'latest.pt'
        source_sha = sha256(source)
        payload = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
        assert 'optimizer' in payload, 'Source must be the full resumable checkpoint'
        policy = {k: v for k, v in payload.items() if k != 'optimizer'}
        target, tmp = dst / 'policy.pt', dst / 'policy.pt.tmp'
        torch.save(policy, tmp)
        tmp.replace(target)
        restored = torch.load(target, map_location='cpu', mmap=True, weights_only=False)
        import fastwam
        root = Path(fastwam.__file__).parents[2]
        assert root.name == 'LoopWAM_initvar', f'loader must be the v0-init-variants worktree: {fastwam.__file__}'
        head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'],
                               capture_output=True, text=True).stdout.strip()
        assert head == REVISION and not dirty, 'loader worktree must be clean at the training revision'
        from fastwam.models.wan22.loopwam import _validate_checkpoint_depth, create_loopwam
        _validate_checkpoint_depth(restored)
        same(policy, restored)
        assert restored['format_version'] == 'loopwam-s-v1' and restored['version'] == 'v0'
        assert (restored['video_loops'], restored['action_loops'], restored['loop_alignment']) == (4, 4, 'late')
        vae = root / 'checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'
        model = create_loopwam(checkpoint_path=str(target), vae_path=str(vae), model_dtype=torch.float32, device='cpu')
        same(dict(model.mot.state_dict()), dict(policy['mot']), 'loaded/mot')
        same(dict(model.proprio_encoder.state_dict()), dict(policy['proprio_encoder']), 'loaded/proprio_encoder')
        assert (model.mot.loops, model.mot.action_loops) == (4, 4)
        count = sum(p.numel() for p in model.policy_parameters())
        assert count == PARAMETERS, count
        step = payload['step']
        del model, restored, policy, payload
        return dict(source_sha256=source_sha, source_bytes=source.stat().st_size, export_sha256=sha256(target),
                    export_bytes=target.stat().st_size, step=step)

    def timing_summary(self, final_timing, updates):
        launches = [json.loads(l) for l in (self.run / 'launches.jsonl').read_text().splitlines()]
        epoch1 = [r['seconds'] for r in updates if r['epoch'] == 1]
        later = [r['seconds'] for r in updates if r['epoch'] > 1]
        return dict(final_timing, launches=launches, gpu_names=sorted({l['gpu_name'] for l in launches}),
                    segments=1, total_training_seconds_all_segments=final_timing['elapsed_training_seconds'],
                    kept_update_seconds_sum=float(np.sum([r['seconds'] for r in updates])),
                    epoch1_mean_update_seconds=float(np.mean(epoch1)), epochs2_10_mean_update_seconds=float(np.mean(later)),
                    timing_note=('One uninterrupted segment. The run shared its node with another training run; the '
                                 'latent cache was seeded from the v0 4/2 run\'s cache (identical provenance), and clips '
                                 'with cached latents were not decoded (--skip-cached-video-decode; first-update losses '
                                 'are bit-identical to decoding).'))

    def readme(self, timing, evaluation):
        hours = timing['total_training_seconds_all_segments'] / 3600
        s = evaluation
        suites = ', '.join(f"{SUITE_NAMES[k]} {100 * s['per_suite'][k]['success_rate']:.2f}%" for k in SUITES)
        ref = ', '.join(f"{SUITE_NAMES[k]} {100 * REFERENCE_EVAL['per_suite'][k]:.2f}%" for k in SUITES)
        seeds = ', '.join(f"{seed}: {100 * v['success_rate']:.2f}%" for seed, v in s['per_seed'].items())
        parity = s['runner_parity']
        return f"""# LoopWAM v0 (video 4 / action 4), initialization variant `{self.variant}` — all four LIBERO suites

Same model, data and training as `libero-all-suites/v0-video4-action4`; only the student initialization differs:
{DESCRIPTION[self.variant]} `init_variant_manifest.json` records the initialization artifact (SHA-256 `{self.init_sha[:12]}...`),
its per-block source plan and tensor checks; it was built by `scripts/build_init_variant.py` of the
[v0-init-variants branch](https://github.com/anhdao69/LoopWAM_NT/tree/v0-init-variants) (`{REVISION[:7]}`).

Ten-epoch full-suite training (Spatial, Object, Goal and LIBERO-10; 1,712 demonstrations, 277,713 windows), training seed
42, global batch 128 (2 GPUs x microbatch 8 x accumulation 8), 21,700 updates, AdamW (0.9, 0.95), weight decay 0.01,
LR 1e-4 cosine with 5% warm-up, clip 1.0, FP32 master weights + BF16 compute, 584,536,135 policy parameters. Data,
normalization (`{NORMALIZATION[:12]}...`), VAE, schedule and precision equal the v0 release; `fairness.json` lists the
checked fields and two documented exceptions (T5 text-embedding files and the canonical donor artifact were regenerated
on this cluster from the same sources). Training took {hours:.1f} h on 2x {' / '.join(timing['gpu_names'])} in one segment.

## Evaluation (matched LIBERO protocol, 6,000 episodes)

**{100 * s['success_rate']:.2f}% ({s['successes']:,}/6,000)** — {suites}. Evaluation seeds {seeds}.
The released `v0-video4-action4` scored 94.92% under the same protocol ({ref}).

Protocol (the campaign's `lightwam_20261009_matched`): four suites x ten tasks x official initial states 0-49 x evaluation
seeds 42/43/44; one environment per task/seed (env.seed(base seed)) reused for its 50 initial states; global seed set once
per task/seed and every action query uses the base seed; 30 settling steps; 400 (Spatial/Object/Goal) or 700 (Long)
policy steps; ten actions executed per 32-action chunk; ten denoising steps, CFG 1, FP32 weights with BF16 autocast;
success is the environment done flag. Inference replays a CUDA graph of the unchanged native computation, verified
bit-exact against eager inference on real observations ({s['cuda_graph_gate']['queries']}/{s['cuda_graph_gate']['queries']} queries).
The same runner scored the released 4/4 policy {parity['h100_successes']}/500 on Long seed 42 versus {parity['campaign_successes']}/500
in the campaign. Evaluated on 4x NVIDIA H100 (videos not saved); `evaluation_summary.json` has per-suite and per-seed
counts, the protocol and the runner hashes. One training seed: differences between variants of a few points are not
established as significant.

## Files

`policy.pt` is the native `loopwam-s-v1` checkpoint with the Adam optimizer state removed; every tensor and metadata field was
checked for exact equality with the final (evaluated) training checkpoint, and the native loader of revision `{REVISION}`
rebuilt the 4/4 model from it. Load with that revision or a compatible later one: `create_loopwam(checkpoint_path=..., vae_path=...)`.
Use it for inference or fresh-optimizer fine-tuning, not exact optimizer-state resume. Keep `dataset_stats.json` matched to it.
Wan VAE, text embeddings and observation preprocessing are external. See `export.json` for source/export SHA-256 hashes.
"""

    def row(self, evaluation):
        s = evaluation
        return (f'| [{self.repo_path}]({self.repo_path}) | v0 init `{self.variant}` | 4/4 | 584,536,135 '
                f"| {100 * s['success_rate']:.2f}% ({s['successes']:,}/6,000; 4 suites x seeds 42/43/44) |\n")

    def update_root(self, api, info, evaluation, workdir):
        """Add to checkpoint_index.json and the README tables on top of the current head (retries on concurrent commits)."""
        from huggingface_hub import CommitOperationAdd, hf_hub_download
        row = self.row(evaluation)
        sentence = (f' `init_variants/{self.variant}` was trained on two H100s (2 GPUs x microbatch 8 x accumulation 8) and '
                    f"scored {100 * evaluation['success_rate']:.2f}% on the matched 6,000-episode four-suite evaluation.")
        for attempt in range(6):
            head = api.model_info(REPO).sha
            files = {name: Path(hf_hub_download(REPO, name, revision=head, token=api.token,
                                                local_dir=workdir / f'head_{attempt}')).read_text()
                     for name in ('README.md', 'checkpoint_index.json')}
            index = json.loads(files['checkpoint_index.json'])
            if any(c.get('repo_path') == self.repo_path for c in index['checkpoints']):
                return dict(index_revision=head, already_indexed=True)
            index['checkpoints'].append(info)
            index['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            text = files['README.md']
            text, n = re.subn(r'This repository contains \*\*\d+ checkpoints\*\*',
                              f"This repository contains **{len(index['checkpoints'])} checkpoints**", text, count=1)
            lines = text.splitlines(keepends=True)
            anchors = [i for i, line in enumerate(lines) if line == NVP_ROW]
            assert anchors, 'no_video_pretraining rows not found in the root README'
            for i in reversed(anchors):   # after the block of init-variant rows that follows each anchor
                j = i + 1
                while j < len(lines) and lines[j].startswith('| [libero-all-suites/init_variants/'):
                    j += 1
                lines.insert(j, row)
            para = [i for i, line in enumerate(lines) if line.startswith(SECTION_START)]
            assert len(para) == 1, 'init-variant section paragraph not found'
            lines[para[0]] = lines[para[0]].rstrip('\n') + sentence + '\n'
            text = ''.join(lines)
            ops = [CommitOperationAdd('README.md', text.encode()),
                   CommitOperationAdd('checkpoint_index.json', (json.dumps(index, indent=2) + '\n').encode())]
            try:
                commit = api.create_commit(REPO, operations=ops, commit_message=f'Index {self.repo_path}', parent_commit=head)
                return dict(index_revision=commit.oid, already_indexed=False, readme_count_updated=bool(n),
                            readme_rows_added=len(anchors))
            except Exception as error:  # concurrent commit or transient failure: rebuild from the new head
                print(json.dumps(dict(event='index_retry', attempt=attempt, error=str(error)[:300])), flush=True)
                time.sleep(30 * (attempt + 1))
        raise RuntimeError('Could not update the root index')

    def publish(self, dry_run):
        dst = self.run / 'publish' / self.repo_path
        dst.mkdir(parents=True, exist_ok=True)
        self.status('verifying_training')
        manifest, timing, updates = self.verify_training()
        data_manifest = read(self.train / 'data' / 'data_manifest.json')
        assert data_manifest['normalization_sha256'] == NORMALIZATION == sha256(self.train / 'data' / 'dataset_stats.json')
        fair = self.fairness()
        infofile = dst / 'export.json'
        if infofile.exists() and (dst / 'policy.pt').exists():
            info = read(infofile)
            assert sha256(dst / 'policy.pt') == info['export_sha256'] and sha256(self.train / 'latest.pt') == info['source_sha256']
        else:
            self.status('exporting')
            hashes = self.export(dst)
            assert hashes['step'] == PLANNED
            evaluation = self.evaluation(hashes['source_sha256'])
            tsum = self.timing_summary(timing, updates)
            info = dict(id=f'v0_{self.variant}_full_libero_h100', repo_path=self.repo_path, training_path=str(self.train),
                        **hashes, format='loopwam-s-v1', optimizer_included=False, version='v0', video_loops=4,
                        action_loops=4, loop_alignment='late', parameters=PARAMETERS, init_variant=self.variant,
                        initialization_sha256=self.init_sha, initialization_mode=self.init_mode, dataset_scope='full_libero',
                        epochs=10, global_batch=128, training_seed=42, evaluation_status='matched_libero_6000_h100',
                        success_rate=evaluation['success_rate'], successes=evaluation['successes'], episodes=6000,
                        per_suite_success_rate={k: v['success_rate'] for k, v in evaluation['per_suite'].items()},
                        training_seconds=tsum['total_training_seconds_all_segments'], resume_starts=0,
                        tensor_equality_verified=True, native_loader_verified=True, source_revision=REVISION,
                        compatible_code_revision=REVISION, code_branch='v0-init-variants',
                        normalization_sha256=NORMALIZATION, fairness_check_passed=fair['passed'],
                        gpu=' + '.join(tsum['gpu_names']), world_size=2, microbatch=8, gradient_accumulation=8,
                        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            shutil.copy2(self.train / 'data' / 'dataset_stats.json', dst / 'dataset_stats.json')
            shutil.copy2(self.train / 'data' / 'data_manifest.json', dst / 'data_manifest.json')
            shutil.copy2(self.train / 'manifest.json', dst / 'training_manifest.json')
            shutil.copy2(WORK / 'init' / f'{self.variant}.manifest.json', dst / 'init_variant_manifest.json')
            write(dst / 'training_timing.json', tsum)
            write(dst / 'fairness.json', fair)
            write(dst / 'evaluation_summary.json', evaluation)
            (dst / 'README.md').write_text(self.readme(tsum, evaluation))
            write(infofile, info)
        evaluation = read(dst / 'evaluation_summary.json')
        self.status('export_verified', export_sha256=info['export_sha256'], bytes=info['export_bytes'])
        if dry_run:
            self.status('dry_run_complete', stage_dir=str(dst))
            return
        from huggingface_hub import HfApi
        api = HfApi(token=TOKEN.read_text().strip())
        assert api.whoami()['name'] == 'anhdao69'
        local = {p.name: dict(size=p.stat().st_size, sha256=sha256(p), blob=git_blob_sha1(p) if p.stat().st_size < 50_000_000 else None)
                 for p in sorted(dst.iterdir()) if p.is_file() and not p.name.endswith('.tmp')}
        remote = {f.rfilename: f for f in api.model_info(REPO, files_metadata=True).siblings
                  if f.rfilename.startswith(self.repo_path + '/')}
        policy_remote = remote.get(f'{self.repo_path}/policy.pt')
        if policy_remote is not None and policy_remote.lfs is not None:
            remote_sha = policy_remote.lfs.sha256 if hasattr(policy_remote.lfs, 'sha256') else policy_remote.lfs['sha256']
            # Never replace a different policy already published at this path.
            assert remote_sha == local['policy.pt']['sha256'], f'A different policy.pt already exists at {self.repo_path}'
        for attempt in range(5):
            try:
                self.status('uploading', files=len(local), bytes=sum(m['size'] for m in local.values()), attempt=attempt)
                api.upload_folder(repo_id=REPO, folder_path=dst, path_in_repo=self.repo_path, allow_patterns=list(local),
                                  commit_message=f'Add {self.repo_path} (four-suite v0 4/4 init variant, matched 6,000-episode evaluation)')
                break
            except Exception as error:
                print(json.dumps(dict(event='upload_retry', attempt=attempt, error=str(error)[:300])), flush=True)
                time.sleep(60 * (attempt + 1))
        else:
            raise RuntimeError('Upload failed')
        self.status('verifying_hub')
        remote = {f.rfilename: f for f in api.model_info(REPO, files_metadata=True).siblings}
        for name, meta in local.items():
            entry = remote.get(f'{self.repo_path}/{name}')
            assert entry is not None and entry.size == meta['size'], f'Missing or wrong-size Hub file: {name}'
            if entry.lfs is not None:
                remote_sha = entry.lfs.sha256 if hasattr(entry.lfs, 'sha256') else entry.lfs['sha256']
                assert remote_sha == meta['sha256'], f'Hub LFS checksum mismatch: {name}'
            else:
                assert entry.blob_id == meta['blob'], f'Hub blob mismatch: {name}'
        self.status('indexing')
        with tempfile.TemporaryDirectory(dir=self.run) as tmp:
            index = self.update_root(api, info, evaluation, Path(tmp))
        info_remote = api.model_info(REPO)
        result = dict(repo=REPO, repo_path=self.repo_path, revision=info_remote.sha, private=info_remote.private,
                      verified_files=local, **index, completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(self.run / 'upload_verification.json', result)
        (self.run / 'PUBLISHED').write_text(info_remote.sha + '\n')
        self.status('complete', revision=info_remote.sha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=sorted(DESCRIPTION))
    ap.add_argument('--dry-run', action='store_true', help='verify and export only; no Hub writes')
    args = ap.parse_args()
    torch.set_num_threads(2)
    Publisher(args.variant).publish(args.dry_run)


if __name__ == '__main__':
    main()
