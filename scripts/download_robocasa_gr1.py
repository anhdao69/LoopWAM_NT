#!/usr/bin/env python3
"""Download ONLY the 24 RoboCasa-GR1 tabletop `gr1_unified.*_1000` folders.

The folder list is starVLA's mixture `fourier_gr1_unified_1000`
(examples/simBenchmarks/Robocasa_tabletop/train_files/data_registry/data_config.py, identical
to its download_gr00t_ft_data.py). The 10k-demo `gr1_arms_waist.*` folders are never matched.
The HF revision is pinned so the data identity is reproducible; every folder is verified after
download (1000 episodes, parquet + mp4 per episode, LeRobot v2.0 metadata incl. meta/stats.json).

python scripts/download_robocasa_gr1.py --out /mnt/data/.../robocasa_gr1 [--verify-only]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

REPO_ID = 'nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim'
REVISION = 'ea7ac0b68f87da62f1e726771bba0fe74300802f'   # dataset HEAD on 2026-11-01

GR1_TASKS = (
    'PnPBottleToCabinetClose', 'PnPCanToDrawerClose', 'PnPCupToDrawerClose', 'PnPMilkToMicrowaveClose',
    'PnPPotatoToMicrowaveClose', 'PnPWineToCabinetClose',
    'PosttrainPnPNovelFromCuttingboardToBasketSplitA', 'PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA',
    'PosttrainPnPNovelFromCuttingboardToPanSplitA', 'PosttrainPnPNovelFromCuttingboardToPotSplitA',
    'PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA', 'PosttrainPnPNovelFromPlacematToBasketSplitA',
    'PosttrainPnPNovelFromPlacematToBowlSplitA', 'PosttrainPnPNovelFromPlacematToPlateSplitA',
    'PosttrainPnPNovelFromPlacematToTieredshelfSplitA', 'PosttrainPnPNovelFromPlateToBowlSplitA',
    'PosttrainPnPNovelFromPlateToCardboardboxSplitA', 'PosttrainPnPNovelFromPlateToPanSplitA',
    'PosttrainPnPNovelFromPlateToPlateSplitA', 'PosttrainPnPNovelFromTrayToCardboardboxSplitA',
    'PosttrainPnPNovelFromTrayToPlateSplitA', 'PosttrainPnPNovelFromTrayToPotSplitA',
    'PosttrainPnPNovelFromTrayToTieredbasketSplitA', 'PosttrainPnPNovelFromTrayToTieredshelfSplitA')
FOLDERS = tuple(f'gr1_unified.{t}_GR1ArmsAndWaistFourierHands_1000' for t in GR1_TASKS)
assert len(FOLDERS) == 24 and len(set(FOLDERS)) == 24


def verify_folder(root: Path) -> dict:
    meta = root / 'meta'
    info = json.loads((meta / 'info.json').read_text())
    for name in ('info.json', 'episodes.jsonl', 'tasks.jsonl', 'modality.json', 'stats.json'):
        if not (meta / name).is_file():
            raise FileNotFoundError(f'{root.name}: missing meta/{name}')
    episodes = [json.loads(l) for l in (meta / 'episodes.jsonl').read_text().splitlines() if l.strip()]
    parquet = sorted(root.glob('data/chunk-*/episode_*.parquet'))
    videos = sorted(root.glob('videos/chunk-*/observation.images.ego_view/episode_*.mp4'))
    if not (len(episodes) == len(parquet) == len(videos) == info['total_episodes'] == 1000):
        raise ValueError(f'{root.name}: episodes {len(episodes)} parquet {len(parquet)} videos {len(videos)} '
                         f'info {info["total_episodes"]}')
    if not str(info['codebase_version']).startswith('v2.'):
        raise ValueError(f'{root.name}: unexpected codebase_version {info["codebase_version"]}')
    return dict(folder=root.name, codebase_version=info['codebase_version'], fps=info['fps'],
                episodes=len(episodes), frames=int(sum(e['length'] for e in episodes)),
                total_frames_info=info['total_frames'], tasks=sum(1 for l in (meta / 'tasks.jsonl').open() if l.strip()))


def _retry(fn, what, attempts=50):
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:                # network hiccups: retry with backoff
            print(json.dumps(dict(event='retry', what=what, attempt=attempt, error=repr(exc)[:300])), flush=True)
            time.sleep(min(60.0, random.uniform(1.0, 5.0) * attempt))
    raise RuntimeError(f'giving up: {what}')


def download(out: Path, workers: int):
    """Per-folder tree listing + per-file download.

    The repo is too large for repo_info().siblings (truncated), so snapshot_download silently
    matches nothing; list each folder's tree instead (what starVLA's list_repo_files does).
    Files already present with the expected size are skipped, so the script is resumable.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    files = []
    for folder in FOLDERS:
        entries = _retry(lambda: list(api.list_repo_tree(REPO_ID, repo_type='dataset', revision=REVISION,
                                                          path_in_repo=folder, recursive=True)), folder)
        files += [(e.path, e.size) for e in entries if getattr(e, 'size', None) is not None
                  and not hasattr(e, 'tree_id')]
    if any(not path.startswith(FOLDERS) for path, _ in files):
        raise RuntimeError('Listing returned a path outside the 24 folders')
    total = sum(s for _, s in files)
    todo = [(p, s) for p, s in files if not ((out / p).is_file() and (out / p).stat().st_size == s)]
    print(json.dumps(dict(event='listing', files=len(files), gb=round(total / 1e9, 2), todo=len(todo))), flush=True)

    def fetch(path, size):
        _retry(lambda: hf_hub_download(REPO_ID, path, repo_type='dataset', revision=REVISION, local_dir=str(out)), path)
        if (out / path).stat().st_size != size:
            raise RuntimeError(f'size mismatch: {path}')
        return path

    started, done = time.time(), 0
    with ThreadPoolExecutor(workers) as pool:
        for future in as_completed([pool.submit(fetch, p, s) for p, s in todo]):
            future.result()
            done += 1
            if done % 500 == 0 or done == len(todo):
                print(json.dumps(dict(event='progress', done=done, of=len(todo),
                                      minutes=round((time.time() - started) / 60, 1))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    out = Path(a.out)
    if not a.verify_only:
        download(out, a.workers)
    present = sorted(d.name for d in out.iterdir() if d.is_dir() and not d.name.startswith('.'))
    unexpected = [d for d in present if d not in FOLDERS]
    if unexpected:
        raise SystemExit(f'Unexpected folders present: {unexpected}')
    rows = [verify_folder(out / f) for f in FOLDERS]
    summary = dict(repo=REPO_ID, revision=REVISION, folders=len(rows), episodes=sum(r['episodes'] for r in rows),
                   frames=sum(r['frames'] for r in rows), per_folder=rows)
    (out / 'download_manifest.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_folder'}), flush=True)


if __name__ == '__main__':
    main()
