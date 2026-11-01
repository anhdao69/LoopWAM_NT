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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    out = Path(a.out)
    if not a.verify_only:
        from huggingface_hub import snapshot_download
        for attempt in range(1, 31):
            try:
                snapshot_download(REPO_ID, repo_type='dataset', revision=REVISION, local_dir=str(out),
                                  allow_patterns=[f'{f}/*' for f in FOLDERS], max_workers=a.workers)
                break
            except Exception as exc:            # network hiccups: resume (completed files are skipped)
                print(json.dumps(dict(event='retry', attempt=attempt, error=repr(exc)[:300])), flush=True)
                time.sleep(min(60, 5 * attempt))
        else:
            raise SystemExit('download failed after 30 attempts')
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
