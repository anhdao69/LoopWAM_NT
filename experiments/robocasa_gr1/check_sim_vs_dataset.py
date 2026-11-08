#!/usr/bin/env python3
"""Check (b): RoboCasa-GR1 sim reset observation vs the training dataset (runs in the robocasa venv).

Per task: build the GR00T env (seeded), reset, and compare with frame 0 of the first K dataset episodes:
  * instruction: env annotation.human.coarse_action == dataset tasks.jsonl string (prompt-cache hit)
  * image: both sim views (ego_view_bg_crop_pad_res256_freq20 = the starVLA-served view, and
    ego_view_pad_res256_freq20) vs dataset ego_view: black-border rows/cols profile, mean |diff| to the
    nearest dataset first frame, mean-row-profile correlation; side-by-side PNG
  * state: sim state.* (29-D) vs dataset frame-0 observation.state (selected 29-D): distance to nearest
  * spaces: env action/observation keys and dims match the 29-D layout
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import cv2
import gymnasium as gym
import numpy as np
import pyarrow.parquet as pq

import robocasa  # noqa: F401
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401

GROUPS = ('left_arm', 'right_arm', 'left_hand', 'right_hand', 'waist')
DIMS = dict(left_arm=7, right_arm=7, left_hand=6, right_hand=6, waist=3)
SRC = dict(left_arm=(0, 7), right_arm=(22, 29), left_hand=(7, 13), right_hand=(29, 35), waist=(41, 44))
VIEWS = ('video.ego_view_bg_crop_pad_res256_freq20', 'video.ego_view_pad_res256_freq20')


def first_frame(path):
    with av.open(str(path)) as c:
        for f in c.decode(c.streams.video[0]):
            return f.to_ndarray(format='rgb24')


def border(img, thr=8):
    """Number of (near-)black rows at top/bottom and cols at left/right."""
    dark_r = img.max(axis=(1, 2)) <= thr; dark_c = img.max(axis=(0, 2)) <= thr

    def run(x):
        n = 0
        for v in x:
            if not v:
                break
            n += 1
        return n
    return dict(top=run(dark_r), bottom=run(dark_r[::-1]), left=run(dark_c), right=run(dark_c[::-1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, help='robocasa_gr1 dataset root')
    p.add_argument('--tasks', default='PnPCupToDrawerClose,PosttrainPnPNovelFromPlateToPanSplitA')
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    report = {}
    for task in a.tasks.split(','):
        folder = Path(a.root) / f'gr1_unified.{task}_GR1ArmsAndWaistFourierHands_1000'
        env_name = f'gr1_unified/{task}_GR1ArmsAndWaistFourierHands_Env'
        tasks = [json.loads(l)['task'] for l in (folder / 'meta/tasks.jsonl').read_text().splitlines() if l.strip()]
        info = json.loads((folder / 'meta/info.json').read_text())
        frames, states = [], []
        for ep in range(a.episodes):
            chunk = ep // info['chunks_size']
            frames.append(first_frame(folder / info['video_path'].format(episode_chunk=chunk, video_key='observation.images.ego_view', episode_index=ep)))
            st = np.stack(pq.read_table(folder / info['data_path'].format(episode_chunk=chunk, episode_index=ep),
                                        columns=['observation.state']).column(0).to_numpy(zero_copy_only=False))[0]
            states.append(np.concatenate([st[SRC[g][0]:SRC[g][1]] for g in GROUPS]))
        frames, states = np.stack(frames), np.stack(states)
        env = gym.make(env_name, enable_render=True, seed=a.seed * 1000)
        obs, _ = env.reset()
        sim_state = np.concatenate([np.asarray(obs[f'state.{g}'], np.float64).reshape(-1) for g in GROUPS])
        rec = dict(env_name=env_name, dataset_tasks=tasks, sim_instruction=obs['annotation.human.coarse_action'],
                   instruction_in_dataset=obs['annotation.human.coarse_action'] in tasks,
                   action_space={k: list(v.shape) for k, v in env.action_space.items()},
                   observation_keys=sorted(obs.keys()),
                   state_dims={g: int(np.asarray(obs[f'state.{g}']).size) for g in GROUPS},
                   dataset_border=[border(f) for f in frames[:3]], views={})
        rec['action_dims_match'] = all(rec['action_space'].get(f'action.{g}') == [DIMS[g]] for g in GROUPS)
        rec['state_dims_match'] = rec['state_dims'] == DIMS
        d_state = np.abs(states - sim_state).max(1)
        rec['state'] = dict(sim=sim_state.tolist(), nearest_dataset_ep=int(d_state.argmin()), nearest_max_abs=float(d_state.min()),
                            dataset_spread_max_abs=float(np.abs(states - states[0]).max()))
        mean_rows = frames.astype(np.float32).mean(axis=(0, 2, 3))
        for v in VIEWS:
            img = np.asarray(obs[v])
            diffs = np.abs(frames.astype(np.float32) - img.astype(np.float32)).mean(axis=(1, 2, 3))
            rows = img.astype(np.float32).mean(axis=(1, 2))
            rec['views'][v] = dict(shape=list(img.shape), border=border(img), nearest_mean_abs_diff=float(diffs.min()),
                                   median_mean_abs_diff=float(np.median(diffs)),
                                   row_profile_corr=float(np.corrcoef(rows, mean_rows)[0, 1]))
            cv2.imwrite(str(out / f'{task}_{v.split(".")[1]}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f'{task}_dataset_ep0.png'), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
        strip = np.concatenate([frames[int(np.abs(frames.astype(np.float32) - np.asarray(obs[VIEWS[0]], np.float32)).mean(axis=(1, 2, 3)).argmin())]]
                               + [np.asarray(obs[v]) for v in VIEWS], axis=1)
        cv2.imwrite(str(out / f'{task}_compare_dataset_bgcrop_pad.png'), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        env.close()
        report[task] = rec
        print(json.dumps({task: dict(instruction_in_dataset=rec['instruction_in_dataset'], action_dims_match=rec['action_dims_match'],
                                     state_dims_match=rec['state_dims_match'], state=rec['state']['nearest_max_abs'],
                                     views={v: rec['views'][v]['nearest_mean_abs_diff'] for v in VIEWS},
                                     border={v: rec['views'][v]['border'] for v in VIEWS}, dataset_border=rec['dataset_border'][0])}), flush=True)
    (out / 'check_b_report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
