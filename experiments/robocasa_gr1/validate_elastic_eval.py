#!/usr/bin/env python3
"""Validate a finished RoboCasa-GR1 official evaluation of a GR1 elastic export and print its result table.

Copy of gr1_eval/tools/gr1_validate.py (the official GR1 v0 eval validator) for elastic_policy_server.py: the server
metadata must name the expected schedule (loops), NFE = schedule length and the export's sha256 instead of NFE 10.
Usage: validate_elastic_eval.py OUT EXPECTED_STEP --schedule NAME --export-sha256 SHA [--render egl] [--json report.json]
Checks per task cell (n12/<task>.json): protocol (50 episodes, n_envs 1, n_action_steps 12, max_episode_steps 720,
seed 0), export step/sha + normalization sha + schedule + NFE + CFG 1 + noise seed 42 from the server metadata, query count
consistent with 12 env steps per query over 720-step episodes, exactly 50 finished videos whose success flags match
the JSON, render backend, and that every prompt was served from the training text cache (server logs).
"""
import argparse, glob, json, os, re, sys

NORM = 'bad54856a2835c9a65c5b6288cd6c8e55660df2d3990777009f22842197be9eb'
p = argparse.ArgumentParser()
p.add_argument('out'); p.add_argument('expected_step', type=int)
p.add_argument('--render', default='egl'); p.add_argument('--json')
p.add_argument('--schedule', required=True); p.add_argument('--export-sha256', required=True)
a = p.parse_args()
LOOPS = {f'flat_k{k}_n{n}': [k] * n for k in (1, 2, 4) for n in (2, 5, 10)}[a.schedule]
problems, rows, final_succ = [], [], {}
summ = json.load(open(f'{a.out}/summary.json'))['by_n_action_steps']['12']
cells = sorted(glob.glob(f'{a.out}/n12/*.json'))
if len(cells) != 24 or summ['tasks'] != 24 or not summ['complete']:
    problems.append(f'expected 24 complete task cells, found {len(cells)} (summary tasks={summ["tasks"]}, complete={summ["complete"]})')
for f in cells:
    d = json.load(open(f)); task = os.path.basename(f)[:-5]; m = d['server_metadata']; c = d['client']
    exp = dict(n_episodes=50, n_envs=1, n_action_steps=12, max_episode_steps=720, seed=0)
    for k, v in exp.items():
        if d.get(k) != v: problems.append(f'{task}: {k}={d.get(k)} != {v}')
    for k, v in dict(step=a.expected_step, normalization_sha256=NORM, num_inference_steps=len(LOOPS), cfg_scale=1.0,
                     noise_seed=42, action_chunk_size=32, video_key='video.ego_view', schedule=a.schedule, loops=LOOPS,
                     export_sha256=a.export_sha256, transport_kind='T0').items():
        if m.get(k) != v: problems.append(f'{task}: server {k}={m.get(k)} != {v}')
    if not 50 * 60 <= c['queries'] <= 50 * 61:
        problems.append(f'{task}: {c["queries"]} policy queries, expected 3000-3050 for 50 x 720 steps / 12')
    if d['env'].get('MUJOCO_GL') != a.render: problems.append(f'{task}: MUJOCO_GL={d["env"].get("MUJOCO_GL")}')
    succ = d['successes']
    if len(succ) != 50 or sum(succ) != round(d['success_rate'] * 50): problems.append(f'{task}: successes list inconsistent')
    vids = glob.glob(f'{a.out}/videos/n12/{task}/*.mp4')
    fin = [v for v in vids if re.search(r'_success[01]\.mp4$', v)]
    if len(fin) != 50 or len(vids) != 50: problems.append(f'{task}: {len(fin)} finished / {len(vids)} total videos, expected 50')
    # starVLA's VideoRecordingWrapper names a video by the success flag of its last rendered step, while the official
    # metric (simulation_env.py) is any-step success: so, in episode order (one client per task, mtime order),
    # every _success1 video must be a JSON success; the reverse need not hold (success reached, then undone).
    flags = [v.endswith('_success1.mp4') for v in sorted(fin, key=os.path.getmtime)]
    if len(flags) == len(succ) and any(f and not s for f, s in zip(flags, succ)):
        problems.append(f'{task}: a final-step-success video is not an any-step success in the JSON')
    final_succ[task] = sum(flags)
    if summ['per_task'].get(task, {}).get('successes') != sum(succ): problems.append(f'{task}: summary.json disagrees')
    rows.append((task.replace('_GR1ArmsAndWaistFourierHands_Env', ''), sum(succ), d['success_rate'], c['mean_roundtrip_s'],
                 d['wall_seconds'], c['instructions']))
logs = glob.glob(f'{a.out}/logs/server_*.log')
modes = {}
for lf in logs:
    for mo in re.finditer(r"prompt .* -> (\w+)", open(lf, errors='replace').read()): modes[mo.group(1)] = modes.get(mo.group(1), 0) + 1
if set(modes) - {'cache', 'memo'}: problems.append(f'prompt sources {modes}: some prompts were not from the training cache')
for lf in logs:
    t = open(lf, errors='replace').read()
    bad = [l for l in t.splitlines() if re.search(r'policy server error|RuntimeError|ValueError|CUDA error|out of memory', l)]
    if bad: problems.append(f'{os.path.basename(lf)}: {bad[:2]}')
mean = sum(r[2] for r in rows) / max(1, len(rows))
print(f'{a.out}  step {a.expected_step}  schedule {a.schedule}  tasks {len(rows)}  mean success {100 * mean:.2f}%  '
      f'({sum(r[1] for r in rows)}/{50 * len(rows)} episodes)  summary mean {100 * summ["mean_success"]:.2f}%')
print(f'prompt sources: {modes}; server logs: {len(logs)}; success still true at the final step: '
      f'{sum(final_succ.values())}/{sum(r[1] for r in rows)} any-step successes')
for r in sorted(rows):
    print(f'  {r[0]:<55} {r[1]:>2}/50  {100 * r[2]:5.1f}%   roundtrip {r[3]:.2f}s  wall {r[4] / 3600:.2f}h')
print('VALID' if not problems else 'PROBLEMS:\n  ' + '\n  '.join(problems))
if a.json:
    json.dump(dict(out=a.out, step=a.expected_step, schedule=a.schedule, export_sha256=a.export_sha256,
                   successes=sum(r[1] for r in rows), episodes=50 * len(rows), mean_success=mean, valid=not problems, problems=problems,
                   per_task={r[0]: dict(successes=r[1], success_rate=r[2]) for r in rows}), open(a.json, 'w'), indent=1)
sys.exit(0 if not problems else 1)
