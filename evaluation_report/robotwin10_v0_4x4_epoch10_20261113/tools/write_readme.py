#!/usr/bin/env python3
"""Write README.md of robotwin10_v0_4x4_epoch10_20261113 from results.json and task_results.csv (no hand-typed numbers)."""
import csv
import json
from pathlib import Path

D = Path(__file__).resolve().parents[1]
R = json.loads((D / 'results.json').read_text())
T = list(csv.DictReader(open(D / 'task_results.csv')))
S = R['summary']
REPO = [('robotwin10_v0_4x4', 'original local'), ('robotwin10_v0_4x4_remote189', 'server189 rerun'),
        ('robotwin10_v0_4x4_remote137_20261105', 'server137 rerun')]
TASKS = sorted({r['task'] for r in T})


def row(task, setting):
    return next(r for r in T if r['task'] == task and r['setting'] == setting)


def p(x):
    return f'{x:.2g}' if x < 0.001 else f'{x:.3f}'


ck = R['checkpoint']; tr = R['training']; ev = R['evaluation']
L = [f'# RoboTwin-10 v0 4/4, epoch 10 — H100 evaluation', '',
     f'Completed {ev["completed"]} on {ev["hardware"]}. All 20 task/setting cells are complete: 2,000 scored episodes '
     f'(10 tasks × clean/randomized × 100 accepted episodes, seed 42, NFE 10), one attempt per cell, no resumes.', '',
     '| Setting | Successes / episodes | SR | 95% CI (Wilson) |', '| --- | ---: | ---: | --- |']
for name, label in (('clean', 'demo_clean'), ('randomized', 'demo_randomized'), ('combined', 'combined')):
    e = S[name]['epoch10']
    L.append(f'| {label} | {e["successes"]}/{e["episodes"]} | **{e["sr_percent"]:.2f}%** | {e["wilson95"][0]:.2f}–{e["wilson95"][1]:.2f} |')
L += ['', '## Per task (successes out of 100 accepted episodes)', '',
      '| Task | Clean | Randomized | Combined SR | Clean, epoch 5 (same harness) | Randomized, epoch 5 (same harness) |',
      '| --- | ---: | ---: | ---: | ---: | ---: |']
for t in TASKS:
    c, r = row(t, 'demo_clean'), row(t, 'demo_randomized')
    comb = (int(c['epoch10_successes']) + int(r['epoch10_successes'])) / 2
    L.append(f'| {t} | {c["epoch10_successes"]} | {r["epoch10_successes"]} | {comb:.2f}% | '
             f'{c["epoch05_same_harness_successes"]} | {r["epoch05_same_harness_successes"]} |')
L.append(f'| **mean** | **{S["clean"]["epoch10"]["sr_percent"]:.1f}** | **{S["randomized"]["epoch10"]["sr_percent"]:.1f}** | '
         f'**{S["combined"]["epoch10"]["sr_percent"]:.2f}%** | {S["clean"]["epoch05_same_harness"]["sr_percent"]:.1f} | '
         f'{S["randomized"]["epoch05_same_harness"]["sr_percent"]:.1f} |')
L += ['', '## Epoch 10 versus epoch 5 on the same harness (paired)', '',
      'The epoch-5 parent (`' + ck['parent_epoch05_sha256'][:16] + '…`) was evaluated with the same H100 harness, policy code, '
      'scene-seed list, per-seed instructions and query noise between ' + R['epoch05_same_harness_evaluation']['started'] + ' and ' + R['epoch05_same_harness_evaluation']['completed'] + ' (' + str(R['epoch05_same_harness_evaluation']['attempts']) + ' attempts for 20 cells; interrupted cells resumed at the next scene seed, so no seed is scored twice). '
      'Pairs are the scene seeds '
      'accepted in both runs; RoboTwin\'s expert check is not fully deterministic, so a few seeds per cell differ. On every '
      f'shared seed both runs received the same instruction ({R["instruction_agreement_on_shared_seeds"]["same"]:,}/'
      f'{R["instruction_agreement_on_shared_seeds"]["shared"]:,}).', '',
      '| Setting | Epoch 5 | Epoch 10 | Shared seeds | Epoch 10 only | Epoch 5 only | Paired difference | Exact McNemar p |',
      '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for name in ('clean', 'randomized', 'combined'):
    s = S[name]; pr = s['paired']
    L.append(f'| {name} | {s["epoch05_same_harness"]["sr_percent"]:.2f}% | {s["epoch10"]["sr_percent"]:.2f}% | {pr["shared_seeds"]:,} | '
             f'{pr["epoch10_only"]} | {pr["epoch05_only"]} | {pr["difference_pp"]:+.2f} pp | {p(pr["mcnemar_p"])} |')
D10 = {(r['task'], r['setting']): int(r['difference_pp']) for r in T}
assert [D10[(t, s)] for t in ('hanging_mug', 'open_microwave', 'turn_switch', 'move_stapler_pad')
        for s in ('demo_clean', 'demo_randomized')] == [10, 10, 10, 11, 8, 3, 5, 7]
assert D10[('stamp_seal', 'demo_randomized')] == -6 == min(D10.values())
assert sorted(D10.values())[-2:] == [10, 11]
assert [r['task'] + r['setting'] for r in T if float(r['mcnemar_p']) < 0.05] == ['hanging_mugdemo_randomized']
L += ['', 'Per task, the largest gains are on the hard tasks: hanging_mug (+10 / +10), open_microwave (+10 / +11), turn_switch '
      '(+8 / +3) and move_stapler_pad (+5 / +7) clean / randomized; stamp_seal randomized is the largest drop (−6). Single '
      'task/setting cells have 100 episodes, and only hanging_mug randomized reaches p < 0.05 on its own (p = '
      f'{p(float(row("hanging_mug", "demo_randomized")["mcnemar_p"]))}, not corrected for 20 cells). Per-cell counts, '
      'shared seeds and p-values are in [task_results.csv](task_results.csv).', '',
      '## Context: the same epoch-5 checkpoint in this report collection', '',
      'Three earlier evaluations in this folder tree used the identical epoch-5 checkpoint (their final audits record SHA-256 '
      f'`{ck["parent_epoch05_sha256"][:16]}…`). With this harness\'s own epoch-5 run, there are four evaluations of epoch 5 '
      'on four hosts:', '',
      '| Checkpoint / evaluation | Clean | Randomized | Combined | Successes | Evidence |', '| --- | ---: | ---: | ---: | ---: | --- |']
for f, desc in REPO:
    L.append(f'| epoch 5 — {desc} | {S["clean"]["repo_" + f]["sr_percent"]:.2f}% | {S["randomized"]["repo_" + f]["sr_percent"]:.2f}% | '
             f'{S["combined"]["repo_" + f]["sr_percent"]:.2f}% | {S["combined"]["repo_" + f]["successes"]}/2000 | '
             f'[report](../{f}/final_report.md) |')
e5 = S['combined']['epoch05_same_harness']
L.append(f'| epoch 5 — H100, this harness | {S["clean"]["epoch05_same_harness"]["sr_percent"]:.2f}% | '
         f'{S["randomized"]["epoch05_same_harness"]["sr_percent"]:.2f}% | {e5["sr_percent"]:.2f}% | {e5["successes"]}/2000 | '
         '[episodes](reference_epoch05_episodes.csv) |')
e10 = S['combined']['epoch10']
L.append(f'| **epoch 10 — H100, this harness** | **{S["clean"]["epoch10"]["sr_percent"]:.2f}%** | '
         f'**{S["randomized"]["epoch10"]["sr_percent"]:.2f}%** | **{e10["sr_percent"]:.2f}%** | {e10["successes"]}/2000 | '
         '[episodes](episodes.csv) |')
e5s = [S['combined']['repo_' + f]['sr_percent'] for f, _ in REPO] + [e5['sr_percent']]
L += ['', f'The four epoch-5 evaluations span {min(e5s):.2f}–{max(e5s):.2f}% combined; epoch 10 is '
      f'{e10["sr_percent"] - max(e5s):+.2f} to {e10["sr_percent"] - min(e5s):+.2f} pp above them. Only the H100 pair is '
      'episode-paired; the other hosts used unseeded instruction sampling and different evaluation code, so they are '
      'aggregate references. The epoch-10 row is not added to the consolidated `all_results` exports, which are built '
      'on ws by `all_results/build_snapshot.py`.', '',
      '## Model', '',
      f'- Checkpoint: `epoch_10.pt`, SHA-256 `{ck["sha256"]}`, update {ck["step"]:,}. Published on Hugging Face as '
      f'`{ck["hf_repo"]}` / `{ck["hf_path"]}` (revision `{ck["hf_revision"][:8]}`; `policy.pt` has the same SHA-256; '
      '[upload verification](evidence/training/hf_upload_verification_epoch_10.json)).',
      f'- Training: {tr["recipe"]}. Updates {tr["updates"][0]:,}–{tr["updates"][1]:,}, {tr["training_hours"]:.2f} h, '
      f'{tr["windows_seen"]:,} windows seen in total. Code `{tr["git_revision"][:7]}` with the uncommitted continuation '
      'patch (per-file hashes in [manifest.json](evidence/training/manifest.json)).',
      '- Parent check: the continuation resumed from the epoch-5 `latest.pt` (update 45,445, 679 AdamW states), whose '
      f'tensors equal `epoch_05.pt` `{ck["parent_epoch05_sha256"][:16]}…` ([parent_verification.json](evidence/training/parent_verification.json), '
      '[resume audit](evidence/training/resume_audit_u45445.json)).',
      '- Mean training loss by epoch: ' + ', '.join(f'epoch {k} {v:.4f}' for k, v in tr['mean_loss_by_epoch'].items()) +
      '.', '',
      '## Protocol', '']
pr = R['protocol']
L += [f'- Harness: {pr["harness"]}; policy {pr["policy"]}; v0 with video/action loops 4/4.',
      f'- Seed {pr["seed"]}, first scene seed {pr["first_scene_seed"]:,}, expert check on, {pr["instruction_type"]} '
      f'instructions (pool {pr["instruction_pool"]}), {pr["episodes_per_cell"]} accepted episodes per task and setting.',
      f'- Predict {pr["action_horizon"]} actions, execute {pr["replan_steps"]}; {pr["num_inference_steps"]} denoising steps '
      f'(NFE 10), CFG {pr["cfg"]}, query noise {pr["noise"]}; renderer {pr["shader"]}.',
      '- The harness files and policy code were last modified before the first epoch-5 attempt, so both runs used the same '
      'files (hashes, modification times and git revision in [provenance.json](provenance.json)). Videos were recorded for every epoch-10 episode; they stay on the cluster and '
      'their names and SHA-256 are listed in [episodes.csv](episodes.csv).', '',
      '## Files', '',
      '- [episodes.csv](episodes.csv): the 2,000 scored epoch-10 episodes (scene seed, outcome, instruction, video SHA-256).',
      '- [reference_epoch05_episodes.csv](reference_epoch05_episodes.csv): the 2,000 scored epoch-5 episodes of the same harness.',
      '- [task_results.csv](task_results.csv) and [results.json](results.json): per-cell counts, paired statistics, references.',
      '- `evidence/epoch10/` and `evidence/epoch05_same_harness/`: per cell `result.json`, the launch `command.json`, the '
      'harness summary and the full `eval.log` (gzip). `evidence/training/`: training manifest, timing, precision, resume '
      'audit, parent check and Hugging Face upload verification.',
      '- [SOURCES.tsv](SOURCES.tsv): every copied file with its cluster path and SHA-256; [tools/](tools/) builds this '
      'folder from the cluster records.',
      '- Independent check: [verify_robotwin10_v0_4x4_epoch10_20261113.py](../verify_robotwin10_v0_4x4_epoch10_20261113.py) '
      'recomputes every number above from the episode files and the per-cell logs.', '']
(D / 'README.md').write_text('\n'.join(L))
print('\n'.join(L))
