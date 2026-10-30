"""Full check of a finished matched run.

Usage: python3 -I verify_run.py RUN_DIR EXPECTED_SHA256 EXPECTED_ACTION_CORE_VIDEO_LOOPS_JSON
EXPECTED_SHA256 is one hex digest, or a JSON object suite -> digest (Light-WAM has one checkpoint per suite).
EXPECTED_ACTION_CORE_VIDEO_LOOPS_JSON is e.g. '[1,2,3,4]', or 'null' for families without it (baselines, Dense).

Checks the complete 6000-episode grid from episodes/*.json (file name vs contents, initial state = episode index,
checkpoint sha), the 120 group files against the episode files, the manifest protocol, every "built" report (sha,
loops, NFE 10, CFG 1, 30 settling steps, replan 10, nothing left unloaded; strict loads for LoopWAM families, whose
loader is ours — FasterWAM and Light-WAM load through their repos' own non-strict loaders), the summary totals, and
that error.json (if any) predates the summary. Prints overall, per-suite and per-seed success rates; with --json
also writes them to the given file.
"""
import json, os, sys
from pathlib import Path

SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
SEEDS = [42, 43, 44]
TASKS = range(10)
EPISODES = range(50)

args = [a for a in sys.argv[1:] if not a.startswith('--json=')]
out_json = next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--json=')), None)
run, sha_arg, loops = Path(args[0]), args[1], json.loads(args[2])
expected = json.loads(sha_arg) if sha_arg.startswith('{') else {s: sha_arg for s in SUITES}
issues = []
bad = issues.append

# Episodes: exactly the full grid, each file consistent with its name.
episodes = {}
for f in (run / 'episodes').iterdir():
    if not f.name.endswith('.json'):
        bad(f'stray file {f.name}')
        continue
    try:
        e = json.loads(f.read_text())
    except Exception as exc:
        bad(f'unreadable {f.name}: {exc}')
        continue
    gid = f"{e['suite']}_seed{e['seed']}_task{e['task']}"
    if e['id'] != gid or f.name != f"{gid}_episode{e['episode']}.json":
        bad(f'name/content mismatch {f.name}')
    if e['initial_state_index'] != e['episode']:
        bad(f'initial state {e["initial_state_index"]} != episode {e["episode"]} in {f.name}')
    if e['checkpoint_sha256'] != expected.get(e['suite']):
        bad(f'checkpoint sha in {f.name}')
    episodes[(e['suite'], e['seed'], e['task'], e['episode'])] = e
grid = {(s, sd, t, ep) for s in SUITES for sd in SEEDS for t in TASKS for ep in EPISODES}
if set(episodes) != grid:
    bad(f'grid: {len(grid - set(episodes))} missing, {len(set(episodes) - grid)} extra')

# Groups: 120 complete groups that agree with the episode files.
groups = list((run / 'groups').glob('*.json'))
if len(groups) != 120:
    bad(f'{len(groups)} group files')
for g in groups:
    rows = json.loads(g.read_text())
    if len(rows) != 50 or sorted(r['episode'] for r in rows) != list(EPISODES):
        bad(f'group {g.name} episodes')
    for r in rows:
        e = episodes.get((r['suite'], r['seed'], r['task'], r['episode']))
        if e is None or any(e[k] != r[k] for k in ('success', 'action_digest', 'policy_steps', 'checkpoint_sha256')):
            bad(f'group {g.name} episode {r["episode"]} differs from its episode file')

# Manifest protocol.
p = json.loads((run / 'manifest.json').read_text())['protocol']
if p['name'] != 'lightwam_20261009_matched':
    bad(f'protocol {p["name"]}')
if p['checkpoint_sha256'] != expected:
    bad('manifest checkpoint sha')
if len(p['groups']) != 120:
    bad('manifest groups')

# Build reports (a line cut off by a crash is reported, not fatal).
builds, unreadable = [], 0
for line in (run / 'events.jsonl').read_text().splitlines():
    if '"built"' not in line:
        continue
    try:
        builds.append(json.loads(line)['report'])
    except Exception:
        unreadable += 1
if not builds:
    bad('no build reports')
for r in builds:
    c = r['config']
    if r['checkpoint_sha256'] != expected.get(r['suite']):
        bad('build sha')
    if r.get('action_core_video_loops') != loops:
        bad(f'build loops {r.get("action_core_video_loops")}')
    if (c['num_inference_steps'], c['text_cfg_scale'], c['num_steps_wait'], c['replan_steps']) != (10, 1.0, 30, 10):
        bad(f'build config {c}')
    if 'loops' in r and not all(x['strict'] and not x['missing'] and not x['unexpected'] for x in r['load_calls']):
        bad('non-strict load')
    if r['unloaded']:
        bad(f'unloaded tensors {r["unloaded"][:3]}')

# Summary.
s = json.loads((run / 'summary.json').read_text())
succ = sum(e['success'] for e in episodes.values())
if s['episodes'] != 6000 or s['successes'] != succ or s['checkpoint_sha256'] != expected:
    bad(f'summary {s["episodes"]} {s["successes"]} vs episodes {succ}')
err = run / 'error.json'
if err.exists() and os.path.getmtime(err) > os.path.getmtime(run / 'summary.json'):
    bad('error.json newer than summary')

rate = lambda keys: 100 * sum(episodes[k]['success'] for k in keys) / max(len(keys), 1)
result = dict(
    run=run.name, passed=not issues, issues=issues, builds=len(builds), unreadable_build_lines=unreadable,
    episodes=len(episodes), successes=succ, success_rate=round(100 * succ / max(len(episodes), 1), 4),
    checkpoint_sha256=expected, action_core_video_loops=loops,
    suites={su: dict(rate=round(rate([k for k in episodes if k[0] == su]), 4),
                     seeds={sd: round(rate([k for k in episodes if k[0] == su and k[1] == sd]), 4) for sd in SEEDS})
            for su in SUITES},
    seeds={sd: round(rate([k for k in episodes if k[1] == sd]), 4) for sd in SEEDS})
print(run.name)
print(f'  builds={len(builds)}' + (f' (+{unreadable} unreadable build lines)' if unreadable else ''))
print(f'  errors={len(issues)}')
for i in issues[:20]:
    print('   ', i)
print(f'  overall {succ}/{len(episodes)} = {100 * succ / max(len(episodes), 1):.2f}%')
for su in SUITES:
    seeds = ' '.join(f'{v:.1f}' for v in result['suites'][su]['seeds'].values())
    print(f'  {su:15s} {result["suites"][su]["rate"]:6.2f}%  seeds {seeds}')
print('  per seed ' + ' '.join(f'{v:.2f}' for v in result['seeds'].values()))
if out_json:
    Path(out_json).write_text(json.dumps(result, indent=2) + '\n')
sys.exit(1 if issues else 0)
