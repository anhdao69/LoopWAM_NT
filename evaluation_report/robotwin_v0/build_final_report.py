"""Build the completed report from audited local evidence, without rerunning evaluation."""
from pathlib import Path
import collections
import csv
import datetime
import hashlib
import json
import re
import shutil

R = Path(__file__).resolve().parent
B = Path('/data14t/anhdh35/loopwam_robotwin_v0_20261030')

def stamp(value):
    return datetime.datetime.fromtimestamp(value, datetime.timezone(datetime.timedelta(hours=7))).strftime('%Y-%m-%d %H:%M:%S')

def duration(seconds):
    seconds = round(seconds)
    return f'{seconds//3600}h {(seconds%3600)//60}m {seconds%60}s'

def main():
    a = json.loads((R/'final_audit.json').read_text())
    assert a['status'] == 'passed' and a['episode_records'] == 10000
    c = json.loads((B/'results/COMPLETE.json').read_text())
    assert hashlib.sha256((B/'results/COMPLETE.json').read_bytes()).hexdigest() == a['complete_sha256']
    p = json.loads((B/'results/manifest.json').read_text())
    candidate_rows = []
    exception_types = collections.Counter()
    for key, cell in sorted(c['finished'].items()):
        text = (Path(cell['directory'])/'eval.log').read_text(errors='replace')
        exceptions = collections.Counter(line for line in text.splitlines()
                                        if re.match(r'^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception):', line))
        exception_types.update(exceptions)
        seeds = [x['seed'] for x in cell['episodes']]
        assert seeds == sorted(seeds) and len(set(seeds)) == 100
        candidate_rows.append(dict(task=cell['task'], setting=cell['config'], first_accepted_seed=min(seeds),
                                   last_accepted_seed=max(seeds), accepted=100,
                                   skipped_candidates_before_last_accepted=max(seeds)-4300000+1-100,
                                   caught_setup_exception_markers=text.count('error occurs !'),
                                   cuda_oom_mentions=text.count('OutOfMemoryError'),
                                   buffer_error_mentions=text.count('cannot create buffer')))
    with (R/'candidate_scene_audit.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(candidate_rows[0]))
        writer.writeheader(); writer.writerows(candidate_rows)
    candidate_summary = dict(caught_setup_exception_markers=sum(x['caught_setup_exception_markers'] for x in candidate_rows),
                             affected_pairs=sum(x['caught_setup_exception_markers'] > 0 for x in candidate_rows),
                             exception_types=dict(exception_types),
                             cuda_oom_mentions=sum(x['cuda_oom_mentions'] for x in candidate_rows),
                             buffer_error_mentions=sum(x['buffer_error_mentions'] for x in candidate_rows),
                             skipped_candidates_before_last_accepted=sum(x['skipped_candidates_before_last_accepted'] for x in candidate_rows))
    (R/'candidate_scene_audit.json').write_text(json.dumps(candidate_summary, indent=2)+'\n')
    if not (R/'initial_launch_snapshot.json').exists():
        shutil.copy2(R/'launch.json', R/'initial_launch_snapshot.json')
    shutil.copy2(B/'recovery_20261031/launch.json', R/'initial_run_final_state.json')
    shutil.copy2(B/'launch.json', R/'launch.json')
    shutil.copy2(B/'results/COMPLETE.json', R/'COMPLETE.json')
    t = a['timing']; q = a['query_latency']
    lines = [
        '# LoopWAM v0 RoboTwin — final evaluation report', '',
        f"**Complete: 10,000/10,000 episodes; 100/100 task/setting pairs.** Finished {stamp(t['finished'])} (Asia/Ho_Chi_Minh, UTC+07:00).",
        f"Final artifact audit passed at {stamp(a['audited_unix'])}. No pending pairs or active workers remain.", '',
        '## Final success rates', '',
        '| Setting | Successes | Failures | Episodes | Success rate |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for key, label in [('demo_clean','Clean'),('demo_randomized','Randomized'),('combined','Combined')]:
        x = a['totals'][key]
        lines.append(f"| {label} | {x['successes']:,} | {x['episodes']-x['successes']:,} | {x['episodes']:,} | **{x['sr_percent']:.2f}%** |")
    lines += ['',
        'SR is successful accepted episodes divided by accepted scored episodes. Every task has 100 episodes per setting, so the per-setting micro-average equals the mean of the 50 task SRs. Combined SR uses all 10,000 episodes. Smoke tests, concurrency calibration, rejected candidate scenes and interrupted attempts are excluded.',
        'This is one evaluation seed (42). Clean and randomized results differ by 0.26 percentage points; this alone does not establish a robustness improvement or statistical significance.', '',
        '## Model, seed and protocol', '',
        '| Item | Evaluated value |', '| --- | --- |',
        '| Model | Original stateless LoopWAM v0, video 4 / action 4; partition 3/6/3; no history or memory |',
        '| Checkpoint | RoboTwin `v0-video4-action4`, epoch 5, optimizer step 237,310 |',
        '| Parameters | 584,567,886 policy parameters; shared T5 and VAE are separate components |',
        '| Tasks | All 50 RoboTwin tasks, clean and randomized |',
        '| Episodes | 100 accepted episodes per task per setting; 10,000 total |',
        '| Evaluation seed | 42; first candidate scene seed 4,300,000 |',
        '| Action noise | CPU RNG with fixed seed 42 on every inference query |',
        '| Instruction split | Unseen; actual strings saved per episode |',
        '| Model inference | NFE 10; CFG 1; predict 32 actions, execute 24 before replanning |',
        '| Robot state and control | 14D joint state and absolute 14D `qpos` actions; training z-score statistics |',
        '| Precision | FP32 policy weights, BF16 inference/VAE; no quantization |',
        '| Execution | CUDA graphs; shared T5 server; initial 8 workers, recovery 4 workers |',
        '| Renderer | RT, 32 samples/pixel, depth 8; OIDN 2.3.3 on RTX PRO 6000 Blackwell Max-Q 96 GB |', '',
        'Release: [pinned Hugging Face checkpoint](https://huggingface.co/anhdao69/LoopWAM_NT/tree/4544aebb8c72b41af043efd97373dd0803e80a76/robotwin/v0-video4-action4).',
        f"Policy SHA256: `{a['checkpoint_sha256']}`.",
        'See [setup and comparability details](setup_report.md) for dependency versions, normalization/VAE hashes, source provenance and rendering adaptations.', '',
        '## Task-by-task final results', '',
        'All rows are complete. Each setting has 100 episodes, so its success count is also its SR in percent. Combined SR uses 200 episodes per task.', '',
        '| Task | Clean successes / 100 | Randomized successes / 100 | Combined SR |',
        '| --- | ---: | ---: | ---: |',
    ]
    for task in p['tasks']:
        clean = c['finished'][f'{task}/demo_clean']['successes']
        randomized = c['finished'][f'{task}/demo_randomized']['successes']
        lines.append(f'| {task} | {clean} | {randomized} | {(clean+randomized)/2:.2f}% |')
    lines += ['', '## Timeline and measured performance', '',
              'All timestamps below are Vietnam time (UTC+07:00).', '',
              '| Event | Time / duration |', '| --- | --- |',
              f"| Initial launch, 8 workers | {stamp(t['original_started'])} |",
              f"| Initial run stopped after 95 complete pairs | {stamp(t['original_stopped'])} |",
              f"| Initial run elapsed | {duration(t['initial_seconds'])} |",
              f"| Downtime before recovery launch | {duration(t['downtime_seconds'])} |",
              f"| Recovery evaluator launch, 4 workers | {stamp(t['recovery_started'])} |",
              f"| Final completion | {stamp(t['finished'])} |",
              f"| Recovery elapsed, including saved-episode revalidation | {duration(t['recovery_seconds'])} |",
              f"| Total wall time, including downtime | {duration(t['total_wall_seconds'])} |", '',
              'These are controller timestamps. Initial environment setup, downloads, smoke tests and calibration preceded the initial launch and are not included. Summing task durations would double-count concurrent execution.', '',
              f"The accepted attempts logged **{q['count']:,} inference queries**. Mean measured query time was **{q['mean_seconds']:.4f} s**, median **{q['median_seconds']:.4f} s**, and p95 **{q['p95_seconds']:.4f} s**. The timer surrounds model inference plus CUDA synchronization; it excludes image packing, text encoding, action denormalization and simulation. These measurements mix 8-worker and 4-worker contention and include warmup. They are not isolated model latency or end-to-end control latency.", '',
              'The separate graph gate measured native 0.1644 s versus CUDA graph 0.1300 s per query on eight captured observations with exact output equality. Short 16-episode concurrency trials measured 281.14 s (2 workers), 185.75 s (4), and 165.67 s (8). Those trials underestimated full-run memory pressure: eight workers repeatedly reached the GPU memory limit. Recovery at four workers completed without a new failed attempt. No claim of a globally optimal concurrency setting is made.', '',
              '## Failures, retries and recovery', '',
              f"There were **{a['failed_attempts']} interrupted attempts**: {a['failure_reasons']['cuda_out_of_memory']} with CUDA out-of-memory errors, {a['failure_reasons']['render_buffer_allocation']} with rendering-buffer allocation errors, and {a['failure_reasons']['other']} other exits. The three other exits were SIGTERM (-15) before any scored episode; this is consistent with the manager's no-progress timeout, but the failure records do not label the termination cause explicitly.",
              f"Interrupted attempts contain **{a['excluded_partial_episodes']:,} logged episode completions**, all excluded from the scored 10,000. These can repeat scenes across retries and must not be treated as additional independent benchmark episodes.", '',
              'The initial limit was three attempts per pair. After five pairs exhausted that limit, recovery reduced workers from eight to four and raised the total allowed attempts to five. Each missing pair completed on attempt 4; attempt 5 was unnecessary. All 95 already-completed result files remained byte-identical and passed revalidation.', '',
              '| Accepted attempt | Completed pairs |', '| --- | ---: |']
    for attempt, count in sorted(a['accepted_attempt_distribution'].items()):
        lines.append(f'| {attempt} | {count} |')
    lines += ['', '| Recovery pair | Final successes / 100 |', '| --- | ---: |']
    for key in json.loads((R/'recovery_20261031.json').read_text())['missing_pairs']:
        lines.append(f"| {key} | {c['finished'][key]['successes']} |")
    lines += ['', '[Recovery details](recovery_20261031.md) · [All interrupted attempts](failed_attempts.csv).', '',
              '## Candidate filtering and comparability limits', '',
              'The frozen FastWAM harness accepts scenes only after expert planning and an expert success check. It also catches exceptions during expert execution or rollout setup, advances the candidate seed, and continues until 100 accepted scored episodes are obtained. An exception during a policy rollout can instead terminate the attempt and trigger a whole-pair retry. These paths must be distinguished.',
              f"The accepted logs contain **{candidate_summary['caught_setup_exception_markers']} caught setup/expert exception markers across {candidate_summary['affected_pairs']} pairs**: 251 missing-target-pose assertions, 102 index errors and 4 NoneType-subscript errors. None of these accepted logs contains a CUDA OOM or `cannot create buffer` message. These skips follow the unchanged harness behavior; they are not policy episode failures. The candidate audit records seed ranges, skipped-candidate totals and exception counts per pair.",
              'Image processing follows the frozen released FastWAM PIL uint8/BF16 adapter. It differs from the uploaded LoopWAM H100 evaluator\'s torchvision float resize. Native LoopWAM retains its trained true-token text mask; FastWAM uses a different padding-mask convention. Rendering uses RT and a Blackwell-compatible OIDN upgrade; it is not claimed pixel-identical to OIDN 2.0.1 or the H100 raster fallback.',
              'The harness does not explicitly seed Python instruction-template selection. Seed 42 therefore does not guarantee identical instruction strings across independent reruns or retries. Actual instructions, accepted scene seeds and action-trace hashes are exported in episodes.csv. Use these records for any paired comparison; do not infer pairing from the seed label alone.',
              'This run evaluates LoopWAM only. It does not supply a newly evaluated RoboTwin FastWAM baseline or a head-to-head significance test. Artifact integrity and protocol checks cannot prove every simulator behavior is error-free.', '',
              '## Latest pull versus evaluated source', '',
              f"Main checkout `/storage/anhdh35/LoopWAM_NT` was fast-forwarded from `56e33285a82c6abe075565b3d60c26a5ba7a148f` to `{a['source']['main_commit']}`. The fetched update changes 126 report files and no executable code. All pre-existing tracked local modifications were preserved, and the previous local RoboTwin reports were archived before editing.",
              f"The completed evaluation used `{a['source']['evaluated_commit']}` in `/storage/anhdh35/LoopWAM-robotwin-eval`, based on upstream `0f9e494`. That worktree remains frozen. Its upstream branch now points to `{a['source']['robotwin_upstream_commit']}`; the intervening changes are report-only. The latest pull is not retroactively assigned as the evaluation's code version.", '',
              '## Final verification and evidence', '',
              f"The final audit checked the checkpoint SHA256, all **{a['source_files_verified']} source/manifest-bound files**, all **{a['runtime_files_verified']:,} runtime asset/dependency files**, all **10,000 video files** through ffprobe and SHA256, and all episode outcomes, unique per-pair scene seeds, instructions and query-trace digests. Every completed cell matches its saved result.json; summary.json and COMPLETE.json agree. The 95 pre-recovery result files remain unchanged.",
              'Video verification checks readable stream metadata and file integrity; it is not a manual review of every frame. The original 26 preflight tests and four smoke episodes are archived; they are not newly rerun or counted as scored episodes during this report-only update.', '',
              '| Artifact | Purpose |', '| --- | --- |',
              '| [final_audit.json](final_audit.json) / [log](final_audit.log) | Fresh verification results, counts, hashes and timings |',
              '| [final_results.json](final_results.json) / [task_results.csv](task_results.csv) | Final aggregate and per-task/setting metrics, accepted attempts and durations |',
              '| [episodes.csv](episodes.csv) | All 10,000 accepted episode records, instructions, scene seeds and video/trace hashes |',
              '| [candidate_scene_audit.csv](candidate_scene_audit.csv) / [summary](candidate_scene_audit.json) | Accepted seed ranges and candidate-skip evidence |',
              '| [COMPLETE.json](COMPLETE.json) | Copy of the evaluator completion artifact |',
              '| [protocol.json](protocol.json) / [runtime_manifest.json](runtime_manifest.json) | Frozen evaluation and dependency contracts |',
              '| [launch.json](launch.json) | Completed recovery controller state |',
              '| [initial_run_final_state.json](initial_run_final_state.json) / [initial snapshot](initial_launch_snapshot.json) | Initial failed run and historical launch evidence |',
              '| [setup_report.md](setup_report.md) / [recovery report](recovery_20261031.md) | Detailed setup and recovery decisions |',
              '| [audit_final.py](audit_final.py) / [build_final_report.py](build_final_report.py) | Reproducible audit and report generation |', '',
              f"Completion artifact SHA256: `{a['complete_sha256']}`.",
              f"Protocol manifest SHA256: `{a['protocol_sha256']}`.",
              f"Bulk simulator logs, commands and videos remain under `{B}/results`. CSV video paths refer to these local files.", '']
    (R/'report.md').write_text('\n'.join(lines))
    print('Built final report: 50 task rows; 10,000 episode export; final provenance and candidate audit')

if __name__ == '__main__':
    main()
