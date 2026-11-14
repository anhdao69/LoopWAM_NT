# RoboTwin-50 elastic students on sn4622128794: E0-epoch10-teacher-5ep at NFE 5 and 10, and E1 epoch 1

**Status: complete.** Nine evaluations, each scoring **10,000 episodes**: all 50 tasks × clean and randomized ×
100 accepted episodes, seed 42. Every evaluation passed its final audit and its seed check against the reference.
They ran from 2026-11-07 to 2026-11-14 on `sn4622128794` (2× RTX PRO 6000 Blackwell Max-Q, 96 GB; a host shared with
other users).

- **Main:** E0-epoch10-teacher-5ep, the five-epoch student distilled from the v0 epoch-10 teacher, at six schedules:
  `[1]×5`, `[1]×10`, `[2]×5`, `[2]×10`, `[4]×5` and `[4]×10`. The checkpoint bytes are the same as in
  [robotwin_elastic_e0_teacher10_5ep_20261111](../robotwin_elastic_e0_teacher10_5ep_20261111/final_report.md), which
  evaluated `[1,1]`, `[2,2]` and `[4,4]` at NFE 2 on other hosts. Together they cover all nine schedules.
- **Also:** E1 epoch 1 (`RT50-ep5-E1`, distilled from the v0 epoch-5 teacher) at `[1,1]`, `[4,4]` and `[4]×5`. These
  were complete and audited on 2026-11-11 and are published here for the first time.

**Findings.**
- **Action loops raise success; going from 5 to 10 denoising steps does not.** At NFE 5, 1 → 2 → 4 loops give
  89.43 → 90.74 → 91.29%. Going from NFE 5 to NFE 10 changes the overall SR by +0.05, −0.06 and −0.06 pp at 1, 2 and
  4 loops.
- **At 4 loops the student is close to its teacher.** `[4]×5` reaches 91.29% and `[4]×10` 91.23%, against 91.87% for
  the v0 epoch-10 teacher. On the scenes both runs accepted, the gaps are 0.67 and 0.62 pp.
- **The cheaper schedule is as good.** `[4]×5` matches `[4]×10` at half the denoising steps.

Each result is one evaluation at one seed (see [Uncertainty](#uncertainty-and-comparability)).

## E0-epoch10-teacher-5ep: six schedules

<!-- TABLE:e0_main -->

`[k]×N` runs `k` action-core loops at each of `N` denoising steps (NFE `N`). The video expert always runs 4 loops. "Same
scenes" restricts the student and the teacher to the scene seeds both runs accepted in the same task/setting cell (see
[Comparability](#uncertainty-and-comparability)).

### All nine schedules of this checkpoint

<!-- TABLE:e0_grid -->

¹ NFE 2 results come from
[robotwin_elastic_e0_teacher10_5ep_20261111](../robotwin_elastic_e0_teacher10_5ep_20261111/final_report.md).
- **Same:** checkpoint bytes and protocol (50 tasks × 2 settings × 100 accepted episodes, seed 42, predict 32 /
  execute 24).
- **Different:** hosts (`[1,1]` on VMO-ANHDH35-D, `[2,2]` and `[4,4]` on server137), with CUDA graphs and 4 workers.

So a difference between the NFE-2 column and the other two combines the schedule with the host and deployment.

### Changes between schedules

<!-- TABLE:e0_deltas -->

Task comparisons use each task's SR over its 200 episodes.

### Student versus teacher

<!-- TABLE:e0_teacher -->

The teacher reference is [robotwin_v0_epoch10](../robotwin_v0_epoch10/report.md): 91.87%, run on the campaign evaluator
(commit `0a0c6e57`) on another RTX PRO 6000 host, with CUDA graphs and 4 workers.

Tasks with the largest gap between `[4]×5` and the teacher (the six largest deficits and the four largest gains):

<!-- TABLE:e0_task_gaps -->

### All 50 tasks

Each entry is clean/randomized successes out of 100, followed by the task SR in percent over its 200 episodes. Every task
and setting is included; nothing is replaced by a result from another run. The same numbers, with more columns, are in
[task_comparison.csv](task_comparison.csv).

<!-- TABLE:e0_tasks -->

## E1 epoch 1 (distilled from the v0 epoch-5 teacher)

<!-- TABLE:e1_main -->

The teacher reference is [robotwin_v0](../robotwin_v0/report.md) (91.73%).

² E0 epoch 1 is the T0 student distilled from the same epoch-5 teacher for one epoch (update 47,462). It was evaluated
in [robotwin_elastic_e0_20261107](../robotwin_elastic_e0_20261107/report.md) on other hosts (`[1,1]` and `[4]×5` on
VMO-ANHDH35-D, `[4,4]` on server137). E1 uses the T1 transport instead (see the E1 model card on Hugging Face). At the
same three schedules, E1 and E0 epoch 1 are within 0.5 pp of each other, which is within the cross-run scale described
below.

## Checkpoints

| | E0-epoch10-teacher-5ep | E1 |
|---|---|---|
| Hugging Face file (`anhdao69/LoopWAM_NT`) | `robotwin/Elastic/E0-epoch10-teacher-5ep/epoch_05.pt` | `robotwin/Elastic/E1/epoch_01.pt` |
| Pinned revision | `b72f355505d98c473fb91e45d10e16d4cc7ce536` | `a03864919c3932c2af686d137cdab75328e41da7` |
| SHA-256 | `07d24ff77410e516ac90d24d76d651e2102b09b0458d9d5d86ea18dff39e823e` | `9eba3aade1417b4425be5593c88e81441cf71bb4b6f16a8d81012ad6146e1bcb` |
| Size (bytes) | 2,340,331,885 | 2,340,331,629 |
| Recipe / transport | E0, T0 | E1, T1 |
| Student epoch / update | 5 / 237,310 | 1 / 47,462 |
| Teacher | v0 4/4 epoch 10, update 474,620, `70cf5af7…` | v0 4/4 epoch 5, update 237,310, `38975220…` |
| Training code | `elastic_robotwin` at `2c78ff6` | `elastic_robotwin` at `2c78ff6` |
| Normalization (`dataset_stats.json`) | `7a02c46c…`, identical for both and their teachers | same |

"epoch10" in the E0 release name refers to the teacher. The student is the epoch-5 export of a five-epoch run. Both files
were downloaded at the pinned revisions and checked against their sizes and `SHA256SUMS` before use.

## Protocol

This host reproduces the local RTX v0 protocol of [robotwin_v0](../robotwin_v0/report.md). Its evaluator was rebuilt from
the campaign evaluator (commit `0a0c6e57`) and checked against the campaign's recorded hashes:
- 777/777 evaluator source files;
- 212/212 Python package pins and 109/109 simulator environment files;
- 20,841 asset files;
- 160 CuRobo sources.

Details are in [setup/robotwin_local_setup.md](setup/robotwin_local_setup.md).

| Setting | Value |
|---|---|
| Tasks and settings | All 50 RoboTwin tasks; `demo_clean` and `demo_randomized` |
| Episodes | 100 accepted per task/setting; 10,000 per evaluation |
| Seed | 42. Candidate scenes start at 100000 × (1 + 42) = 4,300,000 and are accepted only if the scripted expert succeeds (original harness) |
| Instructions | Unseen instruction type, pool of 100; the harness draws them with unseeded Python `random`; the strings are recorded per episode |
| Action chunk | Predict 32, execute 24, then replan |
| Action noise | Fixed seed 42 for every policy query |
| Image input | Released FastWAM PIL uint8 preprocessing with BF16 image arithmetic |
| Text encoder | UMT5-xxl via one shared server per queue. It uses the lean loader, which builds the model on the meta device and loads the same bf16 weights; contexts and masks are bit-equal to the standard loader on 4 prompts (`verification/t5_lean_full_gpu.json`) |
| Inference | Eager (`LOOPWAM_CUDA_GRAPH=0`). The gates show the CUDA graph and eager outputs are bit-equal on `[4]×10` |
| Renderer | RT shader, 32 samples/pixel, depth 8, OIDN 2.3.3. An attempt is rejected if its log contains `OIDN Error`, `cannot create buffer`, `OutOfMemoryError` or `out of memory` |
| Scheduling | One orchestrator per queue, up to 6 workers per GPU. A worker starts only when GPU memory and host RAM allow, and cells run longest first. In practice 5–7 workers ran at a time. Scheduling does not change how a cell is evaluated |

**Differences from the teacher references.** These runs used eager inference, the lean T5 loader and a different worker
count. The teacher references used CUDA graphs and the standard loader. The gates above show the first two give identical
outputs, and worker count only changes scheduling.

The 789 runtime files that determine an episode had identical SHA-256 hashes in all nine run manifests:
- the evaluator, policy and transport code;
- 702 RoboTwin task and harness files;
- the UMT5 tokenizer;
- the environment manifest.

All 789 were unchanged when this report was exported. [setup/evaluation_code.tar.gz](setup/evaluation_code.tar.gz)
contains these files, along with the gate, audit and setup scripts. Two kinds are left out: the four public tokenizer
files, whose hashes are in every `manifest.json`, and the environment manifest, which is in
[verification/runtime_manifest.json.gz](verification/runtime_manifest.json.gz).

## Verification before scoring

| Check | E0-epoch10-teacher-5ep | E1 |
|---|---|---|
| Static export check (CPU): release SHA-256; 339 frozen video tensors and the proprio encoder bit-equal to the FP32 teacher; 338/338 student tensors trained and finite; step, kind, epoch and contract hashes | 22/22 ([record](verification/e0ep10_static.json)) | 22/22 ([record](verification/e1_static.json)) |
| GPU gates on 4 real harness observations (2 clean, 2 randomized), maximum absolute difference 0 | 10/10 ([gates.json](verification/gates_e0_teacher10_5ep/gates.json)) | 9/9 ([gates.json](verification/gates_e1/gates.json)) |
| Smoke run through the full queue (2 tasks × 2 settings × 2 episodes), audited, scene seeds equal to the reference | `[1]×5` and `[4]×10`: 16/16 episodes ([records](verification/smoke_e0_teacher10_5ep)) | `[1,1]` and `[4,4]`: 16/16 ([records](verification/smoke_e1)) |

The gates check five things:
- the text contexts are deterministic, finite and padded;
- the teacher in the merged evaluator is bit-equal to the campaign tree;
- the teacher's rollout through the elastic input path equals the deployed v0 `infer_action`;
- the student's inputs equal the teacher's;
- the student's rollouts on every evaluated schedule are finite, deterministic and consistent with the policy's query path.

For E0, a tenth check confirms that the student and v0 phases loaded the same teacher.

The gates also show that the CUDA graph equals eager on `[4]×10`, and that the pristine `elastic_robotwin` training tree at
`2c78ff6` produces bit-equal rollouts to the evaluator tree. The gate script names its student phases `e1_*` for any
student.

## Run record

<!-- TABLE:runs -->

Accepted cell-hours add up the durations of the accepted attempts, which ran in parallel; they are not elapsed time.

Every cell is scored from exactly one complete accepted attempt. A rejected attempt is discarded whole, with all its
partial episodes. The rejections were:

- **E1 queue (2026-11-07 22:22 → 11-08 22:04).** E1 `[1,1]` lost 6 attempts to GPU-memory faults while the GPUs were
  shared with other jobs. Four workers crashed with `cannot create buffer`. Two attempts finished all 100 episodes but
  were rejected because their logs contain `OIDN Error: out of memory`, meaning some frames were rendered without
  denoising. A host reboot on 11-09 09:20 UTC stopped E1 `[4]×5` at
  93/100 cells. It was resumed on 11-11 18:08 and finished at 18:44; the 4 attempts the reboot interrupted were
  rejected.
- **E0 queue (one orchestrator for all six schedules, started 11-11 18:48).** The kernel OOM-killed 10 workers while
  host RAM was short: 6 in `[1]×5`, 3 in `[1]×10` and 1 in `[4]×10`. The response was in three steps:
  - after the first kill, RAM admission was tightened (relaunch at 19:20);
  - from 11-12 05:17, a watchdog (`oom_guard.sh`) made the worker with the fewest finished episodes the kernel's first
    choice;
  - on 11-13 18:27 the queue was restarted once more to clear the shared T5 server's prompt cache, which grows without
    bound.

  The two relaunches interrupted 6 and 5 running attempts. One `[2]×10` attempt was stopped by the orchestrator's
  30-minute no-progress watchdog at 94/100 episodes.

The OOM watchdog only sets `oom_score_adj` on this queue's own processes; it does not change evaluation. The kernel log is
not readable for this account. So these records show only that every OOM kill recorded for the queue's unit corresponds
to one of the rejected attempts listed above. They cannot show what the kernel did to other processes on the host.

## Final audit

`robotwin_eval_audit.py` ([final_audit.json](e0_teacher10_5ep/flat_k4_n10/final_audit.json) for each run) re-derives
every accepted cell from its own files:
- `COMPLETE.json`, `manifest.json` and `result.json` agree on checkpoint, schedule, coverage and 100 episodes;
- the success counters, seeds and per-query action traces, re-derived from `eval.log`, equal the stored records;
- every video is readable by `ffprobe`, and no render or GPU-memory fault string appears in the accepted log;
- every episode ran the contract schedule with noise seed 42, and scene seeds rise strictly from 4,300,000.

The reference seed check (`reference_seed_check.json.gz`) compares the accepted seeds with the teacher campaign.

When the records were exported for this report (`tools/export_records.py`), each run was checked again:
- it has exactly 100 cells of 100 episodes;
- each accepted `result.json` is byte-for-byte equal to its `COMPLETE.json` entry;
- success counts equal the episode flags, and seeds are unique within each cell;
- all 90,000 videos exist and are non-empty;
- one video per cell (900 in total) still has its recorded SHA-256;
- all 789 runtime files are unchanged.

## Uncertainty and comparability

**One run per schedule.** There are no repeat seeds, so there are no error bars from repeated runs. As a rough scale,
the binomial standard error of an SR near 90% over 10,000 episodes is about 0.3 pp. A difference between two independent
runs therefore needs to exceed about 0.8 pp to stand out at roughly 95%. Episodes are not exchangeable across tasks, so
this is only a guide, not a test. By that guide:
- the NFE changes (|Δ| ≤ 0.06 pp) are well inside it;
- 1 → 2 loops (+1.20 to +1.31 pp) and 1 → 4 loops (+1.75 to +1.86 pp) are outside it;
- 2 → 4 loops (+0.55 pp at NFE 5 and 10) and the 4-loop gap to the teacher (0.6–0.7 pp) are inside it.

**Scenes are not perfectly paired.** A scene is accepted only if the scripted expert succeeds, and that check is not
reproducible for hard-planning tasks: CuRobo keeps its random state across candidates and stops a plan after a 10 s
timeout ([verification/curobo_nondeterminism.json](verification/curobo_nondeterminism.json)). So two runs share most,
but not all, scene seeds, and the "same scenes" comparison covers about 93% of episodes. Instructions are drawn from
unseeded Python `random`, so a shared seed can still come with a different instruction string. The teacher campaigns had
the same properties. A same-scene comparison is a better-matched description, not an exact pairing.

**Cross-host comparisons.** The teacher references and the NFE-2 and E0 epoch-1 context columns were run on other hosts
with CUDA graphs. The checkpoint, protocol and evaluator files are the same, but GPU, worker count and scheduling
differ.

## Files

| Path | Content |
|---|---|
| `e0_teacher10_5ep/<schedule>/`, `e1_epoch1/<schedule>/` | `episodes.csv` (10,000 accepted episodes: seed, success, instruction, video SHA-256, query count, action-trace digest), `task_results.csv`, `final_results.json` (totals, comparison, timing, rejected attempts), `final_audit.json`, `manifest.json` (protocol, code and runtime hashes), `launches.json`, `COMPLETE.json.gz` and `reference_seed_check.json.gz` (gzip of the originals; SHA-256 of the originals in `SOURCES.tsv`) |
| `results.json`, `task_comparison.csv` | All numbers in this README, built from the exported episodes and the referenced reports |
| `verification/` | Static export checks, GPU gate records (JSON only; tensors not copied), smoke audits, T5 loader check, renderer preflight, CuRobo nondeterminism, the environment runtime manifest (gzip) |
| `setup/` | `evaluation_code.tar.gz` (code as run), `code_provenance.json` (SHA-256 of every archived file, checkout state), `robotwin_local_setup.md` (setup, gates, hazards, queue operation) |
| `tools/` | `export_records.py` (exports and rechecks the records on sn4622128794), `build_report.py` and `README.template.md` (build this README, `results.json` and `task_comparison.csv`) |
| `SOURCES.tsv` | Every exported file with its source path on sn4622128794, SHA-256 and size of the original bytes, and how it was derived |

Not copied: rollout videos (about 2 GB per evaluation), per-cell worker logs and the checkpoints. They remain on
sn4622128794 under `/data/anh/robotwin_eval/results/{e1_flat_*,e0ep10_flat_*}` and `/data/anh/robotwin_eval/checkpoints`.

To check independently: `python evaluation_report/verify_robotwin_elastic_sn4622_20261114.py`. It recomputes every
total, the same-scene comparisons and the main tables from the episode files.
