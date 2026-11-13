# Elastic E0 — server137 parallel evaluation

The user's final destination is **anhdh35@10.148.182.137**, replacing the earlier server189 proposal. Server137 was idle at setup and has the same RTXPRO6000BlackwellMax-Q96GB/580.178.04driver as local. Server189 is not used for scored evaluation.

## Schedule ownership

| Schedule | NFE | Host | Workers | Episodes |
| --- | ---: | --- | ---: | ---: |
|[1,1]|2|Local, current evaluator preserved|4|10,000|
|[4,4]|2|10.148.182.137|4|10,000|
|[4]×5|5|Local, after[1,1]|4|10,000|

Every schedule covers50tasks,clean/randomized,100acceptedepisodes/task/setting,seed42. E0epoch01/update47462,checkpointSHA256`abc3c0075a6e1f3ce380be8d6c7b6b9d8bb697ac21cf3f2d40a93efb821b539a`. Four video loops remain fixed. Original normalization,FP32policyweights/BF16compute,PILuint8camera packing,CPUactionnoise42everyquery,predict32/execute24,scene admission,task horizons,unseen instruction pool,RT32/depth8/OIDNcompatibility andlogging are preserved.

## Verification

- 908 native/evaluator/code files match the frozen local checkout.
- 21,127runtime/assets files checked (26.71GiB); six YAMLassetroots undergo documented path-only relocation. All13releasefilesverified.
- Python3.10.22,PyTorch2.7.1+cu128,CUDA12.8 andcheckeddependencyversionsmatchlocal.
- **Eight action outputs and four T5 contexts match local references bit-for-bit** onserver137,zero maximum difference.
- Native/CUDAgraph parity also passes on137 for all3schedules and8realinputs with42/43/44 noise seeds; scored seed remains42 only.
- 40native/adaptertests pass. Four additional unscored smoke episodes validate the deployed simulator andpolicy path before scored launch.
- Seven scheduling/reporting tests pass. Independent review fixes cover atomic concurrent reports, stale-snapshot disclosure, and duplicate-lock status protection. A real subprocess test proves that retiring the scheduling PID leaves its evaluator alive.

## Control and reporting

The old supervisor is retired only after readiness gates pass. Its separately sessioned local evaluator/workers continue without interruption. An external coordinator adopts,validates andfinalizes[1,1],then starts[4]×5. The original frozen local scripts andruntime-manifestbytes remain unchanged. Immutable host assignments prevent a duplicate local[4,4].

Server137runs a detached locked supervisor. Its model/runtime/source/controlfiles are frozen before scored launch. All100cells and10,000episodevideos/actiontraces are audited percompleted schedule. Partial andfailedattempts are excluded; renderer/resourceerrors invalidate attempts; threeattempts/cell and1,800-second no-progress timeout apply. Remote reports mirror locally every60seconds; progress-snapshotage is shown independently of successfultransferage.

Remote bulk directory: `/media/vmo-perception/disk_2/anhdh35/loopwam_robotwin_elastic_e0_20261108`. Local coordinator: `/data14t/anhdh35/loopwam_robotwin_elastic_e0_20261107/parallel137_20261108/start_local.sh`.

Historical Python instruction-template randomness remains unseeded as before. Matching fixed-input inference andprotocol does not promise identical full simulator trajectories.

[Live combined report](../report.md) · [Cross-host parity](cross_host_gate.json) · [Native/graph parity](graph_gate.json) · [Source/runtime check](source_check.json) · [Review](review.md) · [Host assignments](assignments.json) · [Remote live evidence](remote_mirror/report.md).

## Validated handoff and smoke results

All four smoke cells completed with valid videos, episode records and action traces, and clean renderer logs. Two succeeded and two failed the task; all four are excluded from the scored benchmark. The handoff retired only the old scheduling supervisor (PID 2055346); the original local evaluator (PID 2058639, start ticks 52433888) remained alive. New local coordinator PID: 3311080. Remote queue PID: 429621. Report sync PID: 3311365. These PIDs are launch evidence, not a promise that a process is currently alive; consult the live report for current status.

[Smoke validation](smoke_validation.json) · [Handoff receipt](handoff.json) · [Remote dispatch authorization](dispatch_authorization.json).

## Scored evaluation launch confirmed

The remote evaluator (PID 429666) is running the full 50-task manifest with seed 42, action depths `[4,4]`, NFE 2 and 100 episodes per task and setting. All four workers have produced policy queries and completed scored episodes. At launch verification, 8 episodes had completed (7 successes); this small initial sample is not a benchmark result. No failed attempts or renderer resource errors were present. [Deployment evidence](deployment_verification.json). The local `[1,1]` evaluator continues under its original PID.
