# Preparation review and validation ledger

The requested models use an existing frozen evaluator. Operational changes are isolated in new run directories; no benchmark/model source edits or branch merge is required.

Independent review found two restart defects in reused queue patterns: regenerating runtime-manifest creation timestamps changed the bound hash, and rerunning an already-completed evaluator could replace COMPLETE.json without fields required by finalization. Both were fixed before launch, with regression tests observed failing against prior behavior and passing after the fixes. Bound runtime manifests are preserved byte-for-byte on restart; completed evidence is validated and finalized without invoking the evaluator again. No review findings were deferred.

The reviewer verified variants, expected step45445, ten tasks, seed42/NFE10, policy symlinks, denominators, strict loading, graph gating, smoke exclusion, remote path relocation and renderer guard. Remote mirroring was implemented afterward and verified by a successful rsync with sync_status.json.

Execution assignments: local96GB/four workers for4/4; free remote18948GB/two workers for4/2. Busy remote137 was left running its existing workload. Source checks validate nine training hashes. Different raw stats encoding is allowed only after equality of parsed values; new release donor-artifact/data-manifest hash differences are explicitly reported.

See restart_tests_red.log and restart_tests_green.log for the regression evidence. Per-host evaluator tests, strict-load, runtime verification, graph and smoke evidence are in each run report. Launch and scored-progress verification follows preflight.

Local preflight completed: 38 evaluator tests, 2 renderer-guard tests, strict load, exact graph gate, four valid smoke episodes (3 successes, excluded). Full scored queue launched with4 workers; actual manifest matched baseline protocol in17 fields with normalization JSON equality. Remote runtime and38 tests passed; strict-load and graph equality passed, smoke pending at this ledger update.

Remote preflight completed: 38 evaluator tests, 2 renderer-guard tests, strict load, exact graph gate and four valid smoke episodes (4 successes, excluded). Both full benchmark evaluator PIDs are alive; launch manifests match the same-variant baseline protocol. Remote reports sync successfully. No evaluation-source changes or outstanding review findings.
