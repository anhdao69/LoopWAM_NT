# Elastic E0 parallel handoff to server137

The user's final instruction is to run on anhdh35@10.148.182.137 instead of server189. Server137 is now idle and has the same RTXPRO6000BlackwellMax-Q96GB and580.178.04driver as local. Server189 is not used for scored evaluation.

1. Preserve the live local [1,1]/NFE2 evaluator, workers, results and frozen source/runtime.
2. Copy the pinned E0 checkpoint and exact native evaluator into an isolated directory on137. Check26.71GiB of runtime/assets, all13releasefiles,908codefiles, dependency versions and path-only relocations.
3. Require native/graph exact parity and cross-host bit-exact action/T5 reference parity on137; run40native tests and four unscored smoke cells.
4. Assign [4,4]/NFE2 exclusively to137 with4workers. Local retains[1,1], then[4]x5/NFE5 with4workers. Same50tasks,clean/randomized,100episodes/task/setting,seed42,predict32/execute24,FP32weights/BF16compute andRT32/depth8.
5. Retire only the original scheduling PID, preserving its separately sessioned evaluator. New coordinator adopts/validates/finalizes[1,1], then runs[4]x5. Original inference code and original runtime-manifest bytes stay unchanged.
6. Review generic control logic, fix concurrent-report writes and stale-snapshot reporting, test PIDidentity/exclusiveownership/failedcompletion and live evaluator survival before handoff.
7. Bind remote code/runtime/control files before the scored launch. Finalize each full10,000episode result with native coverage validation and full video/action/runtime audit. Mirror137reports and record progress-snapshot age independently fromrsyncage.

Historical Python instruction-template selection remains unseeded as before. Identical inference on fixed inputs does not promise identical full simulator trajectories. No mixed-host shards or duplicate scored schedules.
