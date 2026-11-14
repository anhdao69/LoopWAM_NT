# Eight-worker memory incident and four-worker recovery

At approximately00:54–00:58 on2026-11-09 (Vietnam time), the status check found GPU exhaustion in the longer v0 4/1 epoch10 run. The16-episode eight-worker calibration had passed at peak72,518MiB, but the full-length mixed-task workload later exhausted the96GB GPU. Failure logs contain CUDA out-of-memory, renderer buffer-allocation failures and OIDN out-of-memory errors. The short trial underestimated sustained memory requirements. The underlying source of additional allocations has not been attributed to a specific component; the resource-overcommit failure is established by the logs.

Four worker attempts had already failed after35,26,43 and75 logged episodes. The supervisor was stopped gracefully; the controller, all simulator workers and shared T5 process exited, and GPU memory returned to52MiB. The stop was initiated by the assistant as corrective maintenance under the existing evaluation request, not a user cancellation; the generic signal-handler message `stopped_by_user` in the historical queue log reflects its implementation.

Both completed cells were re-audited with native completed_records and ffprobe, including video hashes, scene seeds, instructions and ordered action traces, and passed renderer-error checks:

| Cell | Successes / episodes |
| --- | ---: |
| move_stapler_pad / randomized | 73/100 |
| press_stapler / clean | 83/100 |
| Retained total | 156/200 (78.00%) |

All12 failed/interrupted attempts, comprising486 logged completed episodes, remain on disk and are excluded from scored results. Their task/setting cells restart from episode zero with the original seed sequence. These exclusions are based on incomplete/error evidence, not success rate. The remaining1,800 valid episodes still need to be completed.

The same checkpoint SHA, native model/evaluator, NFE10, seed42,32/24 prediction/execution horizon, normalization, RT32/depth8 rendering, task list and100episodes/cell are retained. No bound code or runtime asset was changed. Only dynamic calibration selection and worker-count launch metadata were corrected from8 to4, with the original values saved in resource_recovery.json. Native manifest equality still gates reuse. Four workers have a clean short profile and a prior complete2,000-episode4/1 run on this host. Fresh runtime-hash verification and CUDA-graph parity checks passed before resumption.

The replacement supervisor isPID1275661. The resumed evaluator identity is in launch.json. Runtime already spent in the first scoring segment is preserved in resource_recovery.json; the finalizer's controller elapsed time refers to the replacement segment and must be added to that earlier segment when reporting total wall time. The first segment began atUnix1794157629.4085567. All calibration and aborted-attempt time is separate from valid scored throughput.

[Recovery evidence](resource_recovery.json) · [Retained-cell audit](resource_recovery_validated_cells.json) · [Current progress](report.md)
