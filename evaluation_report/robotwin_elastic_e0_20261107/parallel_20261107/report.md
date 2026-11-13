**Superseded by the user’s final server137 instruction. Server189 was not launched for scored evaluation. See [server137 setup](../parallel137_20261108/report.md).**

# Elastic E0 — additional-server readiness

The requested idle-server check found189 available (RTXPRO5000Blackwell48GB), while137 (RTXPRO6000Blackwell96GB) was serving an unrelated SimpleMemVLN job. No existing job was stopped.

The pinned E0 checkpoint and exact frozen evaluation code were transferred to `/media/newdisk/loopwam_robotwin_elastic_e0_20261107` on189. All13 release files,908 evaluator/code files and21,127 portable runtime/asset files were checked. Six asset YAML files differ only in their documented absolute root paths. Python3.10.22,PyTorch2.7.1+cu128,CUDA12.8,and the checked dependency versions match.

Local GPU/driver: RTXPRO6000BlackwellMax-Q96GB/580.178.04. Remote GPU/driver: RTXPRO5000Blackwell48GB/570.211.01. These hardware/driver differences are explicitly recorded; identical protocol does not imply identical trajectories.

## Validation

-40 native Elastic/adapter tests passed on189.
-Native/CUDAgraph outputs matched exactly on189 for all3schedules and8real observations, including reverse-order replay.
-Local repeated native inference matched its saved reference bit-for-bit.
-Cross-host inference did **not** match bit-for-bit: eight action outputs have maximum absolute differences0.0078125–0.010498046875 and mean absolute differences approximately0.00123–0.00174 in normalized action units.
-The tested VAE image features match exactly, including when fed directly to the remote action policy. Action differences persist after that boundary.
-Four T5 text contexts differ as well; masks match. Maximum element differences range0.04296875–1.6474609375, with means0.000338–0.001684 across all elements. These context values are not robot commands.
-Profiling shows different BF16 matrix-multiplication kernel choices. This supports a hardware/runtime numerical explanation; it does not establish that every observed difference comes from one particular library or driver component.
-A separate diagnostic deterministic-cuDNN setting did not restore parity and was **not adopted**. Production inference settings and the local queue remain unchanged.
-Four unscored simulator smoke cells completed and passed independent video/episode/renderer-log validation; two policy successes and two policy failures. Smoke episodes are excluded from campaign SR. See [smoke validation](remote_snapshot/smoke_validation.json).

## Execution decision

No scored evaluation has been started on189 because the requested strict identity check failed. The existing local supervisor/evaluator and queued schedules remain untouched. A choice was requested between requiring exact cross-host outputs and accepting the same scientific protocol with explicitly documented GPU numerical differences.

If the latter is accepted, the prepared next assignment is `[4,4]`,NFE2,50tasks,clean/randomized,100episodes/task/setting,seed42,10,000episodes,on189 with2workers. The local `[1,1]` run would stay intact, then run `[4]×5` with4workers. Handoff and exclusive schedule ownership must be completed before any remote scored launch to avoid duplicates.

Evidence: [readiness](readiness.json),[cross-host gate](remote_snapshot/cross_host_gate.json),[full diagnostics](remote_snapshot/diagnosis.json),[native/graph gate](remote_snapshot/graph_gate.json),[source check](remote_snapshot/source_check.json),[remote tests](remote_snapshot/tests.log),[handoff plan](plan.md).
