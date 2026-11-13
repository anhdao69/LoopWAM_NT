# [4,4] continuation setup and verification

The same E0 teacher-epoch10 / student-five-epoch release is evaluated with action depths[4,4],NFE2. Full RoboTwin50, clean+randomized,100episodes per task/setting,seed42,10000total. Video loops4; predict32/execute24.

The model source, environment, assets, dataset statistics, checkpoint, camera preprocessing, renderer, CPU action noise, instruction generation and scoring code are inherited unchanged from the [1,1]/[2,2] campaign. The new campaign uses a byte-identical 4MiB copy of the immutable source snapshot (large model/assets remain shared) and uses hardlinks for the verified release files, avoiding another weight download. CUDAgraphs,sharedT5 and fourworkers on server137 RTXPRO6000 remain enabled.

The queue first verifies that both predecessors are complete and their final audits cover100settings and10000videos each with the correct checkpoint. Each COMPLETE.json is hashed against its audit. Missing/incomplete evidence or SSH interruptions keep it waiting; inconsistent completed evidence raises an error. GPUwork follows those checks and requires60000MiB free. It performs a new [4,4] native/graph numerical comparison on eight saved simulatorinputs, then eight excluded smokeepisodes, then the10000episode evaluation and full artifact audit.

The copied cross_host_gate.json is historical evidence for[1,1]/[2,2], not a claim of a [4,4] cross-host check. Only server137 runs[4,4]. Python instruction-template RNG remains unseeded as in previous evaluations; this does not guarantee exactly paired instructions between models.

The isolated queue/contract changes leave the original running campaign and its frozen files unchanged. Separate runtime manifest,dependency receipt,PID identity,queue lock,logs,live report,per-task records,videos and final audit are retained. Transient network errors wait automatically. Detached execution survives this chat disconnect; automatic reboot restart is not configured.

## Verification record

New[4,4]contract/assignment/dependency tests:18failed before implementation, then all passed. All36campaign tests passed. Broader inference/pipeline check:50passed,6failed due to previously known absent training-only helpers in this evaluation snapshot. The failures are:

- test_continuation_budget_and_cosine_endpoints
- test_handoff_rejects_wrong_boundary_or_contract
- test_stage_survives_controller_detach_and_prevents_duplicate_child
- test_stage_records_child_failure
- test_stage_resolves_python_after_environment_and_stops_on_source_failure
- test_nested_slurm_launch_does_not_inherit_controller_resource_limits

These training-controller tests were also excluded in the preceding campaign; no training code is needed for this evaluation. The scoped inference/queue suite excludes those and the training-stage recovery test. Full outputs are retained in tests_final.log and tests_scoped.log.

The graph/smoke gates run automatically after predecessors finish; their results must be read from graph_gate.json and smoke_flat_k4_n2.json rather than assumed passed at queue creation.

Final scoped result: **50passed,7deselected**. Independent review and both fixed findings are recorded in [review.md](review.md).

## Migration to server137, November12

The user requested server137 after the localGPU became occupied by SimpleMemVLN. Local waiting queuePID3995961 was stopped with PIDidentity verification; no active evaluator or scored episodes existed. No unrelated GPU processes were stopped. Its logs/receipt are preserved under local_retired/.

The same984sourcefiles and exact checkpointSHA256 were checked on137. Its already-audited environment/assets are inherited; all evaluation settings remain unchanged. Both previous audits are already complete. Their immutable receipt is SHA256-pinned, and server137 rechecks its own current completion/manifest directly. The local completion/manifest was freshly reverified before transfer. This avoids requiring reverseSSH for completed dependencies.

Remote scoped tests:55passed,7training-only deselected. The new remote host/dependency tests first failed6cases before implementation, then passed. The [4,4] numerical gate passed on137 with zero output difference; its native calls were depth4 at timesteps1000 and500, each with stepcondition0.5. The smoke gate is required before scoring. Reports synchronize from137 every30seconds; sync_status.json records sync health.

Launch verified:server137queuePID3093141, evaluatorPID3098414. Eight smokeepisodes passed operational/artifact validation and are excluded. Full50tasks,2settings,100episodes each,seed42,[4,4],NFE2,predict32execute24 confirmed in the scored manifest. See launch_validation.json and current_results.json for evidence.
