# Parallel supervisor review

Fresh-context read-only reviewer: /root/elastic_parallel_review. Generic scheduling logic reviewed before deployment; native source and historical evaluator remained unchanged. User subsequently selected server137 instead of189; all host paths, ownership and commands were migrated and checked before launch.

Important findings:
- Concurrent report publication used a shared temporary filename and could interrupt supervision. Fixed with serialized report updates and unique temporary files for atomic JSON publication. Regression reproduced failure before fix and passed after.
- Successful rsync could conceal old remote progress. Fixed: report snapshot age independently of transfer age, account for clock offset, and mark active snapshots older than180seconds stale. Regression RED→GREEN.

Minor finding also fixed: a duplicate invocation failing its queue lock no longer publishes error status over the live supervisor. A real duplicate invocation against the held local lock raised BlockingIOError and left status bytes unchanged.

Five handoff tests plus two report regressions passed. Handoff tests use real processes to prove retiring the scheduling PID preserves the evaluator and rejects stale PID identity. Completion guard refuses failed or absent COMPLETE evidence. Import checks verify the coordinator uses frozen local helpers, not the similarly named remote templates.

No deferred findings. Remote hardware/source/inference/simulator gates are additionally required before launch. Server189 remains unused for scored evaluation.
