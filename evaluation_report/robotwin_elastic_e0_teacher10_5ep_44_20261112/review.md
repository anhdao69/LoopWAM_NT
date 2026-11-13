# Independent review and fixes

Reviewer: e0_44_continuation_review. No critical issues. Two important resume issues were fixed before launch.

1. Repo symlink resolved the T5 server path into the original campaign, preventing owned-orphan cleanup after a crash. Fixed by copying the 4MiB source snapshot byte-for-byte and adjusting its deployment symlink to the new policy directory. All984sourcefiles verified against the original source manifest. The original ownership predicate returnedfalse before this change andtrue afterward; ownership checks were not weakened. Largeassets/checkpoints remain shared.
2. Volatile predecessor queuePID/audit timestamps could invalidate an otherwise identical frozen dependency receipt on restart. Fixed by retaining only stable audited identity/counts plus the exact manifest and COMPLETE hash. A regressiontest failed before implementation and passes after; it also rejects an altered COMPLETE artifact and preserves receipt bytes.

Final scoped suite:50passed,7training-only tests deselected. Full broader-run failures remain documented in setup_report.md.

The reviewer declined [4,4]cross-host equivalence because this is single-host, and deferred numerical/performance judgment to the mandatory runtime graphgate. This is appropriate: no such cross-host claim is made; graph/smoke gates block scored work until they pass.
