# v0 evaluation — stop after Ka=2

User changed the queue on 27 October 2026: keep Ka=2 running; remove Ka=3,5,6. Ka=1 is already complete. No further evaluation will launch.

Kv=4; NFE10; seeds42/43/44; four suites × ten tasks × 50 episodes/seed = 6,000 per retained condition (12,000 total).

| Ka | Status | Episodes | Successes | SR |
| --- | --- | --- | --- | --- |
| 1 | complete | 6000/6000 | 738 | 12.30% |
| 2 | complete | 6000/6000 | 3334 | 55.57% |

Ka=3,5,6: removed at user request, without scored episodes. Dense-S12/S30 were also removed earlier.
The retained Ka1/Ka2 results are complete. Original evaluator source, active runner, checkpoint and frozen protocol remain unchanged.
[Queue change record](stop_after_ka2.json) · [Live state](queue_status.json) · [Original frozen protocol](protocol.json)

Updated index note (November7): separate later matched-baseline Ka3/5/6 results are listed in [the consolidated report](../all_results.md); removal from this original queue is not a claim that no other campaign evaluated them.
