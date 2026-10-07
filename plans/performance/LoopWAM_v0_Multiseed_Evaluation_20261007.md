# LoopWAM v0 evaluation across seeds — October 7, 2026

The active queue was restarted at the user's request on **worker-1**, interactive
allocation **4728**, step **4728.2**, with two H100s. Its fresh output root is
`/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/v0_multiseed_job4728_worker1_20261007`.
All three seeds restart from episode zero with the same completed v0 checkpoint.
The earlier worker-0 step **4719.8** was cancelled (signal 15) when allocation
4719 ended; it saved two partial-run videos. Those outputs are retained and
excluded from the new queue's results. The original launch record follows.
The worker-1 queue began at approximately **00:47 UTC October 7**, with an
estimated finish around **02:47–03:47 UTC** (10:47–11:47 PM New York time October 6).

The user requested evaluation of the existing trained v0 checkpoint across
several seeds. The queue runs evaluation seeds **42, 43 and 44**, sequentially,
using both H100s in interactive allocation **4719**, step **4719.8**.

The policy is the original LIBERO-Long v0 trained with seed 42, the 344/44
demonstration split, global batch 128, ten epochs and 7,250 updates. Its final
checkpoint SHA-256 matches the previously audited checkpoint:
`d75adef4068d74827921ea35a6bd8d36b5e6ad8b0b84de8dc7ad9e9766486125`.

Each seed evaluates all ten tasks with ten episodes per task, producing
**300 episodes and videos** across the queue. The initial-state indices 0–9
are identical across seeds. Simulator and diffusion-sampling randomness use
the evaluation seed; episode seed is `seed + task_id*100000 + episode_index*1000`,
and action replans add their index. Inference uses four loops, ten diffusion
steps, a 700-policy-step limit, 30 settling steps and ten actions per replan.

The runner validates checkpoint/source hashes, each requested seed, two-GPU
execution, unique task/episode pairs, initial states and every saved video before
advancing. Errors stop the queue and are recorded in `pipeline_status.json`.
After all runs complete, `results.json` records each success rate and their mean
and sample standard deviation. These quantify evaluation randomness for one
fixed trained policy.

Evaluation source is pinned at `b12085164218211c6b404a7ce2dd92a9dd7d36bf`.
The runner and launcher are archived in
[`../evidence/v0_multiseed_20261007`](../evidence/v0_multiseed_20261007).

Server output:
`/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/v0_multiseed_job4719_20261007`.
Per-seed directories are `seed_42`, `seed_43`, and `seed_44`. Videos stay on the
server. The interactive allocation remains available after the queue exits.

The previous v0 100-episode evaluation took 44 minutes on two H100s. Estimated
queue duration is **2–3 hours**, depending on rollout lengths and storage load.
The queue started at approximately **00:45 UTC October 7** (8:45 PM New York
time October 6); estimated finish is **02:45–03:45 UTC October 7** (10:45–11:45 PM
New York time October 6). Final results are pending.
