# RoboCasa-GR1 epochs 9 and 10: independent reevaluation

Started 2026-11-08. **Status: both official evaluations running; no new complete success result yet.** Local epoch 9: supervisor 1080416, driver 1080852. Server189 epoch 10: supervisor 1697417, driver 1698068. Both full 720-step smoke tests passed before scoring. Live status, task cells and logs are synchronized into this report every 60 seconds; videos remain on their source disks. See `local_live_status.json`, `server189_live_status.json` and `sync_completed_utc.txt`.

Release: [anhdao69/LoopWAM_NT, Robocasa-GR1](https://huggingface.co/anhdao69/LoopWAM_NT/tree/9fd190b3698ab97a209a0229dc46d0c89a3fd978/Robocasa-GR1), pinned revision `9fd190b3698ab97a209a0229dc46d0c89a3fd978`.

| Checkpoint | Update | SHA-256 | Existing release H100 reference |
|---|---:|---|---:|
| epoch_09.pt | 211644 | `694a679b4c07934b57995dc3a2ab62e2f65357cea29108b109b30049531e2748` | 626/1200 = 52.17% |
| epoch_10.pt | 235160 | `cb7b9c1969037a7ab6f8a62d9f639dc4e4528ad7d65f2776da2fe94286d41679` | 620/1200 = 51.67% |

Both files are 2,339,059,493 bytes. Downloaded over verified HTTPS via server137, then transferred using SSH. Each hash and size agrees with both the pinned Hugging Face LFS metadata and release upload receipt. Server137 is only a download relay; these RoboCasa runs never use its GPU.

Protocol matches the release and previous official evaluator: all 24 GR1 tasks, 50 episodes per task, one environment per client, 720 environment steps per episode, action prefix 12 from a predicted chunk of 32, raw state sent, ego RGB at 256×256, any-step success, 10 denoising steps, CFG 1.0. **Environment seed 0; policy noise seed 42 for every query.** Simulator episodes consume the same environment-owned RNG across resets. EGL rendering and all 1,200 episode videos are retained. Runs use the unchanged `d937167ba61a0df4a9818d1917de6c7155400dff` evaluator, the exact VAE and normalization, and all 187 verified training prompt-cache files. T5 fallback is disabled so a cache miss fails visibly.

Local runtime: `/data14t/anhdh35/robocasa_gr1_epochs09_10_20261108`. Epoch 9 is assigned to the local RTX PRO 6000 96 GB (six policy servers, 24 simultaneous task clients). Epoch 10 is assigned to server189 (`vm-perception-ws@10.148.183.189`), isolated root `/media/newdisk/robocasa_gr1_epochs09_10_20261108`, RTX PRO 5000 48 GB (four policy servers, eight simultaneous task clients; bounded for its 62 GiB system RAM). Each host has a private CUDA MPS daemon. Different concurrency changes scheduling only; task seeds, checkpoints, simulator, model, and evaluation protocol remain fixed. GPU architecture can still cause small floating-point differences from the H100 reference.

Each checkpoint must pass input checks plus a complete 720-step, one-episode smoke test before its official run begins. The smoke is separate from scored results. Each official output records an immutable `binding.json`; resume checks checkpoint identity and preserves complete task cells. Interrupted task cells restart at episode zero, with partial videos moved aside. The supervisor records `STATUS.json` every ten seconds, retries up to three times, and validates all task metadata and episode videos before writing `COMPLETE.json`. Original results and working trees are preserved.

Commands on the relevant host (set `R` to its runtime root):

```bash
python3 "$R/scripts/status.py"
python3 "$R/scripts/stop.py" 9 # use 10 on server189
# Resume epoch 9 after confirming it is stopped:
rm -f "$R/results/official_/epoch_09/STOP" "$R/results/smoke_/epoch_09/STOP"
setsid nohup bash "$R/scripts/launch.sh" 9 24 6 7600 >> "$R/logs/epoch09_supervisor.log" 2>&1 < /dev/null &
# Server189 resume: epoch 10, 8 clients, 4 servers, base port7700.
```

Official output: `results/official_/epoch_09` or `results/official_/epoch_10`. Evidence: `evidence/inputs_epochNN.json`, pinned `hf_tree.json`, runtime freezes, asset checksums, rendering gate, and `results/smoke_/epoch_NN/VALIDATED.json`. Official completion requires `validation.json` with `valid: true`, 24 complete task cells, and 1,200 finished videos.

Observed early throughput (local 13:52 UTC): epoch 9 had 99 finished videos after 7.15 minutes (~13.85 episodes/minute), a simple projection of ~79 minutes remaining. This is provisional until complete task cells exist and later task timings stabilize. Epoch10 started later; no reliable throughput estimate yet. Server189's clock is 219.62 seconds behind local; `evidence/clock_offset.json` records the measurement, and sync timestamps use local UTC.

Controller review fixes: verification precedes reading the bound receipt; checkpoint bytes additionally match pinned LFS metadata; staged source changes are rejected; all nonterminal phases can be stopped; owned child process groups receive bounded TERM/KILL cleanup; previous attempt logs are retained under `attempt_logs/`. Independent review is clear (`review.json`). Four CPU lifecycle tests pass on both hosts; archived-log recovery and missing-log rejection regression checks also pass. Final validation reads both current and archived prompt logs, requires matching server metadata for each accepted task cell, and reports historical failures separately. Local epoch 9 had already started when these fixes were reviewed, so its in-memory supervisor is protected by `scripts/guard_legacy_epoch09.py`, which preserves this healthy attempt, intercepts retries, cleans its owned process group after unexpected exit, and hands off once to the reviewed runner. Explicit stop is respected. The original first-attempt checkpoint binding is verified and unchanged.

The server's legacy metadata lists a default action prefix of 16; the unchanged native client explicitly sends 12 on every request and checks the returned action shapes. Scored task JSON records 12. The model always predicts 32 actions; the validator audits the executed prefix 12 and query count. No policy code or native scoring code was changed.

Live [SUMMARY.md](SUMMARY.md) is generated automatically from native task JSON. The local legacy guard is PID 1088562; its safety handoff avoids restarting the healthy current attempt. Both smoke tests used 60 queries, taking 36.1s locally and 36.6s on server189. Existing H100 release percentages in the table are reference results, not this rerun.
