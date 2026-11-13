# Elastic E0 full RoboTwin evaluation

1. Preserve local results and fast-forward main; isolate latest elastic_robotwin source.
2. Download pinned native E0 epoch_01 release; verify all files, parent and normalization, loader tensors, VAE and inherited runtime.
3. Adapt original full50 simulator and policy input path to native TransportPolicy.rollout with 14D interfaces. Schedules flat_k1_n2, flat_k4_n2, flat_k4_n5, in that order. Video prefill remains four loops; step condition is 1/NFE.
4. Test schedule guards, shape/noise, source rollout arguments, renderer error rejection and resume integrity. Compare eager and CUDA graph outputs on eight real inputs for each schedule; require bit equality. Preserve FP32 weights/BF16 compute and original PIL image arithmetic.
5. Run four simulator smoke cells per schedule, excluded from score; confirm all 50 tasks and 100 clean/randomized cells per schedule. Freeze code/runtime once tests and independent review pass.
6. Launch persistent sequential queue on local GPU, four workers, seed42,100 accepted episodes/task/setting. Log videos, scene seeds, instructions, actions and latency. Reject renderer failures; retry incomplete cells at most three times.
7. Finalize each schedule with hashes, video/action trace checks, SR by task/setting, and auto-updated Markdown report. 10,000 scored episodes per schedule; 30,000 total.

No training, model precision changes, scene admission changes, checkpoint edits or changes to old evaluation trees. Instruction-template generation remains unseeded as in the historical runs; cross-run exact pairing is not claimed.
