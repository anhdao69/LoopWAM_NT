# LoopWAM v0 throughput optimization

User authorizes benchmarking larger microbatches/lower accumulation, ZeRO-1 and ZeRO-2, and deploying the fastest correct configuration. Preserve global128, architecture, dataset, normalization, optimizer hyperparameters, fresh initialization and full10-epoch budget. Only use2H100s in job4689; do not touch SimpleMemVLN.

Baseline: FP32 policy/AdamW states + BF16autocast, DDP, microbatch2, accumulation32. ~9.75s/update, ~27GB reserved per GPU. Prior sampled GPU busy fraction93-95%; low memory occupancy is not the same as low GPU compute utilization.

- Measure live GPU duty cycle and stage timings.
- Secure latest durable checkpoint and stop only training step4689.49 for uncontended benchmarks.
- Benchmark DDP mb2/4/8, higher fit limits, fused optimizer, ZeRO1/2 with matched precision and128 effective batch. Warmup then at least3 timed updates, record memory and real-data throughput. OOM outcomes are recorded.
- Profile data/VAE/forward/backward/optimizer; evaluate further changes only when supported by measurements.
- Verify chosen changes and start from canonical Wan donor initialization with fresh optimizer/scheduler/data progress; never resume the previous checkpoint.
- Compare fresh-run measured timing, persist benchmark evidence and report total/remaining projection.
