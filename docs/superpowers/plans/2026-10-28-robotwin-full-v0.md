# Full RoboTwin original LoopWAM v0 implementation plan

User-approved final recipe: original LoopWAM/LoopMoT v0,3/6/3 loops4/4,no ChronoLoop model,memory0/history0. Fresh canonical Wan2.1 donors and seeded14D interfaces;27500demos/50tasks/6075103anchors. Five epochs, each anchor once, per-epoch randperm(seed42+epoch). Exactly128 real windows/full update across8GPUs,95 in each epoch tail.47462updates/epoch,237310total,30375515windows. Accumulation as needed. AdamW lr1e-4 allparams,betas(.9,.95),eps1e-8,decay.01,clip1,warmup5%(11865updates),cosinefloor1%(final1e-6).FP32weights/moments BF16 compute. Retain exact epoch3/4/5 weights, plus recovery checkpoints.

Architecture: dedicated original-v0 robot factory and trainer, with original forward/loss unchanged; dataset/cache helpers reused from completed full50 preparation. Validated supplied-latents override prevents encoding dummy video. Fresh CPU-seeded14D action encoder/head/proprio;673 other tensors exactlymatch canonicaldonors. Strict robotcheckpoint format and loader, original LoopMoT, no trained LIBERO parent. Loss sums actual local windows then scales8/global_real_count before DDP average. Deterministic per-window/epoch noise preserves resume/layout comparability.

- [x] Original model adapter, cache bypass,14Dseed,loss normalization,depth/scale guards unit tests.
- [x] Exact sampler tests: once-per-epoch,seed42+epoch,tail95,resume cursor;5epoch budget237310.
- [x] Cache/readiness/init audit and both priorLIBERO queues verified/uploaded, canceled5248/4925; new8GPU allocation5489 running.
- [ ] CPU/GPU suite, native checkpoint reload and optimizer resume against continuous reference, real95-tail DDP/compile smoke.
- [ ] Fixed128 benchmark micro4/8 withoutcheckpointing,16withcheckpointing; compiled fastest eagerlayout. Select fastest measured verified layout.
- [ ] Pin source/config/hash gates, fresh production, verify live updates; push implementation/evidence/report and estimate total runtime.

Global constraints: no home storage; no mutation of shared cache/data/normalization; no trained parent; no production before all gates pass. Cache1.12TB stays on disk, larger than960GB allocationRAM. Original outputs preserved. Tests on this branch exclude17 optional ChronoLoop-specific GPU-parent tests because no ChronoLoop model is used.
