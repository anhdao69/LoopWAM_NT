# Analysis plan — matched LIBERO evaluation of 2026-11-01 (written before any scored episode)

Requested by the user on 2026-11-01: (1) all-suites v0 4/2; (2) E0 at K_a = 3, NFE 10; (3) E0 at K_a = 5 and 6, NFE 10;
(4) base v0 4/4 at NFE 2 and NFE 5, no distillation; (5) base v0 at [1,1], no distillation. Seven conditions, each the
full matched protocol `lightwam_20261009_matched`: 4 suites x 10 tasks x initial states 0-49 x seeds 42/43/44 =
6,000 episodes.

| Condition | Weights (sha256) | Evaluator | Schedule (action loops per denoising step) |
|---|---|---|---|
| v0 4/2 | HF `libero-all-suites/v0-video4-action2/policy.pt` (`33476d58…`) | local matched harness (`matched_eval_v042`, family `loopwam_v0_v4a2`, LoopWAM-chrono code, eager native `infer_action`) | trained 4/2, NFE 10 |
| E0 [3]x10 | HF `Elastic/E0/epoch_10.pt` (`5569c47f…`) | elastic matched runner (`LoopWAM-elastic` @ `eval/matched-pro6000-20261101`, CUDA graph, bit-exact gated) | flat_k3_n10 (test-time, untrained K_a) |
| E0 [5]x10 | same | same | flat_k5_n10 (test-time, deeper than trained) |
| E0 [6]x10 | same | same | flat_k6_n10 (test-time, deeper than trained) |
| Z10 [4,4] | HF `v0-video4-action4/policy.pt` (`1c659a7c…`), unmodified | elastic runner, `TransportInference(zero_shot_kind=T0)` (no step condition, no transport) | parent2 |
| Z10 [4]x5 | same | same | parent5 |
| Z10 [1,1] | same | same | flat_k1_n2 |

Reporting rules (as in the elastic campaign's predeclared analysis, `scripts/elastic_eval_report.py`):
- A condition is reported only when all 6,000 episodes exist and every group record matches its checkpoint sha256 and
  schedule; per-suite and per-seed rates are reported for every condition.
- All contrasts below are secondary and unadjusted (no primary hypothesis is declared for this batch). Estimator: difference
  of success rates (= mean of per-cluster differences, cluster = (suite, task, initial state) with its three seeds);
  95% percentile interval from 10,000 cluster-bootstrap resamples; two-sided paired cluster sign-flip permutation test
  (10,000 flips); each interval/test seeded by (20261101, CRC32 of its name). Paired contrasts need identical
  episode keys. A nonsignificant difference is not equivalence.
- Same-host contrasts (this RTX PRO 6000 host, same regenerated prompt cache):
  * E0 [k]x10 - base v0 at K_a = k, NFE 10, for k = 3, 5, 6 (local runs loopwam_v0_ka3/ka5/ka6; the base-v0 K_a
    definition is bit-identical to the one used here, gate G3);
  * Z10 [4,4] - Z10 [4]x5 - Z10 [1,1] relative to each other (descriptive).
- Cross-host contrasts (labelled as such: H100 vs RTX PRO 6000, and the H100 runs read the training prompt-cache bytes):
  * Z10 at s - E0 at s (H100 elastic evaluation) for s in {[4,4], [4]x5, [1,1]}: the plan's "benefit of adaptation";
  * Z10 at s - base v0 4/4 at NFE 10 (campaign "LoopWAM full 4/4", 94.92%);
  * E0 [k]x10 - E0 [4]x10 (H100);
  * v0 4/2 - campaign v0 4/4, 4/1, 2/2 and 1/4 ("LoopWAM full …" records).
- Timing (episodes/hour, per-query seconds) is reported only as throughput of this shared host, never as model latency
  (other users' jobs share both GPUs; CUDA MPS concurrency).
