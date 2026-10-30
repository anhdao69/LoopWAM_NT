# Verified campaign evidence — source 3a89bfa

Jobs 4770 and 4771, two H100s each. Metadata snapshot at production launch,
2026-10-09 approximately 01:19 UTC. Training metrics under concat_long/mix_long
are partial initial evidence, not final results.

- job4770 / job4771: all backend timings, native cold/warm smokes, simulator
  episodes/manifests, fairness, diagnostics, prepared/release manifests.
- operations: submitted batch scripts, actual launch scripts, reviewed audit,
  failed native compilation logs and diagnostic provenance, release helper.
- Native Inductor numerical gate **failed**; eager-only production audit passed.
  No failed native result is presented as a successful compiled check.
- Report: plans/performance/LoopWAM_v0_V4A1_KVconcat_20261008.md.

Weights, latent tensors, simulator videos and action-trace arrays stay on the
server. Benchmark OOM trials are retained as expected sizing failures. Operational
scripts are archived for review and are not intended to be rerun over existing
outputs. CPU/GPU suite logs are in the sibling pinned_3a89bfa directory.
