# Independent review and fixes

Reviewer: fresh-context /root/elastic_e0_review (read-only). No critical findings. Native14Dinput/noise, PIL/BF16preprocessing, FP32parameters, schedule/NFEpropagation and native stepconditioning matched the release. Simulator/textsource and graphparity evidence confirmed.

Findings and disposition:
1. Important: supervisor45-second shutdown could interrupt sequential child cleanup. Fixed: parallel child termination,125-second supervisor grace, persistent child PID/starttime records and guarded orphan cleanup on resume. Real SIGTERM-ignoring subprocess test fails before fix and passes afterward.
2. Stale summary after recovered completion. Fixed: evaluator republishes summary; audited overview reads COMPLETE. Regression reproduces stale partialSR and verifies corrected completion.
3. Interrupted attempts missing from failure totals. Fixed: inventory every nonaccepted attempt directory and count its logged episodes, including directories without failure.json. Regression covers interruptedattemptwithoutfailurefile.
4. CLIdefault checkpointstep237310 was teacher default. Corrected to47462 for E0. Queue already passed47462 explicitly.

No deferred findings. Changes preserve native model mathematics. Five regression tests: RED→GREEN; combined targeted suite52passed.

Review scope decisions: preserve explicitly established unseeded Python instruction generation, simulator admission, precision, renderer, policy mathematics and action-hash traceformat. Smoke throughput and finalSR remain empirical runtime observations. No training/selection-quality claims are made.
