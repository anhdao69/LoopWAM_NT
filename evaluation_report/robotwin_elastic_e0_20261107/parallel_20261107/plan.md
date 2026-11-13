# Elastic E0 parallel server handoff

User requests idle-server execution while preserving evaluation correctness. Existing authorization covers transferring this checkpoint/code and launching evaluation on previously used189/137 hosts.189 is idle;137 has an unrelated active GPUjob and is not selected.

- Preserve running [1,1], immutable source and current runtime manifest.
- Transfer the exact pinned E0 release and native evaluator to a new directory on189, reuse previously validated assets/environment only after content/version checks. Record unavoidable GPU/driver/path differences.
- Compare eight fixed real-input outputs against local native inference, run native/graph parity and four simulator smoke cells on189. Same50tasks,bothsettings,100episodes,seed42,NFE2,[4,4],renderer,normalization,noise,observation/actionhorizons.
- Only after all gates pass, replace the old scheduling supervisor without stopping its evaluator/workers. A new external coordinator adopts and finalizes [1,1], reserves [4,4] exclusively for189, then launches [4]x5 locally. Old frozen code remains untouched.
- Maintain periodic remote reports and a combined local dashboard; completion requires independent final audits. No duplicate or mixed-host result cells within a schedule.
- Test handoff PIDidentity, live-child adoption, failed-completion refusal, exact schedule ownership and restarted supervision. One fresh-context final review before handoff.

Tradeoff: remote189 has RTXPRO5000/48GB and2workers; local RTXPRO6000/96GB uses4workers. Hardware and historical unseeded Python instruction templates prevent promising identical rollouts. All scientific protocol choices and data bytes remain matched; numerical gate evidence will be recorded explicitly.
