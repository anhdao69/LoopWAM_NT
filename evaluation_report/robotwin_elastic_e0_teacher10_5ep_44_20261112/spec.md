# E0 teacher10 five-epoch [4,4] continuation

User request: after the current [1,1] and [2,2] evaluations finish, evaluate [4,4].

Use the identical pinned E0-epoch10-teacher-5ep epoch_05.pt (SHA256 07d24ff77410e516ac90d24d76d651e2102b09b0458d9d5d86ea18dff39e823e). This is student epoch5/update237310 from teacher epoch10/update474620.

All50 tasks, clean and randomized,100accepted episodes/task/setting, seed42, 10000scored episodes. Actiondepths[4,4], NFE2, fourvideo loops, stepcondition0.5, predict32/execute24. Same frozen code/environment, FP32weights/BF16compute, original camera packing, CPUactionnoise, renderer and instruction conventions. Server137 RTXPRO6000, fourworkers, sharedT5, CUDAgraphs.

Approach: separate campaign directory with the byte-identical copied inference snapshot and shared weight storage. Existing campaign scripts and manifests remain untouched. Waiting for both final audits and queue completion honors sequencing. A fresh [4,4] native-versus-graph check and eight excluded smokeepisodes precede scoring. The prior cross-host check covers only[1,1]/[2,2]; [4,4] runs on onehost and needs no cross-host claim.

Start requires matching checkpoint, 100verified pairs,10000verified videos and successful queue completion on both predecessors. Transient SSH failures keep the job waiting. GPUadmission requires60000MiB free. Single queue lock prevents duplicate execution. Finalaudit and live report use the existing pipeline.

Migration requested 2026-11-12: local waiting queue retired before any scored work. Both predecessors already completed and audited. Their verified receipt is transferred and SHA256-pinned in host_config; server137 additionally rechecks its original live manifest/COMPLETE against that receipt. Source, checkpoint and protocol stay unchanged.
