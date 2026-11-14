# Dense-S12 epoch10 implementation plan

> Required execution skill: superpowers:executing-plans; verification-before-completion. Isolated deployment; reuse existing frozen evaluator.

**Goal:** Verified epoch10-only download and autonomous2000episodeevaluation onidlelocalGPU.
**Architecture:** Adapt prior epoch10launcher identity toDenseS12; unchangednative evaluator; fixed4worker preflight;automaticfinalaudit/report.
**Tech stack:** Python3.10/PyTorch2.7.1cu128,CUDAgraphs,RoboTwin,existing supervisor.
**Spec:** spec.md

## Constraints/review focus

Epoch10only;90890updates;DenseS12architecture;pincheckpoint;preserveNFE10/seed42/actions32/24;fourworkers. Verifywrongcheckpoint/reuse rejection,8workerselection rejection,renderererrors,ownership/duplicatequeue safety,unchangedruntime andsource,smokeexclusion,automaticfinalaudit.

## Task1 — download and identity

- Download11epoch10releasefiles via137HTTPS relay;HF/LFS/Gitblob plusSHA256SUMS verification;rehashafterlocaltransfer.
- Adapt isolated launcher/config/CPUguards; keepinference codeunchanged;verifysourcehashes againstrelease andepoch5normalization/protocol.

## Task2 — safe fast launch gates

- Red/green regressions:calibrationneverlaunches8workers;rejectcached8workerselection;rejectrenderererrors;archiveincompletecalibrationbeforeretiming.
- Run38nativeevaluationtests andownedchild/rendererchecks.
- CPUstrictload epoch10/update90890/DenseS12/FP32/finiteweights/normalization/VAE;native-vsgraph exactactionparityoneightrealobservationsbothorders.
- Independentnarrowdeploymentreview;freezehost/source/assets/filesmanifest.

## Task3 — launch and verify

- Detachqueue;require4workersmoke (4tasks×2settings×2episodes=16excludedepisodes);keepfixed4workerswithmemoryheadroom.
- Verifyfullten-taskmanifest,actualpolicyqueriesandscoredprogress;retainlogsandautomaticfinalizer;publishsetup/liveMarkdowninrequested evaluation_reportfolder.

Ruling: useralreadyrequestedexecution;proceedwithoutadditionalapproval. PriorDenseS12ten-workerandv04/1eight-workerlongrunsfailedmemorydespitepassingshortcalibration. Useonlythevalidatedfour-workerpath. No mainrepositoryorbenchmarkmathchanges.
