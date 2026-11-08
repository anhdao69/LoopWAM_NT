# Live reevaluation summary

Updated 2026-11-08T16:30:33.848528+00:00 (local UTC). Source report sync runs every 60 seconds.

| Epoch | Host | Phase | Finished episode videos | Complete task cells | Any-step success |
|---|---|---|---:|---:|---|
| 9 | Local RTX PRO 6000 | complete | 1200/1200 | 24/24 | 628/1200 = 52.33% (validated) |
| 10 | Server189 RTX PRO 5000 | complete | 1200/1200 | 24/24 | 600/1200 = 50.00% (validated) |

Success comes only from native task JSON, never from video filenames. Each task writes its JSON after all 50 episodes finish. Partial completed-task rates are not the full 24-task result. Smoke episodes are excluded. Environment seed 0, policy noise seed 42, 24 tasks × 50 episodes, 720 steps, action prefix 12/32, NFE 10, CFG 1, all videos.

| Epoch | Completed task | Successes | Rate |
|---|---|---:|---:|
| 9 | PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env | 23/50 | 46.00% |
| 9 | PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env | 36/50 | 72.00% |
| 9 | PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env | 17/50 | 34.00% |
| 9 | PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env | 23/50 | 46.00% |
| 9 | PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env | 18/50 | 36.00% |
| 9 | PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env | 28/50 | 56.00% |
| 9 | PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env | 35/50 | 70.00% |
| 9 | PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env | 27/50 | 54.00% |
| 9 | PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env | 29/50 | 58.00% |
| 9 | PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env | 33/50 | 66.00% |
| 9 | PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env | 21/50 | 42.00% |
| 9 | PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env | 27/50 | 54.00% |
| 9 | PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env | 22/50 | 44.00% |
| 9 | PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env | 30/50 | 60.00% |
| 9 | PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env | 13/50 | 26.00% |
| 9 | PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env | 28/50 | 56.00% |
| 9 | PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env | 22/50 | 44.00% |
| 9 | PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env | 28/50 | 56.00% |
| 9 | PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env | 39/50 | 78.00% |
| 9 | PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env | 26/50 | 52.00% |
| 9 | PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env | 31/50 | 62.00% |
| 9 | PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env | 32/50 | 64.00% |
| 9 | PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env | 24/50 | 48.00% |
| 9 | PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env | 16/50 | 32.00% |
| 10 | PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env | 23/50 | 46.00% |
| 10 | PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env | 32/50 | 64.00% |
| 10 | PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env | 18/50 | 36.00% |
| 10 | PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env | 24/50 | 48.00% |
| 10 | PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env | 19/50 | 38.00% |
| 10 | PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env | 18/50 | 36.00% |
| 10 | PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env | 27/50 | 54.00% |
| 10 | PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env | 21/50 | 42.00% |
| 10 | PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env | 37/50 | 74.00% |
| 10 | PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env | 26/50 | 52.00% |
| 10 | PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env | 24/50 | 48.00% |
| 10 | PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env | 25/50 | 50.00% |
| 10 | PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env | 26/50 | 52.00% |
| 10 | PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env | 29/50 | 58.00% |
| 10 | PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env | 14/50 | 28.00% |
| 10 | PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env | 30/50 | 60.00% |
| 10 | PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env | 19/50 | 38.00% |
| 10 | PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env | 22/50 | 44.00% |
| 10 | PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env | 34/50 | 68.00% |
| 10 | PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env | 33/50 | 66.00% |
| 10 | PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env | 33/50 | 66.00% |
| 10 | PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env | 32/50 | 64.00% |
| 10 | PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env | 24/50 | 48.00% |
| 10 | PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env | 10/50 | 20.00% |
