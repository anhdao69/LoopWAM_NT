| Task | ws epoch 4 | ws epoch 5 | StarVLA-OFT-Qwen3 | GR00T-N1.6 |
|---|---|---|---|---|
| PnPBottleToCabinetClose | 46 | 44 | 30 | 52 |
| PnPCanToDrawerClose | 56 | 56 | 76 | 13 |
| PnPCupToDrawerClose | 54 | 38 | 44 | 8 |
| PnPMilkToMicrowaveClose | 52 | 44 | 44 | 14 |
| PnPPotatoToMicrowaveClose | 18 | 30 | 32 | 42 |
| PnPWineToCabinetClose | 34 | 34 | 36 | 16 |
| PosttrainPnPNovelFromCuttingboardToBasketSplitA | 60 | 38 | 50 | 58 |
| PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA | 44 | 36 | 40 | 46 |
| PosttrainPnPNovelFromCuttingboardToPanSplitA | 58 | 64 | 70 | 68 |
| PosttrainPnPNovelFromCuttingboardToPotSplitA | 60 | 58 | 54 | 65 |
| PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA | 44 | 32 | 38 | 46 |
| PosttrainPnPNovelFromPlacematToBasketSplitA | 54 | 60 | 32 | 58 |
| PosttrainPnPNovelFromPlacematToBowlSplitA | 38 | 28 | 58 | 58 |
| PosttrainPnPNovelFromPlacematToPlateSplitA | 50 | 48 | 52 | 63 |
| PosttrainPnPNovelFromPlacematToTieredshelfSplitA | 20 | 22 | 24 | 28 |
| PosttrainPnPNovelFromPlateToBowlSplitA | 56 | 44 | 60 | 57 |
| PosttrainPnPNovelFromPlateToCardboardboxSplitA | 46 | 34 | 50 | 44 |
| PosttrainPnPNovelFromPlateToPanSplitA | 38 | 38 | 66 | 51 |
| PosttrainPnPNovelFromPlateToPlateSplitA | 60 | 74 | 68 | 79 |
| PosttrainPnPNovelFromTrayToCardboardboxSplitA | 34 | 54 | 44 | 52 |
| PosttrainPnPNovelFromTrayToPlateSplitA | 56 | 56 | 56 | 71 |
| PosttrainPnPNovelFromTrayToPotSplitA | 64 | 62 | 62 | 64 |
| PosttrainPnPNovelFromTrayToTieredbasketSplitA | 50 | 44 | 54 | 57 |
| PosttrainPnPNovelFromTrayToTieredshelfSplitA | 18 | 30 | 30 | 32 |
| **Average** | **46.25** | **44.50** | 48.8 | 47.6 |

Success rate (%) over 50 rollouts per task (any-step success, 720 steps, n_action_steps 12, n_envs 1, seed 0, EGL); reference columns from starVLA examples/simBenchmarks/Robocasa_tabletop/README.md @13d8ad4.
- ws epoch 4: step 94064, validation VALID, dir `/storage/anhdh35/robocasa_gr1_eval/results/official_epoch04`
- ws epoch 5: step 117580, validation VALID, dir `/storage/anhdh35/robocasa_gr1_eval/results/official_epoch05`
