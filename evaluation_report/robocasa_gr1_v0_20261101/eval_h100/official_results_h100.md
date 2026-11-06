| Task | H100 epoch 4 | H100 epoch 5 | StarVLA-OFT-Qwen3 | GR00T-N1.6 |
|---|---|---|---|---|
| PnPBottleToCabinetClose | 54 | 48 | 30 | 52 |
| PnPCanToDrawerClose | 50 | 58 | 76 | 13 |
| PnPCupToDrawerClose | 36 | 32 | 44 | 8 |
| PnPMilkToMicrowaveClose | 52 | 58 | 44 | 14 |
| PnPPotatoToMicrowaveClose | 20 | 32 | 32 | 42 |
| PnPWineToCabinetClose | 22 | 46 | 36 | 16 |
| PosttrainPnPNovelFromCuttingboardToBasketSplitA | 50 | 56 | 50 | 58 |
| PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA | 48 | 34 | 40 | 46 |
| PosttrainPnPNovelFromCuttingboardToPanSplitA | 52 | 48 | 70 | 68 |
| PosttrainPnPNovelFromCuttingboardToPotSplitA | 56 | 54 | 54 | 65 |
| PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA | 30 | 36 | 38 | 46 |
| PosttrainPnPNovelFromPlacematToBasketSplitA | 52 | 50 | 32 | 58 |
| PosttrainPnPNovelFromPlacematToBowlSplitA | 40 | 32 | 58 | 58 |
| PosttrainPnPNovelFromPlacematToPlateSplitA | 52 | 52 | 52 | 63 |
| PosttrainPnPNovelFromPlacematToTieredshelfSplitA | 22 | 16 | 24 | 28 |
| PosttrainPnPNovelFromPlateToBowlSplitA | 46 | 42 | 60 | 57 |
| PosttrainPnPNovelFromPlateToCardboardboxSplitA | 34 | 42 | 50 | 44 |
| PosttrainPnPNovelFromPlateToPanSplitA | 42 | 46 | 66 | 51 |
| PosttrainPnPNovelFromPlateToPlateSplitA | 56 | 68 | 68 | 79 |
| PosttrainPnPNovelFromTrayToCardboardboxSplitA | 44 | 48 | 44 | 52 |
| PosttrainPnPNovelFromTrayToPlateSplitA | 60 | 52 | 56 | 71 |
| PosttrainPnPNovelFromTrayToPotSplitA | 60 | 54 | 62 | 64 |
| PosttrainPnPNovelFromTrayToTieredbasketSplitA | 50 | 44 | 54 | 57 |
| PosttrainPnPNovelFromTrayToTieredshelfSplitA | 30 | 22 | 30 | 32 |
| **Average** | **44.08** | **44.58** | 48.8 | 47.6 |

Success rate (%) over 50 rollouts per task (any-step success, 720 steps, n_action_steps 12, n_envs 1, seed 0, EGL); reference columns from starVLA examples/simBenchmarks/Robocasa_tabletop/README.md @13d8ad4.
- H100 epoch 4: step 94064, validation VALID, dir `/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robocasa_gr1_v0_20261101/eval/official_epoch04`
- H100 epoch 5: step 117580, validation VALID, dir `/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robocasa_gr1_v0_20261101/eval/official_epoch05`
