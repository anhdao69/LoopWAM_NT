| Task | H100 ep4 | H100 ep5 | ws ep4 | ws ep5 | StarVLA-OFT-Qwen3 | GR00T-N1.6 |
|---|---|---|---|---|---|---|
| PnPBottleToCabinetClose | 54 | 48 | 46 | 44 | 30 | 52 |
| PnPCanToDrawerClose | 50 | 58 | 56 | 56 | 76 | 13 |
| PnPCupToDrawerClose | 36 | 32 | 54 | 38 | 44 | 8 |
| PnPMilkToMicrowaveClose | 52 | 58 | 52 | 44 | 44 | 14 |
| PnPPotatoToMicrowaveClose | 20 | 32 | 18 | 30 | 32 | 42 |
| PnPWineToCabinetClose | 22 | 46 | 34 | 34 | 36 | 16 |
| PosttrainPnPNovelFromCuttingboardToBasketSplitA | 50 | 56 | 60 | 38 | 50 | 58 |
| PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA | 48 | 34 | 44 | 36 | 40 | 46 |
| PosttrainPnPNovelFromCuttingboardToPanSplitA | 52 | 48 | 58 | 64 | 70 | 68 |
| PosttrainPnPNovelFromCuttingboardToPotSplitA | 56 | 54 | 60 | 58 | 54 | 65 |
| PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA | 30 | 36 | 44 | 32 | 38 | 46 |
| PosttrainPnPNovelFromPlacematToBasketSplitA | 52 | 50 | 54 | 60 | 32 | 58 |
| PosttrainPnPNovelFromPlacematToBowlSplitA | 40 | 32 | 38 | 28 | 58 | 58 |
| PosttrainPnPNovelFromPlacematToPlateSplitA | 52 | 52 | 50 | 48 | 52 | 63 |
| PosttrainPnPNovelFromPlacematToTieredshelfSplitA | 22 | 16 | 20 | 22 | 24 | 28 |
| PosttrainPnPNovelFromPlateToBowlSplitA | 46 | 42 | 56 | 44 | 60 | 57 |
| PosttrainPnPNovelFromPlateToCardboardboxSplitA | 34 | 42 | 46 | 34 | 50 | 44 |
| PosttrainPnPNovelFromPlateToPanSplitA | 42 | 46 | 38 | 38 | 66 | 51 |
| PosttrainPnPNovelFromPlateToPlateSplitA | 56 | 68 | 60 | 74 | 68 | 79 |
| PosttrainPnPNovelFromTrayToCardboardboxSplitA | 44 | 48 | 34 | 54 | 44 | 52 |
| PosttrainPnPNovelFromTrayToPlateSplitA | 60 | 52 | 56 | 56 | 56 | 71 |
| PosttrainPnPNovelFromTrayToPotSplitA | 60 | 54 | 64 | 62 | 62 | 64 |
| PosttrainPnPNovelFromTrayToTieredbasketSplitA | 50 | 44 | 50 | 44 | 54 | 57 |
| PosttrainPnPNovelFromTrayToTieredshelfSplitA | 30 | 22 | 18 | 30 | 30 | 32 |
| **Average** | **44.08** | **44.58** | **46.25** | **44.50** | 48.8 | 47.6 |

Success rate (%) over 50 rollouts per task (any-step success, 720 steps, n_action_steps 12, n_envs 1, seed 0, EGL); reference columns from starVLA examples/simBenchmarks/Robocasa_tabletop/README.md @13d8ad4.
- H100 ep4: step 94064, validation VALID, dir `/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robocasa_gr1_v0_20261101/eval/official_epoch04`
- H100 ep5: step 117580, validation VALID, dir `/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robocasa_gr1_v0_20261101/eval/official_epoch05`
- ws ep4: step 94064, validation VALID, dir `/storage/anhdh35/robocasa_gr1_eval/results/official_epoch04`
- ws ep5: step 117580, validation VALID, dir `/storage/anhdh35/robocasa_gr1_eval/results/official_epoch05`
