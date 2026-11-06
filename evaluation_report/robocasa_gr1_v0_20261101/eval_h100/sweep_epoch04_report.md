# LoopWAM v0 RoboCasa-GR1 epoch_04 (update 94,064): n_action_steps sweep on ws_anhdh35

The model always predicts its trained 32-action chunk; n = env steps executed per policy query before re-planning (official protocol: 12).

| Task | n 8 | n 12 (official) | n 16 | n 24 | n 32 | StarVLA-OFT-Qwen3 | GR00T-N1.6 |
|---|---|---|---|---|---|---|---|
| PnPBottleToCabinetClose | 40 | 46 | 46 | 48 | 42 | 30 | 52 |
| PnPCanToDrawerClose | 34 | 56 | 78 | 46 | 52 | 76 | 13 |
| PnPCupToDrawerClose | 44 | 54 | 50 | 52 | 42 | 44 | 8 |
| PnPMilkToMicrowaveClose | 46 | 52 | 52 | 44 | 44 | 44 | 14 |
| PnPPotatoToMicrowaveClose | 16 | 18 | 22 | 16 | 22 | 32 | 42 |
| PnPWineToCabinetClose | 20 | 34 | 26 | 26 | 38 | 36 | 16 |
| PosttrainPnPNovelFromCuttingboardToBasketSplitA | 52 | 60 | 50 | 50 | 58 | 50 | 58 |
| PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA | 44 | 44 | 48 | 36 | 34 | 40 | 46 |
| PosttrainPnPNovelFromCuttingboardToPanSplitA | 70 | 58 | 38 | 56 | 60 | 70 | 68 |
| PosttrainPnPNovelFromCuttingboardToPotSplitA | 56 | 60 | 54 | 46 | 52 | 54 | 65 |
| PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA | 26 | 44 | 32 | 46 | 40 | 38 | 46 |
| PosttrainPnPNovelFromPlacematToBasketSplitA | 56 | 54 | 54 | 62 | 50 | 32 | 58 |
| PosttrainPnPNovelFromPlacematToBowlSplitA | 42 | 38 | 44 | 38 | 20 | 58 | 58 |
| PosttrainPnPNovelFromPlacematToPlateSplitA | 54 | 50 | 52 | 52 | 48 | 52 | 63 |
| PosttrainPnPNovelFromPlacematToTieredshelfSplitA | 20 | 20 | 20 | 6 | 26 | 24 | 28 |
| PosttrainPnPNovelFromPlateToBowlSplitA | 34 | 56 | 30 | 22 | 26 | 60 | 57 |
| PosttrainPnPNovelFromPlateToCardboardboxSplitA | 44 | 46 | 46 | 32 | 34 | 50 | 44 |
| PosttrainPnPNovelFromPlateToPanSplitA | 34 | 38 | 32 | 34 | 48 | 66 | 51 |
| PosttrainPnPNovelFromPlateToPlateSplitA | 66 | 60 | 70 | 68 | 64 | 68 | 79 |
| PosttrainPnPNovelFromTrayToCardboardboxSplitA | 50 | 34 | 58 | 50 | 40 | 44 | 52 |
| PosttrainPnPNovelFromTrayToPlateSplitA | 46 | 56 | 54 | 60 | 60 | 56 | 71 |
| PosttrainPnPNovelFromTrayToPotSplitA | 62 | 64 | 50 | 66 | 54 | 62 | 64 |
| PosttrainPnPNovelFromTrayToTieredbasketSplitA | 38 | 50 | 48 | 36 | 48 | 54 | 57 |
| PosttrainPnPNovelFromTrayToTieredshelfSplitA | 18 | 18 | 28 | 18 | 32 | 30 | 32 |
| **Average** | **42.17** | **46.25** | **45.08** | **42.08** | **43.08** | 48.8 | 47.6 |

Success rate (%) over 50 rollouts per task (any-step success, 720 steps, n_envs 1, seed 0, EGL, same code d937167 and checkpoint); n 12 is the official run results/official_epoch04. Reference columns (n_action_steps 12) from starVLA examples/simBenchmarks/Robocasa_tabletop/README.md @13d8ad4.
- n 8: step 94064, validation VALID, dir `/storage/anhdh35/robocasa_gr1_eval/results/sweep_epoch04`
- n 12 (official): step 94064, validation VALID, dir `/storage/anhdh35/robocasa_gr1_eval/results/official_epoch04`
- n 16: step 94064, validation VALID, dir `/storage/anhdh35/robocasa_gr1_eval/results/sweep_epoch04`
- n 24: step 94064, validation VALID, dir `/storage/anhdh35/robocasa_gr1_eval/results/sweep_epoch04`
- n 32: step 94064, validation VALID, dir `/storage/anhdh35/robocasa_gr1_eval/results/sweep_epoch04`
