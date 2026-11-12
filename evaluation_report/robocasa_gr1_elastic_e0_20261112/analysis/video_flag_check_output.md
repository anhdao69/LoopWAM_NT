# Validator flags: which episodes

Output of `video_flag_check.py`, run read-only on the H100 login node on 2026-11-12 (the rollout videos are kept on the H100 and are not stored in this report).

```
e01_flat_k1_n2 PnPMilkToMicrowaveClose videos 50 json successes 16 success1 videos 17
  frames per video (set over the 50 videos): [360]
  episode index 47 (0-based): JSON success False, video flag 1, frames 360, bytes 5245381, file eb4669c8-5543-4369-a541-06a1b02b85d5_success1.mp4
  JSON successes whose video flag is 0 (success reached, not held to the last rendered step): 0
e05_flat_k1_n5 PosttrainPnPNovelFromPlateToCardboardboxSplitA videos 50 json successes 18 success1 videos 13
  frames per video (set over the 50 videos): [360]
  episode index 46 (0-based): JSON success False, video flag 1, frames 360, bytes 4768393, file c13f3a56-3390-4121-912a-d8b208f4610b_success1.mp4
  JSON successes whose video flag is 0 (success reached, not held to the last rendered step): 6
```
