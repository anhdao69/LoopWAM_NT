"""Task selections for ChronoLoop robot runs (see plans/performance/ChronoLoop_Robot_Setup_20261015.md)."""

# LiLa-WAM (arXiv 2608.03701), Sec. 4.2: "For ablation studies, we additionally construct a subset of
# 10 representative tasks". Same list in github.com/teee000/LiLa-WAM data-200-10-taskcond/meta.json.
LILAWAM_10 = ('adjust_bottle', 'grab_roller', 'hanging_mug', 'move_stapler_pad', 'open_microwave',
              'press_stapler', 'scan_object', 'stack_blocks_two', 'stamp_seal', 'turn_switch')

# Five hardest RoboTwin 2.0 tasks by the trimmed mean (best and worst dropped) of the ten clean/randomized
# success rates of pi0.5, Motus, X-VLA, GigaWorld-Policy and LiLa-WAM in LiLa-WAM Table 8 (all 50 tasks):
# hanging_mug 27.6, blocks_ranking_size 63.5, scan_object 65.0, place_can_basket 69.5, handover_mic 72.2.
HARD_5 = ('hanging_mug', 'blocks_ranking_size', 'scan_object', 'place_can_basket', 'handover_mic')

# Ten-task selection for the HiPerGator CL-A run (October 22, 2026): HARD_5 without scan_object,
# adjust_bottle / open_microwave / turn_switch from LILAWAM_10, and lift_pot, beat_block_hammer, stack_blocks_three.
ROBOTWIN_10B = ('adjust_bottle', 'place_can_basket', 'lift_pot', 'handover_mic', 'open_microwave', 'turn_switch',
                'beat_block_hammer', 'hanging_mug', 'blocks_ranking_size', 'stack_blocks_three')

# Five-task coverage subset for the ChronoLoop variant sweep on HiPerGator (October 22, 2026): short single-stage
# (adjust_bottle, lift_pot, beat_block_hammer), articulated (open_microwave) and long multi-stage (stack_blocks_three).
ROBOTWIN5_COVERAGE_V1 = ('adjust_bottle', 'lift_pot', 'open_microwave', 'beat_block_hammer', 'stack_blocks_three')

# name -> (tasks, demos). The paper trains on "50 clean demonstrations and 500 randomized
# demonstrations" per task, jointly; the FastWAM release holds exactly these 550 per task.
ROBOTWIN_SUBSETS = {
    'robotwin10': (LILAWAM_10, 'clean+randomized'),
    'robotwin10_clean': (LILAWAM_10, 'clean'),
    'robotwin5hard': (HARD_5, 'clean+randomized'),
    'robotwin10b': (ROBOTWIN_10B, 'clean+randomized'),
    'robotwin5cov': (ROBOTWIN5_COVERAGE_V1, 'clean+randomized'),
}
# Subsets whose latent cache is built with the resize chain on the GPU (same ops; see chronoloop_robot.py).
ROBOTWIN_PREPROCESS = {'robotwin5hard': 'cuda', 'robotwin10b': 'cuda', 'robotwin5cov': 'cuda'}

# RoboDojo (25 FPS): actions executed per replanning query, i.e. the TBPTT traversal stride.
ROBODOJO_REPLAN_STRIDE = 24
