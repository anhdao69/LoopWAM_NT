"""Task selections for ChronoLoop robot runs (see plans/performance/ChronoLoop_Robot_Setup_20261015.md)."""

# LiLa-WAM (arXiv 2608.03701), Sec. 4.2: "For ablation studies, we additionally construct a subset of
# 10 representative tasks". Same list in github.com/teee000/LiLa-WAM data-200-10-taskcond/meta.json.
LILAWAM_10 = ('adjust_bottle', 'grab_roller', 'hanging_mug', 'move_stapler_pad', 'open_microwave',
              'press_stapler', 'scan_object', 'stack_blocks_two', 'stamp_seal', 'turn_switch')

# Five hardest RoboTwin 2.0 tasks by the trimmed mean (best and worst dropped) of the ten clean/randomized
# success rates of pi0.5, Motus, X-VLA, GigaWorld-Policy and LiLa-WAM in LiLa-WAM Table 8 (all 50 tasks):
# hanging_mug 27.6, blocks_ranking_size 63.5, scan_object 65.0, place_can_basket 69.5, handover_mic 72.2.
HARD_5 = ('hanging_mug', 'blocks_ranking_size', 'scan_object', 'place_can_basket', 'handover_mic')

# name -> (tasks, demos). The paper trains on "50 clean demonstrations and 500 randomized
# demonstrations" per task, jointly; the FastWAM release holds exactly these 550 per task.
ROBOTWIN_SUBSETS = {
    'robotwin10': (LILAWAM_10, 'clean+randomized'),
    'robotwin10_clean': (LILAWAM_10, 'clean'),
    'robotwin5hard': (HARD_5, 'clean+randomized'),
}
# Subsets whose latent cache is built with the resize chain on the GPU (same ops; see chronoloop_robot.py).
ROBOTWIN_PREPROCESS = {'robotwin5hard': 'cuda'}

# RoboDojo (25 FPS): actions executed per replanning query, i.e. the TBPTT traversal stride.
ROBODOJO_REPLAN_STRIDE = 24
