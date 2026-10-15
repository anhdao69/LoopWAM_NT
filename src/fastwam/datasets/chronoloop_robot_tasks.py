"""Task selections for ChronoLoop robot runs (see plans/performance/ChronoLoop_Robot_Setup_20261015.md)."""

# LiLa-WAM (arXiv 2608.03701), Sec. 4.2: "For ablation studies, we additionally construct a subset of
# 10 representative tasks". Same list in github.com/teee000/LiLa-WAM data-200-10-taskcond/meta.json.
LILAWAM_10 = ('adjust_bottle', 'grab_roller', 'hanging_mug', 'move_stapler_pad', 'open_microwave',
              'press_stapler', 'scan_object', 'stack_blocks_two', 'stamp_seal', 'turn_switch')

# name -> (tasks, demos). The paper trains on "50 clean demonstrations and 500 randomized
# demonstrations" per task, jointly; the FastWAM release holds exactly these 550 per task.
ROBOTWIN_SUBSETS = {
    'robotwin10': (LILAWAM_10, 'clean+randomized'),
    'robotwin10_clean': (LILAWAM_10, 'clean'),
}

# RoboDojo (25 FPS): actions executed per replanning query, i.e. the TBPTT traversal stride.
ROBODOJO_REPLAN_STRIDE = 24
