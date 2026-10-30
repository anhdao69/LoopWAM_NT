# Rejected evaluator controls

These results used the earlier episode-parallel pool, which changed LIBERO's
per-task environment lifecycle. The unchanged aligned 4/1 checkpoint scored
79/100 at seed 42 rather than the historical 81/100. Do not use these control
results in model comparisons, significance tests, or final success-rate tables.

They are retained as failure evidence. The corrected task-sequence pool passed
exact 100-episode baseline reproduction; see `../evaluator_reproduction/`.
