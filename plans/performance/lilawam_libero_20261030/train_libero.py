"""Run the public LiLa-WAM train.py unchanged except that its dataset factory reads LIBERO (libero_data.py).
Same CLI as train.py: --config --norm_stats_path --save_dir [--resume]."""
import os, runpy, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import libero_data  # noqa: E402  (puts the LiLa-WAM repo on sys.path)
from dataloader import dataset as D  # noqa: E402
D.create_dataset = libero_data.create_dataset
runpy.run_path(os.path.join(libero_data.LILA, "train.py"), run_name="__main__")
