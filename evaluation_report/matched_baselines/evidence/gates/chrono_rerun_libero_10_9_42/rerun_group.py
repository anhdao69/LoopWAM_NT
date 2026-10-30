"""Rerun the campaign group libero_10:9:42 (10 episodes) exactly as gates.py's campaign gate does, writing digests
to a scratch file instead of the gate JSON."""
import hashlib
import json
import sys

sys.path.insert(0, '/data/anh/code/wam_baselines/matched_eval_partition')
import torch  # noqa: E402
from gates import run_episodes  # noqa: E402
from families import FAMILIES  # noqa: E402

name, out = sys.argv[1], sys.argv[2]
torch.set_num_threads(4)
family = FAMILIES[name]()
family.build('libero_spatial', text_encoder=False, init_seed=2)  # gates.py rolls out with its last build
family.check_protocol_config()
text_cache, _ = family.load_text_cache()
state = dict(digest=hashlib.sha256(), queries=0)


def observer(action, seconds):
    state['digest'].update(action.tobytes())
    state['queries'] += 1


rollout = family.install_policy(observer, text_cache)
rows = run_episodes(family, rollout, 'libero_10', 9, 42, 10, state)
with open(out, 'w') as f:
    json.dump(dict(family=name, repo=str(family.repo), rows=rows), f, indent=1)
