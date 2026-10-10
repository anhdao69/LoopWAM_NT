#!/usr/bin/env python3
"""Compare cached RT windows with the existing decoded parent dataset."""
import json
from pathlib import Path
import numpy as np
import torch
from fastwam.datasets.transport_cached import TransportCachedDataset
from fastwam.datasets.loopwam_long import LoopWAMLongDataset, CAMERAS
from fastwam.datasets.lerobot3.lerobot_dataset import LeRobotDataset
from fastwam.models.wan22.residual_transport import gripper_command
from train_transport import PARENT_RUN

def main():
    dataset=TransportCachedDataset(PARENT_RUN/"train/data/data_manifest.json",PARENT_RUN/"latents")
    checked=[]
    for suite in dataset.manifest["suites"]:
        details=dataset.manifest["coverage"][suite]
        root=Path(details["dataset_dir"])
        fps=dataset.manifest["fps"]
        delta={k:[t/fps for t in range(0,33,4)] for k in CAMERAS}
        delta.update(action=[t/fps for t in range(32)],**{"observation.state":[t/fps for t in range(33)]})
        original=LoopWAMLongDataset(LeRobotDataset(str(root),root=root,episodes=details["train_episodes"],
            delta_timestamps=delta,video_backend="pyav"),dataset.manifest["text_cache_dir"],dataset.stats)
        for i in (0,31,len(original)//2,len(original)-1):
            a=original[i]; b=dataset[details["global_index_start"]+i]
            for key in ("action","action_is_pad","context","context_mask"):
                torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
            torch.testing.assert_close(a["proprio"][:1],b["proprio"],rtol=0,atol=0)
            checked.append([suite,i])
    # Actual dataset actions, compared with deployed evaluator's raw > .5 convention.
    raw=torch.from_numpy(np.concatenate([ep[0] for ep in dataset.episodes],0))
    indices=torch.linspace(0,len(raw)-1,1000).long()
    selected=raw[indices]
    norm=dataset.transform.normalizer["action"]
    command=gripper_command(norm.forward(selected),norm.scale,norm.offset)
    assert torch.equal(torch.where(command>=0,1.,-1.),torch.where(selected[:,6]>.5,-1.,1.))
    result=dict(windows=len(dataset),episodes=len(dataset.episodes),exact_window_checks=checked,
                gripper_actions_checked=1000,gripper_signs_match=True)
    Path("plans/residual_transport_evidence/data_verification.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
if __name__=="__main__": main()
