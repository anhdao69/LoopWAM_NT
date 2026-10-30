"""LIBERO adapter for the public LiLa-WAM training code (github.com/teee000/LiLa-WAM @339860f, which ships RoboTwin only).

Reads OUR LIBERO data converted by convert_ours_to_lilawam_hdf5.py (authors' LIBERO HDF5 layout):
  <root>/<suite>/<task>/demo_clean/data/episode_XXXXX.hdf5 : action (T,7), proprio (T,8), observation/{agentview,robot0_eye_in_hand}/rgb PNG
Only data access changes; sampling, normalization, augmentation and the model are the public code's.
- The paper's LIBERO policy sees the third-person and wrist views at 256x256. The public model takes one image, so the
  two views are placed side by side into one 512x256 (W x H) RGB image: [agentview | wrist]. DINOv3 (patch 16, RoPE)
  then sees 32x16 patches; the future-feature target is the same composite at t + future_frame_offset.
- Images are decoded to RGB explicitly (our PNGs are standard-colour); the same composite() is used at eval time.
- task_name = "<suite>/<task dir>" so the 40 LIBERO tasks are distinct task-condition keys.
"""
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

LILA = "/groups/yshang/an221229/code/wam_baselines_20261030/LiLa-WAM"
if LILA not in sys.path:
    sys.path.insert(0, LILA)
from dataloader import dataset as D  # noqa: E402

CAMS = ("agentview", "robot0_eye_in_hand")


def decode_rgb(buf):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(bytes(buf), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def composite(agent_rgb, wrist_rgb, size=256):
    """Two HxWx3 uint8 RGB views -> one size x 2*size x 3 image [agentview | wrist]."""
    def fit(x):
        return x if x.shape[:2] == (size, size) else cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(np.concatenate([fit(agent_rgb), fit(wrist_rgb)], axis=1))


class LiberoTaskDataset(D.RobotWinTaskDataset):
    def _load_episodes(self):
        for root in self.dataset_dirs:
            for task_dir in sorted(d for d in root.iterdir() if d.is_dir()):
                for p in sorted((task_dir / "demo_clean" / "data").glob("*.hdf5")):
                    self.all_episodes.append({"episode_name": p.stem, "task_name": f"{root.name}/{task_dir.name}",
                                              "hdf5_path": str(p), "split": "demo_clean"})
        if not self.all_episodes:
            raise ValueError(f"No episodes in {self.dataset_dirs}")

    def _build_index_map(self):
        self.valid_indices, self.episode_metadata, off = [], [], 0
        for ep in self.all_episodes:
            with h5py.File(ep["hdf5_path"], "r") as f:
                n = f["action"].shape[0]
            if n < 2:
                continue
            self.episode_metadata.append({"hdf5_path": ep["hdf5_path"], "task_name": ep["task_name"], "length": n,
                                          "global_start": off, "global_end": off + n})
            self.valid_indices.append(np.arange(off, off + n, dtype=np.int64))
            off += n
        self.valid_indices = np.concatenate(self.valid_indices)
        self._ep_end_bounds = np.array([e["global_end"] for e in self.episode_metadata])

    def _frames(self, root, t_idx):
        idx = np.asarray(t_idx)
        u = np.unique(idx)
        a, w = root["observation/agentview/rgb"][u], root["observation/robot0_eye_in_hand/rgb"][u]
        comp = np.stack([composite(decode_rgb(x), decode_rgb(y)) for x, y in zip(a, w)])
        return comp[np.searchsorted(u, idx)]

    def _load_hdf5_data(self, hdf5_path, query_indices):
        out = {}
        with h5py.File(hdf5_path, "r") as root:
            for key, src, name in (("action", "action", "action_sequence"), ("state", "proprio", "state")):
                idx = np.asarray(query_indices[key]); u = np.unique(idx)
                out[name] = torch.from_numpy(root[src][u][np.searchsorted(u, idx)]).float()
            cam0 = self.camera_names[0]
            out["frame"] = {cam0: self._frames(root, query_indices[cam0])}
            if self.use_future_feat and "future_frame" in query_indices:
                out["future_frame"] = self._frames(root, query_indices["future_frame"])
        return out


def create_dataset(config, val=False):
    from omegaconf import OmegaConf
    ff = config.model.get("future_feat", {})
    return LiberoTaskDataset(
        dataset_dir=list(config.dataset.dataset_dir), data_mode="clean",
        indices_config=OmegaConf.to_container(config.dataset.indices_config, resolve=True), val=val,
        image_aug=config.dataset.image_aug and not val, camera_names=list(config.dataset.camera_names),
        image_size=tuple(config.dataset.image_size),
        task_cond_dir=config.dataset.task_cond_dir if config.model.get("use_task_cond", False) else None,
        use_future_feat=bool(ff.get("enabled", False)) if ff else False,
        future_frame_offset=config.dataset.get("future_frame_offset", None))
