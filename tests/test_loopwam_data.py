"""Episode isolation, normalization provenance, and temporal sample contracts."""
import hashlib
import importlib.util

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


def test_data_adapter_exists():
    assert importlib.util.find_spec("fastwam.datasets.loopwam_long") is not None


def test_split_is_stratified_deterministic_and_disjoint():
    from fastwam.datasets.loopwam_long import split_episodes
    rows = [{"episode_index": i, "tasks": [f"task-{i // 50}"], "length": 40}
            for i in range(100)]
    train, val, counts = split_episodes(rows, seed=42)
    assert (train, val, counts) == split_episodes(list(reversed(rows)), seed=42)
    assert len(train) == 90 and len(val) == 10 and not set(train) & set(val)
    assert set(train) | set(val) == set(range(100))
    assert all(c == {"available": 50, "train": 45, "validation": 5} for c in counts.values())
    assert split_episodes(rows, seed=43)[1] != val


def test_normalization_excludes_heldout_extreme(tmp_path):
    from fastwam.datasets.loopwam_long import compute_train_stats
    data = tmp_path / "data" / "chunk-000"
    data.mkdir(parents=True)
    pq.write_table(pa.table({"episode_index": [0, 0, 1],
                            "action": [[-1.] * 7, [1.] * 7, [999.] * 7],
                            "observation.state": [[-2.] * 8, [2.] * 8, [999.] * 8]}),
                   data / "file-000.parquet")
    stats = compute_train_stats(tmp_path, [0])
    assert stats["num_episodes"] == 1 and stats["num_transition"] == 2
    assert stats["action"]["default"]["global_max"] == [1.] * 7
    assert stats["state"]["default"]["global_min"] == [-2.] * 8


def test_sample_preserves_text_mask_padding_and_camera_order(tmp_path):
    from fastwam.datasets.loopwam_long import LoopWAMLongDataset, DEFAULT_PROMPT
    prompt = DEFAULT_PROMPT.format(task="test task")
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    mask = torch.arange(128) < 7
    torch.save({"context": torch.ones(128, 4096), "mask": mask},
               tmp_path / f"{digest}.t5_len128.wan22ti2v5b.pt")
    # The reader fixture represents a two-step remainder; real decoder coverage
    # is exercised separately against the remote LIBERO files.
    row = {"task": "test task", "episode_index": torch.tensor(0),
           "frame_index": torch.tensor(38), "timestamp": torch.tensor(1.9),
           "action": torch.ones(32, 7), "observation.state": torch.ones(33, 8),
           "action_is_pad": torch.arange(32) >= 2,
           "observation.state_is_pad": torch.arange(33) >= 2,
           "observation.images.image_is_pad": torch.arange(9) >= 1,
           "observation.images.image": torch.zeros(9, 3, 224, 224),
           "observation.images.wrist_image": torch.ones(9, 3, 224, 224)}
    stats = {kind: {"default": {"global_min": [-1.] * dim, "global_max": [1.] * dim}}
             for kind, dim in (("action", 7), ("state", 8))}
    dataset = LoopWAMLongDataset([row], tmp_path, stats)
    sample = dataset[0]
    assert sample["video"].shape == (3, 9, 224, 448)
    assert torch.all(sample["video"][..., :224] == -1)
    assert torch.all(sample["video"][..., 224:] == 1)
    assert torch.equal(sample["context_mask"], mask)
    assert torch.all(sample["context"][~mask] == 0)
    assert torch.all(sample["action"][2:, :6] == 0)
    assert torch.all(sample["action"][2:, 6] == 1)
    assert sample["proprio"].shape == (32, 8)
    assert sample["proprio_is_pad"].shape == (32,)
    assert sample["action_is_pad"].sum() == 30
    assert sample["image_is_pad"].sum() == 8
    torch.testing.assert_close(dataset.denormalize_action(sample["action"]), sample["action"])


def test_content_manifest_detects_modified_payload(tmp_path):
    from fastwam.datasets.loopwam_long import dataset_content_manifest
    path=tmp_path/'videos'/'clip.mp4'; path.parent.mkdir()
    path.write_bytes(b'original')
    first=dataset_content_manifest(tmp_path)
    path.write_bytes(b'modified')
    second=dataset_content_manifest(tmp_path)
    assert first.keys()==second.keys()
    assert first['videos/clip.mp4']['sha256']!=second['videos/clip.mp4']['sha256']
