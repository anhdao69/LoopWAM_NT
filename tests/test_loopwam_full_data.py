"""All-suite coverage, shared normalization, and immutable cache provenance."""
import hashlib
import json
import sys
import types

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


@pytest.fixture
def full_data(tmp_path, monkeypatch):
    from fastwam.datasets.loopwam_long import CAMERAS, DEFAULT_PROMPT

    root = tmp_path / "datasets"
    cache = tmp_path / "text"
    cache.mkdir()
    for suite, value in zip(SUITES, (-8., -4., 0., 8.)):
        directory = root / f"{suite}_no_noops_lerobot"
        meta = directory / "meta/episodes/chunk-000"
        data = directory / "data/chunk-000"
        meta.mkdir(parents=True)
        data.mkdir(parents=True)
        tasks = ["shared task", f"{suite} task"]
        (directory / "meta/info.json").write_text(json.dumps({
            "codebase_version": "v3.0", "total_episodes": 2,
            "total_frames": 4, "fps": 20}))
        pq.write_table(pa.table({"episode_index": [0, 1], "tasks": [[t] for t in tasks],
                                 "length": [2, 2]}), meta / "file-000.parquet")
        pq.write_table(pa.table({"episode_index": [0, 0, 1, 1],
                                 "action": [[value] * 7, [value + 1] * 7] * 2,
                                 "observation.state": [[value * 2] * 8, [value * 2 + 2] * 8] * 2}),
                       data / "file-000.parquet")
        for task in tasks:
            digest = hashlib.sha256(DEFAULT_PROMPT.format(task=task).encode()).hexdigest()
            torch.save({"context": torch.ones(128, 4096), "mask": torch.arange(128) < 7},
                       cache / f"{digest}.t5_len128.wan22ti2v5b.pt")

    # Keep parquet selection and transforms real; replace the external video
    # reader because these fixtures deliberately have no encoded camera files.
    class Reader:
        def __init__(self, repo_id, *, root, episodes, delta_timestamps, video_backend):
            self.root = root
            self.episodes = episodes
            self.rows = [r for r in pq.read_table(root / "data/chunk-000/file-000.parquet").to_pylist()
                         if r["episode_index"] in episodes]
            self.tasks = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            item = {"task": self.tasks[row["episode_index"]]["tasks"][0],
                    "episode_index": row["episode_index"], "frame_index": index % 2,
                    "action": torch.tensor([row["action"]] * 32),
                    "observation.state": torch.tensor([row["observation.state"]] * 33),
                    "action_is_pad": torch.zeros(32, dtype=torch.bool),
                    "observation.state_is_pad": torch.zeros(33, dtype=torch.bool)}
            for camera in CAMERAS:
                item[camera] = torch.zeros(9, 3, 16, 16)
                item[f"{camera}_is_pad"] = torch.zeros(9, dtype=torch.bool)
            return item

    module = types.ModuleType("fastwam.datasets.lerobot3.lerobot_dataset")
    module.LeRobotDataset = Reader
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return root, cache, tmp_path / "output"


def test_full_builder_uses_every_suite_episode_and_one_global_normalization(full_data):
    from fastwam.datasets.loopwam_long import build_full_libero_datasets

    root, cache, output = full_data
    train, validation, manifest = build_full_libero_datasets(root, cache, output)
    assert len(train) == 16 and len(validation) == 0
    assert manifest["dataset_scope"] == "full_libero" and manifest["split"] == "all_train"
    assert manifest["suites"] == list(SUITES)
    assert len(manifest["train_episodes"]) == len(set(manifest["train_episodes"])) == 8
    assert manifest["validation_episodes"] == []
    assert manifest["train_windows"] == 16 and manifest["validation_windows"] == 0
    assert len(manifest["text_cache_files"]) == 5
    assert len(manifest["content_files"]) == 4
    assert all(manifest["coverage"][suite]["train_episodes"] == [0, 1] for suite in SUITES)
    stats = json.loads((output / "dataset_stats.json").read_text())
    assert stats["num_episodes"] == 8 and stats["num_transition"] == 16
    assert stats["action"]["default"] == {"global_min": [-8.] * 7, "global_max": [9.] * 7}
    assert stats["state"]["default"] == {"global_min": [-16.] * 8, "global_max": [18.] * 8}
    assert all(dataset.stats == stats for dataset in train.datasets)
    assert [train[i]["prompt"].split(": ")[-1] for i in (2, 6, 10, 14)] == [f"{s} task" for s in SUITES]
    torch.testing.assert_close(train[0]["action"], torch.full((32, 7), -1.))
    torch.testing.assert_close(train[13]["action"], torch.full((32, 7), 1.))
    assert manifest["normalization_source"] == "training episodes only"
    assert manifest["normalization_sha256"] == hashlib.sha256((output / "dataset_stats.json").read_bytes()).hexdigest()


def test_full_builder_rejects_missing_task_cache_before_persisting(full_data):
    from fastwam.datasets.loopwam_long import DEFAULT_PROMPT, build_full_libero_datasets

    root, cache, output = full_data
    digest = hashlib.sha256(DEFAULT_PROMPT.format(task="libero_goal task").encode()).hexdigest()
    (cache / f"{digest}.t5_len128.wan22ti2v5b.pt").unlink()
    with pytest.raises(FileNotFoundError, match="text cache"):
        build_full_libero_datasets(root, cache, output)
    assert not (output / "data_manifest.json").exists()


def test_full_builder_rejects_missing_training_frames(full_data):
    from fastwam.datasets.loopwam_long import build_full_libero_datasets

    root, cache, output = full_data
    path = root / "libero_goal_no_noops_lerobot/data/chunk-000/file-000.parquet"
    pq.write_table(pq.read_table(path).slice(0, 3), path)
    with pytest.raises(ValueError, match="frame|transition"):
        build_full_libero_datasets(root, cache, output)


def test_full_manifest_cache_identity_is_stable_and_binds_suite_order(full_data):
    from fastwam.datasets.loopwam_long import build_full_libero_datasets
    from fastwam.datasets.loopwam_latent_cache import latent_cache_provenance

    root, cache, output = full_data
    _, _, manifest = build_full_libero_datasets(root, cache, output)
    _, _, repeated = build_full_libero_datasets(root, cache, output)
    assert manifest == repeated
    _, _, elsewhere = build_full_libero_datasets(root, cache, output.parent / "other")
    provenance = latent_cache_provenance(manifest, "vae")
    assert provenance == latent_cache_provenance(elsewhere, "vae")
    changed = {**manifest, "suites": list(reversed(SUITES))}
    assert provenance != latent_cache_provenance(changed, "vae")
    assert "loopwam_long.py" in provenance["implementation_sha256"]
