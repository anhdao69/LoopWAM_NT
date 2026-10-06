"""Explicit episode splits and train-only normalization for the LoopWAM screen.

Both local LeRobot v2.1 and v3.0 layouts are supported. Source datasets are
never rewritten. The output manifest records the actual available demos;
the currently distributed FastWAM Long set contains 388, rather than 500.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import torch
from torchvision.transforms.functional import resize

from fastwam.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
CAMERAS = ("observation.images.image", "observation.images.wrist_image")


def split_episodes(rows, seed=42):
    """Sorted episode IDs, with a deterministic 90/10 split within each task."""
    by_task = defaultdict(list)
    for row in rows:
        if len(row["tasks"]) != 1:
            raise ValueError("The Long split requires exactly one task per episode")
        by_task[row["tasks"][0]].append(int(row["episode_index"]))
    train, validation, counts = [], [], {}
    for task, ids in sorted(by_task.items()):
        if len(ids) < 2 or len(set(ids)) != len(ids):
            raise ValueError(f"Task needs at least two distinct episodes: {task}")
        # Hash ordering is stable across Python/NumPy versions and input order.
        ids.sort(key=lambda ep: hashlib.sha256(f"{seed}:{task}:{ep}".encode()).digest())
        n_train = min(len(ids) - 1, max(1, (9 * len(ids)) // 10))
        train.extend(ids[:n_train])
        validation.extend(ids[n_train:])
        counts[task] = {"available": len(ids), "train": n_train,
                        "validation": len(ids) - n_train}
    if set(train) & set(validation):
        raise ValueError("Episode IDs are not globally unique")
    return sorted(train), sorted(validation), counts


def compute_train_stats(dataset_dir, episode_ids):
    """Global min/max from actual training rows, before any window padding."""
    paths = sorted((Path(dataset_dir) / "data").glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No data parquet files in {dataset_dir}")
    table = pads.dataset([str(p) for p in paths], format="parquet").to_table(
        columns=["episode_index", "action", "observation.state"],
        filter=pads.field("episode_index").isin(episode_ids))
    found = set(table["episode_index"].to_pylist())
    if found != set(episode_ids):
        raise ValueError("Training episode IDs do not match the parquet rows")
    stats = {"num_episodes": len(episode_ids), "num_transition": len(table)}
    for kind, column, dim in (("action", "action", 7), ("state", "observation.state", 8)):
        values = np.asarray(table[column].to_pylist(), dtype=np.float32)
        if values.shape != (len(table), dim) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {column} rows: {values.shape}")
        stats[kind] = {"default": {"global_min": values.min(0).tolist(),
                                   "global_max": values.max(0).tolist()}}
    return stats


class LoopWAMLongDataset(torch.utils.data.Dataset):
    """Apply the retained FastWAM transform to an explicitly selected reader."""

    def __init__(self, reader, text_cache_dir, stats):
        self.reader = reader
        self.text_cache_dir = Path(text_cache_dir)
        self.stats = stats
        self.normalizer = {
            kind: SingleFieldLinearNormalizer({
                key.removeprefix("global_"): torch.tensor(value, dtype=torch.float32)
                for key, value in stats[kind]["default"].items()}, mode="min/max")
            for kind in ("action", "state")}
        self._text_cache = {}

    def __len__(self):
        return len(self.reader)

    def denormalize_action(self, actions):
        norm = self.normalizer["action"]
        return (actions - norm.offset.to(actions.device)) / norm.scale.to(actions.device)

    def _text(self, prompt):
        if prompt not in self._text_cache:
            digest = hashlib.sha256(prompt.encode()).hexdigest()
            path = self.text_cache_dir / f"{digest}.t5_len128.wan22ti2v5b.pt"
            payload = torch.load(path, map_location="cpu", weights_only=True)
            context, mask = payload["context"], payload["mask"].bool()
            if context.shape != (128, 4096) or mask.shape != (128,) or not mask.any():
                raise ValueError(f"Invalid text cache: {path}")
            context = context.clone()
            context[~mask] = 0
            self._text_cache[prompt] = (context, mask)
        return self._text_cache[prompt]

    def __getitem__(self, index):
        # Decoder failures propagate; they must not silently change the manifest.
        row = self.reader[index]
        views = []
        for key in CAMERAS:
            frames = row[key]
            if frames.dtype == torch.uint8:
                frames = frames.float() / 255
            if frames.shape[:2] != (9, 3):
                raise ValueError(f"Expected nine CHW RGB frames for {key}: {frames.shape}")
            views.append(resize(frames, [224, 224], antialias=True))
        video = (torch.cat(views, dim=-1) * 2 - 1).permute(1, 0, 2, 3).contiguous()
        action_pad = row["action_is_pad"].bool()
        action = row["action"].clone().float()
        # Delta pose padding is zero; the absolute gripper retains its last value.
        action[action_pad, :6] = 0
        action = self.normalizer["action"].forward(action)
        proprio = self.normalizer["state"].forward(row["observation.state"].float())[:-1]
        prompt = DEFAULT_PROMPT.format(task=row["task"])
        context, context_mask = self._text(prompt)
        return {"video": video, "action": action, "proprio": proprio,
                "prompt": prompt, "context": context, "context_mask": context_mask,
                "action_is_pad": action_pad,
                "image_is_pad": row[f"{CAMERAS[0]}_is_pad"].bool(),
                "proprio_is_pad": row["observation.state_is_pad"].bool()[:-1],
                "episode_index": int(row["episode_index"]),
                "frame_index": int(row["frame_index"])}


def _persist_json(path, payload):
    text = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    try:
        with path.open("x") as stream:
            stream.write(text)
    except FileExistsError:
        if path.read_text() != text:
            raise ValueError(f"Refusing to overwrite a different data manifest: {path}")
    return hashlib.sha256(text.encode()).hexdigest()


def dataset_content_manifest(root):
    """Bind each run to the exact local parquet and compressed RGB video bytes."""
    root = Path(root)
    result = {}
    for directory, suffix in [('data', '*.parquet'), ('videos', '*.mp4')]:
        for path in sorted((root / directory).rglob(suffix)):
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            result[str(path.relative_to(root))] = {'bytes':path.stat().st_size, 'sha256':digest.hexdigest()}
    return result


def build_long_datasets(dataset_dir, text_cache_dir, output_dir, seed=42):
    """Return ``(train, validation, manifest)`` and save split/stat provenance.

    Call once on rank zero before other distributed ranks read the artifacts,
    or give each independent builder its own output directory.
    """
    root = Path(dataset_dir).resolve()
    output_dir = Path(output_dir)
    info_bytes = (root / "meta/info.json").read_bytes()
    info = json.loads(info_bytes)
    version = info["codebase_version"]
    if version == "v3.0":
        from fastwam.datasets.lerobot3.lerobot_dataset import LeRobotDataset
        paths = sorted((root / "meta/episodes").glob("*/*.parquet"))
        rows = pq.read_table([str(p) for p in paths],
                             columns=["episode_index", "tasks", "length"]).to_pylist()
    elif version == "v2.1":
        from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset
        rows = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines()]
    else:
        raise ValueError(f"Unsupported LeRobot version: {version}")
    if len(rows) != info["total_episodes"] or sum(r["length"] for r in rows) != info["total_frames"]:
        raise ValueError("Episode metadata does not match info.json")
    train_ids, val_ids, counts = split_episodes(rows, seed)
    stats = compute_train_stats(root, train_ids)
    fps = int(info["fps"])
    delta_timestamps = {key: [t / fps for t in range(0, 33, 4)] for key in CAMERAS}
    delta_timestamps.update({"action": [t / fps for t in range(32)],
                             "observation.state": [t / fps for t in range(33)]})
    datasets = []
    for ids in (train_ids, val_ids):
        reader = LeRobotDataset(str(root), root=root, episodes=ids,
                                delta_timestamps=delta_timestamps, video_backend="pyav")
        dataset = LoopWAMLongDataset(reader, text_cache_dir, stats)
        for task in counts:
            dataset._text(DEFAULT_PROMPT.format(task=task))
        datasets.append(dataset)
    train, validation = datasets
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_hash = _persist_json(output_dir / "dataset_stats.json", stats)
    cache_files = {}
    for task in counts:
        prompt = DEFAULT_PROMPT.format(task=task)
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        path = Path(text_cache_dir) / f"{digest}.t5_len128.wan22ti2v5b.pt"
        cache_files[path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                 "valid_tokens": int(train._text(prompt)[1].sum())}
    manifest = {
        "dataset_dir": str(root), "codebase_version": version, "seed": seed,
        "content_files": dataset_content_manifest(root),
        "info_sha256": hashlib.sha256(info_bytes).hexdigest(),
        "episode_metadata_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "split_algorithm": "SHA256(seed:task:episode_id) sorted, floor(0.9*n) train",
        "task_counts": counts, "train_episodes": train_ids, "validation_episodes": val_ids,
        "train_windows": len(train), "validation_windows": len(validation),
        "normalization_sha256": stats_hash, "normalization_source": "training episodes only",
        "normalization_path": str((output_dir / "dataset_stats.json").resolve()),
        "text_cache_dir": str(Path(text_cache_dir).resolve()),
        "text_cache_files": cache_files,
        "text_encoder_provenance": {
            "expected_encoder": "models_t5_umt5-xxl-enc-bf16.pth",
            "expected_encoder_sha256": "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d",
            "source": "Identical official Wan2.1-T2V-1.3B and Wan2.2-TI2V-5B Hub LFS hashes, audited 2026-10-05",
            "cache_payload_binds_encoder_hash": False,
            "limitation": "Existing context/mask-only caches have no embedded encoder or tokenizer provenance",
        },
        "camera_order": list(CAMERAS), "fps": fps,
        "video_offsets": list(range(0, 33, 4)), "action_offsets": list(range(32)),
        "padding_policy": "all anchors, repeat last frame, zero padded delta pose, retain gripper",
        "complete_500_demo_set": len(rows) == 500 and all(v["available"] == 50 for v in counts.values()),
    }
    _persist_json(output_dir / "data_manifest.json", manifest)
    return train, validation, manifest


FULL_LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def build_full_libero_datasets(dataset_root, text_cache_dir, output_dir, seed=42):
    """Train on every available demonstration in all four local v3.0 suites.

    Concatenation order is fixed by ``FULL_LIBERO_SUITES``. Its global indices
    identify latent-cache entries; episode IDs in the manifest are qualified by
    suite because the source datasets reuse their local episode numbers. The
    seed is recorded for training reproducibility and does not select episodes.
    No validation demonstrations are withheld in this all-training protocol.
    """
    from fastwam.datasets.lerobot3.lerobot_dataset import LeRobotDataset

    root = Path(dataset_root).resolve()
    output_dir = Path(output_dir)
    coverage, suite_stats, content_files, metadata_files = {}, [], {}, {}
    task_counts, train_episodes, info_hashes, episode_hashes = {}, [], {}, {}
    tasks_to_suites = defaultdict(list)
    fps = None
    for suite in FULL_LIBERO_SUITES:
        directory = root / f"{suite}_no_noops_lerobot"
        info_bytes = (directory / "meta/info.json").read_bytes()
        info = json.loads(info_bytes)
        if info["codebase_version"] != "v3.0":
            raise ValueError(f"Full LIBERO requires v3.0: {suite}")
        paths = sorted((directory / "meta/episodes").glob("*/*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No episode metadata in {directory}")
        rows = pq.read_table([str(p) for p in paths],
                             columns=["episode_index", "tasks", "length"]).to_pylist()
        rows.sort(key=lambda row: int(row["episode_index"]))
        ids = [int(row["episode_index"]) for row in rows]
        if not ids or len(set(ids)) != len(ids) or any(row["length"] <= 0 for row in rows):
            raise ValueError(f"Invalid or duplicate episode metadata: {suite}")
        if len(rows) != info["total_episodes"] or sum(row["length"] for row in rows) != info["total_frames"]:
            raise ValueError(f"Episode metadata does not match info.json: {suite}")
        counts = {}
        for row in rows:
            if len(row["tasks"]) != 1 or not isinstance(row["tasks"][0], str) or not row["tasks"][0].strip():
                raise ValueError(f"Exactly one nonempty task is required per episode: {suite}")
            task = row["tasks"][0]
            counts.setdefault(task, {"available": 0, "train": 0, "validation": 0})
            counts[task]["available"] += 1
            counts[task]["train"] += 1
        for task, count in sorted(counts.items()):
            task_counts[f"{suite}/{task}"] = count
            tasks_to_suites[task].append(suite)
        stats = compute_train_stats(directory, ids)
        if stats["num_transition"] != info["total_frames"]:
            raise ValueError(f"Training frame count does not match metadata: {suite}")
        suite_stats.append(stats)
        suite_fps = int(info["fps"])
        if suite_fps <= 0 or (fps is not None and fps != suite_fps):
            raise ValueError("All LIBERO suites must have the same positive fps")
        fps = suite_fps
        info_hashes[suite] = hashlib.sha256(info_bytes).hexdigest()
        episode_hashes[suite] = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        content_files.update({f"{directory.name}/{path}": value
                              for path, value in dataset_content_manifest(directory).items()})
        # Include task and video/episode indexing metadata used by the reader,
        # as well as the historical info/episode-summary hashes below.
        for path in sorted((directory / "meta").rglob("*")):
            if path.is_file():
                metadata_files[str(path.relative_to(root))] = {
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        start = sum(item["train_windows"] for item in coverage.values())
        coverage[suite] = {
            "dataset_dir": str(directory), "codebase_version": "v3.0",
            "available_episodes": len(ids), "available_windows": stats["num_transition"],
            "train_episodes": ids, "validation_episodes": [],
            "train_windows": stats["num_transition"], "validation_windows": 0,
            "global_index_start": start, "global_index_stop": start + stats["num_transition"],
            "task_counts": counts, "info_sha256": info_hashes[suite],
            "episode_metadata_sha256": episode_hashes[suite],
        }
        train_episodes.extend(f"{suite}:{episode}" for episode in ids)

    stats = {"num_episodes": sum(s["num_episodes"] for s in suite_stats),
             "num_transition": sum(s["num_transition"] for s in suite_stats)}
    for kind in ("action", "state"):
        stats[kind] = {"default": {
            "global_min": np.min([s[kind]["default"]["global_min"] for s in suite_stats], axis=0).tolist(),
            "global_max": np.max([s[kind]["default"]["global_max"] for s in suite_stats], axis=0).tolist()}}

    # Validate every task before persisting artifacts or opening video readers.
    text_loader = LoopWAMLongDataset([], text_cache_dir, stats)
    cache_files = {}
    for task, suites in sorted(tasks_to_suites.items()):
        prompt = DEFAULT_PROMPT.format(task=task)
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        path = Path(text_cache_dir) / f"{digest}.t5_len128.wan22ti2v5b.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing text cache for {suites}: {task!r}: {path}")
        context, mask = text_loader._text(prompt)
        if not torch.isfinite(context).all():
            raise ValueError(f"Nonfinite text cache for {suites}: {path}")
        cache_files[path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                 "valid_tokens": int(mask.sum())}
    delta_timestamps = {key: [t / fps for t in range(0, 33, 4)] for key in CAMERAS}
    delta_timestamps.update({"action": [t / fps for t in range(32)],
                             "observation.state": [t / fps for t in range(33)]})
    datasets = []
    for suite, details in coverage.items():
        directory = Path(details["dataset_dir"])
        reader = LeRobotDataset(str(directory), root=directory, episodes=details["train_episodes"],
                                delta_timestamps=delta_timestamps, video_backend="pyav")
        dataset = LoopWAMLongDataset(reader, text_cache_dir, stats)
        dataset._text_cache = text_loader._text_cache
        if len(dataset) != details["train_windows"]:
            raise ValueError(f"Reader windows do not cover every training frame: {suite}")
        datasets.append(dataset)
    train = torch.utils.data.ConcatDataset(datasets)
    validation = torch.utils.data.Subset(train, [])
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_hash = _persist_json(output_dir / "dataset_stats.json", stats)
    manifest = {
        "dataset_scope": "full_libero", "split": "all_train",
        "suites": list(FULL_LIBERO_SUITES), "coverage": coverage,
        "dataset_dir": str(root), "codebase_version": "v3.0", "seed": seed,
        "available_episodes": stats["num_episodes"], "available_windows": stats["num_transition"],
        "content_files": content_files, "metadata_files": metadata_files,
        "info_sha256": hashlib.sha256(json.dumps(info_hashes, sort_keys=True).encode()).hexdigest(),
        "episode_metadata_sha256": hashlib.sha256(json.dumps(episode_hashes, sort_keys=True).encode()).hexdigest(),
        "split_algorithm": "all available episodes; fixed suite order and ascending local episode IDs",
        "task_counts": task_counts, "train_episodes": train_episodes, "validation_episodes": [],
        "train_windows": len(train), "validation_windows": 0,
        "normalization_sha256": stats_hash, "normalization_source": "training episodes only",
        "normalization_scope": "global across all four LIBERO suites",
        "normalization_path": str((output_dir / "dataset_stats.json").resolve()),
        "text_cache_dir": str(Path(text_cache_dir).resolve()), "text_cache_files": cache_files,
        "text_encoder_provenance": {
            "expected_encoder": "models_t5_umt5-xxl-enc-bf16.pth",
            "expected_encoder_sha256": "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d",
            "source": "Identical official Wan2.1-T2V-1.3B and Wan2.2-TI2V-5B Hub LFS hashes, audited 2026-10-05",
            "cache_payload_binds_encoder_hash": False,
            "limitation": "Existing context/mask-only caches have no embedded encoder or tokenizer provenance",
        },
        "camera_order": list(CAMERAS), "fps": fps,
        "video_offsets": list(range(0, 33, 4)), "action_offsets": list(range(32)),
        "padding_policy": "all anchors, repeat last frame, zero padded delta pose, retain gripper",
        "complete_500_demo_set": all(len(details["train_episodes"]) == 500 and
            all(count["available"] == 50 for count in details["task_counts"].values())
            for details in coverage.values()),
    }
    _persist_json(output_dir / "data_manifest.json", manifest)
    return train, validation, manifest
