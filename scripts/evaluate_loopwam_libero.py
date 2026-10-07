#!/usr/bin/env python3
"""Final Dense-S12 / LoopWAM LIBERO rollouts, independently task-sharded by torchrun.

Example: torchrun --standalone --nproc_per_node=2 scripts/evaluate_loopwam_libero.py
  --checkpoint RUN/latest.pt --vae-path CHECKPOINTS/Wan2.1_VAE.pth --output-dir RUN/rollouts
Use --simulator-preflight for a CPU-only simulator/render/video dependency check.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def checkpoint_policy_spec(payload):
    """Use the checkpoint's declared architecture and full trained depth."""
    version = payload.get("version")
    if (payload.get("format_version") != "loopwam-s-v1"
            or version not in {"dense_s30", "dense_s12", "v0", "v1", "v2"}):
        raise ValueError("Expected a supported dense or LoopWAM student checkpoint")
    loops = payload.get("trained_max_loops")
    expected_loops = 1 if version in {"dense_s12", "dense_s30"} else 4
    if (type(loops) is not int or type(payload.get("inference_loops")) is not int
            or loops != expected_loops or payload.get("inference_loops") != loops):
        raise ValueError("Checkpoint depth differs from its approved architecture")
    if version == "dense_s30":
        from fastwam.models.wan22.loopwam_init import target_configs
        video, action = target_configs(30)
        architecture = payload.get("architecture", {})
        if (architecture.get("target_video_config") != video
                or architecture.get("target_action_config") != action):
            raise ValueError("Dense-S30 checkpoint requires both complete 30-layer target configs")
    from fastwam.models.wan22.loopwam import _validate_checkpoint_depth
    _validate_checkpoint_depth(payload)
    return version, loops


def validate_checkpoint(payload, data, stats_hash, *, smoke=False, suite="libero_10"):
    """Incomplete checkpoints may only produce explicitly labelled smoke results."""
    version, _ = checkpoint_policy_spec(payload)
    state = payload.get("training_state") or {}
    contract = state.get("contract") or {}
    if suite not in LIBERO_SUITES:
        raise ValueError(f"Unsupported evaluation suite: {suite}")
    full_libero = (data.get("dataset_scope") == "full_libero"
                   or contract.get("dataset_scope") == "full_libero")
    planned_updates = 7250
    if full_libero:
        if (data.get("dataset_scope") != "full_libero"
                or contract.get("dataset_scope") != "full_libero"
                or data.get("split") != "all_train" or contract.get("epochs") != 10):
            raise ValueError("Full LIBERO training contract requires all_train and ten epochs")
        for source in (data, contract):
            suites = source.get("suites")
            if (not isinstance(suites, list) or len(suites) != len(LIBERO_SUITES)
                    or any(name not in suites for name in LIBERO_SUITES)):
                raise ValueError("Full LIBERO suite coverage must include all four trained suites")
        train_windows = data.get("train_windows")
        if type(train_windows) is not int or train_windows < 1:
            raise ValueError("Full LIBERO training contract requires positive train_windows")
        planned_updates = math.ceil(train_windows / 128) * 10
    elif suite != "libero_10":
        raise ValueError("Legacy Long checkpoint does not cover the requested suite")
    if (data.get("normalization_source") != "training episodes only"
            or stats_hash != data.get("normalization_sha256")
            or stats_hash != contract.get("normalization_sha256")):
        raise ValueError("Training normalization provenance/hash mismatch")
    world = contract.get("world")
    microbatch = contract.get("microbatch")
    valid_batch = (type(world) is int and world > 0
                   and type(microbatch) is int and microbatch > 0
                   and 128 % (world * microbatch) == 0)
    if (contract.get("version") != version or contract.get("planned_updates") != planned_updates
            or contract.get("global_batch") != 128 or not valid_batch
            or contract.get("train_windows") != data.get("train_windows")
            or contract.get("seed") != 42):
        raise ValueError("Checkpoint training contract differs from the approved run")
    if payload.get("action_loops", payload.get("inference_loops")) != payload.get("inference_loops"):
        if (contract.get("video_loops") != payload.get("video_loops")
                or contract.get("action_loops") != payload.get("action_loops")
                or contract.get("loop_alignment") != "late"):
            raise ValueError("Asymmetric training and checkpoint loop contracts differ")
    if payload.get("step") != state.get("update"):
        raise ValueError("Checkpoint step and training update disagree")
    if smoke:
        return contract
    if payload.get("step") != planned_updates or state.get("epoch") != 9:
        raise ValueError(f"Final evaluation requires complete {planned_updates}-update, 10-epoch training")
    if state.get("windows_seen") != 10 * data["train_windows"]:
        raise ValueError("Final training windows_seen does not cover ten complete epochs")
    if state.get("next_micro") != math.ceil(data["train_windows"] / (world * microbatch)):
        raise ValueError("Final checkpoint does not complete the last epoch")
    return contract


def shard_tasks(task_ids, rank, world):
    if world < 1 or not 0 <= rank < world or len(task_ids) != len(set(task_ids)):
        raise ValueError("Invalid rank/world or duplicate tasks")
    if not task_ids or any(task < 0 or task >= 10 for task in task_ids):
        raise ValueError("LIBERO suite task IDs must be in 0..9")
    return list(task_ids[rank::world])


class ObservationAdapter:
    def __init__(self, stats, text_cache_dir, text_files=None):
        # Reuse the exact training normalization and cached-T5 reader.
        from fastwam.datasets.loopwam_long import LoopWAMLongDataset
        self.dataset = LoopWAMLongDataset([], text_cache_dir, stats)
        self.text_files = text_files
        for kind, dim in (("action", 7), ("state", 8)):
            for key in ("global_min", "global_max"):
                value = np.asarray(stats[kind]["default"][key])
                if value.shape != (dim,) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid {kind} normalization statistics")

    def normalize_state(self, state):
        value = torch.as_tensor(state, dtype=torch.float32)
        if value.shape != (8,) or not torch.isfinite(value).all():
            raise ValueError("Expected finite current 8D proprioception")
        return self.dataset.normalizer["state"].forward(value)

    def libero_actions(self, action):
        action = action.detach().to(device="cpu", dtype=torch.float32)
        if action.shape != (32, 7) or not torch.isfinite(action).all():
            raise ValueError("Expected finite 32x7 predicted actions")
        action = self.dataset.denormalize_action(action).numpy().copy()
        # Raw LeRobot gripper is 0=closed, 1=open (also documented by the
        # existing evaluator). Audit of all 104280 parquet rows: mean absolute
        # finger qpos is [.01754,.01827] for g=0, [.03752,.03756] for g=1.
        # LIBERO uses +1=closed, -1=open. Exactly .5 chooses closed.
        action[:, -1] = np.where(action[:, -1] > .5, -1., 1.)
        return action

    def text(self, task):
        from fastwam.datasets.loopwam_long import DEFAULT_PROMPT
        prompt = DEFAULT_PROMPT.format(task=task)
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        name = f"{digest}.t5_len128.wan22ti2v5b.pt"
        if self.text_files is not None:
            expected = self.text_files.get(name, {}).get("sha256")
            if expected is None or sha256_file(self.dataset.text_cache_dir / name) != expected:
                raise ValueError(f"Text cache differs from training provenance: {name}")
        return self.dataset._text(prompt)

    def observation(self, obs):
        from torchvision.transforms.functional import resize
        from experiments.libero.libero_utils import get_libero_image, quat2axisangle
        images = get_libero_image(obs)
        views = [resize(torch.from_numpy(images[key]).permute(2, 0, 1).float() / 255,
                        [224, 224], antialias=True) for key in ("image", "wrist_image")]
        image = (torch.cat(views, dim=-1) * 2 - 1).contiguous()
        state = np.concatenate([obs["robot0_eef_pos"],
                                quat2axisangle(np.array(obs["robot0_eef_quat"], copy=True)),
                                obs["robot0_gripper_qpos"]])
        return image, self.normalize_state(state)


def run_episode(env, initial_state, predict, *, seed, max_steps=700, wait_steps=30,
                replan_steps=10, frame_fn=None):
    """Respect terminal states, including during settling; never step after done."""
    if not 1 <= replan_steps <= 32 or max_steps < 1 or wait_steps < 0:
        raise ValueError("Invalid rollout horizon or settling steps")
    if frame_fn is None:
        from experiments.libero.libero_utils import get_libero_image
        frame_fn = get_libero_image
    env.reset()
    obs = env.set_init_state(initial_state)
    frames = [frame_fn(obs)]
    steps = settled = replans = 0
    done = False
    success = False
    for _ in range(wait_steps):
        obs, _, done, _ = env.step([0., 0., 0., 0., 0., 0., -1.])
        settled += 1
        frames.append(frame_fn(obs))
        success = bool(env.check_success())
        if done or success:
            break
    pending = []
    while steps < max_steps and not done and not success:
        if not pending:
            actions = np.asarray(predict(obs, seed + replans))
            if actions.shape != (32, 7) or not np.isfinite(actions).all():
                raise ValueError("Policy returned invalid action chunk")
            pending = actions[:replan_steps].tolist()
            replans += 1
        obs, _, done, _ = env.step(pending.pop(0))
        steps += 1
        frames.append(frame_fn(obs))
        success = bool(env.check_success())
    return dict(success=success, terminal=bool(done), steps=steps, wait_steps=settled,
                replans=replans, seed=seed, timeout=steps == max_steps and not success), frames


def initial_states(task):
    from libero.libero import get_libero_path
    path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    # LIBERO ships trusted NumPy arrays in these torch archives.
    return torch.load(path, map_location="cpu", weights_only=False)


def simulator_preflight(output, tasks, seed, suite_name="libero_10"):
    from libero.libero import benchmark
    from experiments.libero.libero_utils import get_libero_env, save_rollout_video
    suite = benchmark.get_benchmark_dict()[suite_name]()
    results = []
    output.mkdir(parents=True, exist_ok=True)
    for task_id in tasks:
        env, description = get_libero_env(suite.get_task(task_id), 256, seed)
        try:
            result, frames = run_episode(env, initial_states(suite.get_task(task_id))[0],
                lambda obs, seed: np.tile([0., 0., 0., 0., 0., 0., -1.], (32, 1)),
                seed=seed, max_steps=2, wait_steps=1)
            result["video"] = save_rollout_video(output, frames, f"preflight_task{task_id}",
                                                   result["success"], description)
            result.update(suite=suite_name, task_id=task_id)
            results.append(result)
        finally:
            env.close()
    write_json(output / "simulator_preflight.json", {
        "mode": "simulator_preflight", "suite": suite_name, "episodes": results})
    print(json.dumps({"event": "simulator_preflight_complete", "output": str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--suite", choices=LIBERO_SUITES, default="libero_10")
    parser.add_argument("--vae-path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", help="Training run data manifest/stats directory; defaults beside checkpoint")
    parser.add_argument("--stats", help="Training dataset_stats.json; its directory also supplies data_manifest.json")
    parser.add_argument("--text-cache-dir", help="Defaults to the training manifest cache directory")
    parser.add_argument("--tasks", "--task-ids", default=",".join(map(str, range(10))))
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--wait-steps", type=int, default=30)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--simulator-preflight", action="store_true")
    args = parser.parse_args()
    tasks = [int(value) for value in args.tasks.split(",")]
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    assigned = shard_tasks(tasks, rank, world)
    if args.episodes_per_task < 1 or args.max_steps < 1 or not 0 <= args.wait_steps or not 1 <= args.replan_steps <= 32:
        parser.error("Invalid episode count, step budget or action replan horizon")
    output = Path(args.output_dir).resolve()
    if args.simulator_preflight:
        if world != 1:
            parser.error("Run simulator preflight as a single CPU process")
        simulator_preflight(output, tasks, args.seed, args.suite)
        return
    if not args.checkpoint or not args.vae_path:
        parser.error("--checkpoint and --vae-path are required for policy rollouts")
    if not args.smoke and (args.max_steps != 700 or args.wait_steps != 30 or args.replan_steps != 10):
        parser.error("Final evaluation uses 700 steps, 30 settling steps, and replan every 10 actions")
    checkpoint = Path(args.checkpoint).resolve()
    data_dir = Path(args.data_dir) if args.data_dir else (Path(args.stats).parent if args.stats else checkpoint.parent / "data")
    data = json.loads((data_dir / "data_manifest.json").read_text())
    stats_path = Path(args.stats) if args.stats else data_dir / "dataset_stats.json"
    stats_hash = sha256_file(stats_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    contract = validate_checkpoint(payload, data, stats_hash, smoke=args.smoke, suite=args.suite)
    version, loops = checkpoint_policy_spec(payload)
    checkpoint_step = payload["step"]
    del payload
    train_manifest = json.loads((checkpoint.parent / "manifest.json").read_text())
    vae_hash = sha256_file(args.vae_path)
    if vae_hash != train_manifest["asset_sha256"]["vae"]:
        raise ValueError("VAE differs from the one used during training")
    checkpoint_hash = sha256_file(checkpoint)
    if not args.smoke:
        timing = json.loads((checkpoint.parent / "timing.json").read_text())
        if timing.get("status") != "complete":
            raise ValueError("Training run has not reported complete status")
    import torch.distributed as dist
    if world > 1:
        dist.init_process_group("gloo", timeout=timedelta(hours=2))
    try:
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            if list(output.glob("rank*.json")) or (output / "manifest.json").exists():
                raise ValueError("Output already contains a rollout run; choose a fresh directory")
            versions = {}
            for package in ("torch", "torchvision", "numpy", "mujoco", "robosuite"):
                try:
                    versions[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    versions[package] = "unknown"
            manifest = dict(mode="smoke" if args.smoke else "final_rollout", suite=args.suite,
                checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash, checkpoint_step=checkpoint_step,
                version=version, loops=loops, video_loops=loops, action_loops=payload.get('action_loops',loops), inference_steps=10, cfg=1.0, action_chunk=32,
                protocol=dict(max_policy_steps=args.max_steps, settling_steps=args.wait_steps,
                              replan_steps=args.replan_steps, camera_resolution=256,
                              model_camera_size=[224, 224], concatenation="horizontal",
                              horizon_policy="700 policy steps and 30 settling steps for every suite"),
                normalization_sha256=stats_hash, vae_sha256=vae_hash, training_contract=contract,
                world_size=world, task_assignment={str(r): shard_tasks(tasks, r, world) for r in range(world)},
                arguments=vars(args), package_versions=versions, cuda=torch.version.cuda,
                git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                script_sha256=sha256_file(__file__), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                gripper_convention="dataset 0=closed, 1=open; LIBERO +1=closed, -1=open; threshold >0.5 opens",
                seed_formula="seed + task_id*100000 + episode_index*1000 + replan_index",
                text_cache_files=data["text_cache_files"])
            write_json(output / "manifest.json", manifest)
        if world > 1:
            dist.barrier()
        torch.cuda.set_device(local)
        torch.set_num_threads(4)
        from fastwam.models.wan22.loopwam import create_loopwam
        from libero.libero import benchmark
        from experiments.libero.libero_utils import get_libero_env, save_rollout_video
        adapter = ObservationAdapter(json.loads(stats_path.read_text()),
            args.text_cache_dir or data["text_cache_dir"], data["text_cache_files"])
        model = create_loopwam(checkpoint_path=str(checkpoint), vae_path=args.vae_path,
            version=version, loops=loops, device=f"cuda:{local}", model_dtype=torch.float32).eval()
        suite = benchmark.get_benchmark_dict()[args.suite]()
        episodes = []
        video_dir = output / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        for task_id in assigned:
            task = suite.get_task(task_id)
            context, mask = adapter.text(task.language)
            states = initial_states(task)
            if len(states) < args.episodes_per_task:
                raise ValueError("Requested more episodes than distinct LIBERO initial states")
            env, description = get_libero_env(task, 256, args.seed)
            try:
                def predict(obs, sampler_seed):
                    image, proprio = adapter.observation(obs)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        action = model.infer_action(input_image=image, proprio=proprio,
                            context=context, context_mask=mask, action_horizon=32,
                            num_inference_steps=10, text_cfg_scale=1.0, seed=sampler_seed)["action"]
                    return adapter.libero_actions(action)
                for episode in range(args.episodes_per_task):
                    seed = args.seed + task_id * 100000 + episode * 1000
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    env.seed(seed)
                    started = time.perf_counter()
                    result, frames = run_episode(env, states[episode], predict, seed=seed,
                        max_steps=args.max_steps, wait_steps=args.wait_steps, replan_steps=args.replan_steps)
                    result.update(suite=args.suite, task_id=task_id, episode_index=episode, initial_state_index=episode,
                        task_description=description, rank=rank, checkpoint_sha256=checkpoint_hash,
                        duration_seconds=time.perf_counter() - started, mode="smoke" if args.smoke else "final_rollout")
                    result["video"] = save_rollout_video(video_dir, frames, f"task{task_id}_trial{episode}",
                                                           result["success"], description)
                    write_json(output / "episodes" / f"task{task_id}_episode{episode}.json", result)
                    episodes.append(result)
                    print(json.dumps(dict(event="episode_complete", **result)), flush=True)
            finally:
                env.close()
        write_json(output / f"rank{rank}.json", {"rank": rank, "suite": args.suite, "checkpoint_sha256": checkpoint_hash,
                                                   "episodes": episodes})
        if world > 1:
            dist.barrier()
        if rank == 0:
            combined = []
            for worker in range(world):
                worker_result = json.loads((output / f"rank{worker}.json").read_text())
                if worker_result["checkpoint_sha256"] != checkpoint_hash:
                    raise ValueError("Workers evaluated different checkpoints")
                if (worker_result.get("suite") != args.suite
                        or any(row.get("suite") != args.suite for row in worker_result["episodes"])):
                    raise ValueError("Workers evaluated different suites")
                combined.extend(worker_result["episodes"])
            expected = {(task, episode) for task in tasks for episode in range(args.episodes_per_task)}
            actual = [(row["task_id"], row["episode_index"]) for row in combined]
            if len(actual) != len(expected) or set(actual) != expected:
                raise ValueError("Worker results have missing or duplicate episodes")
            successes = sum(row["success"] for row in combined)
            summary = dict(mode="smoke" if args.smoke else "final_rollout", suite=args.suite,
                checkpoint_sha256=checkpoint_hash, version=version, loops=loops, action_loops=payload.get('action_loops',loops), checkpoint_step=checkpoint_step,
                total_episodes=len(combined), successes=successes, success_rate=successes / len(combined),
                per_task={str(task): dict(episodes=args.episodes_per_task,
                    successes=sum(row["success"] for row in combined if row["task_id"] == task),
                    success_rate=sum(row["success"] for row in combined if row["task_id"] == task) / args.episodes_per_task)
                    for task in tasks}, episodes=sorted(combined, key=lambda row: (row["task_id"], row["episode_index"])))
            write_json(output / "summary.json", summary)
            print(json.dumps({"event": "rollouts_complete", "successes": successes,
                              "episodes": len(combined), "output": str(output)}), flush=True)
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
