"""Protect simulator evaluation against silently invalid reported success rates."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_loopwam_libero.py"
    assert path.exists(), "Standalone rollout runner is missing"
    spec = importlib.util.spec_from_file_location("loopwam_rollout", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contract_fixture():
    data = dict(train_windows=92700, normalization_sha256="stats", normalization_source="training episodes only")
    contract = dict(world=2, microbatch=8, global_batch=128, seed=42,
                    train_windows=92700, planned_updates=7250, version="v1", normalization_sha256="stats")
    checkpoint = dict(format_version="loopwam-s-v1", version="v1", step=7250, trained_max_loops=4, inference_loops=4,
                      training_state=dict(update=7250, epoch=9, next_micro=5794, windows_seen=927000,
                                          contract=contract))
    return checkpoint, data


def test_rejects_partial_or_wrong_version_checkpoint_but_allows_explicit_smoke():
    m = runner()
    checkpoint, data = contract_fixture()
    m.validate_checkpoint(checkpoint, data, "stats")
    partial = copy.deepcopy(checkpoint)
    partial["step"] = partial["training_state"]["update"] = 12
    with pytest.raises(ValueError, match="complete"):
        m.validate_checkpoint(partial, data, "stats")
    m.validate_checkpoint(partial, data, "stats", smoke=True)
    checkpoint["version"] = "v0"
    with pytest.raises(ValueError, match="contract"):
        m.validate_checkpoint(checkpoint, data, "stats", smoke=True)


def test_rejects_stats_mismatch_and_incomplete_data_exposure():
    m = runner()
    checkpoint, data = contract_fixture()
    with pytest.raises(ValueError, match="normalization"):
        m.validate_checkpoint(checkpoint, data, "different")
    checkpoint["training_state"]["windows_seen"] -= 1
    with pytest.raises(ValueError, match="windows"):
        m.validate_checkpoint(checkpoint, data, "stats")


def test_workers_cover_each_requested_task_exactly_once():
    m = runner()
    assert m.shard_tasks([0, 2, 3, 7, 9], 0, 2) == [0, 3, 9]
    assert m.shard_tasks([0, 2, 3, 7, 9], 1, 2) == [2, 7]
    with pytest.raises(ValueError):
        m.shard_tasks([0, 0], 0, 2)


def test_train_minmax_state_and_libero_gripper_conversion():
    m = runner()
    stats = {kind: {"default": {"global_min": [-2.] * dim, "global_max": [2.] * dim}}
             for kind, dim in (("state", 8), ("action", 7))}
    stats["state"]["default"]["global_min"][-1] = 3.
    stats["state"]["default"]["global_max"][-1] = 3.
    stats["action"]["default"]["global_min"][-1] = 0.
    stats["action"]["default"]["global_max"][-1] = 1.
    adapter = m.ObservationAdapter(stats, ".")
    state = adapter.normalize_state(np.array([2., -2., 0., 1., -1., 100., 0., 3.]))
    torch.testing.assert_close(state, torch.tensor([1., -1., 0., .5, -.5, 5., 0., 0.]))
    actions = torch.zeros(32, 7)
    actions[0, 0] = .5
    actions[:2, -1] = torch.tensor([-1., 1.])
    actions[2:5, -1] = torch.tensor([-.02, 0., .02])
    converted = adapter.libero_actions(actions)
    np.testing.assert_allclose(converted[0], [1., 0., 0., 0., 0., 0., 1.])
    assert converted[1, -1] == -1.
    np.testing.assert_array_equal(converted[2:5, -1], [1., 1., -1.])
    assert torch.equal(actions[:2, -1], torch.tensor([-1., 1.]))


def test_episode_stops_at_terminal_and_never_executes_unused_chunk_actions():
    m = runner()
    class Environment:
        def __init__(self): self.actions = []
        def reset(self): pass
        def set_init_state(self, state): return {"step": 0}
        def step(self, action):
            self.actions.append(action)
            return {"step": len(self.actions)}, 0., len(self.actions) == 3, {}
        def check_success(self): return len(self.actions) == 3
    env = Environment()
    result, frames = m.run_episode(env, None, lambda obs, seed: np.ones((32, 7)),
                                   seed=42, max_steps=700, wait_steps=0,
                                   frame_fn=lambda obs: obs["step"])
    assert result["success"] and result["steps"] == 3 and result["replans"] == 1
    assert len(env.actions) == 3 and frames == [0, 1, 2, 3]


def test_timeout_is_failure_and_replans_after_ten_actions():
    m = runner()
    class Environment:
        def __init__(self): self.steps = 0
        def reset(self): pass
        def set_init_state(self, state): return {}
        def step(self, action): self.steps += 1; return {}, 0., False, {}
        def check_success(self): return False
    seeds = []
    def policy(obs, seed): seeds.append(seed); return np.zeros((32, 7))
    result, _ = m.run_episode(Environment(), None, policy, seed=10, max_steps=21,
                               wait_steps=5, frame_fn=lambda obs: None)
    assert not result["success"] and result["steps"] == 21
    assert result["wait_steps"] == 5 and seeds == [10, 11, 12]


def test_text_cache_preserves_padding_mask_and_rejects_modified_training_asset(tmp_path):
    m = runner()
    from fastwam.datasets.loopwam_long import DEFAULT_PROMPT
    task = "put the cup down"
    digest = hashlib.sha256(DEFAULT_PROMPT.format(task=task).encode()).hexdigest()
    name = f"{digest}.t5_len128.wan22ti2v5b.pt"
    path = tmp_path / name
    mask = torch.arange(128) < 5
    torch.save({"context": torch.ones(128, 4096), "mask": mask}, path)
    stats = {kind: {"default": {"global_min": [-1.] * dim, "global_max": [1.] * dim}}
             for kind, dim in (("state", 8), ("action", 7))}
    adapter = m.ObservationAdapter(stats, tmp_path, {name: {"sha256": m.sha256_file(path)}})
    context, actual_mask = adapter.text(task)
    assert torch.equal(actual_mask, mask)
    assert torch.all(context[:5] == 1) and torch.all(context[5:] == 0)
    torch.save({"context": torch.zeros(128, 4096), "mask": mask}, path)
    with pytest.raises(ValueError, match="Text cache"):
        adapter.text(task)


def test_camera_rotation_order_and_current_eight_dimensional_proprio():
    m = runner()
    stats = {kind: {"default": {"global_min": [-1.] * dim, "global_max": [1.] * dim}}
             for kind, dim in (("state", 8), ("action", 7))}
    adapter = m.ObservationAdapter(stats, ".")
    primary = np.zeros((256, 256, 3), dtype=np.uint8)
    primary[:128] = 255
    obs = dict(agentview_image=primary, robot0_eye_in_hand_image=np.zeros_like(primary),
               robot0_eef_pos=np.array([.1, .2, .3]), robot0_eef_quat=np.array([0., 0., 0., 1.]),
               robot0_gripper_qpos=np.array([.04, -.04]))
    image, proprio = adapter.observation(obs)
    assert image.shape == (3, 224, 448)
    torch.testing.assert_close(image[:, 0, :224], -torch.ones(3, 224))
    torch.testing.assert_close(image[:, -1, :224], torch.ones(3, 224))
    assert torch.all(image[:, :, 224:] == -1)
    torch.testing.assert_close(proprio, torch.tensor([.1, .2, .3, 0., 0., 0., .04, -.04]))


def test_terminal_during_settling_never_calls_policy_or_counts_terminal_as_success():
    m = runner()
    class Environment:
        def reset(self): pass
        def set_init_state(self, state): return {}
        def step(self, action): return {}, 0., True, {}
        def check_success(self): return False
    def predict(obs, seed): raise AssertionError("Terminal environment cannot run policy")
    result, _ = m.run_episode(Environment(), None, predict, seed=0, frame_fn=lambda obs: None)
    assert result["terminal"] and not result["success"]
    assert result["wait_steps"] == 1 and result["steps"] == 0


@pytest.mark.parametrize("version,loops", [("dense_s12", 1), ("v0", 4), ("v1", 4), ("v2", 4)])
@pytest.mark.parametrize("world,next_micro", [(1, 11588), (2, 5794), (4, 2897)])
def test_completed_models_use_checkpoint_version_and_training_world(version, loops, world, next_micro):
    m = runner()
    checkpoint, data = contract_fixture()
    checkpoint.update(version=version, trained_max_loops=loops, inference_loops=loops)
    checkpoint["training_state"]["contract"].update(version=version, world=world)
    checkpoint["training_state"]["next_micro"] = next_micro
    contract = m.validate_checkpoint(checkpoint, data, "stats")
    assert contract["version"] == version and contract["world"] == world
    assert m.checkpoint_policy_spec(checkpoint) == (version, loops)
    checkpoint["training_state"]["next_micro"] -= 1
    with pytest.raises(ValueError, match="last epoch"):
        m.validate_checkpoint(checkpoint, data, "stats")


@pytest.mark.parametrize("version,trained,inference", [
    ("dense_s12", 4, 4), ("dense_s12", 1, 4), ("v2", 1, 1),
    ("v2", 4, 1), ("unknown", 4, 4),
])
def test_rejects_incompatible_checkpoint_depth_even_in_smoke(version, trained, inference):
    m = runner()
    checkpoint, data = contract_fixture()
    checkpoint.update(version=version, trained_max_loops=trained, inference_loops=inference)
    checkpoint["training_state"]["contract"]["version"] = version
    with pytest.raises(ValueError):
        m.validate_checkpoint(checkpoint, data, "stats", smoke=True)


@pytest.mark.parametrize("world,microbatch", [(0, 8), (-1, 8), (4, 0), (3, 8), (4, 64)])
def test_rejects_invalid_training_parallelism_even_in_smoke(world, microbatch):
    m = runner()
    checkpoint, data = contract_fixture()
    checkpoint["training_state"]["contract"].update(world=world, microbatch=microbatch)
    with pytest.raises(ValueError, match="contract"):
        m.validate_checkpoint(checkpoint, data, "stats", smoke=True)


def test_four_rank_smoke_and_final_rollouts_cover_every_gpu_and_task():
    m = runner()
    assert [m.shard_tasks([0, 1, 2, 3], rank, 4) for rank in range(4)] == [[0], [1], [2], [3]]
    assert [m.shard_tasks(list(range(10)), rank, 4) for rank in range(4)] == [
        [0, 4, 8], [1, 5, 9], [2, 6], [3, 7]]
