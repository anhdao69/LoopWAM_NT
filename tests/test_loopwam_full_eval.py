"""Full LIBERO evaluation requires every trained suite and all ten epochs."""
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest


SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_loopwam_libero.py"
    spec = importlib.util.spec_from_file_location("full_libero_rollout", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def full_contract():
    # 1025 windows need nine global batches per epoch, including the remainder.
    data = dict(train_windows=1025, normalization_sha256="stats",
                normalization_source="training episodes only", split="all_train",
                dataset_scope="full_libero", suites=SUITES.copy())
    contract = dict(world=4, microbatch=8, global_batch=128, seed=42,
                    train_windows=1025, planned_updates=90, version="v0",
                    normalization_sha256="stats", dataset_scope="full_libero",
                    epochs=10, suites=SUITES.copy())
    payload = dict(format_version="loopwam-s-v1", version="v0", step=90,
                   trained_max_loops=4, inference_loops=4,
                   training_state=dict(update=90, epoch=9, next_micro=33,
                                       windows_seen=10250, contract=contract))
    return payload, data


@pytest.mark.parametrize("suite", SUITES)
def test_accepts_complete_dynamic_budget_for_each_trained_suite(suite):
    payload, data = full_contract()
    contract = runner().validate_checkpoint(payload, data, "stats", suite=suite)
    assert contract["planned_updates"] == 90


@pytest.mark.parametrize("updates", [80, 81, 89, 91, 7250])
def test_rejects_budget_that_drops_or_combines_epoch_remainders(updates):
    payload, data = full_contract()
    payload["step"] = payload["training_state"]["update"] = updates
    payload["training_state"]["contract"]["planned_updates"] = updates
    with pytest.raises(ValueError, match="contract"):
        runner().validate_checkpoint(payload, data, "stats")


@pytest.mark.parametrize("field,value", [("epoch", 8), ("windows_seen", 10249), ("next_micro", 32)])
def test_rejects_final_checkpoint_with_incomplete_data_exposure(field, value):
    payload, data = full_contract()
    payload["training_state"][field] = value
    with pytest.raises(ValueError, match="complete|windows|last epoch"):
        runner().validate_checkpoint(payload, data, "stats")


@pytest.mark.parametrize("owner", ["data", "contract"])
@pytest.mark.parametrize("suites", [None, SUITES[:-1], SUITES + ["libero_90"], SUITES + ["libero_10"]])
def test_rejects_missing_or_wrong_suite_coverage_even_for_smoke(owner, suites):
    payload, data = full_contract()
    target = data if owner == "data" else payload["training_state"]["contract"]
    if suites is None:
        target.pop("suites")
    else:
        target["suites"] = suites
    with pytest.raises(ValueError, match="suite"):
        runner().validate_checkpoint(payload, data, "stats", smoke=True)


@pytest.mark.parametrize("owner,field,value", [
    ("data", "split", "train"), ("data", "dataset_scope", None),
    ("contract", "dataset_scope", None), ("contract", "epochs", 9),
    ("data", "train_windows", 0), ("data", "train_windows", True),
])
def test_rejects_inconsistent_full_training_provenance(owner, field, value):
    payload, data = full_contract()
    target = data if owner == "data" else payload["training_state"]["contract"]
    target[field] = value
    with pytest.raises(ValueError):
        runner().validate_checkpoint(payload, data, "stats", smoke=True)


def test_smoke_allows_partial_full_checkpoint_but_not_final_report():
    payload, data = full_contract()
    payload["step"] = payload["training_state"]["update"] = 1
    runner().validate_checkpoint(payload, data, "stats", smoke=True)
    with pytest.raises(ValueError, match="complete"):
        runner().validate_checkpoint(payload, data, "stats")


def test_legacy_long_checkpoint_cannot_evaluate_untrained_suite():
    payload, data = full_contract()
    for target in [data, payload["training_state"]["contract"]]:
        target.pop("dataset_scope")
        target.pop("suites")
    payload["training_state"]["contract"]["planned_updates"] = 7250
    with pytest.raises(ValueError, match="suite"):
        runner().validate_checkpoint(payload, data, "stats", smoke=True, suite="libero_goal")


@pytest.mark.parametrize("suite", SUITES)
def test_cli_preflight_uses_selected_suite_and_records_episode_identity(tmp_path, monkeypatch, suite):
    module = runner()
    class Environment:
        def reset(self): pass
        def set_init_state(self, state): return {}
        def step(self, action): return {}, 0, False, {}
        def check_success(self): return False
        def close(self): pass
    class Suite:
        def get_task(self, task_id): return types.SimpleNamespace(language=f"{suite}:{task_id}")
    benchmark = types.SimpleNamespace(get_benchmark_dict=lambda: {suite: Suite})
    monkeypatch.setitem(sys.modules, "libero.libero", types.SimpleNamespace(benchmark=benchmark))
    monkeypatch.setitem(sys.modules, "experiments.libero.libero_utils", types.SimpleNamespace(
        get_libero_env=lambda task, size, seed: (Environment(), task.language),
        get_libero_image=lambda obs: np.zeros((1, 1, 3)),
        save_rollout_video=lambda output, frames, name, success, description: str(output / f"{name}.mp4")))
    monkeypatch.setattr(module, "initial_states", lambda task: [None])
    monkeypatch.setattr(sys, "argv", ["evaluate", "--suite", suite, "--tasks", "0,9",
                                    "--simulator-preflight", "--output-dir", str(tmp_path)])
    module.main()
    report = json.loads((tmp_path / "simulator_preflight.json").read_text())
    assert report["suite"] == suite
    assert [(row["suite"], row["task_id"]) for row in report["episodes"]] == [(suite, 0), (suite, 9)]
