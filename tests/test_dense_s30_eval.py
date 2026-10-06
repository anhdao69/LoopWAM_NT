"""Dense-S30 evaluator accepts only its complete checkpoint architecture."""
import pytest
from scripts.evaluate_loopwam_libero import checkpoint_policy_spec, validate_checkpoint
from fastwam.models.wan22.loopwam_init import target_configs


def checkpoint():
    video, action = target_configs(30)
    data = dict(train_windows=92678, normalization_sha256='stats', normalization_source='training episodes only')
    contract = dict(world=2, microbatch=4, global_batch=128, seed=42, train_windows=92678,
                    planned_updates=7250, version='dense_s30', normalization_sha256='stats')
    payload = dict(format_version='loopwam-s-v1', version='dense_s30', step=7250,
        trained_max_loops=1, inference_loops=1,
        architecture=dict(target_video_config=video, target_action_config=action),
        training_state=dict(update=7250, epoch=9, next_micro=11585, windows_seen=926780, contract=contract))
    return payload, data


def test_complete_dense30_and_partial_smoke():
    payload, data = checkpoint()
    assert checkpoint_policy_spec(payload)==('dense_s30',1)
    validate_checkpoint(payload,data,'stats')
    payload['step']=payload['training_state']['update']=10
    with pytest.raises(ValueError,match='complete'):validate_checkpoint(payload,data,'stats')
    validate_checkpoint(payload,data,'stats',smoke=True)


@pytest.mark.parametrize('field', ['target_video_config','target_action_config'])
def test_rejects_truncated_or_misdeclared_dense30(field):
    payload, _ = checkpoint()
    payload['architecture'][field]['num_layers']=12
    with pytest.raises(ValueError,match='30-layer'):checkpoint_policy_spec(payload)


@pytest.mark.parametrize('field', ['trained_max_loops','inference_loops'])
def test_rejects_loop_depth_mismatch(field):
    payload, _ = checkpoint();payload[field]=4
    with pytest.raises(ValueError,match='depth'):checkpoint_policy_spec(payload)
