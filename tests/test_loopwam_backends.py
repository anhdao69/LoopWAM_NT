import pytest
import torch

from fastwam.training_backends import make_deepspeed_config, policy_parameters_fp32


@pytest.mark.parametrize('stage', [1, 2])
@pytest.mark.parametrize('microbatch', [2, 4, 8, 16, 32, 64])
def test_global_batch_and_precision_contract(stage, microbatch):
    config = make_deepspeed_config(stage, microbatch)
    assert config['train_micro_batch_size_per_gpu'] * config['gradient_accumulation_steps'] * 2 == 128
    assert config['train_batch_size'] == 128
    assert config['zero_optimization']['stage'] == stage
    assert config['fp16']['enabled'] is False and config['bf16']['enabled'] is False
    assert config['torch_autocast'] == {'enabled': True, 'dtype': 'bfloat16', 'lower_precision_safe_modules': []}
    assert config['data_types']['grad_accum_dtype'] == config['communication_data_type'] == 'fp32'
    assert config['gradient_clipping'] == 1.0


@pytest.mark.parametrize('kwargs', [dict(stage=3), dict(stage=True), dict(microbatch=3),
    dict(microbatch=0), dict(world_size=0), dict(bucket_size=-1), dict(gradient_clipping=float('nan'))])
def test_invalid_settings_fail_early(kwargs):
    values = dict(stage=1, microbatch=8)
    values.update(kwargs)
    with pytest.raises(ValueError):
        make_deepspeed_config(**values)


def test_optimizer_policy_coverage_and_frozen_bf16_module():
    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = torch.nn.Linear(2, 2)
            self.vae = torch.nn.Linear(2, 2).to(torch.bfloat16).requires_grad_(False)
        def policy_parameters(self):
            return list(self.policy.parameters())
    model = TinyPolicy()
    assert len(policy_parameters_fp32(model)) == 2
    model.alias = model.policy
    assert len(policy_parameters_fp32(model)) == 2
    assert model.vae.weight.dtype == torch.bfloat16
    model.policy_parameters = lambda: [model.policy.weight]
    with pytest.raises(ValueError, match='cover'):
        policy_parameters_fp32(model)
    model.policy_parameters = lambda: [model.policy.weight, model.policy.weight, model.policy.bias]
    with pytest.raises(ValueError, match='deduplicated'):
        policy_parameters_fp32(model)


def test_installed_deepspeed_accepts_configuration():
    # CPU-only config validation. World size one permits parsing without a
    # distributed accelerator; the production harness initializes NCCL first.
    deepspeed = pytest.importorskip('deepspeed')
    from deepspeed.runtime.config import DeepSpeedConfig
    for stage in (1, 2):
        config = DeepSpeedConfig(make_deepspeed_config(stage, 8, world_size=1))
        assert config.torch_autocast_enabled
        assert config.torch_autocast_dtype == torch.bfloat16
        assert config.grad_accum_dtype == 'fp32'
        assert config.communication_data_type == torch.float32
        assert config.zero_config.stage == stage
