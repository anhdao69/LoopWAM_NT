from types import SimpleNamespace

from scripts.train_loopwam import training_contract


def test_full_training_contract_binds_scope_and_complete_budget():
    args = SimpleNamespace(microbatch=8, global_batch=128, seed=42, version='v0', epochs=10)
    data = dict(normalization_sha256='abc', dataset_scope='full_libero',
                suites=['libero_spatial', 'libero_object', 'libero_goal', 'libero_10'])
    contract = training_contract(args, 4, 277713, 21700, data)
    assert contract['dataset_scope'] == 'full_libero'
    assert contract['epochs'] == 10
    assert contract['suites'] == data['suites']
    assert contract['planned_updates'] == 21700
    assert contract['train_windows'] == 277713


def test_legacy_checkpoint_contract_preserved():
    args = SimpleNamespace(microbatch=8, global_batch=128, seed=42, version='v0', epochs=10)
    contract = training_contract(args, 2, 92678, 7250, {'normalization_sha256': 'abc'})
    assert contract == dict(world=2, microbatch=8, global_batch=128, seed=42,
                           version='v0', train_windows=92678, planned_updates=7250,
                           normalization_sha256='abc')
