import importlib.util
import torch

def test_cached_dataset_implemented():
    assert importlib.util.find_spec("fastwam.datasets.transport_cached") is not None

def test_random_streams_independent_of_batch_layout():
    assert importlib.util.find_spec("train_transport") is not None
    from train_transport import sample_noise
    a=sample_noise([9,4,6,1],epoch=2,seed=42)
    b=[sample_noise(ids,epoch=2,seed=42) for ids in ([9,4],[6,1])]
    for i in range(3): assert torch.equal(a[i],torch.cat([x[i] for x in b]))

def test_experiment_gates_preserved():
    assert importlib.util.find_spec("transport_experiments") is not None
    from transport_experiments import EXPERIMENTS,eligible
    assert {k for k in EXPERIMENTS if eligible(k,{})}=={"RT-A","RT-B4"}
    assert not eligible("RT-A2",{"gate0":True})
    assert eligible("RT-A2",{"gate0":True,"gate1":True,"round1_complete":True})
    assert not eligible("RT+B2",{"gate0":True,"gate1":True,"round1_complete":True})

def test_inference_adapter_is_available():
    assert importlib.util.find_spec("fastwam.models.wan22.transport_inference") is not None

def test_submission_refuses_missing_gate_evidence(tmp_path):
    assert importlib.util.find_spec("submit_transport") is not None
    from submit_transport import planned_runs
    assert planned_runs({})==["RT-A","RT-B4"]
    assert "RT-A2" not in planned_runs({"gate0":True,"round1_complete":True})

def test_stack_requires_final_b2a_identity():
    import train_transport as t
    assert hasattr(t,"validate_stacked_checkpoint")
    from transport_experiments import EXPERIMENTS
    import pytest
    with pytest.raises(ValueError):
        t.validate_stacked_checkpoint(dict(training_state=dict(epoch=10),transport_config=EXPERIMENTS["RT-A"]))
    t.validate_stacked_checkpoint(dict(training_state=dict(epoch=10),transport_config=EXPERIMENTS["RT-B2a"]))
    with pytest.raises(ValueError):
        t.validate_stacked_checkpoint(dict(training_state=dict(epoch=9),transport_config=EXPERIMENTS["RT-B2a"]))

def test_rt_epoch_checkpoint_evaluation_contract():
    import evaluate_loopwam_libero as e
    assert hasattr(e,"validate_transport_checkpoint")
    from transport_experiments import EXPERIMENTS
    import pytest
    c=dict(config=EXPERIMENTS["RT-A"],epochs=10,world=2,microbatch=64,global_batch=128,
           windows=277713,planned_updates=21700,normalization_sha256="stats")
    payload=dict(format_version="loopwam-s-v1",version="v0",trained_max_loops=4,inference_loops=4,
        step=17360,transport_config=EXPERIMENTS["RT-A"],
        training_state=dict(epoch=8,update=17360,windows_seen=2221704,contract=c))
    data=dict(dataset_scope="full_libero",train_windows=277713,normalization_sha256="stats",
              suites=["libero_spatial","libero_object","libero_goal","libero_10"])
    assert e.validate_transport_checkpoint(payload,data,"stats")==c
    with pytest.raises(ValueError): e.validate_transport_checkpoint(payload,data,"wrong")
    payload["training_state"]["windows_seen"]-=1
    with pytest.raises(ValueError): e.validate_transport_checkpoint(payload,data,"stats")

def test_parent_loader_rejects_silent_transport_drop(tmp_path):
    from fastwam.models.wan22.loopwam import create_loopwam
    import pytest
    path=tmp_path/"rt.pt"
    torch.save(dict(format_version="loopwam-s-v1",transport_config={"kind":"T4"}),path)
    with pytest.raises(ValueError,match="TransportInference"):
        create_loopwam(checkpoint_path=path,vae_path="unused")

def test_retry_only_terminal_failures():
    import submit_transport as s
    assert hasattr(s,"retryable_state")
    assert s.retryable_state("FAILED")
    assert s.retryable_state("TIMEOUT")
    assert not s.retryable_state("RUNNING")
    assert not s.retryable_state("COMPLETED")
    assert not s.retryable_state("PENDING")


def test_single_gpu_rt_epoch_evaluation_contract():
    import evaluate_loopwam_libero as e
    from transport_experiments import EXPERIMENTS
    config=EXPERIMENTS["RT-A2"]
    contract=dict(config=config,epochs=10,world=1,microbatch=64,global_batch=128,
        windows=277713,planned_updates=21700,normalization_sha256="stats")
    payload=dict(format_version="loopwam-s-v1",version="v0",trained_max_loops=4,inference_loops=4,
        step=17360,transport_config=config,
        training_state=dict(epoch=8,update=17360,windows_seen=2221704,contract=contract))
    data=dict(dataset_scope="full_libero",train_windows=277713,normalization_sha256="stats",
        suites=["libero_spatial","libero_object","libero_goal","libero_10"])
    assert e.validate_transport_checkpoint(payload,data,"stats")==contract

def test_single_gpu_sampler_matches_two_gpu_global_batches():
    from train_loopwam import ExactDistributedBatches
    single=list(ExactDistributedBatches(277713,64,0,1,42))
    pair=[list(ExactDistributedBatches(277713,64,r,2,42)) for r in range(2)]
    for update in range(len(pair[0])):
        serial=single[2*update:2*update+2]
        serial=[i for batch in serial for i in batch]
        distributed=pair[0][update]+pair[1][update]
        assert serial==distributed


def test_explicit_phase_two_pair_does_not_unlock_other_runs():
    from submit_transport import planned_runs
    assert planned_runs({},independent_phase2_pair=True)==["RT-A2","RT-B2a"]
    assert planned_runs({})==["RT-A","RT-B4"]
