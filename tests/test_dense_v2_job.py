import json
import pytest
from pathlib import Path
from scripts.run_dense_v2_job import select_candidates, verify_training, verify_evaluation, training_command


def trial(version="dense_s12", backend="ddp", micro=16, seconds=2.):
    return dict(version=version, backend=backend, microbatch=micro, world_size=4,
        global_batch=128, accumulation=32//micro, loops=1 if version=="dense_s12" else 4,
        steady_seconds=seconds, records=[dict(loss=.3,grad_norm=1.,seconds=seconds)]*5,
        precision=dict(policy_dtype="float32", optimizer_moment_dtypes=["torch.float32"]),
        fused=True, structured_attention=True)


def test_selects_fastest_real_four_rank_measurement():
    assert select_candidates([trial(seconds=3),trial(backend="zero1",seconds=2)],"dense_s12")[0]["backend"]=="zero1"
    for key,value in [("version","v1"),("world_size",2),("loops",4),("global_batch",256),
                      ("steady_seconds",float("nan")),("microbatch",32)]:
        bad=trial();bad[key]=value
        with pytest.raises(ValueError):select_candidates([bad],"dense_s12")
    bad=trial();bad["records"][0]=dict(loss=float("nan"),grad_norm=1.,seconds=2)
    with pytest.raises(ValueError):select_candidates([bad],"dense_s12")
    with pytest.raises(ValueError):select_candidates([],"dense_s12")


def setup_training(path,version="dense_s12"):
    path.mkdir(exist_ok=True)
    payloads=dict(manifest=dict(version=version,resume=None,epochs=10,global_batch=128,
        world_size=4,microbatch=16,gradient_accumulation=2,loops=1 if version=="dense_s12" else 4,
        policy_parameters=584536135,planned_updates=7250,planned_windows=926780,train_windows=92678),
        timing=dict(status="complete",completed_updates=7250,windows_seen=926780),
        trainer_state=dict(update=7250,windows_seen=926780,epoch=9))
    for name,data in payloads.items():(path/(name+".json")).write_text(json.dumps(data))
    (path/"latest.pt").write_bytes(b"checkpoint")


def test_training_gate_requires_fresh_full_correct_architecture(tmp_path):
    setup_training(tmp_path)
    assert verify_training(tmp_path,"dense_s12")["loops"]==1
    for name,key,value in [("manifest","resume","old.pt"),("manifest","version","v2"),
                           ("manifest","world_size",2),("manifest","loops",4),
                           ("timing","status","smoke_complete"),("trainer_state","epoch",8),
                           ("trainer_state","windows_seen",926779)]:
        setup_training(tmp_path)
        f=tmp_path/(name+".json"); d=json.loads(f.read_text());d[key]=value;f.write_text(json.dumps(d))
        with pytest.raises(ValueError):verify_training(tmp_path,"dense_s12")


def test_production_command_never_resumes_or_stops_at_smoke_budget(tmp_path):
    args=training_command(trial(),tmp_path/"train",tmp_path/"latents")
    assert "--resume" not in args and "--max-updates" not in args
    assert args[args.index("--version")+1]=="dense_s12"
    assert args[args.index("--global-batch")+1]=="128"
    assert args[args.index("--epochs")+1]=="10"
    assert "--smoke" in args
    smoke=training_command(trial(),tmp_path/"smoke",tmp_path/"cold",smoke_updates=3)
    assert smoke[smoke.index("--max-updates")+1]=="3"


def setup_evaluation(tmp_path):
    manifest=dict(mode="final_rollout",version="dense_s12",loops=1,checkpoint="/a/latest.pt",checkpoint_sha256="abc")
    video=tmp_path/"video.mp4";video.write_bytes(b"video")
    rows=[dict(task_id=i,episode_index=j,mode="final_rollout",checkpoint_sha256="abc",video=str(video))
          for i in range(10) for j in range(10)]
    summary=dict(mode="final_rollout",version="dense_s12",checkpoint_step=7250,checkpoint_sha256="abc",
        total_episodes=100,successes=7,success_rate=.07,episodes=rows,
        per_task={str(i):dict(episodes=10) for i in range(10)})
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    (tmp_path/"summary.json").write_text(json.dumps(summary))
    return summary


def test_evaluation_gate_rejects_missing_episodes_smoke_or_wrong_model(tmp_path):
    summary=setup_evaluation(tmp_path)
    assert verify_evaluation(tmp_path,"dense_s12",Path("/a/latest.pt"))["total_episodes"]==100
    with pytest.raises(ValueError):verify_evaluation(tmp_path,"dense_s12",Path("/different/latest.pt"))
    for key,value in [("mode","smoke"),("version","v2"),("total_episodes",99),("checkpoint_step",3)]:
        summary=setup_evaluation(tmp_path);summary[key]=value
        (tmp_path/"summary.json").write_text(json.dumps(summary))
        with pytest.raises(ValueError):verify_evaluation(tmp_path,"dense_s12",Path("/a/latest.pt"))
    summary=setup_evaluation(tmp_path);summary["episodes"][-1]=summary["episodes"][0]
    (tmp_path/"summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError,match="Duplicate"):verify_evaluation(tmp_path,"dense_s12",Path("/a/latest.pt"))
    setup_evaluation(tmp_path);(tmp_path/"video.mp4").unlink()
    with pytest.raises(ValueError,match="video missing"):verify_evaluation(tmp_path,"dense_s12",Path("/a/latest.pt"))


def test_failed_stage_never_advances_to_evaluation_or_v2(tmp_path,monkeypatch):
    import scripts.run_dense_v2_job as mod
    monkeypatch.setattr(mod,"source_hashes",lambda:{"source":"verified"})
    (tmp_path/"prepared.json").write_text(json.dumps(dict(source_hashes={"source":"verified"},
        selections={v:dict(selected=trial(version=v)) for v in mod.VERSIONS})))
    pipeline=mod.Pipeline(tmp_path)
    stages=[]
    def fail(stage,cmd,allow_oom=False):
        stages.append(stage)
        raise RuntimeError("training failed")
    monkeypatch.setattr(pipeline,"run",fail)
    with pytest.raises(RuntimeError,match="training failed"):pipeline.production()
    assert stages==["dense_s12_training"]


def test_production_refuses_changed_code_and_existing_output(tmp_path,monkeypatch):
    import scripts.run_dense_v2_job as mod
    monkeypatch.setattr(mod,"source_hashes",lambda:{"source":"changed"})
    (tmp_path/"prepared.json").write_text(json.dumps(dict(source_hashes={"source":"verified"})))
    pipeline=mod.Pipeline(tmp_path)
    with pytest.raises(ValueError,match="Source changed"):pipeline.production()
    monkeypatch.setattr(mod,"source_hashes",lambda:{"source":"verified"})
    (tmp_path/"dense_s12_train").mkdir()
    with pytest.raises(ValueError,match="Production outputs exist"):pipeline.production()


def test_benchmarks_must_match_current_model_and_backend_sources():
    from scripts.run_dense_v2_job import verify_benchmark_sources
    names=["src/fastwam/models/wan22/loopwam.py","src/fastwam/models/wan22/loop_mot.py",
           "src/fastwam/models/wan22/loopwam_init.py","src/fastwam/models/wan22/fastwam.py",
           "src/fastwam/training_backends.py"]
    current={name:"current" for name in names}
    candidate=dict(source_sha256=dict(current))
    verify_benchmark_sources(candidate,current)
    candidate["source_sha256"][names[0]]="stale"
    with pytest.raises(ValueError,match="Benchmark source"):verify_benchmark_sources(candidate,current)
    candidate["source_sha256"]={}
    with pytest.raises(ValueError,match="Benchmark source"):verify_benchmark_sources(candidate,current)


def test_speed_selection_aggregates_repeats_and_retains_loader_setting():
    a=trial(seconds=1.);b=trial(seconds=3.);c=trial(backend="zero1",seconds=1.8)
    selected=select_candidates([a,b,c],"dense_s12")
    assert selected[0]["backend"]=="zero1"
    assert selected[1]["steady_seconds"]==2.
    candidate=trial();candidate["workers"]=8
    cmd=training_command(candidate,Path("fresh"),Path("cache"))
    assert cmd[cmd.index("--workers")+1]=="8"
