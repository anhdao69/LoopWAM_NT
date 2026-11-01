"""RoboCasa-GR1 v0: 44->29 selection, normalization guard, batching, model interfaces, windows."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from fastwam.datasets.lerobot.transforms import gr1

MODALITY = {
    'state': {'left_arm': {'original_key': 'observation.state', 'start': 0, 'end': 7},
              'left_hand': {'original_key': 'observation.state', 'start': 7, 'end': 13},
              'left_leg': {'original_key': 'observation.state', 'start': 13, 'end': 19},
              'neck': {'original_key': 'observation.state', 'start': 19, 'end': 22},
              'right_arm': {'original_key': 'observation.state', 'start': 22, 'end': 29},
              'right_hand': {'original_key': 'observation.state', 'start': 29, 'end': 35},
              'right_leg': {'original_key': 'observation.state', 'start': 35, 'end': 41},
              'waist': {'original_key': 'observation.state', 'start': 41, 'end': 44}},
    'video': {'ego_view': {'original_key': 'observation.images.ego_view'}},
    'annotation': {'human.coarse_action': {'original_key': 'annotation.human.coarse_action'}},
}
MODALITY['action'] = {k: dict(v, original_key='action') for k, v in MODALITY['state'].items()}


def _gr1_vector(n, rng):
    x = rng.normal(size=(n, 44))
    x[:, list(gr1.DROPPED_INDICES)] = 0.0           # legs/neck are identically zero in the data
    return x


def test_select_indices_are_exact_and_match_modality():
    expected = [*range(0, 7), *range(22, 29), *range(7, 13), *range(29, 35), *range(41, 44)]
    assert list(gr1.SELECT_INDICES) == expected
    assert gr1.indices_from_modality(MODALITY, 'action') == tuple(expected)
    assert gr1.indices_from_modality(MODALITY, 'state') == tuple(expected)
    assert sorted(gr1.DROPPED_INDICES) == [*range(13, 22), *range(35, 41)]
    assert gr1.OUTPUT_SLICES == dict(left_arm=(0, 7), right_arm=(7, 14), left_hand=(14, 20), right_hand=(20, 26), waist=(26, 29))
    # starVLA's RoboCasa client splits the 29-D chunk with exactly these slices.
    x = np.arange(44.0)[None]
    y = gr1.select(x)[0]
    assert y[0:7].tolist() == list(range(0, 7)) and y[7:14].tolist() == list(range(22, 29))
    assert y[14:20].tolist() == list(range(7, 13)) and y[20:26].tolist() == list(range(29, 35))
    assert y[26:29].tolist() == [41, 42, 43]
    bad = json.loads(json.dumps(MODALITY)); bad['action']['waist']['start'] = 40
    with pytest.raises(ValueError):
        gr1.indices_from_modality(bad, 'action')


def test_select_expand_and_group_roundtrip():
    rng = np.random.default_rng(0)
    x = _gr1_vector(5, rng)
    assert np.array_equal(gr1.expand(gr1.select(x)), x)
    t = torch.from_numpy(x).float()
    assert torch.equal(gr1.expand(gr1.select(t)), t)
    y = gr1.select(x)
    groups = gr1.split_groups(y)
    assert list(groups) == list(gr1.GROUP_ORDER)
    assert np.array_equal(gr1.join_groups(groups), y)
    # The env's state.* groups come straight from the 44-D slices.
    env_state = {g: x[:, lo:hi] for g, (lo, hi) in gr1.SOURCE_SLICES.items()}
    assert np.array_equal(gr1.join_groups(env_state), y)
    with pytest.raises(ValueError):
        gr1.select(np.zeros((2, 29)))
    with pytest.raises(ValueError):
        gr1.expand(np.zeros((2, 44)))


def test_transform_hook_forward_backward():
    rng = np.random.default_rng(1)
    a, s = _gr1_vector(32, rng), _gr1_vector(33, rng)
    batch = {'action': {'default': torch.from_numpy(a)}, 'state': {'default': torch.from_numpy(s)}}
    tr = gr1.GR1ArmsWaistSelect()
    out = tr.forward(batch)
    assert out['action']['default'].shape == (32, 29) and out['state']['default'].shape == (33, 29)
    back = tr.backward(out)
    assert torch.equal(back['action']['default'], torch.from_numpy(a)) and torch.equal(back['state']['default'], torch.from_numpy(s))


def _stats29(rng, n=1000):
    from fastwam.datasets.robocasa_gr1 import compute_stats
    a, s = gr1.select(_gr1_vector(n, rng)), gr1.select(_gr1_vector(n, rng))
    a[:, 3] = 0.25              # constant action dim
    s[:, 5] = 1.0 + 1e-6 * rng.random(n)   # near-constant state dim (range < guard)
    return a, s, compute_stats(a, s)


def test_minmax_epsilon_guard_and_roundtrip():
    from fastwam.datasets.robocasa_gr1 import build_normalizer
    rng = np.random.default_rng(2)
    a, s, stats = _stats29(rng)
    assert stats['normalization']['guarded_dims'] == {'action': [3], 'state': [5]}
    norm = build_normalizer(stats)
    na = norm['action'].forward(torch.from_numpy(a).float())
    ns = norm['state'].forward(torch.from_numpy(s).float())
    assert torch.isfinite(na).all() and torch.isfinite(ns).all()
    keep = [i for i in range(29) if i != 3]
    assert na[:, keep].min() >= -1 - 1e-5 and na[:, keep].max() <= 1 + 1e-5
    assert torch.allclose(na[:, 3], torch.zeros(len(a)), atol=1e-6)                 # constant -> 0
    assert ns[:, 5].abs().max() < 1e-4                                              # no 1/range blow-up
    assert torch.allclose(norm['action'].backward(na), torch.from_numpy(a).float(), atol=1e-5)
    assert torch.allclose(norm['state'].backward(ns), torch.from_numpy(s).float(), atol=1e-5)


def test_normalization_and_slicing_roundtrip_for_serving():
    """(c) dataset 44-D -> 29-D -> normalized -> model space -> de-normalized -> action.* groups."""
    from fastwam.datasets.robocasa_gr1 import build_normalizer
    rng = np.random.default_rng(3)
    _, _, stats = _stats29(rng)
    norm = build_normalizer(stats)
    raw44 = _gr1_vector(32, rng)
    lo = np.asarray(stats['action']['default']['global_min']); hi = np.asarray(stats['action']['default']['global_max'])
    raw44[:, list(gr1.SELECT_INDICES)] = np.clip(raw44[:, list(gr1.SELECT_INDICES)], lo, hi)
    x29 = torch.from_numpy(gr1.select(raw44)).float()
    model_space = norm['action'].forward(x29)
    groups = gr1.split_groups(norm['action'].backward(model_space).numpy())
    for g, (a, b) in gr1.SOURCE_SLICES.items():
        assert np.allclose(groups[g], raw44[:, a:b], atol=1e-5), g
    assert np.allclose(gr1.expand(gr1.join_groups(groups)), raw44, atol=1e-5)


def test_processor_hook_from_yaml_config(tmp_path):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    rng = np.random.default_rng(4)
    _, _, stats = _stats29(rng)
    (tmp_path / 's.json').write_text(json.dumps(stats))
    cfg = OmegaConf.create({'data': OmegaConf.load(Path(__file__).parents[1] / 'configs/data/robocasa_gr1.yaml')})
    train = cfg.data.train
    assert train.num_frames == 33 and train.action_video_freq_ratio == 4 and list(train.video_size) == [256, 256]
    assert train.val_set_proportion == 0.0 and train.processor.norm_default_mode == 'min/max'
    assert len(train.dataset_dirs) == 24 and train.processor.delta_action_dim_mask is None
    proc = instantiate(train.processor)
    proc.set_normalizer_from_stats(load_dataset_stats_from_json(str(tmp_path / 's.json')))
    proc.train()
    data = {'action': {'default': torch.from_numpy(_gr1_vector(32, rng)).float()},
            'state': {'default': torch.from_numpy(_gr1_vector(33, rng)).float()},
            'images': {'ego_view': torch.randint(0, 256, (9, 3, 256, 256), dtype=torch.uint8)},
            'image_is_pad': torch.zeros(9, dtype=torch.bool), 'action_is_pad': torch.zeros(32, dtype=torch.bool),
            'state_is_pad': torch.zeros(33, dtype=torch.bool), 'task': 'unlocked_waist: x', 'idx': 0}
    proc.num_image_steps = 9
    out = proc.preprocess(data)
    assert out['action'].shape == (32, 29) and out['proprio'].shape == (33, 29)
    assert out['pixel_values'].shape == (1, 9, 3, 256, 256)


def test_exact_256_batches_cover_once_on_4_and_8_ranks():
    from fastwam.datasets.loopwam_robot_batches import epoch_rank_batches, training_budget
    size = 256 * 3 + 37
    for world in (4, 8):
        ranks = [list(epoch_rank_batches(size, 256, world, r, 42)) for r in range(world)]
        rows = [sum([ranks[r][i] for r in range(world)], []) for i in range(4)]
        assert [len(x) for x in rows] == [256, 256, 256, 37]
        assert sum(rows, []) == torch.randperm(size, generator=torch.Generator().manual_seed(42)).tolist()
        assert all(len(ranks[r][0]) == 256 // world for r in range(world))
    assert training_budget(size, 256, 5) == (4, 20, 37)


def test_preprocess_frames_gr1_scaling_and_shape():
    from fastwam.datasets.robocasa_gr1 import preprocess_frames_gr1
    x = torch.zeros(2, 3, 256, 256, dtype=torch.uint8); x[1] = 255
    y = preprocess_frames_gr1(x)
    assert y.dtype == torch.bfloat16 and y[0].eq(-1).all() and y[1].eq(1).all()
    with pytest.raises(ValueError):
        preprocess_frames_gr1(torch.zeros(1, 3, 224, 224, dtype=torch.uint8))


def test_gr1_interfaces_parameters_and_cached_latents():
    from fastwam.models.wan22.loopwam_gr1 import fresh_gr1_interfaces, GR1LoopWAM, PARAMETERS
    from fastwam.models.wan22.loopwam_robot import PARAMETERS as P14
    assert PARAMETERS - P14 == 15 * 512 + 15 * 512 + 15 + 15 * 4096 == 76815

    def model():
        m = nn.Module(); m.action_expert = nn.Module(); m.action_expert.hidden_dim = 8
        m.action_expert.action_encoder = nn.Linear(7, 8); m.action_expert.head = nn.Linear(8, 7)
        m.proprio_encoder = nn.Linear(8, 16); return m
    a, b = model(), model(); fresh_gr1_interfaces(a, 42); fresh_gr1_interfaces(b, 42)
    assert a.action_expert.action_encoder.in_features == 29 and a.action_expert.head.out_features == 29
    assert a.proprio_encoder.in_features == 29 and a.proprio_encoder.out_features == 16
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    m = object.__new__(GR1LoopWAM)
    lat = torch.randn(2, 16, 3, 32, 32)
    assert m._encode_training_video({'latents': lat}, None) is lat
    for bad in ({'latents': torch.zeros(2, 16, 3, 48, 40)}, {}):
        with pytest.raises(ValueError):
            m._encode_training_video(bad, None)


@pytest.mark.parametrize('field,bad', [('inference_loops', 3), ('trained_max_loops', 3), ('exit_weight_scale', .5),
                                       ('robot_dim', 14), ('latent_shape', [16, 3, 48, 40])])
def test_gr1_loader_rejects_changed_contract(field, bad):
    from fastwam.models.wan22.loopwam_gr1 import validate_gr1_payload, FORMAT
    from fastwam.models.wan22.loopwam_init import target_configs, DONOR_INDICES
    v, a = target_configs(12); a['action_dim'] = 29
    p = dict(format_version=FORMAT, version='v0', video_loops=4, action_loops=4, memory_tokens=0, history_frame=0,
             inference_loops=4, trained_max_loops=4, exit_weight_scale=1., robot_dim=29, latent_shape=[16, 3, 32, 32],
             architecture=dict(target_video_config=v, target_action_config=a, robot_proprio_dim=29,
                               donor_indices=list(DONOR_INDICES), initialization_mode='fresh_wan_donors_no_embodied_pretraining'))
    validate_gr1_payload(p); p[field] = bad
    with pytest.raises(ValueError):
        validate_gr1_payload(p)


def _fake_dataset(root: Path, folders, lengths, rng):
    import pyarrow as pa
    import pyarrow.parquet as pq
    raw = {}
    for f, lens in zip(folders, lengths):
        d = root / f
        (d / 'meta').mkdir(parents=True); (d / 'data/chunk-000').mkdir(parents=True)
        info = dict(codebase_version='v2.0', fps=20.0, total_episodes=len(lens), total_frames=sum(lens), chunks_size=1000,
                    data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
                    video_path='videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
                    features={'observation.images.ego_view': {'shape': [256, 256, 3]}, 'action': {'shape': [44]},
                              'observation.state': {'shape': [44]}})
        (d / 'meta/info.json').write_text(json.dumps(info))
        (d / 'meta/modality.json').write_text(json.dumps(MODALITY))
        (d / 'meta/stats.json').write_text('{}')
        (d / 'meta/tasks.jsonl').write_text(json.dumps(dict(task_index=0, task='')) + '\n' +
                                            json.dumps(dict(task_index=1, task=f'unlocked_waist: do {f}')) + '\n')
        (d / 'meta/episodes.jsonl').write_text(''.join(json.dumps(dict(episode_index=i, length=n)) + '\n' for i, n in enumerate(lens)))
        for ep, n in enumerate(lens):
            a, s = _gr1_vector(n, rng), _gr1_vector(n, rng)
            raw[(f, ep)] = (a, s)
            pq.write_table(pa.table({'observation.state': list(s), 'action': list(a), 'timestamp': np.arange(n) / 20.0,
                                     'task_index': np.ones(n, np.int64), 'episode_index': np.full(n, ep, np.int64),
                                     'index': np.arange(n, dtype=np.int64), 'annotation.human.coarse_action': np.ones(n, np.int64)}),
                           d / f'data/chunk-000/episode_{ep:06d}.parquet')
    return raw


def test_gr1_windows_end_to_end(tmp_path):
    from fastwam.datasets.robocasa_gr1 import GR1Spec, GR1Windows, build_normalizer
    from fastwam.datasets.loopwam_long import DEFAULT_PROMPT
    rng = np.random.default_rng(5)
    folders = ('fa', 'fb')
    raw = _fake_dataset(tmp_path / 'root', folders, [[40, 5], [33]], rng)
    text_dir = tmp_path / 'text'; text_dir.mkdir()
    spec = GR1Spec(name='robocasa_gr1', root=str(tmp_path / 'root'), folders=folders, text_cache_dir=str(text_dir))
    w = GR1Windows(spec, cache_dir=tmp_path / 'shared', threads=2, text_threads=1, require_stats=False)
    assert len(w) == 78 and [e[2:] for e in w.episodes] == [(0, 40), (40, 5), (45, 33)]
    assert w.texts == ['unlocked_waist: do fa', 'unlocked_waist: do fb'] and w.dropped_max_abs == dict(action=0.0, state=0.0)
    a_all = np.concatenate([gr1.select(raw[k][0]) for k in [('fa', 0), ('fa', 1), ('fb', 0)]])
    assert np.allclose(w.actions.numpy(), a_all.astype(np.float32))
    stats = w._stats64
    assert np.allclose(stats['action']['default']['global_min'], a_all.min(0))
    for pr in w.prompts():
        mask = torch.zeros(128, dtype=torch.bool); mask[:7] = True
        torch.save(dict(context=torch.randn(128, 4096).to(torch.bfloat16), mask=mask),
                   text_dir / f'{hashlib.sha256(pr.encode()).hexdigest()}.{spec.text_suffix}')
    w.stats, w.normalizer = stats, build_normalizer(stats)
    f = w.fields(42)                                   # episode (fa, 1): global 40..44, 3 frames from its end
    assert f['action'].shape == (32, 29) and f['proprio'].shape == (32, 29)
    assert f['action_is_pad'].tolist() == [False] * 3 + [True] * 29
    assert f['image_is_pad'].tolist() == [False] + [True] * 8
    expected = w.normalizer['action'].forward(torch.from_numpy(gr1.select(raw[('fa', 1)][0][[2, 3, 4] + [4] * 29])).float())
    assert torch.allclose(f['action'], expected)
    assert w.prompt(42) == DEFAULT_PROMPT.format(task='unlocked_waist: do fa')
    assert w.prompt(50) == DEFAULT_PROMPT.format(task='unlocked_waist: do fb')
    assert (f['context'][7:] == 0).all() and f['context_mask'].sum() == 7
    # Packed reload is identical.
    w2 = GR1Windows(spec, cache_dir=tmp_path / 'shared', threads=2, text_threads=1, require_stats=False)
    assert torch.equal(w2.actions, w.actions) and np.array_equal(w2.task_index, w.task_index) and w2.texts == w.texts
