import numpy as np
import pytest
import torch

from fastwam.datasets.loopwam_latent_cache import LoopWAMLatentCache


SHAPE = (2, 3, 1, 2)
PROVENANCE = {'vae_sha256': 'native-vae', 'dataset_revision': 'immutable-data', 'seed': 42}


class Encoder:
    def __init__(self):
        self.calls = []
    def __call__(self, video):
        self.calls.append(video.clone())
        return (video.reshape(len(video), *SHAPE) * .5).bfloat16().float()


def clips():
    return torch.arange(5 * 12, dtype=torch.float32).reshape(5, 12)


def cache(path, **kwargs):
    return LoopWAMLatentCache(path, 5, PROVENANCE, device='cpu', latent_shape=SHAPE, **kwargs)


def test_first_encode_partial_hits_dummy_and_all_hit_order(tmp_path):
    encoder = Encoder()
    data = clips()
    with cache(tmp_path, create=True) as store:
        first = store.encode_batch(torch.tensor([2, 0]), data[[2, 0]], encoder)
        torch.testing.assert_close(first, (data[[2, 0]] * .5).reshape(2, *SHAPE), rtol=0, atol=0)
        mixed = store.encode_batch(torch.tensor([0, 3, -1, 2]), data[[0, 3, 4, 2]], encoder)
        torch.testing.assert_close(mixed, (data[[0, 3, 4, 2]] * .5).reshape(4, *SHAPE), rtol=0, atol=0)
        torch.testing.assert_close(encoder.calls[-1], data[[3, 4]])
        assert store.stats['hits'] == 2 and store.stats['misses'] == 3
        assert store.stats['dummy_encodes'] == 1 and store.stats['encoder_samples'] == 4
        calls = len(encoder.calls)
        ordered = store.encode_batch(torch.tensor([3, 2, 0]), data[[3, 2, 0]], encoder)
        assert len(encoder.calls) == calls
        torch.testing.assert_close(ordered, (data[[3, 2, 0]] * .5).reshape(3, *SHAPE), rtol=0, atol=0)
        assert ordered.dtype == torch.float32
        store.encode_batch(torch.tensor([-1]), data[[4]], encoder)
        assert len(encoder.calls) == calls + 1
        assert store._valid.tolist() == [1, 0, 1, 1, 0]


def test_reopen_reads_persisted_payload_and_does_not_truncate(tmp_path):
    with cache(tmp_path, create=True) as store:
        original = store.encode_batch(torch.tensor([4]), clips()[[4]], Encoder())
    # create=True on an already initialized identical cache must also preserve it.
    for create in (False, True):
        with cache(tmp_path, create=create) as store:
            result = store.encode_batch(torch.tensor([4]), clips()[[4]], lambda _: pytest.fail('Unexpected re-encoding'))
            torch.testing.assert_close(result, original, rtol=0, atol=0)
            assert store.stats['hits'] == 1 and store.stats['misses'] == 0
            store.reset_stats()
            assert store.stats['hits'] == 0


def test_provenance_and_file_size_mismatches_fail(tmp_path):
    with cache(tmp_path, create=True):
        pass
    with pytest.raises(ValueError, match='provenance'):
        LoopWAMLatentCache(tmp_path, 5, {**PROVENANCE, 'vae_sha256': 'other'},
                          create=True, device='cpu', latent_shape=SHAPE)
    with (tmp_path / 'latents.uint16').open('r+b') as stream:
        stream.truncate(2)
    with pytest.raises(ValueError, match='sizes'):
        cache(tmp_path)


def test_failed_encoder_does_not_publish_validity(tmp_path):
    with cache(tmp_path, create=True) as store:
        for bad in (torch.full((1, *SHAPE), .1), torch.full((1, *SHAPE), float('nan'))):
            with pytest.raises(ValueError, match='losslessly'):
                store.encode_batch(torch.tensor([1]), clips()[[1]], lambda _: bad)
            assert not store._valid.any()
        good = store.encode_batch(torch.tensor([1]), clips()[[1]], Encoder())
        assert torch.isfinite(good).all() and store._valid[1] == 1


def test_unpublished_payload_is_a_miss_and_rank_open_requires_creation(tmp_path):
    with pytest.raises(FileNotFoundError, match='rank zero'):
        cache(tmp_path)
    with cache(tmp_path, create=True) as store:
        store._latents[1] = np.uint16(65535)
        store._latents.flush()
        encoder = Encoder()
        result = store.encode_batch(torch.tensor([1]), clips()[[1]], encoder)
        assert len(encoder.calls) == 1
        torch.testing.assert_close(result, (clips()[[1]] * .5).reshape(1, *SHAPE), rtol=0, atol=0)


def test_invalid_indices_and_closed_cache_fail(tmp_path):
    with cache(tmp_path, create=True) as store:
        for indices in (torch.tensor([-2]), torch.tensor([5])):
            with pytest.raises(IndexError):
                store.encode_batch(indices, clips()[[0]], Encoder())
        with pytest.raises(ValueError, match='integer'):
            store.encode_batch(torch.tensor([1.]), clips()[[0]], Encoder())
    with pytest.raises(RuntimeError, match='closed'):
        store.encode_batch(torch.tensor([0]), clips()[[0]], Encoder())


def test_provenance_ignores_only_run_specific_normalization_path():
    from fastwam.datasets.loopwam_latent_cache import latent_cache_provenance
    first = dict(normalization_path='/run/bench/data/stats.json', normalization_sha256='same',
                 train_episodes=[1, 3], data_files={'data.parquet': 'hash'})
    second = {**first, 'normalization_path': '/run/production/data/stats.json'}
    assert latent_cache_provenance(first, 'vae') == latent_cache_provenance(second, 'vae')
    assert latent_cache_provenance(first, 'vae') != latent_cache_provenance({**second, 'normalization_sha256': 'different'}, 'vae')
    assert first['normalization_path'] == '/run/bench/data/stats.json'
