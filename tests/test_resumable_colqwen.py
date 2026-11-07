import numpy as np
import pytest
import hashlib

from paper_agent.resumable_colqwen import ResumableColQwenStore


def test_interrupted_page_resumes_only_identical_vectors(tmp_path, monkeypatch):
    contract = {'model_revision': 'fixed', 'source_hash': 'corpus', 'dimension': 2}
    path = tmp_path / 'colqwen'
    store = ResumableColQwenStore(path, contract, create=True)
    vectors = np.tile(np.array([[1., 0.]], dtype=np.float32), (513, 1))
    original = store.collection.upsert
    calls = 0
    def crash_after_one_batch(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('simulated interruption')
        return original(**kwargs)
    monkeypatch.setattr(store.collection, 'upsert', crash_after_one_batch)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        store.add_page('1_1', 'paper1', 'a' * 64, vectors)
    resumed = ResumableColQwenStore(path, contract)
    with pytest.raises(ValueError, match='unfinished'):
        resumed.seal()
    with pytest.raises(ValueError, match='different source'):
        resumed.add_page('1_1', 'paper1', 'b' * 64, vectors)
    resumed.add_page('1_1', 'paper1', 'a' * 64, vectors)
    resumed.add_page('1_1', 'paper1', 'a' * 64, vectors)
    resumed.add_page('1_2', 'paper1', 'c' * 64, [[0., 1.]])
    resumed.seal()
    reopened = ResumableColQwenStore(path, contract)
    assert reopened.collection.count() == 514
    assert reopened.get_page('1_1').shape == (513, 2)
    assert [page for page, _ in reopened.rerank([[0., 1.]], ['1_1', '1_2'])] == ['1_2', '1_1']
    with pytest.raises(ValueError, match='immutable'):
        reopened.add_page('1_3', 'paper1', 'd' * 64, [[1., 0.]])


def test_realistic_float_vectors_roundtrip_and_detect_tampering(tmp_path):
    contract = {'model_revision': 'fixed', 'source_hash': 'corpus', 'dimension': 128}
    store = ResumableColQwenStore(tmp_path / 'vectors', contract, create=True)
    rng = np.random.default_rng(42)
    values = rng.standard_normal((8, 128)).astype(np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    values = values.astype(np.float16).astype(np.float32)
    store.add_page('1_1', 'paper1', 'a' * 64, values)
    np.testing.assert_allclose(store.get_page('1_1'), values, atol=1e-6, rtol=0)
    store.seal()
    changed = values[0].copy()
    changed[0] += 1e-4
    store.collection.update(ids=['1_1:000000'], embeddings=[changed.tolist()])
    with pytest.raises(ValueError, match='checksum mismatch'):
        store.get_page('1_1')


def test_sealed_patch_provenance_is_bound_and_validated(tmp_path):
    contract = {'model_revision': 'fixed', 'source_hash': 'corpus', 'dimension': 2}
    path = tmp_path / 'vectors'
    store = ResumableColQwenStore(path, contract, create=True)
    store.add_page('1_1', 'paper1', 'a' * 64, [[1., 0.]])
    manifest = path / 'patch-provenance.jsonl'
    manifest.write_text('{"image_token_positions":[0]}\n', encoding='utf-8')
    fingerprint = hashlib.sha256(manifest.read_bytes()).hexdigest()
    store.seal(patch_provenance_sha256=fingerprint)
    ResumableColQwenStore(path, contract)
    manifest.write_text('{}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='patch provenance mismatch'):
        ResumableColQwenStore(path, contract)


def test_marker_publication_never_overwrites_existing_data(tmp_path):
    marker = tmp_path / 'existing.json'
    marker.write_text('{"old":true}', encoding='utf-8')
    with pytest.raises(FileExistsError):
        ResumableColQwenStore._write_new(marker, {'new': True})
    assert marker.read_text(encoding='utf-8') == '{"old":true}'
    assert len(list(tmp_path.glob('existing.json.*.pending'))) == 1
