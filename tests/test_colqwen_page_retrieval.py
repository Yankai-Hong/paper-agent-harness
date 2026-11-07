import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from paper_agent.colqwen_page_retrieval import ColQwenCandidateRetriever
from paper_agent.resumable_colqwen import ResumableColQwenStore


def test_candidate_page_search_validates_source_and_scores_real_store(tmp_path):
    images = tmp_path / 'images'
    images.mkdir()
    sources = {}
    for page, vector in [('one', [1., 0.]), ('two', [0., 1.])]:
        (images / page).write_bytes(page.encode())
        fingerprint = hashlib.sha256(page.encode()).hexdigest()
        sources[page] = {'page_id': page, 'paper_id': 'p', 'page_number': 1, 'image_path': page, 'image_sha256': fingerprint}
    manifest = tmp_path / 'pages.jsonl'
    manifest.write_text(''.join(json.dumps(x) + '\n' for x in sources.values()), encoding='utf-8')
    contract = {'model_revision': 'model', 'base_revision': 'base', 'source_hash': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                'dimension': 2, 'weight_sha256': {'weight': 'hash'}, 'max_visual_tokens': 256}
    store = ResumableColQwenStore(tmp_path / 'index', contract, create=True)
    for page, vector in [('one', [1., 0.]), ('two', [0., 1.])]:
        fingerprint = sources[page]['image_sha256']
        store.add_page(page, 'p', fingerprint, np.array([vector], dtype=np.float32))
    store.seal()
    encoder = SimpleNamespace(model_revision='model', base_revision='base', weight_sha256={'weight': 'hash'},
                              max_visual_tokens=256, encode_query=lambda q: [[0., 1.]])
    retriever = ColQwenCandidateRetriever(store, encoder, manifest, images, max_candidates=2)
    result = retriever.search_text('question', ['one', 'one', 'two'])
    assert [x['page_id'] for x in result['hits']] == ['two', 'one']
    assert result['hits'][0]['candidate_rank'] == 2 and result['hits'][0]['citation_granularity'] == 'page'
    assert result['scope'] == 'candidate_only' and result['candidate_count'] == 2
    (images / 'two').write_bytes(b'changed')
    with pytest.raises(ValueError, match='image changed'):
        retriever.search_text('question', ['two'])
    encoder.model_revision = 'other'
    with pytest.raises(ValueError, match='revisions differ'):
        ColQwenCandidateRetriever(store, encoder, manifest, images)
    encoder.model_revision = 'model'
    manifest.write_text(manifest.read_text(encoding='utf-8') + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='source manifest'):
        ColQwenCandidateRetriever(store, encoder, manifest, images)
