import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path('scripts').resolve()))
from colqwen_candidate_eval import install_patch_provenance
from colqwen_smoke import sha


def test_patch_positions_bind_source_grid_and_vector_checkpoint(tmp_path):
    source, index, provenance = tmp_path / 'source', tmp_path / 'index', tmp_path / 'patch'
    (source / 'pages').mkdir(parents=True)
    index.mkdir()
    provenance.mkdir()
    (source / 'contract.json').write_text('{}', encoding='utf-8')
    marker = source / 'pages' / '1_1.json'
    marker.write_text(json.dumps({'vectors_sha256': 'vector-hash', 'image_grid_thw': [[1, 2, 4]], 'token_count': 6}), encoding='utf-8')
    data = {'page_id': '1_1', 'paper_id': 'paper1', 'image_sha256': 'a' * 64,
            'encoding_contract_sha256': sha(source / 'contract.json'), 'encoded_marker_sha256': sha(marker),
            'vector_file_sha256': 'vector-hash', 'image_grid_thw': [[1, 2, 4]],
            'spatial_merge_size': 2, 'merged_grid_thw': [1, 1, 2], 'active_token_count': 6, 'image_token_positions': [2, 3]}
    sidecar = provenance / '1_1.json'
    sidecar.write_text(json.dumps(data), encoding='utf-8')
    pages = {'1_1': {'paper_id': 'paper1', 'image_sha256': 'a' * 64}}
    fingerprint = install_patch_provenance(index, source, provenance, ['1_1'], pages)
    assert sha(index / 'patch-provenance.jsonl') == fingerprint
    data['image_token_positions'] = [2, 2]
    sidecar.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='Invalid patch provenance'):
        install_patch_provenance(index, source, provenance, ['1_1'], pages)
