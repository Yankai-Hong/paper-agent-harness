"""Import real PDF page-token checkpoints into a sealed Chroma collection."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from colqwen_encode_candidates import check_marker
from colqwen_smoke import rows, sha
from paper_agent.resumable_colqwen import ResumableColQwenStore


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('prepared', 'encoded', 'index', 'report', 'adapter'):
        p.add_argument('--' + key, required=True)
    a = p.parse_args()
    import psutil
    source, encoded, index = Path(a.prepared), Path(a.encoded), Path(a.index)
    contract = json.loads((encoded / 'contract.json').read_text(encoding='utf-8'))
    if contract['source_hash'] != sha(source / 'pages.jsonl'):
        raise ValueError('PDF source manifest changed')
    adapter = Path(a.adapter)
    if sha(adapter / 'adapter_model.safetensors') != contract['weight_sha256']['adapter_model.safetensors']:
        raise ValueError('Adapter weights changed')
    processor = json.loads((adapter / 'preprocessor_config.json').read_text(encoding='utf-8'))
    merge = processor.get('merge_size')
    if type(merge) is not int or merge < 1:
        raise ValueError('Invalid processor spatial merge size')
    pages = rows(source / 'pages.jsonl')
    store = ResumableColQwenStore(index, contract, create=not index.exists())
    started = time.perf_counter()
    provenance = []
    for page in pages:
        key = page['page_id']
        expected = {'page_id': key, 'paper_id': page['paper_id'], 'image_sha256': page['image_sha256']}
        if not check_marker(encoded / 'pages', key, expected):
            raise ValueError('Missing page vector: ' + key)
        path = (source / page['image_path']).resolve()
        if not path.is_relative_to(source.resolve()) or sha(path) != page['image_sha256']:
            raise ValueError('PDF page image changed')
        marker_path = encoded / 'pages' / (key + '.json')
        marker = json.loads(marker_path.read_text(encoding='utf-8'))
        positions, grid, count = marker.get('image_token_positions'), marker.get('image_grid_thw'), marker['token_count']
        if (marker.get('active_token_count') != count or not isinstance(positions, list)
                or positions != sorted(set(positions)) or not positions
                or any(type(x) is not int or not 0 <= x < count for x in positions)
                or not isinstance(grid, list) or len(grid) != 1 or len(grid[0]) != 3
                or any(type(x) is not int or x < 1 for x in grid[0])
                or grid[0][1] % merge or grid[0][2] % merge
                or len(positions) != grid[0][0] * (grid[0][1] // merge) * (grid[0][2] // merge)):
            raise ValueError('Invalid page token mask or grid')
        with np.load(encoded / 'pages' / marker['vector_file'], allow_pickle=False) as archive:
            vectors = np.asarray(archive['vectors'], dtype=np.float32)
        if vectors.shape != (count, contract['dimension']):
            raise ValueError('Page vector dimensions differ')
        if not store.final.exists():
            store.add_page(key, page['paper_id'], page['image_sha256'], vectors)
        else:
            store.get_page(key)
        provenance.append({**expected, 'encoding_contract_sha256': sha(encoded / 'contract.json'),
                           'encoded_marker_sha256': sha(marker_path), 'vector_file_sha256': marker['vectors_sha256'],
                           'image_grid_thw': grid, 'image_token_positions': positions,
                           'active_token_count': count, 'order': 'model input tokens; no pixel bbox claim'})
    content = ''.join(json.dumps(x, sort_keys=True, separators=(',', ':')) + '\n' for x in provenance).encode('utf-8')
    fingerprint = hashlib.sha256(content).hexdigest()
    sidecar = index / 'patch-provenance.jsonl'
    if sidecar.exists():
        if sha(sidecar) != fingerprint:
            raise ValueError('Existing provenance differs')
    else:
        with sidecar.open('xb') as stream:
            stream.write(content)
    if not store.final.exists():
        store.seal(patch_provenance_sha256=fingerprint)
    report = {'status': 'complete', 'pages': len(pages), 'tokens': store.collection.count(),
              'seal_sha256': sha(store.final), 'provenance_sha256': fingerprint,
              'preprocessor_config_sha256': sha(adapter / 'preprocessor_config.json'), 'spatial_merge_size': merge,
              'elapsed_seconds': time.perf_counter() - started, 'rss_bytes_final': psutil.Process().memory_info().rss,
              'index_bytes': sum(p.stat().st_size for p in index.rglob('*') if p.is_file())}
    with Path(a.report).open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
