"""Compare PDF candidate MaxSim with the official multi-vector processor."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from colqwen_encode_candidates import check_marker
from colqwen_smoke import rows, sha

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from paper_agent.colqwen_query_encoder import ColQwenQueryEncoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('prepared', 'encoded', 'base', 'adapter', 'query', 'output'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--page-id', action='append', required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    import torch
    source, encoded = Path(a.prepared), Path(a.encoded)
    contract = json.loads((encoded / 'contract.json').read_text(encoding='utf-8'))
    if contract['source_hash'] != sha(source / 'pages.jsonl') or len(a.page_id) != len(set(a.page_id)):
        raise ValueError('PDF source or page identities invalid')
    pages = {x['page_id']: x for x in rows(source / 'pages.jsonl')}
    encoder = ColQwenQueryEncoder(a.base, a.adapter, contract, device=a.device)
    query = encoder.encode_query(a.query)
    results = []
    for key in a.page_id:
        page = pages[key]
        if not check_marker(encoded / 'pages', key, {'page_id': key, 'paper_id': page['paper_id'],
                                                       'image_sha256': page['image_sha256']}):
            raise ValueError('Missing page checkpoint')
        image_path = (source / page['image_path']).resolve()
        if not image_path.is_relative_to(source.resolve()) or sha(image_path) != page['image_sha256']:
            raise ValueError('PDF page image changed')
        marker = json.loads((encoded / 'pages' / (key + '.json')).read_text(encoding='utf-8'))
        with np.load(encoded / 'pages' / marker['vector_file'], allow_pickle=False) as archive:
            vectors = np.asarray(archive['vectors'], dtype=np.float32)
        numpy_score = float(np.max(query @ vectors.T, axis=1).sum(dtype=np.float64))
        official = float(encoder.processor.score_multi_vector([torch.from_numpy(query)],
                         [torch.from_numpy(vectors)], device='cpu')[0, 0])
        if abs(numpy_score - official) > 1e-3:
            raise ValueError('MaxSim differs from official processor')
        results.append({'page_id': key, 'image_sha256': page['image_sha256'], 'tokens': len(vectors),
                        'numpy_score': numpy_score, 'official_score': official,
                        'absolute_difference': abs(numpy_score - official)})
    report = {'scope': 'real PDF page candidates; padding removed; page-level only',
              'query_sha256': hashlib.sha256(a.query.encode('utf-8')).hexdigest(),
              'query_tokens': len(query), 'results': results}
    with Path(a.output).open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
