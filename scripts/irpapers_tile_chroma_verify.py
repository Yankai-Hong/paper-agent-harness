"""Compare actual Chroma tile rankings with the encoded candidate probe."""
import argparse
import json
from pathlib import Path

import numpy as np
from irpapers_optimize import read_frozen
from irpapers_tile_probe import sha
from paper_agent.vector_store import ChromaVectorStore
from paper_agent.visual_retrieval import SubprocessEncoder


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip(): yield json.loads(line)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--visual-python', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root = Path('indexes/irpapers-clip-tiles-probe-20261107')
    metadata = json.loads((root / 'metadata.json').read_text(encoding='utf-8'))
    if (sha(root / 'mapping.jsonl') != metadata['contract']['mapping_sha256']
            or sha('runs/irpapers-tile-probe-20261107/plan.json') != metadata['contract']['plan_sha256']
            or sha('runs/irpapers-tile-probe-20261107/encoded/vectors.npy') != metadata['source_vectors_sha256']):
        raise ValueError('Frozen tile inputs do not match the index')
    read_frozen('runs/irpapers-chroma-baseline-v2-20261106')
    mapping = list(rows(root / 'mapping.jsonl'))
    plan = json.loads(Path('runs/irpapers-tile-probe-20261107/plan.json').read_text(encoding='utf-8'))
    records = {r['case_id']: r for r in rows('runs/irpapers-chroma-baseline-v2-20261106/text-results.jsonl') if r['partition'] == 'dev'}
    vectors = np.load('runs/irpapers-tile-probe-20261107/encoded/vectors.npy', allow_pickle=False)
    encoder = SubprocessEncoder(args.visual_python, 'models/multimodal-v1/clip-image-v1',
                                'models/multimodal-v1/clip-text-v1', 'cuda:0')
    if encoder.provenance != metadata['contract']['model_provenance']:
        raise ValueError('CLIP snapshot mismatch')
    store = ChromaVectorStore(root / 'vectors', metadata['contract'])
    queries = encoder.encode_texts([records[c['case_id']]['question'] for c in plan['cases']])
    result = []
    for case, vector in zip(plan['cases'], queries):
        found = store.search(vector, len(mapping))
        top = [key for key, _ in found]
        exact = np.argsort(-(vectors @ (vector / np.linalg.norm(vector))), kind='stable')
        expected = [mapping[int(i)]['tile_id'] for i in exact]
        result.append({'case_id': case['case_id'], 'tile_top5_set_matches': set(top[:5]) == set(expected[:5]),
                       'tile_top20_set_matches': set(top[:20]) == set(expected[:20]),
                       'search_strategy': store.last_search_mode})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump({'scope': '222 development-probe tiles only', 'results': result}, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__': main()
