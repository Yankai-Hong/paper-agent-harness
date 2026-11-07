"""Persist an independently encoded CLIP tile probe in a fresh Chroma index."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from irpapers_tile_probe import sha
from paper_agent.vector_store import ChromaVectorStore


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--encoded', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    source, output = Path(args.encoded), Path(args.output)
    metadata = json.loads((source / 'results.json').read_text(encoding='utf-8'))
    if sha(source / 'vectors.npy') != metadata['vectors_sha256'] or sha(source / 'mapping.jsonl') != metadata['mapping_sha256']:
        raise ValueError('Tile encoding hash mismatch')
    with (source / 'mapping.jsonl').open(encoding='utf-8') as stream:
        mapping = [json.loads(line) for line in stream if line.strip()]
    vectors = np.load(source / 'vectors.npy', allow_pickle=False)
    if vectors.shape != (len(mapping), metadata['dimension']) or len(mapping) != metadata['tiles']:
        raise ValueError('Tile vectors do not match mappings')
    if output.exists():
        raise ValueError('Refusing to overwrite an existing tile index')
    output.mkdir(parents=True)
    contract = {'collection': 'paper_clip_tile_probe_v1', 'kind': 'clip_page_tiles_development_probe',
                'mapping_sha256': metadata['mapping_sha256'], 'plan_sha256': metadata['plan_sha256'],
                'model_provenance': metadata['model_provenance'], 'dimension': metadata['dimension'],
                'normalization': 'l2', 'metric': 'cosine'}
    store = ChromaVectorStore(output / 'vectors', contract, create=True)
    ids = [r['tile_id'] for r in mapping]
    metas = [{'paper_id': r['paper_id'], 'page_id': r['page_id'], 'page_number': r['page_number'],
              'x0': r['box_pixels'][0], 'y0': r['box_pixels'][1], 'x1': r['box_pixels'][2], 'y1': r['box_pixels'][3],
              'image_sha256': r['image_sha256']} for r in mapping]
    store.upsert(ids, vectors, metas)
    store.seal(ids)
    shutil.copyfile(source / 'mapping.jsonl', output / 'mapping.jsonl')
    result = {'contract': contract, 'count': len(ids), 'source_vectors_sha256': metadata['vectors_sha256'],
              'mapping_sha256': sha(output / 'mapping.jsonl'), 'vector_manifest_sha256': sha(output / 'vectors' / 'vector-store.json'),
              'scope': 'candidate-only development tile probe; not a full corpus index'}
    with (output / 'metadata.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
