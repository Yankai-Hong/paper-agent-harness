"""Build a separate CLIP page collection from verified rendered PDF pages."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('prepared', 'output', 'visual-python', 'image-model', 'text-model'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    from paper_agent.vector_store import ChromaVectorStore
    from paper_agent.visual_retrieval import SubprocessEncoder
    source = Path(a.prepared)
    binding = json.loads((source / 'binding.json').read_text(encoding='utf-8'))
    mapping = source / 'pages.jsonl'
    if binding.get('status') != 'sealed' or binding.get('pages_sha256') != sha(mapping):
        raise ValueError('Prepared page source mismatch')
    with mapping.open(encoding='utf-8') as stream:
        pages = [json.loads(line) for line in stream if line.strip()]
    paths = []
    for page in pages:
        path = (source / page['image_path']).resolve()
        if not path.is_relative_to(source.resolve()) or sha(path) != page['image_sha256']:
            raise ValueError('Rendered page source changed')
        paths.append(path)
    encoder = SubprocessEncoder(a.visual_python, a.image_model, a.text_model, device=a.device)
    started = time.perf_counter()
    vectors = encoder.encode_images(paths)
    if len(vectors) != len(pages):
        raise ValueError('Unexpected page vector count')
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(mapping, output / 'mapping.jsonl')
    contract = {'collection': 'paper_docling_page_clip_v1', 'kind': 'clip_page_image',
                'source_hash': sha(mapping), 'mapping_hash': sha(output / 'mapping.jsonl'),
                'model_provenance': encoder.provenance, 'dimension': int(vectors.shape[1]),
                'normalization': 'l2', 'metric': 'cosine', 'preprocessing': 'fixed CLIP model processor'}
    store = ChromaVectorStore(output / 'vectors', contract, create=True)
    ids = [page['page_id'] for page in pages]
    store.upsert(ids, vectors)
    store.seal(ids)
    metadata = {'backend': 'chroma', 'pages_sha256': sha(mapping), 'mapping_sha256': sha(output / 'mapping.jsonl'),
                'model_provenance': encoder.provenance, 'dimension': int(vectors.shape[1]),
                'vector_contract': contract, 'vector_manifest_sha256': sha(output / 'vectors' / 'vector-store.json'),
                'pages': len(pages), 'elapsed_seconds': time.perf_counter() - started}
    with (output / 'metadata.json').open('x', encoding='utf-8') as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'pages': len(pages), 'elapsed_seconds': metadata['elapsed_seconds']}))


if __name__ == '__main__':
    main()
