"""Verify reused text vectors and trace each table representation to Docling."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('old-index', 'new-index', 'evidence', 'evidence-root', 'output'):
        p.add_argument('--' + key, required=True)
    a = p.parse_args()
    from paper_agent.storage import read_jsonl
    from paper_agent.vector_store import ChromaVectorStore
    from paper_agent.visual_evidence import validate_image
    old, new = Path(a.old_index), Path(a.new_index)
    old_meta = json.loads((old / 'metadata.json').read_text(encoding='utf-8'))
    new_meta = json.loads((new / 'metadata.json').read_text(encoding='utf-8'))
    old_chunks, new_chunks = read_jsonl(old / 'chunks.jsonl'), read_jsonl(new / 'chunks.jsonl')
    if new_chunks[:len(old_chunks)] != old_chunks:
        raise ValueError('Original text evidence changed')
    old_store = ChromaVectorStore(old / 'vectors', old_meta['vector_contract'])
    new_store = ChromaVectorStore(new / 'vectors', new_meta['vector_contract'])
    ids = [x['chunk_id'] for x in old_chunks]
    def vectors(store):
        result = store.collection.get(ids=ids, include=['embeddings'])
        if set(result['ids']) != set(ids):
            raise ValueError('Missing reused body vector')
        return {key: np.asarray(value, dtype=np.float32) for key, value in zip(result['ids'], result['embeddings'])}
    first, second = vectors(old_store), vectors(new_store)
    difference = max(float(np.max(np.abs(first[key] - second[key]))) for key in ids)
    if difference > 1e-6:
        raise ValueError('Reused body vectors changed beyond tolerance')
    sources = {(r.get('paper_id'), r.get('chunk_id')): r for r in read_jsonl(a.evidence)}
    tables = new_chunks[len(old_chunks):]
    for table in tables:
        source = sources.get((table['paper_id'], table['source_visual_chunk_id']))
        if not source or source.get('table_text_status') != 'docling_extracted_unverified':
            raise ValueError('Table representation lacks its Docling source')
        validate_image(source, a.evidence_root)
        content = source['table_text'].strip()
        caption = (source.get('caption') or '').strip()
        if caption and content.startswith(caption):
            content = content[len(caption):].lstrip()
        if (table['text'] != content or table.get('parse_status') != 'docling_table_text_unverified'
                or any(table.get(k) != source.get(k) for k in ('paper_id', 'page', 'bbox', 'page_size', 'pdf_sha256'))):
            raise ValueError('Table text or source location differs')
    report = {'status': 'complete', 'body_chunks': len(ids), 'table_texts': len(tables),
              'body_vector_max_abs_difference': difference, 'meaning': 'source and vector checks; not semantic validation'}
    with Path(a.output).open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
