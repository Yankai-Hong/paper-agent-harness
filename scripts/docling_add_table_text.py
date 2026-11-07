"""Extend one shared text index with table text while reusing body vectors."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('text-index', 'prepared', 'evidence', 'evidence-root', 'output', 'binding-output'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    from paper_agent.retrieval import BGEEncoder
    from paper_agent.storage import digest, read_jsonl, write_json, write_jsonl
    from paper_agent.vector_store import ChromaVectorStore
    from paper_agent.visual_evidence import validate_image
    root, source = Path(a.text_index), Path(a.prepared)
    metadata = json.loads((root / 'metadata.json').read_text(encoding='utf-8'))
    binding = json.loads((source / 'binding.json').read_text(encoding='utf-8'))
    chunks = read_jsonl(root / 'chunks.jsonl')
    if (metadata.get('backend') != 'chroma' or metadata.get('corpus_hash') != digest(chunks)
            or binding.get('status') != 'sealed' or binding.get('pages_sha256') != sha(source / 'pages.jsonl')
            or binding.get('text_chunks_sha256') != sha(root / 'chunks.jsonl')
            or binding.get('text_corpus_hash') != metadata['corpus_hash']):
        raise ValueError('Text index and PDF source binding mismatch')
    pages = read_jsonl(source / 'pages.jsonl')
    locators = {(r['paper_id'], r['pdf_sha256'], r['page_number']): r['page_id'] for r in pages}
    if len(locators) != len(pages):
        raise ValueError('Ambiguous page provenance')
    existing = {x['chunk_id'] for x in chunks}
    additions = []
    for visual in read_jsonl(a.evidence):
        if visual.get('evidence_type') != 'table' or not isinstance(visual.get('table_text'), str):
            continue
        page_id = locators.get((visual.get('paper_id'), visual.get('pdf_sha256'), visual.get('page')))
        if page_id is None or visual.get('table_text_status') != 'docling_extracted_unverified':
            continue
        validate_image(visual, a.evidence_root)
        content = visual['table_text'].strip()
        caption = (visual.get('caption') or '').strip()
        if caption and content.startswith(caption):
            content = content[len(caption):].lstrip()
        if not content:
            continue
        chunk_id = 'table-text-' + digest([visual['paper_id'], visual['chunk_id']])[:24]
        if chunk_id in existing:
            raise ValueError('Duplicate table text identity')
        existing.add(chunk_id)
        additions.append({'paper_id': visual['paper_id'], 'chunk_id': chunk_id, 'page_id': page_id,
            'title': visual.get('title'), 'evidence_type': 'text', 'source_kind': 'table_text',
            'text': content, 'page': visual['page'], 'bbox': visual['bbox'], 'page_size': visual['page_size'],
            'coordinate_system': visual['coordinate_system'], 'page_rotation': visual['page_rotation'],
            'pdf_sha256': visual['pdf_sha256'], 'parser': visual.get('parser'), 'docling_ref': visual.get('docling_ref'),
            'docling_label': 'table', 'parse_status': 'docling_table_text_unverified',
            'citation_granularity': 'table', 'source_visual_chunk_id': visual['chunk_id']})
    if not additions:
        raise ValueError('No valid table text sources')
    model_path = Path(metadata['encoder'])
    provenance_path = model_path / 'provenance.json'
    if metadata.get('model_provenance') and (not provenance_path.is_file()
            or json.loads(provenance_path.read_text(encoding='utf-8')) != metadata['model_provenance']):
        raise ValueError('Text encoder provenance differs')
    store = ChromaVectorStore(root / 'vectors', metadata['vector_contract'])
    ids = [x['chunk_id'] for x in chunks]
    retrieved = store.collection.get(ids=ids, include=['embeddings'])
    if set(retrieved['ids']) != set(ids):
        raise ValueError('Existing text vectors incomplete')
    by_id = dict(zip(retrieved['ids'], retrieved['embeddings']))
    body = np.asarray([by_id[key] for key in ids], dtype=np.float32)
    started = time.perf_counter()
    encoder = BGEEncoder(metadata['encoder'], a.device, 2, metadata['max_length'])
    table_vectors = encoder.encode([x['text'] for x in additions])
    combined = chunks + additions
    output, binding_output = Path(a.output), Path(a.binding_output)
    if output.exists() or binding_output.exists():
        raise ValueError('Choose new output paths')
    output.mkdir(parents=True)
    contract = copy.deepcopy(metadata['vector_contract'])
    contract['corpus_hash'] = digest(combined)
    target = ChromaVectorStore(output / 'vectors', contract, create=True)
    target.upsert(ids, body)
    target.upsert([x['chunk_id'] for x in additions], table_vectors)
    target.seal(ids + [x['chunk_id'] for x in additions])
    write_jsonl(output / 'chunks.jsonl', combined)
    updated = copy.deepcopy(metadata)
    updated.update(corpus_hash=contract['corpus_hash'], chunks=len(combined), vector_contract=contract,
                   extended_from={'source_corpus_hash': metadata['corpus_hash'], 'body_vectors_reencoded': False,
                                  'table_source_sha256': sha(a.evidence), 'table_count': len(additions)})
    write_json(output / 'metadata.json', updated)
    new_binding = {**binding, 'text_chunks_sha256': sha(output / 'chunks.jsonl'),
                   'text_corpus_hash': contract['corpus_hash'], 'table_source_sha256': sha(a.evidence),
                   'table_text_count': len(additions)}
    with binding_output.open('x', encoding='utf-8') as stream:
        json.dump(new_binding, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'body_vectors_reused': len(ids), 'new_table_texts': len(additions),
                      'elapsed_seconds': time.perf_counter() - started, 'corpus_hash': contract['corpus_hash']}))


if __name__ == '__main__':
    main()
