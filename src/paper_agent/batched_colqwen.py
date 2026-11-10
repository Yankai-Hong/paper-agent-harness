"""Lossless batched Chroma reads with page/token provenance validation."""
import numpy as np
from .colqwen_store import _matrix


def _read_pages(store, records, require_sealed):
    if (require_sealed and not store.final.is_file()) or not records or len(records) > 32:
        raise ValueError('A sealed store and 1..32 pages are required')
    ids = [f"{r['page_id']}:{i:06d}" for r in records for i in range(r['count'])]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate page token identities')
    result = store.collection.get(ids=ids, include=['embeddings', 'metadatas'])
    lookup = dict(zip(result['ids'], zip(result['embeddings'], result['metadatas'])))
    if len(result['ids']) != len(ids) or set(lookup) != set(ids):
        raise ValueError('Incomplete Chroma page-token batch')
    pages=[]
    for record in records:
        values=[]
        for i in range(record['count']):
            vector, meta = lookup[f"{record['page_id']}:{i:06d}"]
            expected={'page_id':record['page_id'],'paper_id':record['paper_id'],
                      'image_sha256':record['image_sha256'],'token_index':i}
            if meta != expected: raise ValueError('Page token provenance mismatch')
            values.append(vector)
        page=_matrix(values,store.contract['dimension'])
        reference=store._reference(record)
        if reference is None or not np.allclose(page,reference,atol=1e-6,rtol=0):
            raise ValueError('Page tokens differ from hashed input reference')
        pages.append(page)
    return pages


def read_pages(store, records):
    return _read_pages(store, records, True)


def seal_batched(store, *, patch_provenance_sha256, batch_pages=16):
    """Seal only after checking every committed token against its reference."""
    import json
    from .storage import digest
    if store.final.exists() or not 1 <= batch_pages <= 32:
        raise ValueError('Invalid seal request')
    pending={p.stem for p in store.pending.glob('*.json')}
    committed={p.stem for p in store.committed.glob('*.json')}
    if not committed or pending != committed:
        raise ValueError('Cannot seal with missing or unfinished pages')
    records=[json.loads((store.committed/(page+'.json')).read_text(encoding='utf-8')) for page in sorted(committed)]
    if store.collection.count()!=sum(r['count'] for r in records):
        raise ValueError('Unexpected token count or orphan writes')
    for start in range(0,len(records),batch_pages):
        _read_pages(store,records[start:start+batch_pages],False)
        if (start+batch_pages)%400==0: print('seal pages verified',start+batch_pages,'of',len(records),flush=True)
    path=store.root/'patch-provenance.jsonl'
    if not path.is_file() or store._file_hash(path)!=patch_provenance_sha256:
        raise ValueError('Patch provenance file hash mismatch')
    final={'schema':1,'contract_hash':digest(store.contract),'pages':len(records),
           'tokens':store.collection.count(),'page_records_hash':digest(records),
           'patch_provenance_sha256':patch_provenance_sha256}
    store._write_new(store.final,final)
