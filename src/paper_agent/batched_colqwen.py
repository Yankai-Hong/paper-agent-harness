"""Lossless batched Chroma reads with page/token provenance validation."""
import numpy as np
from .colqwen_store import _matrix, _digest


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


def add_pages_batched(store, pages, *, token_batch=512):
    """Commit pages after resumable token writes and a verified batched read."""
    import json
    import re
    if store.final.exists() or not pages or len(pages)>16 or not 1<=token_batch<=512:
        raise ValueError('Invalid page batch')
    if len({p['page_id'] for p in pages})!=len(pages): raise ValueError('Duplicate pages')
    records=[]; arrays=[]
    for page in pages:
        key, paper, fingerprint = page['page_id'], page['paper_id'], page['image_sha256']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',key) or not paper or not re.fullmatch(r'[0-9a-f]{64}',fingerprint):
            raise ValueError('Invalid page source')
        array=_matrix(page['vectors'],store.contract['dimension'])
        record={'page_id':key,'paper_id':paper,'image_sha256':fingerprint,'count':len(array),'vectors_sha256':_digest(array)}
        pending=store.pending/(key+'.json'); committed=store.committed/(key+'.json')
        if pending.exists():
            if json.loads(pending.read_text(encoding='utf-8'))!=record: raise ValueError('Pending page differs')
        else: store._write_new(pending,record)
        store._ensure_reference(record,array)
        if committed.exists():
            old=json.loads(committed.read_text(encoding='utf-8'))
            if any(old.get(k)!=v for k,v in record.items()): raise ValueError('Committed page differs')
        records.append(record); arrays.append(array)
    ids=[]; values=[]; metadata=[]
    for record,array in zip(records,arrays):
        if (store.committed/(record['page_id']+'.json')).exists(): continue
        for i,vector in enumerate(array):
            ids.append(f"{record['page_id']}:{i:06d}"); values.append(vector)
            metadata.append({'page_id':record['page_id'],'paper_id':record['paper_id'],
                             'image_sha256':record['image_sha256'],'token_index':i})
    for start in range(0,len(ids),token_batch):
        stop=min(start+token_batch,len(ids))
        store.collection.upsert(ids=ids[start:stop],embeddings=np.asarray(values[start:stop]).tolist(),
                                metadatas=metadata[start:stop])
    checked=_read_pages(store,records,False)
    for record,page in zip(records,checked):
        committed=store.committed/(record['page_id']+'.json')
        if not committed.exists(): store._write_new(committed,{**record,'stored_vectors_sha256':_digest(page)})


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
