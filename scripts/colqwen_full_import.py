"""Import all real IRPAPERS page tokens into a separately sealed Chroma store."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from colqwen_encode_candidates import check_marker
from colqwen_smoke import rows, sha
from paper_agent.resumable_colqwen import ResumableColQwenStore
from paper_agent.batched_colqwen import seal_batched
from paper_agent.colqwen_store import _digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','encoded','dev-index','test-index','adapter','index'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    import psutil
    source, encoded, index = Path(a.prepared), Path(a.encoded), Path(a.index)
    config = json.loads((encoded / 'contract.json').read_text(encoding='utf-8'))
    pages = rows(source / 'pages.jsonl'); enc = config['encoder']
    if len(pages) != 3230 or sha(source / 'pages.jsonl') != config['source_pages_sha256']:
        raise ValueError('Full source differs')
    preprocessor = Path(a.adapter) / 'preprocessor_config.json'
    if sha(preprocessor) != '4ddb56e63fbb08cde866dde534c8e4d9fbd926df087be33ff7a74a32040432f6':
        raise ValueError('Processor configuration differs from recorded snapshot')
    merge = json.loads(preprocessor.read_text(encoding='utf-8'))['merge_size']
    refs = {}
    for root in (Path(a.dev_index),Path(a.test_index)):
        seal = json.loads((root / 'sealed.json').read_text(encoding='utf-8'))
        if sha(root / 'patch-provenance.jsonl') != seal['patch_provenance_sha256']:
            raise ValueError('Original token provenance differs')
        for r in rows(root / 'patch-provenance.jsonl'):
            refs.setdefault((r['page_id'], r['vector_file_sha256']), r)
    contract = {'model_revision':enc['model_revision'],'base_revision':enc['base_revision'],
                'source_hash':config['source_pages_sha256'],'dimension':enc['dimension'],
                'weight_sha256':enc['weight_sha256'],'max_visual_tokens':enc['max_visual_tokens'],
                'full_encoding_contract_sha256':sha(encoded / 'contract.json'), 'scope':'full_corpus'}
    store = ResumableColQwenStore(index, contract, create=not index.exists())
    started = time.perf_counter(); provenance=[]
    for i, page in enumerate(pages, 1):
        key=page['page_id']; marker_path=encoded / 'pages' / (key+'.json')
        image_path=(source / page['image_path']).resolve()
        if not image_path.is_relative_to(source.resolve()) or sha(image_path)!=page['image_sha256']:
            raise ValueError('Source page image changed: '+key)
        if not check_marker(encoded / 'pages', key, {'page_id':key,'paper_id':page['paper_id'],'image_sha256':page['image_sha256']}):
            raise ValueError('Missing page vector: '+key)
        marker=json.loads(marker_path.read_text(encoding='utf-8'))
        ref=refs.get((key,marker['vectors_sha256']), marker)
        positions=ref.get('image_token_positions'); grid=marker['image_grid_thw']; count=marker['token_count']
        if (not isinstance(positions,list) or positions != sorted(set(positions)) or not positions
            or any(type(x) is not int or not 0 <= x < count for x in positions)
            or len(grid)!=1 or len(grid[0])!=3 or grid[0][1]%merge or grid[0][2]%merge
            or len(positions)!=grid[0][0]*(grid[0][1]//merge)*(grid[0][2]//merge)):
            raise ValueError('Invalid image token mask: '+key)
        with np.load(encoded / 'pages' / marker['vector_file'],allow_pickle=False) as z: vectors=np.asarray(z['vectors'],dtype=np.float32)
        if vectors.shape != (count, enc['dimension']): raise ValueError('Vector shape mismatch')
        if not store.final.exists():
            committed=store.committed/(key+'.json')
            if committed.exists():
                previous=json.loads(committed.read_text(encoding='utf-8'))
                if (previous.get('paper_id')!=page['paper_id'] or previous.get('image_sha256')!=page['image_sha256']
                        or previous.get('count')!=count or previous.get('vectors_sha256')!=_digest(vectors)):
                    raise ValueError('Committed page differs from checkpoint: '+key)
            else: store.add_page(key,page['paper_id'],page['image_sha256'],vectors)
        else: store.get_page(key)
        provenance.append({'page_id':key,'paper_id':page['paper_id'],'image_sha256':page['image_sha256'],
            'marker_sha256':sha(marker_path),'vector_file_sha256':marker['vectors_sha256'],
            'image_grid_thw':grid,'image_token_positions':positions,'active_token_count':count})
        if i%100==0: print('pages verified or committed',i,'of',len(pages),flush=True)
    content=''.join(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n' for r in provenance).encode()
    fingerprint=hashlib.sha256(content).hexdigest(); sidecar=index/'patch-provenance.jsonl'
    if sidecar.exists():
        if sha(sidecar)!=fingerprint: raise ValueError('Full provenance differs')
    else:
        with sidecar.open('xb') as f: f.write(content)
    if not store.final.exists(): seal_batched(store,patch_provenance_sha256=fingerprint)
    print(json.dumps({'status':'complete','pages':len(pages),'tokens':store.collection.count(),
        'seal_sha256':sha(store.final),'elapsed_seconds':time.perf_counter()-started,
        'rss_bytes':psutil.Process().memory_info().rss,'index_bytes':sum(p.stat().st_size for p in index.rglob('*') if p.is_file())}),flush=True)


if __name__ == '__main__': main()
