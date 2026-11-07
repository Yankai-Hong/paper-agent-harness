"""Copy verified legacy FAISS vectors to a new isolated Chroma index."""
import argparse
import copy
import hashlib
import json
import shutil
import time
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['bge','visual','clip-text'],required=True)
    p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    args=p.parse_args(); source=Path(args.source); output=Path(args.output)
    if output.exists():raise ValueError('Refusing to overwrite destination')
    from paper_agent.storage import read_jsonl,digest,write_json
    from paper_agent.visual_retrieval import visual_vector_id
    from paper_agent.clip_text_index import representation_vector_id
    from paper_agent.vector_store import ChromaVectorStore
    import faiss
    meta=json.loads((source/'metadata.json').read_text(encoding='utf8'))
    if meta.get('backend','faiss')!='faiss': raise ValueError('Source is not a legacy FAISS index')
    if args.kind=='bge':
        if not meta.get('dense'):raise ValueError('Source has no dense vectors')
        sidecars=['chunks.jsonl']; items=read_jsonl(source/sidecars[0]); filename='dense.faiss'
        if digest(items)!=meta.get('corpus_hash'):raise ValueError('Text corpus hash mismatch')
        ids=[x['chunk_id'] for x in items]
        contract={'collection':'paper_bge_text_v1','kind':'bge_text','corpus_hash':meta['corpus_hash'],
                  'model_id':meta['encoder'],'model_provenance':meta.get('model_provenance'),'max_length':meta['max_length'],
                  'dimension':meta['dimension'],'normalization':'l2','metric':'cosine'}
    elif args.kind=='visual':
        sidecars=['evidence.jsonl'];items=read_jsonl(source/sidecars[0]);filename='visual.faiss'
        if digest(items)!=meta.get('corpus_hash'):raise ValueError('Visual corpus hash mismatch')
        ids=[visual_vector_id(x) for x in items]
        contract={'collection':'paper_clip_images_v1','kind':'clip_figure_image','corpus_hash':meta['corpus_hash'],
                  'model_provenance':meta['model_provenance'],'dimension':meta['dimension'],'normalization':'l2','metric':'cosine'}
    else:
        sidecars=['sources.jsonl','representations.jsonl'];sources=read_jsonl(source/sidecars[0]);items=read_jsonl(source/sidecars[1]);filename='text.faiss'
        if digest(sources)!=meta.get('source_hash') or digest(items)!=meta.get('representation_hash'):
            raise ValueError('CLIP text source or representation hash mismatch')
        source_by_key={(x['paper_id'],x['chunk_id']):x for x in sources}
        for x in items:
            s=source_by_key.get((x['paper_id'],x['chunk_id']))
            if s is None or not 0<=x['start']<x['end']<=len(s['text']) or s['text'][x['start']:x['end']]!=x['text']:
                raise ValueError('Invalid representation source mapping')
        ids=[representation_vector_id(x) for x in items]
        contract={'collection':'paper_clip_text_v1','kind':'clip_located_text','source_hash':meta['source_hash'],
                  'representation_hash':meta['representation_hash'],'model_provenance':meta['model_provenance'],
                  'dimension':meta['dimension'],'normalization':'l2','metric':'cosine','preprocessing':meta['preprocessing']}
    if len(ids)!=len(set(ids)) or not ids:raise ValueError('Source vector identities not unique')
    index_path=source/filename
    if sha(index_path)!=meta.get('index_sha256'):raise ValueError('Legacy index checksum mismatch')
    index=faiss.read_index(str(index_path))
    if index.metric_type!=faiss.METRIC_INNER_PRODUCT or index.ntotal!=len(ids) or index.d!=meta['dimension']:
        raise ValueError('Legacy index metric, dimension or count mismatch')
    begin=time.monotonic(); output.mkdir(parents=True,exist_ok=False)
    store=ChromaVectorStore(output/'vectors',contract,create=True)
    for start in range(0,len(ids),256):
        vectors=index.reconstruct_n(start,min(256,len(ids)-start))
        store.upsert(ids[start:start+len(vectors)],vectors)
    store.seal(ids)
    for name in sidecars:
        shutil.copyfile(source/name,output/name)
        if sha(source/name)!=sha(output/name):raise ValueError('Copied sidecar checksum mismatch')
    new=copy.deepcopy(meta);new['backend']='chroma';new['vector_contract']=contract
    new['migrated_from']={'backend':'faiss','index_sha256':new.pop('index_sha256'),
                          'metadata_sha256':sha(source/'metadata.json')}
    write_json(output/'metadata.json',new)
    print(json.dumps({'kind':args.kind,'source':str(source),'destination':str(output),'count':len(ids),
                      'dimension':index.d,'elapsed_seconds':time.monotonic()-begin,
                      'vector_hash':store.manifest['vectors_hash']}))


if __name__=='__main__': main()
