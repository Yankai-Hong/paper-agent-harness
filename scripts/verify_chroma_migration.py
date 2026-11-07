"""Read-only same-vector FAISS/Chroma migration verification, not a model benchmark."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['bge','visual','clip-text'],required=True)
    p.add_argument('--source',required=True);p.add_argument('--destination',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();src=Path(args.source);dst=Path(args.destination);out=Path(args.output)
    if out.exists():raise ValueError('Choose a new output file')
    import faiss
    import numpy as np
    from paper_agent.storage import read_jsonl,digest
    from paper_agent.vector_store import ChromaVectorStore,ChromaIndexAdapter
    from paper_agent.visual_retrieval import visual_vector_id
    from paper_agent.clip_text_index import representation_vector_id
    before=json.loads((src/'metadata.json').read_text(encoding='utf8'))
    after=json.loads((dst/'metadata.json').read_text(encoding='utf8'))
    filename={'bge':'dense.faiss','visual':'visual.faiss','clip-text':'text.faiss'}[args.kind]
    sidecar={'bge':'chunks.jsonl','visual':'evidence.jsonl','clip-text':'representations.jsonl'}[args.kind]
    if hashlib.sha256((src/filename).read_bytes()).hexdigest()!=before['index_sha256']:
        raise ValueError('Legacy checksum mismatch')
    items=read_jsonl(src/sidecar); newitems=read_jsonl(dst/sidecar)
    if digest(items)!=digest(newitems):raise ValueError('Source mapping changed')
    ids=([x['chunk_id'] for x in items] if args.kind=='bge' else
         [visual_vector_id(x) for x in items] if args.kind=='visual' else [representation_vector_id(x) for x in items])
    old=faiss.read_index(str(src/filename)); store=ChromaVectorStore(dst/'vectors',after['vector_contract'])
    adapter=ChromaIndexAdapter(store,ids)
    if old.ntotal!=adapter.ntotal or old.d!=adapter.d:raise ValueError('Vector size mismatch')
    max_difference=0.;n=old.ntotal
    for start in range(0,n,256):
        batch=ids[start:start+256]; actual=store.collection.get(ids=batch,include=['embeddings'])
        mapping={key:np.asarray(v,dtype=np.float32) for key,v in zip(actual['ids'],actual['embeddings'])}
        vectors=old.reconstruct_n(start,len(batch))
        max_difference=max(max_difference,float(np.max(np.abs(vectors-np.stack([mapping[key] for key in batch])))))
    indices=sorted(set([0,n//4,n//2,3*n//4,n-1])); comparisons=[]
    for i in indices:
        q=old.reconstruct(i).reshape(1,-1);k=min(5,n)
        a_score,a_pos=old.search(q,k);b_score,b_pos=adapter.search(q,k)
        a=a_pos[0].tolist();b=b_pos[0].tolist()
        score_by_old={int(k):float(v) for k,v in zip(a_pos[0],a_score[0])}
        common=set(a)&set(b)
        comparisons.append({'query_vector_index':i,'faiss_top5':[ids[x] for x in a],
                            'chroma_top5':[ids[x] for x in b], 'exact_order':a==b,
                            'top5_overlap':len(common),'top1_equal':a[0]==b[0],
                            'max_common_score_difference':max((abs(score_by_old[int(pos)]-float(score))
                                for pos,score in zip(b_pos[0],b_score[0]) if int(pos) in common),default=None)})
    result={'kind':args.kind,'count':n,'dimension':old.d,'max_vector_abs_difference':max_difference,
            'faiss_metric':'normalized inner product, exact IndexFlatIP',
            'chroma_metric':'cosine distance converted as similarity=1-distance; HNSW approximate',
            'comparisons':comparisons}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps({'kind':args.kind,'count':n,'max_vector_abs_difference':max_difference,
                      'top1_equal':sum(x['top1_equal'] for x in comparisons),'queries':len(comparisons),
                      'top5_overlap':[x['top5_overlap'] for x in comparisons]}))


if __name__=='__main__':main()
