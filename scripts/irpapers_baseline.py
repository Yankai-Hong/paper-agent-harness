"""Freeze and run full-corpus IRPAPERS text and page-CLIP baselines."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()


def rows(path):
    with Path(path).open(encoding='utf8') as f: return [json.loads(x) for x in f if x.strip()]


def write_new(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f: json.dump(value,f,ensure_ascii=False,indent=2)


def queries(source):
    out=[]
    for label,name in [('standard','irpapers-queries.csv'),('visual','irpapers-visual-queries.csv')]:
        with (Path(source)/name).open(encoding='utf-8-sig',newline='') as f:
            for i,row in enumerate(csv.DictReader(f)):
                out.append({'case_id':f'{label}-{i+1:03d}','group':label,'question':row['question'],
                            'answer':row['answer'],'gold_page':row['dataset_id'],'pdf_id':row['pdf_id'],
                            'query_page_number':row['page_number']})
    return out


def plan_load(path):
    plan=json.loads(Path(path).read_text(encoding='utf8'))
    for x in plan['code_hashes']:
        if sha(x)!=plan['code_hashes'][x]: raise ValueError('Frozen baseline implementation changed: '+x)
    return plan


def freeze(args):
    audit=json.loads((Path(args.prepared)/'audit.json').read_text(encoding='utf8'))
    if audit['files']['irpapers-docs.csv']['sha256']!='8fe316290b55c71bd1641603b5feb02117672028f7d1af45bcad1c111c53611b':
        raise ValueError('Local docs differ from observed HF LFS revision')
    source=Path(args.source)
    blob_expected={'irpapers-queries.csv':'3f13852aaa8fb5490ba817b7b91c7b10d75c2955',
                   'irpapers-visual-queries.csv':'189dcab6ebb09de6206b30086dbf350a37b668ff'}
    for filename in blob_expected:
        if sha(source/filename)!=audit['files'][filename]['sha256']: raise ValueError('Query CSV changed')
        contents=(source/filename).read_bytes()
        if hashlib.sha1(('blob '+str(len(contents))+'\0').encode()+contents).hexdigest()!=blob_expected[filename]:
            raise ValueError('Local query CSV differs from observed HF revision')
    qs=queries(source); pages=rows(Path(args.prepared)/'pages.jsonl')
    by_page={x['page_id']:x for x in pages}; by_pair={(x['pdf_id'],str(x['page_number'])):x['page_id'] for x in pages}
    paper_ids={q['pdf_id'] for q in qs}|{by_page[q['gold_page']]['pdf_id'] for q in qs if q['gold_page'] in by_page}
    # Conflicting dataset_id and (pdf_id,page_number) labels must remain in the
    # same partition, even though their gold cannot be resolved automatically.
    parent={x:x for x in paper_ids}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    for q in qs:
        target=by_page.get(q['gold_page'])
        if target is not None:
            a,b=find(q['pdf_id']),find(target['pdf_id'])
            if a!=b: parent[max(a,b)]=min(a,b)
    groups={}
    for pid in sorted(paper_ids): groups.setdefault(find(pid),[]).append(pid)
    components=sorted(groups.values());random.Random(20261106).shuffle(components)
    split=max(1,int(len(components)*.75));dev={pid for group in components[:split] for pid in group}
    cases=[{'case_id':q['case_id'],'group':q['group'],'partition':'dev' if q['pdf_id'] in dev else 'test',
            'pdf_id':q['pdf_id'],'gold_page':q['gold_page'],'query_page_number':q['query_page_number'],
            'gold_status':('missing' if q['gold_page'] not in by_page else 'identity_conflict'
                if (by_page[q['gold_page']]['pdf_id']!=q['pdf_id'] or str(by_page[q['gold_page']]['page_number'])!=q['query_page_number']) else 'consistent'),
            'alternate_page':by_pair.get((q['pdf_id'],q['query_page_number'])),
            'question_sha256':hashlib.sha256(q['question'].encode()).hexdigest(),
            'answer_sha256':hashlib.sha256(q['answer'].encode()).hexdigest()} for q in qs]
    from paper_agent.visual_retrieval import _model_manifest
    visual_provenance={'image':_model_manifest('models/multimodal-v1/clip-image-v1'),
                       'text':_model_manifest('models/multimodal-v1/clip-text-v1')}
    bge=json.loads(Path('models/bge-m3/provenance.json').read_text(encoding='utf8'))
    bge_weights=sha('models/bge-m3/pytorch_model.bin')
    if bge_weights!=bge['large_file_hashes']['pytorch_model.bin']: raise ValueError('BGE model weight checksum mismatch')
    visual_versions=json.loads(subprocess.check_output(['.venv-mm-20261030/Scripts/python.exe','-c',
        "import importlib.metadata as m,json;print(json.dumps({n:m.version(n) for n in ('sentence-transformers','torch','transformers','Pillow')}))"],text=True))
    names=['scripts/irpapers_prepare.py','scripts/irpapers_baseline.py','scripts/irpapers_clip_pages.py',
           'src/paper_agent/corpus.py','src/paper_agent/retrieval.py','src/paper_agent/vector_store.py',
           'src/paper_agent/storage.py','src/paper_agent/visual_retrieval.py',
           'scripts/clip_worker.py']
    plan={'schema':1,'dataset_revision':audit['hf_revision_observed'], 'source_file_hashes':{k:v['sha256'] for k,v in audit['files'].items()},
          'prepared_pages_sha256':audit['pages_sha256'], 'split':{'method':'paper-disjoint seeded shuffle of union of both query sets',
          'seed':20261106,'dev_fraction_components':.75,'dev_papers':sorted(dev),'test_papers':sorted(set(paper_ids)-dev),
          'conflicting_label_papers_grouped':True},
          'query_git_blob_sha1':blob_expected,'cases':cases,'configuration':{'text_model':'models/bge-m3','bge_provenance_sha256':sha('models/bge-m3/provenance.json'),
          'bge_weights_sha256':bge_weights,'clip_model_provenance':visual_provenance,
          'chunk_tokens':500,'overlap_tokens':80,'bge_max_length':768,'bge_device':'cuda:0','bge_batch_size':2,
          'bm25':'rank_bm25 BM25Okapi lexical-v1', 'dense_backend':'Chroma 0.6.3 HNSW cosine; explicit exact scan of Chroma vectors if full ranking exceeds HNSW limits',
          'chunk_fusion':'project RRF k=60',
          'text_candidate_depth':'all indexed chunks','page_aggregation':'first ranked chunk per page',
          'visual_model':'models/multimodal-v1/clip-image-v1','visual_text_model':'models/multimodal-v1/clip-text-v1',
          'visual_device':'cuda:0','visual_batch_size':8,'visual_backend':'Chroma 0.6.3 HNSW cosine; explicit exact scan of Chroma vectors if full ranking exceeds HNSW limits',
          'page_fusion':'RRF k=60 after page deduplication',
          'retrieval_scope':'all available corpus pages; never filter by gold','metric':'single labeled source page',
          'reference_answers':'scoring only; never included in retrieval inputs'},
          'runtime_versions':{name:importlib.metadata.version(name) for name in ('chromadb','numpy','rank-bm25','torch','transformers','FlagEmbedding')},
          'visual_runtime_versions':visual_versions,
          'code_hashes':{n:sha(n) for n in names}}
    write_new(args.output,plan)
    print(json.dumps({'plan':args.output,'plan_sha256':sha(args.output),'cases':len(cases),
          'dev_papers':len(dev),'test_papers':len(paper_ids)-len(dev),
          'ambiguous_gold_cases':sum(c['gold_status']=='identity_conflict' for c in cases)}))


def chunks(args):
    plan=plan_load(args.plan)
    if sha(Path(args.prepared)/'pages.jsonl')!=plan['prepared_pages_sha256']: raise ValueError('Prepared pages changed')
    from transformers import AutoTokenizer
    from paper_agent.corpus import Chunker
    tokenizer=AutoTokenizer.from_pretrained(plan['configuration']['text_model'],use_fast=True,local_files_only=True)
    chunker=Chunker(500,80,tokenizer,tokenizer_id='BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181')
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    count=empty=0; start=time.monotonic()
    with output.open('x',encoding='utf8',newline='\n') as dest:
        for page in rows(Path(args.prepared)/'pages.jsonl'):
            if not page['text'].strip(): empty+=1; continue
            paper={'paper_id':page['paper_id'],'title':page['title'] or page['paper_id'],
                   'paragraphs':[{'paragraph_id':page['page_id'],'page':page['page_number'],'text':page['text']}]}
            for c in chunker.split(paper):
                c.update(page_id=page['page_id'],page_number=page['page_number'],pdf_id=page['pdf_id'],
                         source_type='irpapers_model_transcription_not_pdf_or_docling',
                         page_charspan=[c['spans'][0]['start'],c['spans'][0]['end']])
                dest.write(json.dumps(c,ensure_ascii=False)+'\n'); count+=1
    write_new(str(output)+'.metadata.json',{'chunks':count,'empty_pages':empty,'sha256':sha(output),
              'plan_sha256':sha(args.plan),'elapsed_seconds':time.monotonic()-start})
    print('chunks',count,'empty_pages',empty,flush=True)


def text_build(args):
    plan=plan_load(args.plan); out=Path(args.output)
    if out.exists(): raise ValueError('Refusing to overwrite text index')
    chunks_meta=json.loads(Path(str(args.chunks)+'.metadata.json').read_text(encoding='utf8'))
    if chunks_meta['sha256']!=sha(args.chunks) or chunks_meta['plan_sha256']!=sha(args.plan): raise ValueError('Chunk provenance mismatch')
    if sha('models/bge-m3/pytorch_model.bin')!=plan['configuration']['bge_weights_sha256']:
        raise ValueError('BGE weights changed after freeze')
    from paper_agent.retrieval import BGEEncoder,build_index
    start=time.monotonic(); enc=BGEEncoder('models/bge-m3','cuda:0',2,768)
    import torch
    torch.cuda.reset_peak_memory_stats()
    meta=build_index(args.chunks,out,enc)
    import psutil
    write_new(out/'benchmark.json',{'plan_sha256':sha(args.plan),'chunks_sha256':sha(args.chunks),
              'elapsed_seconds':time.monotonic()-start,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
              'process_rss_bytes_final':psutil.Process().memory_info().rss,
              'index_bytes_before_benchmark_metadata':sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
              'index_metadata':meta})
    print('text index built',meta['chunks'],'dimension',meta['dimension'],flush=True)


def adopt_text_index(args):
    """Copy verified vectors into a new snapshot after an infrastructure fix."""
    plan=plan_load(args.plan); old=json.loads(Path(args.old_plan).read_text(encoding='utf8'))
    for key in ('source_file_hashes','prepared_pages_sha256','split','cases'):
        if plan[key]!=old[key]:raise ValueError('Reused baseline data or split differs')
    old_config=dict(old['configuration']);new_config=dict(plan['configuration'])
    for key in ('dense_backend','visual_backend'):
        old_config.pop(key,None);new_config.pop(key,None)
    if old_config!=new_config:raise ValueError('Reused baseline model or retrieval configuration differs')
    source=Path(args.source_index);output=Path(args.output)
    if output.exists():raise ValueError('Refusing to overwrite adopted index')
    source_meta=json.loads((source/'metadata.json').read_text(encoding='utf8'))
    old_benchmark=json.loads((source/'benchmark.json').read_text(encoding='utf8'))
    if old_benchmark['plan_sha256']!=sha(args.old_plan) or source_meta.get('backend')!='chroma':
        raise ValueError('Original index does not match prior frozen plan')
    from paper_agent.storage import digest
    from paper_agent.vector_store import ChromaVectorStore
    import numpy as np
    chunks=rows(source/'chunks.jsonl')
    if digest(chunks)!=source_meta['corpus_hash']:raise ValueError('Original text corpus hash mismatch')
    contract=source_meta['vector_contract'];src=ChromaVectorStore(source/'vectors',contract)
    output.mkdir(parents=True,exist_ok=False);dest=ChromaVectorStore(output/'vectors',contract,create=True)
    ids=[x['chunk_id'] for x in chunks];start=time.monotonic()
    for i in range(0,len(ids),256):
        batch=ids[i:i+256];result=src.collection.get(ids=batch,include=['embeddings'])
        by_id={key:np.asarray(v,dtype=np.float32) for key,v in zip(result['ids'],result['embeddings'])}
        if set(by_id)!=set(batch):raise ValueError('Source vector mapping incomplete')
        dest.upsert(batch,np.stack([by_id[key] for key in batch]))
    dest.seal(ids)
    for name in ('chunks.jsonl','metadata.json'):
        shutil.copyfile(source/name,output/name)
        if sha(source/name)!=sha(output/name):raise ValueError('Copied text snapshot mismatch')
    write_new(output/'benchmark.json',{'plan_sha256':sha(args.plan),'source_plan_sha256':sha(args.old_plan),
        'source_index':str(source),'source_vector_hash':src.manifest['vectors_hash'],
        'adopted_vector_hash':dest.manifest['vectors_hash'],'source_chunks_sha256':sha(source/'chunks.jsonl'),
        'adoption_elapsed_seconds':time.monotonic()-start,'original_index_metadata':source_meta})
    print('adopted verified text vectors',len(ids),flush=True)


def metrics(items,ranking_field):
    result={}
    sets=[('standard_all',lambda r:r['group']=='standard'),('standard_dev',lambda r:r['group']=='standard' and r['partition']=='dev'),
          ('standard_test',lambda r:r['group']=='standard' and r['partition']=='test'),
          ('visual_all',lambda r:r['group']=='visual'),('visual_dev',lambda r:r['group']=='visual' and r['partition']=='dev'),
          ('visual_test',lambda r:r['group']=='visual' and r['partition']=='test')]
    seen=set(); union=[]
    for r in items:
        key=(r['gold_page'],' '.join(r['question'].casefold().split()))
        if key not in seen: seen.add(key); union.append(r)
    for label,predicate in sets:
        part=[r for r in items if predicate(r)]; n=len(part)
        entries=[r.get(ranking_field,{}) for r in part]
        known=[x.get('gold_rank') for x in entries]
        result[label]={'denominator':n,'ok':sum(x.get('status')=='ok' for x in entries),
          'gold_in_index':sum(x.get('gold_in_index') is True for x in entries),
          'ambiguous_gold':sum(r.get('gold_status')=='identity_conflict' for r in part),
          'hit_at_1':sum(r is not None and r<=1 for r in known)/n if n else None,
          'hit_at_5':sum(r is not None and r<=5 for r in known)/n if n else None,
          'hit_at_20':sum(r is not None and r<=20 for r in known)/n if n else None,
          'mrr':sum(1/r if r else 0 for r in known)/n if n else None}
    n=len(union); entries=[r.get(ranking_field,{}) for r in union]; ranks=[x.get('gold_rank') for x in entries]
    result['deduplicated_union']={'denominator':n,'ok':sum(x.get('status')=='ok' for x in entries),
        'gold_in_index':sum(x.get('gold_in_index') is True for x in entries),
        'ambiguous_gold':sum(r.get('gold_status')=='identity_conflict' for r in union),
        **{f'hit_at_{k}':sum(r is not None and r<=k for r in ranks)/n if n else None for k in (1,5,20)},
        'mrr':sum(1/r if r else 0 for r in ranks)/n if n else None}
    return result


def page_rank(chunk_ids,by_id):
    seen=set(); out=[]
    for cid in chunk_ids:
        page=by_id[cid]['page_id']
        if page not in seen: seen.add(page); out.append(page)
    return out


def verified_queries(args,plan):
    q=queries(args.source)
    if len(q)!=len(plan['cases']): raise ValueError('Query count changed')
    for row,case in zip(q,plan['cases']):
        if (row['case_id']!=case['case_id'] or row['gold_page']!=case['gold_page'] or row['pdf_id']!=case['pdf_id']
                or row['query_page_number']!=case['query_page_number']
                or hashlib.sha256(row['question'].encode()).hexdigest()!=case['question_sha256']
                or hashlib.sha256(row['answer'].encode()).hexdigest()!=case['answer_sha256']):
            raise ValueError('Frozen query changed')
        row['partition']=case['partition']
        row['gold_status']=case['gold_status'];row['alternate_page']=case['alternate_page']
    return q


def text_eval(args):
    plan=plan_load(args.plan); q=verified_queries(args,plan)
    from paper_agent.retrieval import BGEEncoder,Retriever,rrf
    import numpy as np
    start=time.monotonic(); enc=BGEEncoder('models/bge-m3','cuda:0',2,768)
    unique=list(dict.fromkeys(x['question'] for x in q)); encode_start=time.monotonic()
    vectors=enc.encode(unique); encode_seconds=time.monotonic()-encode_start; cache=dict(zip(unique,vectors))
    class CachedEncoder:
        model_id=enc.model_id; max_length=enc.max_length
        def encode(self,texts): return np.ascontiguousarray([cache[x] for x in texts],dtype=np.float32)
    retriever=Retriever(args.index,CachedEncoder()); available={x['page_id'] for x in retriever.chunks}
    build=json.loads((Path(args.index)/'benchmark.json').read_text(encoding='utf8'))
    if build['plan_sha256']!=sha(args.plan): raise ValueError('Text index belongs to another plan')
    results=[]; total=len(retriever.chunks)
    for idx,item in enumerate(q,1):
        row={k:item[k] for k in ('case_id','group','partition','gold_page','pdf_id','question','gold_status','alternate_page')}
        branch_chunks={}
        for mode in ('bm25','dense'):
            t=time.monotonic()
            try:
                found=retriever.search(item['question'],mode=mode,top_k=min(100,total),candidate_k=total,paper_ids=None)
                branch_chunks[mode]=found['candidate_ids']
                pages=page_rank(found['candidate_ids'],retriever.by_id)
                row[mode]={'status':'ok','gold_in_index':item['gold_page'] in available,
                    'gold_rank':pages.index(item['gold_page'])+1 if item['gold_status']=='consistent' and item['gold_page'] in pages else None,
                    'declared_id_rank':pages.index(item['gold_page'])+1 if item['gold_page'] in pages else None,
                    'alternate_identity_rank':pages.index(item['alternate_page'])+1 if item['alternate_page'] in pages else None,
                    'pages':pages,'top_chunks':[{k:c.get(k) for k in ('paper_id','page_id','chunk_id','rank','bm25_rank','dense_rank','fusion_rank')} for c in found['hits']],
                    'search_strategy':found.get('dense_search_strategy'),
                    'elapsed_seconds':time.monotonic()-t}
            except Exception as exc:
                row[mode]={'status':'error','error_type':type(exc).__name__,'gold_in_index':item['gold_page'] in available,'gold_rank':None,'pages':[],'top_chunks':[],'elapsed_seconds':time.monotonic()-t}
        t=time.monotonic()
        try:
            if row['bm25']['status']!='ok' or row['dense']['status']!='ok': raise ValueError('Required text branch failed')
            # Both full rankings come from Retriever.search with the frozen all-chunk depth.
            # RRF is the same function and inputs as Retriever.search(mode='hybrid').
            bm_chunk_ids=branch_chunks['bm25']; de_chunk_ids=branch_chunks['dense']
            ranked=[key for key,_ in rrf([bm_chunk_ids,de_chunk_ids],k=60)]
            if idx==1:
                check=retriever.search(item['question'],mode='hybrid',top_k=min(100,total),candidate_k=total,paper_ids=None)
                if check['candidate_ids']!=ranked: raise ValueError('Hybrid interface equivalence check failed')
            pages=page_rank(ranked,retriever.by_id); b_rank={c:i+1 for i,c in enumerate(bm_chunk_ids)}; d_rank={c:i+1 for i,c in enumerate(de_chunk_ids)}
            row['hybrid']={'status':'ok','gold_in_index':item['gold_page'] in available,
                'gold_rank':pages.index(item['gold_page'])+1 if item['gold_status']=='consistent' and item['gold_page'] in pages else None,
                'declared_id_rank':pages.index(item['gold_page'])+1 if item['gold_page'] in pages else None,
                'alternate_identity_rank':pages.index(item['alternate_page'])+1 if item['alternate_page'] in pages else None,'pages':pages,
                'top_chunks':[{'paper_id':retriever.by_id[c]['paper_id'],'page_id':retriever.by_id[c]['page_id'],
                    'chunk_id':c,'rank':i+1,'fusion_rank':i+1,'bm25_rank':b_rank.get(c),'dense_rank':d_rank.get(c)}
                    for i,c in enumerate(ranked[:100])],
                'search_strategy':row['dense'].get('search_strategy'),
                'elapsed_seconds':row['bm25']['elapsed_seconds']+row['dense']['elapsed_seconds']+time.monotonic()-t}
        except Exception as exc:
            row['hybrid']={'status':'error','error_type':type(exc).__name__,'gold_in_index':item['gold_page'] in available,
                           'gold_rank':None,'pages':[],'top_chunks':[],'elapsed_seconds':time.monotonic()-t}
        results.append(row)
        if idx%20==0: print('text cases',idx,'of',len(q),flush=True)
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf8') as f:
        for row in results:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    summary={'plan_sha256':sha(args.plan),'index_corpus_hash':retriever.metadata['corpus_hash'],
             'query_count':len(q),'unique_query_count':len(unique),'index_chunks':total,'index_pages':len(available),
             'query_encoding_seconds':encode_seconds,'elapsed_seconds':time.monotonic()-start,
             'metrics':{mode:metrics(results,mode) for mode in ('bm25','dense','hybrid')},
             'results_sha256':sha(output)}
    write_new(str(output)+'.summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def visual_build(args):
    plan=plan_load(args.plan)
    from paper_agent.visual_retrieval import SubprocessEncoder
    enc=SubprocessEncoder(args.visual_python,'models/multimodal-v1/clip-image-v1','models/multimodal-v1/clip-text-v1','cuda:0')
    if enc.provenance!=plan['configuration']['clip_model_provenance']: raise ValueError('Visual model snapshots changed')
    if sha(Path(args.prepared)/'pages.jsonl')!=plan['prepared_pages_sha256']: raise ValueError('Prepared corpus mismatch')
    request=Path(args.request)
    write_new(request,{'prepared':args.prepared,'output':args.checkpoint,'image_model':enc.image_model_path,'device':'cuda:0',
                       'batch_size':8,'model_provenance':enc.provenance,'plan_sha256':sha(args.plan)})
    subprocess.run([args.visual_python,'scripts/irpapers_clip_pages.py','--request',str(request)],check=True)
    import numpy as np
    from paper_agent.vector_store import ChromaVectorStore
    checkpoint=Path(args.checkpoint);metadata=json.loads((checkpoint/'metadata.json').read_text(encoding='utf8'))
    if (metadata['plan_sha256']!=sha(args.plan) or metadata['pages_sha256']!=plan['prepared_pages_sha256']
            or sha(checkpoint/'vectors.npy')!=metadata['vectors_sha256'] or sha(checkpoint/'mapping.jsonl')!=metadata['mapping_sha256']):
        raise ValueError('Page vector checkpoint provenance mismatch')
    mapping=rows(checkpoint/'mapping.jsonl');vectors=np.load(checkpoint/'vectors.npy',mmap_mode='r',allow_pickle=False)
    if vectors.shape!=(len(mapping),metadata['dimension']):raise ValueError('Page vector checkpoint shape mismatch')
    output=Path(args.output)
    if output.exists():raise ValueError('Refusing to overwrite page Chroma index')
    output.mkdir(parents=True,exist_ok=False)
    contract={'collection':'paper_irpapers_page_clip_v1','kind':'clip_page_image','source_hash':metadata['pages_sha256'],
              'mapping_hash':metadata['mapping_sha256'],'model_provenance':enc.provenance,'dimension':metadata['dimension'],
              'normalization':'l2','metric':'cosine','preprocessing':metadata['preprocessing']}
    store=ChromaVectorStore(output/'vectors',contract,create=True);ids=[x['page_id'] for x in mapping]
    start=time.monotonic()
    for i in range(0,len(ids),256):store.upsert(ids[i:i+256],vectors[i:i+256])
    store.seal(ids);shutil.copyfile(checkpoint/'mapping.jsonl',output/'mapping.jsonl')
    if sha(output/'mapping.jsonl')!=metadata['mapping_sha256']:raise ValueError('Page mapping copy mismatch')
    metadata.update(backend='chroma',vector_contract=contract,chroma_build_seconds=time.monotonic()-start,
                    vector_manifest_sha256=sha(output/'vectors'/'vector-store.json'),
                    chroma_index_bytes_before_metadata=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))
    write_new(output/'metadata.json',metadata)
    print('page Chroma index built',len(ids),'dimension',metadata['dimension'],flush=True)


def visual_eval(args):
    plan=plan_load(args.plan); query=verified_queries(args,plan); text_rows=rows(args.text_results)
    text_meta=json.loads(Path(str(args.text_results)+'.summary.json').read_text(encoding='utf8'))
    if text_meta['plan_sha256']!=sha(args.plan) or text_meta['results_sha256']!=sha(args.text_results):
        raise ValueError('Text baseline results do not match frozen plan')
    if len(query)!=len(text_rows): raise ValueError('Text result count mismatch')
    from paper_agent.visual_retrieval import SubprocessEncoder,_normalize
    from paper_agent.retrieval import rrf
    import numpy as np
    from paper_agent.vector_store import ChromaVectorStore,ChromaIndexAdapter
    directory=Path(args.visual_index); meta=json.loads((directory/'metadata.json').read_text(encoding='utf8'))
    mapping=rows(directory/'mapping.jsonl')
    if (meta.get('backend')!='chroma' or meta.get('plan_sha256')!=sha(args.plan)
            or meta['pages_sha256']!=plan['prepared_pages_sha256'] or sha(directory/'mapping.jsonl')!=meta['mapping_sha256']
            or sha(directory/'vectors'/'vector-store.json')!=meta['vector_manifest_sha256']):
        raise ValueError('Visual page index provenance mismatch')
    enc=SubprocessEncoder(args.visual_python,'models/multimodal-v1/clip-image-v1','models/multimodal-v1/clip-text-v1','cuda:0')
    if enc.provenance!=meta['model_provenance'] or enc.provenance!=plan['configuration']['clip_model_provenance']: raise ValueError('Visual models differ from index or frozen plan')
    index=ChromaIndexAdapter(ChromaVectorStore(directory/'vectors',meta['vector_contract']),[m['page_id'] for m in mapping])
    if index.ntotal!=len(mapping) or index.d!=meta['dimension']: raise ValueError('Visual index dimension/count mismatch')
    unique=list(dict.fromkeys(x['question'] for x in query)); start=time.monotonic()
    vectors=enc.encode_texts(unique); cache=dict(zip(unique,vectors)); available={m['page_id'] for m in mapping}
    results=[]
    for item,previous in zip(query,text_rows):
        if item['case_id']!=previous['case_id']:raise ValueError('Text results do not match frozen cases')
        t=time.monotonic(); row={k:item[k] for k in ('case_id','group','partition','gold_page','pdf_id','question','gold_status','alternate_page')}
        try:
            vector=_normalize(cache[item['question']].reshape(1,-1))
            if vector.shape[1]!=index.d: raise ValueError('Visual query dimension mismatch')
            scores,positions=index.search(vector,index.ntotal)
            pages=[mapping[int(i)]['page_id'] for i in positions[0]]
            row['clip']={'status':'ok','gold_in_index':item['gold_page'] in available,
                'gold_rank':pages.index(item['gold_page'])+1 if item['gold_status']=='consistent' and item['gold_page'] in pages else None,
                'declared_id_rank':pages.index(item['gold_page'])+1 if item['gold_page'] in pages else None,
                'alternate_identity_rank':pages.index(item['alternate_page'])+1 if item['alternate_page'] in pages else None,
                'pages':pages,'top_scores':[float(s) for s in scores[0][:20]],
                'search_strategy':index.store.last_search_mode,'elapsed_seconds':time.monotonic()-t}
            old=previous['hybrid']
            if old['status']!='ok': raise ValueError('Text hybrid branch unavailable')
            fused=[page for page,_score in rrf([old['pages'],pages],k=60)]
            row['fusion']={'status':'ok','gold_in_index':item['gold_page'] in set(old['pages']) or item['gold_page'] in available,
                'gold_rank':fused.index(item['gold_page'])+1 if item['gold_status']=='consistent' and item['gold_page'] in fused else None,
                'declared_id_rank':fused.index(item['gold_page'])+1 if item['gold_page'] in fused else None,
                'alternate_identity_rank':fused.index(item['alternate_page'])+1 if item['alternate_page'] in fused else None,'pages':fused,
                'elapsed_seconds':time.monotonic()-t}
        except Exception as exc:
            for name in ('clip','fusion'):
                if name not in row: row[name]={'status':'error','error_type':type(exc).__name__,'gold_in_index':item['gold_page'] in available,'gold_rank':None,'pages':[],'elapsed_seconds':time.monotonic()-t}
        results.append(row)
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf8') as f:
        for row in results:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    summary={'plan_sha256':sha(args.plan),'text_results_sha256':sha(args.text_results),'visual_metadata':meta,
             'query_count':len(query),'elapsed_seconds':time.monotonic()-start,
             'metrics':{mode:metrics(results,mode) for mode in ('clip','fusion')},'results_sha256':sha(output)}
    write_new(str(output)+'.summary.json',summary); print(json.dumps(summary,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='cmd',required=True)
    def add(name,options):
        x=sub.add_parser(name)
        for option in options:x.add_argument('--'+option,required=True)
    add('freeze',['source','prepared','output']); add('chunks',['plan','prepared','output'])
    add('text-build',['plan','chunks','output']); add('adopt-text-index',['plan','old-plan','source-index','output'])
    add('text-eval',['plan','source','index','output'])
    add('visual-build',['plan','prepared','visual-python','request','checkpoint','output'])
    add('visual-eval',['plan','source','text-results','visual-index','visual-python','output'])
    args=p.parse_args(); {'freeze':freeze,'chunks':chunks,'text-build':text_build,'adopt-text-index':adopt_text_index,'text-eval':text_eval,
                          'visual-build':visual_build,'visual-eval':visual_eval}[args.cmd](args)


if __name__=='__main__':main()
