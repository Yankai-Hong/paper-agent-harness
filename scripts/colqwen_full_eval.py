"""Exact full-corpus MaxSim with a verified, lossless Chroma read cache."""
import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
from colqwen_encode_candidates import check_marker
from colqwen_smoke import rows, sha
from irpapers_optimize import read_frozen
from paper_agent.colqwen_store import maxsim
from paper_agent.maxsim_batch import score_maxsim_batch
from paper_agent.full_maxsim import rank_pages
from paper_agent.batched_colqwen import read_pages
from paper_agent.resumable_colqwen import ResumableColQwenStore


def write_new(path, value):
    ResumableColQwenStore._write_new(path, value)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','encoded','index','baseline','dev','test','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--batch-pages',type=int,default=64)
    p.add_argument('--partition',choices=['dev','test'],required=True); p.add_argument('--selection')
    a=p.parse_args()
    if not 1 <= a.batch_pages <= 256: raise ValueError('Invalid page batch bound')
    import psutil, torch
    source, encoded, index, root, out = map(Path,(a.prepared,a.encoded,a.index,a.baseline,a.output))
    freeze=read_frozen(root); pages=rows(source/'pages.jsonl')
    enc=json.loads((encoded/'contract.json').read_text(encoding='utf-8'))
    if len(pages)!=3230 or sha(source/'pages.jsonl')!=enc['source_pages_sha256']:
        raise ValueError('Full corpus mismatch')
    if (sha(Path(a.dev)/'contract.json')!=enc['dev_contract_sha256']
            or sha(Path(a.test)/'contract.json')!=enc['test_contract_sha256']):
        raise ValueError('Query encoder checkpoints differ from full page contract')
    outer=json.loads((index/'contract.json').read_text(encoding='utf-8'))
    store=ResumableColQwenStore(index,outer['contract'])
    if not store.final.is_file(): raise ValueError('Full index not sealed')
    seal=json.loads(store.final.read_text(encoding='utf-8'))
    if seal['pages']!=3230: raise ValueError('Full index page coverage incomplete')
    spec={'scope':'exact full corpus MaxSim, no candidate selection or text tail','pages':3230,
          'source_pages_sha256':sha(source/'pages.jsonl'),'index_seal_sha256':sha(store.final),
          'baseline_freeze_sha256':sha(root/'BASELINE-FROZEN.json'),'baseline_files':freeze['files'],
          'encoding_contract_sha256':sha(encoded/'contract.json'),'implementation_sha256':sha(__file__),
          'maxsim_code_sha256':sha('src/paper_agent/colqwen_store.py'),
          'batch_code_sha256':sha('src/paper_agent/maxsim_batch.py'), 'device':a.device,
          'ranking_code_sha256':sha('src/paper_agent/full_maxsim.py'),
          'batched_read_code_sha256':sha('src/paper_agent/batched_colqwen.py'),
          'batch_pages':a.batch_pages,'torch_version':torch.__version__,'strict_blind_test':False}
    out.mkdir(parents=True,exist_ok=True); plan=out/'plan.json'
    if plan.exists():
        if json.loads(plan.read_text(encoding='utf-8')) != spec: raise ValueError('Evaluation plan changed')
    else: write_new(plan,spec)
    if a.partition=='test':
        if not a.selection: raise ValueError('Frozen development selection required')
        selection=json.loads(Path(a.selection).read_text(encoding='utf-8'))
        if (selection.get('status')!='frozen_after_development'
                or selection.get('dev_results_sha256')!=sha(out/'results-dev.jsonl')
                or selection.get('eval_plan_sha256')!=sha(plan)
                or selection.get('baseline_freeze_sha256')!=sha(root/'BASELINE-FROZEN.json')):
            raise ValueError('Development selection differs')
    records=[]; offsets=[0]
    for page in pages:
        record=json.loads((store.committed/(page['page_id']+'.json')).read_text(encoding='utf-8'))
        if record['paper_id']!=page['paper_id'] or record['image_sha256']!=page['image_sha256']:
            raise ValueError('Index page source mismatch')
        records.append(record); offsets.append(offsets[-1]+record['count'])
    if offsets[-1]!=seal['tokens']: raise ValueError('Token count mismatch')
    cache=out/'vectors.npy'; marks=out/'cache-pages'; marks.mkdir(exist_ok=True)
    shape=(offsets[-1],enc['encoder']['dimension'])
    if cache.exists(): matrix=np.lib.format.open_memmap(cache,mode='r+')
    else: matrix=np.lib.format.open_memmap(cache,mode='w+',dtype=np.float32,shape=shape)
    if matrix.shape!=shape or matrix.dtype!=np.float32: raise ValueError('Cache shape mismatch')
    started=time.perf_counter(); cache_read_start=started
    process=psutil.Process(); sampled_peak_rss=process.memory_info().rss
    for start in range(0,len(pages),16):
        indices=list(range(start,min(start+16,len(pages)))); missing=[]
        for i in indices:
            path=marks/(pages[i]['page_id']+'.json'); block=matrix[offsets[i]:offsets[i+1]]
            if path.exists():
                marker=json.loads(path.read_text(encoding='utf-8'))
                if marker['seal']!=spec['index_seal_sha256'] or marker['sha256']!=hashlib.sha256(block.tobytes()).hexdigest():
                    raise ValueError('Cached page checksum mismatch')
            else: missing.append(i)
        if missing:
            for i, values in zip(missing,read_pages(store,[records[j] for j in missing])):
                matrix[offsets[i]:offsets[i+1]]=values
            matrix.flush()
            for i in missing:
                block=matrix[offsets[i]:offsets[i+1]]
                write_new(marks/(pages[i]['page_id']+'.json'),
                          {'seal':spec['index_seal_sha256'],'sha256':hashlib.sha256(block.tobytes()).hexdigest()})
        sampled_peak_rss=max(sampled_peak_rss,process.memory_info().rss)
        if (start+16)%400==0: print('Chroma pages cached and verified',start+16,'of',len(pages),flush=True)
    cache_seconds=time.perf_counter()-cache_read_start
    queries=rows(root/'text-results.jsonl')
    if len(queries)!=210: raise ValueError('Question coverage differs')
    checkpoints=out/'scores'; checkpoints.mkdir(exist_ok=True); result=[]
    if a.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    queries=[x for x in queries if x['partition']==a.partition]
    for case in queries:
        cid=case['case_id']; folder=Path(a.dev if case['partition']=='dev' else a.test)/'queries'
        expected={'case_id':cid,'question_sha256':hashlib.sha256(case['question'].encode()).hexdigest()}
        if not check_marker(folder,cid,expected): raise ValueError('Query checkpoint missing')
        marker=json.loads((folder/(cid+'.json')).read_text(encoding='utf-8'))
        with np.load(folder/marker['vector_file'],allow_pickle=False) as z: q=np.asarray(z['vectors'],dtype=np.float32)
        ck=checkpoints/(cid+'.json'); signature=sha(plan)
        if ck.exists():
            envelope=json.loads(ck.read_text(encoding='utf-8'))
            if envelope['plan_sha256']!=signature or envelope['query_marker_sha256']!=sha(folder/(cid+'.json')):
                raise ValueError('Scoring checkpoint contract mismatch')
            record=envelope['result']
            if envelope['result_sha256']!=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest():
                raise ValueError('Scoring checkpoint altered')
        else:
            tick=time.perf_counter(); scores=[]
            for start in range(0,len(pages),a.batch_pages):
                stop=min(start+a.batch_pages,len(pages))
                docs=[matrix[offsets[j]:offsets[j+1]] for j in range(start,stop)]
                batch=score_maxsim_batch(q,docs,device=a.device)
                if start==0 and max(abs(batch[j]-maxsim(q,docs[j])) for j in range(min(5,len(docs))))>1e-3:
                    raise ValueError('Batch MaxSim differs from independent reference')
                scores.extend(batch)
            order=rank_pages([page['page_id'] for page in pages],scores)
            ranked=[pages[i]['page_id'] for i in order]
            gold=case['gold_page']; valid=case['gold_status']=='consistent'
            rank=ranked.index(gold)+1 if valid and gold in ranked else None
            record={'case_id':cid,'group':case['group'],'partition':case['partition'],'gold_status':case['gold_status'],
                    'gold_rank':rank,'target_in_index':gold in set(ranked) if valid else None,
                    'pages':ranked,'scores':[scores[i] for i in order], 'query_encode_seconds':marker['encode_seconds'],
                    'top20_sources':[{k:pages[i].get(k) for k in ('page_id','paper_id','page_number','image_path','image_sha256')}
                                     for i in order[:20]],
                    'score_seconds':time.perf_counter()-tick}
            write_new(ck,{'plan_sha256':signature,'query_marker_sha256':sha(folder/(cid+'.json')),
                          'result':record,'result_sha256':hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()})
        result.append(record)
        sampled_peak_rss=max(sampled_peak_rss,process.memory_info().rss)
        if len(result)%10==0: print('full queries scored',len(result),'of',len(queries),flush=True)
    output=out/('results-'+a.partition+'.jsonl')
    if output.exists():
        if rows(output)!=result: raise ValueError('Existing results differ from checkpoints')
    else:
        pending=out/(output.name+'.'+uuid.uuid4().hex+'.pending')
        with pending.open('x',encoding='utf-8') as f:
            for r in result: f.write(json.dumps(r)+'\n')
            f.flush(); os.fsync(f.fileno())
        if os.name=='nt': os.rename(pending,output)
        else: os.link(pending,output); pending.unlink()
    report={'status':'complete','partition':a.partition,'results_sha256':sha(output),'plan_sha256':sha(plan),
            'selection_sha256':sha(a.selection) if a.selection else None,
            'cache_seconds_this_run':cache_seconds,'elapsed_seconds_this_run':time.perf_counter()-started,
            'rss_bytes_final':process.memory_info().rss,'sampled_peak_rss_bytes':sampled_peak_rss,
            'gpu_peak_bytes':torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else None}
    summary_path=out/('summary-'+a.partition+'.json')
    if summary_path.exists():
        saved=json.loads(summary_path.read_text(encoding='utf-8'))
        if saved['results_sha256']!=sha(output) or saved['plan_sha256']!=sha(plan):
            raise ValueError('Existing summary differs')
        report=saved
    else: write_new(summary_path,report)
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
