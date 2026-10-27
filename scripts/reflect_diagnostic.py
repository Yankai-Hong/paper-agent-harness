"""Predeclared small natural Reflect diagnostic plus separate forced-branch check."""
import json
import os
from pathlib import Path
from dataclasses import replace
import time
import hashlib
from paper_agent.providers import APIConfig, CompatibleReasoner, post_json
from paper_agent.pipeline import PaperPipeline, PaperAgentState
from paper_agent.memory_manager import MemoryManager
from paper_agent.memory_stores import RedisWorkingStore, ChromaEpisodes
from paper_agent.retrieval import BGEEncoder, BGEReranker, Retriever
from paper_agent.storage import read_jsonl,write_json,write_jsonl

ROOT=Path('reports/final-acceptance/reflect-diagnostic')


def main():
    global ROOT
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(ROOT))
    ROOT=Path(parser.parse_args().output)
    ROOT.mkdir(parents=True,exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    cfg=json.loads(Path('configs/final-acceptance.json').read_text())
    api=replace(APIConfig.from_env(),max_calls=144,max_tokens=4096,thinking='disabled',timeout=60,max_retries=1,repair_retries=1)
    papers={p['paper_id']:p for p in read_jsonl('data/qasper/papers.jsonl')}
    queries=read_jsonl('data/qasper/retrieval.jsonl')[:3]
    cases=[]
    for i,q in enumerate(queries):
        paper=papers[q['source_paper_id']]
        cases.extend([
            {'id':f'positive-{i}','question':f"In the paper {paper['title']}, {q['query']}",
             'expected':'answer','label_source':'official QASPER answerable; retrieval may still fail','paper_id':paper['paper_id']},
            {'id':f'insufficient-{i}','question':f"In the paper {paper['title']}, give the exact SHA256 checksum of the authors' unpublished raw experiment log. Do not infer it or give a procedure.",
             'expected':'refuse','label_source':'fixed unavailable-artifact diagnostic; agent-authored, not human gold','paper_id':paper['paper_id']}])
    write_jsonl(ROOT/'cases.jsonl',cases)
    write_json(ROOT/'budget.json',{'estimated_requests':48,'maximum_requests_including_retries':144,
        'selection':'first three fixed source questions, plus the same unavailable-log request; frozen before outcomes',
        'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src/paper_agent').glob('*.py')},
        'config':cfg,'model':api.model,'index_metadata':json.loads(Path(cfg['index'],'metadata.json').read_text())})
    count=0
    def reasoner(sid):
        def transport(url,body,headers,timeout):
            nonlocal count
            if count>=144: raise RuntimeError('Diagnostic budget exhausted')
            count+=1
            rec={'request':count,'session':sid,'usage':None}
            start=time.monotonic()
            try:
                response=post_json(url,body,headers,timeout,connect_timeout=15)
                rec.update(usage=response.get('usage'),model=response.get('model'))
                return response
            finally:
                rec['elapsed_seconds']=time.monotonic()-start
                write_json(ROOT/'requests'/f'{count:04d}.json',rec)
        return CompatibleReasoner(api,sid,transport)
    encoder=BGEEncoder(cfg['encoder'],cfg['device'],4)
    retriever=Retriever(cfg['index'],encoder,BGEReranker(cfg['reranker'],cfg['device'],4))
    results=[]
    for name,opts in [('off',{'reflect':False}),('on',{'max_retrieval_attempts':1}),('second',{'max_retrieval_attempts':2})]:
        out=ROOT/name
        manager=MemoryManager(out/'memory',RedisWorkingStore(cfg['redis_url'],namespace='paper-agent:reflect-diagnostic:'+name),
                              ChromaEpisodes(out/'chroma',encoder),fallback=False)
        for row in cases:
            sid=row['id']
            p=PaperPipeline(retriever,reasoner(name+'-'+sid),out/'traces',out/'memory',mode='rerank',
                            memory_manager=manager,allow_degraded=False,**opts)
            r={'group':name,**row,'error':None}
            try:
                answer=p.run(row['question'],PaperAgentState(sid),sid)['answer']
                trace=json.loads((out/'traces'/(sid+'.json')).read_text(encoding='utf-8'))
                r.update(answer=answer,decision_match=answer['decision']==row['expected'],
                         second_retrieval=sum(s['stage']=='Retrieve' for s in trace['stages'])>1)
            except Exception as exc: r.update(error=type(exc).__name__,decision_match=False,second_retrieval=False)
            results.append(r)
            write_jsonl(ROOT/'results.jsonl',results)
        print(name+' done',flush=True)
    # Same real retrieval/storage, deliberately scripted routing: mechanism only.
    class Forced:
        def structured(self,stage,payload):
            if stage=='nlu': return {'intent':'fact','standalone_query':queries[0]['query'],'paper_ids':[cases[0]['paper_id']],'resolved_entities':{}}
            if stage=='reflect': return {'decision':'retrieve_again','next_query':'experimental results','reason':'forced diagnostic'}
            raise AssertionError('Must refuse after bounded retry')
    out=ROOT/'forced'
    manager=MemoryManager(out/'memory',RedisWorkingStore(cfg['redis_url'],namespace='paper-agent:reflect-forced'),ChromaEpisodes(out/'chroma',encoder),fallback=False)
    p=PaperPipeline(retriever,Forced(),out/'traces',out/'memory',mode='rerank',memory_manager=manager,allow_degraded=False)
    answer=p.run('forced retry',PaperAgentState('forced'),'forced')['answer']
    t=json.loads((out/'traces/forced.json').read_text(encoding='utf-8'))
    assert answer['decision']=='refuse' and sum(s['stage']=='Retrieve' for s in t['stages'])==2
    write_json(ROOT/'summary.json',{'natural':{name:{'n':6,'decision_match':sum(r['decision_match'] for r in results if r['group']==name),
        'correct_refusal':sum(r['decision_match'] for r in results if r['group']==name and r['expected']=='refuse'),
        'refusal_denominator':3,'second_retrieval':sum(r['second_retrieval'] for r in results if r['group']==name),
        'failures':sum(bool(r['error']) for r in results if r['group']==name)} for name in ['off','on','second']},
        'forced':{'passed':True,'retrieval_attempts':2,'decision':'refuse','paid_api_calls':0,'not_natural_model_effect':True},
        'requests':count,'note':'Diagnostic decision agreement, not human task-quality accuracy; no requirement for a positive gain.'})


if __name__=='__main__': main()
