"""Paired, synthetic cross-paper questions on the existing QASPER/real BM25 index.

No LLM is invoked. Gold paragraph coverage is a retrieval proxy, not answer quality.
"""
import json
import time
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from paper_agent.evaluation import evidence_units
from paper_agent.pipeline import PaperAgentState,PaperPipeline
from paper_agent.retrieval import Retriever
from paper_agent.storage import read_jsonl,write_json,digest
from paper_agent.observability import local_trace_path

parser=argparse.ArgumentParser()
parser.add_argument('--output',default='reports/revision/real-pairs.json')
args=parser.parse_args()
rows=read_jsonl('data/eval/v0.6/retrieval.jsonl')
seen=set(); fixed=[]
for row in rows:
    if row['source_paper_id'] not in seen:
        fixed.append(row);seen.add(row['source_paper_id'])
    if len(fixed)==6: break
assert len(fixed)==6
retriever=Retriever('indexes/bge')
out=[]
with TemporaryDirectory(prefix='revision-pairs-',dir='.cache') as temporary:
    for i in range(3):
        pair=fixed[2*i:2*i+2]
        ids=[r['source_paper_id'] for r in pair]
        question='For paper '+ids[0]+': '+pair[0]['query']+' For paper '+ids[1]+': '+pair[1]['query']
        class Reasoner:
            def __init__(self): self.calls=[];self.evidence=[]
            def structured(self,stage,payload):
                self.calls.append(stage)
                if stage=='nlu': return {'intent':'compare','standalone_query':question,'paper_ids':ids,'resolved_entities':{},
                    'sub_questions':[{'id':str(j),'query':r['query'],'target':r['query'],'paper_ids':[r['source_paper_id']]} for j,r in enumerate(pair)]}
                if stage=='reflect':
                    self.evidence=payload['evidence']
                    return {'decision':'answer','reason':'fixture decision; support not adjudicated'}
                return {'claims':[{'text':c['text'],'citations':[{'paper_id':c['paper_id'],'chunk_id':c['chunk_id']}]} for c in payload['evidence']]}
        for enabled in (False,True):
            base=Path(temporary)/str(i)/str(enabled)
            reasoner=Reasoner()
            pipeline=PaperPipeline(retriever,reasoner,base/'trace',base/'memory',mode='bm25',decomposition=enabled)
            started=time.perf_counter()
            answer=None;error_type=None
            try:
                answer=pipeline.run(question,PaperAgentState('s'),'turn',case_id=str(i),run_id='real-pairs-fixture')
            except Exception as exc:
                error_type=type(exc).__name__
            elapsed=time.perf_counter()-started
            trace_path=Path(answer['trace_path']) if answer and answer.get('trace_path') else local_trace_path(
                base/'trace',{'session_id':'s','turn_id':'turn','run_id':'real-pairs-fixture'})
            trace=json.loads(trace_path.read_text(encoding='utf-8')) if trace_path.exists() else {}
            units=evidence_units(reasoner.evidence)
            has_evidence=any(s.get('stage')=='Reflect' for s in trace.get('stages',[]))
            goals=[{'query_id':r['query_id'],'paper_id':r['source_paper_id'],
                    'paper_present':any(c['paper_id']==r['source_paper_id'] for c in reasoner.evidence) if has_evidence else None,
                    'gold_paragraph_hit':bool(units & set().union(*map(set,r.get('gold_alternatives') or [r['gold_units']]))) if has_evidence else None } for r in pair]
            out.append({'case':i,'decomposition':enabled,'goals':goals,'decision':answer['answer']['decision'] if answer else None,
                        'error_type':error_type,
                        'llm_requests':0,'scripted_stage_calls':len(reasoner.calls),'provider_tokens':None,
                        'latency_seconds':elapsed,'budget':trace.get('budget',{}).get('used'),
                        'answer_completeness':None,'unsupported_conclusions':None,'false_refusal':None})
write_json(args.output,{'scope':'first six distinct papers in fixed v0.6 QASPER, paired in source order; synthetic questions; real BM25, scripted reasoner',
    'label_source':'QASPER official paragraph labels, no human re-evaluation',
    'dataset_hash':digest(fixed),'corpus_hash':retriever.metadata['corpus_hash'], 'runs':out})
print(json.dumps({'pairs':3,'runs':len(out),'model_requests':0,'output':args.output}))
