"""One fixed public-QASPER pair, two modes, at most six actual API requests.

No private memory, telemetry, Judge, retries or schema repairs. Unknown stays unknown.
"""
from dataclasses import replace
import argparse
import json
from pathlib import Path
import time
from paper_agent.evaluation import evidence_units
from paper_agent.pipeline import PaperAgentState,PaperPipeline
from paper_agent.providers import APIConfig,CompatibleReasoner
from paper_agent.observability import Exporter, local_trace_path
from paper_agent.retrieval import Retriever
from paper_agent.storage import read_jsonl,write_json,digest

parser=argparse.ArgumentParser()
parser.add_argument('--output',required=True,help='New directory; never overwrites a previous run')
output=Path(parser.parse_args().output)
output.mkdir(parents=True,exist_ok=False)
rows=read_jsonl('data/eval/v0.6/retrieval.jsonl')
pair=[]
for row in rows:
    if row['source_paper_id'] not in {p['source_paper_id'] for p in pair}: pair.append(row)
    if len(pair)==2: break
retriever=Retriever('indexes/bge')
config=replace(APIConfig.from_env(),max_calls=6,max_retries=0,repair_retries=0,
               max_tokens=1024,timeout=20,connect_timeout=5)
reasoner=CompatibleReasoner(config,'revision-model-pair')
question='For paper '+pair[0]['source_paper_id']+': '+pair[0]['query']+' For paper '+pair[1]['source_paper_id']+': '+pair[1]['query']
summary={'budget':{'max_actual_requests':6,'per_request_timeout_seconds':20,'per_turn_seconds':65,
                   'max_output_tokens':1024,'retries':0,'repairs':0,'judge_requests':0},
         'sample':'first two distinct papers in fixed QASPER v0.6; synthetic paired question',
         'data_hash':digest(pair),'corpus_hash':retriever.metadata['corpus_hash'],'actual_requests_total':0,'runs':[]}
write_json(output/'summary.json',summary)
for enabled in (False,True):
    start=time.monotonic();before=len(reasoner.calls)
    pipeline=PaperPipeline(retriever,reasoner,output/str(enabled)/'traces',output/str(enabled)/'memory',
                           mode='bm25',decomposition=enabled,exporter=Exporter(enabled=False),
                           budget_limits={'seconds':65,'model_requests':3})
    result=None
    try:
        result=pipeline.run(question,PaperAgentState('public-pair'),turn_id='pair',case_id='pair',run_id='revision-model-pair')
        item={'decomposition':enabled,'decision':result['answer']['decision'],'error_type':None}
    except Exception as exc:
        item={'decomposition':enabled,'decision':None,'error_type':type(exc).__name__}
    trace_path=Path(result['trace_path']) if result and result.get('trace_path') else local_trace_path(
        output/str(enabled)/'traces',{'session_id':'public-pair','turn_id':'pair','run_id':'revision-model-pair'})
    trace_path=trace_path if trace_path.exists() else None
    trace=json.loads(trace_path.read_text(encoding='utf-8')) if trace_path else {}
    evidence_refs=set()
    for stage in trace.get('stages',[]):
        if stage['stage']=='Rerank':
            evidence_refs.update(stage.get('output',{}).get('evidence_ids',[]))
    evidence=[retriever.by_id[c] for c in evidence_refs if c in retriever.by_id]
    units=evidence_units(evidence)
    calls=reasoner.calls[before:]
    item.update(elapsed_seconds=time.monotonic()-start,actual_requests=len(calls),
                known_usage=[c.get('usage') if c.get('usage_complete') else None for c in calls],
                goals=[{'query_id':r['query_id'],'paper_present':None if not trace_path else any(c['paper_id']==r['source_paper_id'] for c in evidence),
                        'gold_paragraph_hit':None if not trace_path else bool(units & set().union(*map(set,r.get('gold_alternatives') or [r['gold_units']]))) } for r in pair],
                completeness=None,unsupported_conclusions=None,false_refusal=None,
                trace_path=str(trace_path) if trace_path else None)
    summary['runs'].append(item)
    summary['actual_requests_total']=len(reasoner.calls)
    write_json(output/'summary.json',summary)
print(json.dumps({'status':'finished','actual_requests':len(reasoner.calls),'output':str(output/'summary.json')}))
