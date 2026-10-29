"""Bounded paired public-QASPER evaluation; no Judge or private memory.

The caller must explicitly select a new output directory. --dry-run prints the
fixed cost limits without loading API credentials or making a network request.
"""
import argparse
import json
import re
import time
from dataclasses import replace
from pathlib import Path
from paper_agent.evaluation import evidence_units
from paper_agent.observability import Exporter
from paper_agent.pipeline import PaperAgentState,PaperPipeline
from paper_agent.providers import APIConfig,CompatibleReasoner
from paper_agent.retrieval import Retriever
from paper_agent.storage import digest,read_jsonl,write_json

MAX_REQUESTS=18
MAX_OUTPUT_TOKENS=4096
MAX_INPUT_CHARS=30000
REQUEST_TIMEOUT=45
TURN_TIMEOUT=140

def fixed_cases(rows):
    pair=[]
    for row in rows:
        if row['source_paper_id'] not in {r['source_paper_id'] for r in pair}:
            pair.append(row)
        if len(pair)==2: break
    if len(pair)!=2: raise ValueError('At least two distinct papers required')
    a,b=pair
    return [
        {'case_id':'simple','question':f"For paper {a['source_paper_id']}: {a['query']}",
         'targets':[a],'label':'official QASPER source evidence; answer decision not independently reviewed'},
        {'case_id':'comparison','question':f"For paper {a['source_paper_id']}: {a['query']} For paper {b['source_paper_id']}: {b['query']}",
         'targets':[a,b],'label':'synthetic pairing of two official QASPER evidence questions'},
        {'case_id':'missing','question':f"For paper {a['source_paper_id']}, what is the exact SHA-256 hexadecimal checksum of its original training dataset?",
         'targets':[a],'label':'author-created missing-evidence probe; absence checked for 64-digit hex literal; semantic ground truth unknown'}]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    out=Path(args.output)
    if out.exists(): raise FileExistsError('Output must be a new directory')
    rows=read_jsonl('data/eval/v0.6/retrieval.jsonl')
    cases=fixed_cases(rows)
    spec={'case_count':len(cases),'paired_conditions':2,'max_actual_requests':MAX_REQUESTS,
          'max_actual_requests_per_turn':3,'max_output_tokens_per_request':MAX_OUTPUT_TOKENS,
          'max_possible_output_tokens':MAX_REQUESTS*MAX_OUTPUT_TOKENS,
          'max_input_chars_per_request':MAX_INPUT_CHARS,'per_request_timeout_seconds':REQUEST_TIMEOUT,
          'per_turn_seconds':TURN_TIMEOUT,'transport_retries':0,'schema_repairs':0,
          'reflect_retrieval_rounds_max':1,'judge_requests':0,'thinking':'disabled',
          'telemetry':'disabled','retrieval':'existing BM25 index','input':'public QASPER only',
          'case_ids':[c['case_id'] for c in cases], 'dataset_hash':digest(cases)}
    if args.dry_run:
        print(json.dumps(spec,ensure_ascii=False));return
    out.mkdir(parents=True,exist_ok=False)
    summary={'budget':spec,'actual_requests_total':0,'runs':[]}
    write_json(out/'summary.json',summary)
    retriever=Retriever('indexes/bge')
    missing_paper=cases[-1]['targets'][0]['source_paper_id']
    corpus=' '.join(c['text'] for c in retriever.chunks if c['paper_id']==missing_paper)
    spec['missing_hex_literal_count']=len(re.findall(r'(?i)(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])',corpus))
    if spec['missing_hex_literal_count']:
        raise ValueError('Missing-evidence probe was invalidated by corpus check')
    config=replace(APIConfig.from_env(),max_calls=MAX_REQUESTS,max_retries=0,repair_retries=0,
                   max_tokens=MAX_OUTPUT_TOKENS,max_input_chars=MAX_INPUT_CHARS,
                   timeout=REQUEST_TIMEOUT,connect_timeout=10,thinking='disabled')
    reasoner=CompatibleReasoner(config,'revision-public-comparison')
    summary['corpus_hash']=retriever.metadata['corpus_hash']
    write_json(out/'summary.json',summary)
    for case in cases:
        for enabled in (False,True):
            condition='split' if enabled else 'baseline'
            folder=out/condition/case['case_id']
            before=len(reasoner.calls); started=time.perf_counter()
            pipeline=PaperPipeline(retriever,reasoner,folder/'traces',folder/'memory',mode='bm25',
                decomposition=enabled,exporter=Exporter(enabled=False),max_retrieval_attempts=1,
                budget_limits={'seconds':TURN_TIMEOUT,'model_requests':3})
            item={'case_id':case['case_id'],'condition':condition,'label_source':case['label'],
                  'decision':None,'error_type':None,'answer_completeness':None,
                  'unsupported_conclusions':None,'false_refusal':None,'label_quality':'not independently verified'}
            try:
                result=pipeline.run(case['question'],PaperAgentState(case['case_id']),turn_id='turn',
                                    case_id=case['case_id'],run_id=condition)
                item['decision']=result['answer']['decision']
            except Exception as exc:
                item['error_type']=type(exc).__name__
            paths=list((folder/'traces').rglob('turn.json')) if (folder/'traces').exists() else []
            if len(paths)>1:
                raise RuntimeError('Ambiguous local trace for case')
            path=paths[0] if paths else None
            trace=json.loads(path.read_text(encoding='utf-8')) if path else {}
            evidence_ids=set()
            for stage in trace.get('stages',[]):
                if stage['stage']=='Rerank': evidence_ids.update(stage.get('output',{}).get('evidence_ids',[]))
            evidence=[retriever.by_id[i] for i in evidence_ids if i in retriever.by_id]
            units=evidence_units(evidence)
            item.update(elapsed_seconds=time.perf_counter()-started,actual_requests=len(reasoner.calls)-before,
                calls=[{k:c.get(k) for k in ('stage','status','error_type','error_category','finish_reason',
                                            'elapsed_ms','usage_complete','usage','model_requested','model_returned')}
                       for c in reasoner.calls[before:]],
                goals=[{'query_id':r['query_id'],'paper_present':None if not path else any(e['paper_id']==r['source_paper_id'] for e in evidence),
                        'gold_paragraph_hit':None if not path or case['case_id']=='missing' else bool(units & set().union(*map(set,r.get('gold_alternatives') or [r['gold_units']]))) }
                       for r in case['targets']],
                retrieval_calls=trace.get('budget',{}).get('used',{}).get('subqueries',0) if path else None,
                commit_status=trace.get('commit_status','unknown'),
                trace_path=str(path) if path else None)
            summary['runs'].append(item)
            summary['actual_requests_total']=len(reasoner.calls)
            write_json(out/'summary.json',summary)
    print(json.dumps({'output':str(out/'summary.json'),'actual_requests_total':len(reasoner.calls)}))

if __name__=='__main__': main()
