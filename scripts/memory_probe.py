"""Paired NLU-only long-context replay probe, not an end-to-end memory benchmark."""
import argparse
import copy
from dataclasses import replace, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from live_validation import configuration
from summarize_live import collect
from paper_agent.providers import CompatibleReasoner
from paper_agent.pipeline import PaperAgentState, compact_state
from paper_agent.storage import read_jsonl, write_json, digest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run_root')
    parser.add_argument('--output',required=True)
    parser.add_argument('--semantic',action='store_true')
    parser.add_argument('--baseline',default='reports/live-final/memory-probe')
    args=parser.parse_args()
    base_config=configuration('deepseek-commented')
    config=replace(base_config,thinking='disabled',max_calls=1)
    predictions=collect(args.run_root)
    rows=read_jsonl('data/dialogues-draft/dialogues.jsonl')
    first=[r for r in rows if r['turn']==1 and predictions.get(r['turn_id'],{}).get('answer',{}).get('decision')=='answer']
    if len(first)<8:
        raise ValueError('Need eight real answered opening turns')
    catalog=[{'paper_id':p['paper_id'],'title':p['title']} for p in read_jsonl('data/qasper/papers.jsonl')]
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    cases=[]
    summary_calls=[]
    # Four overlapping replay contexts, each made from eight real first-turn exchanges.
    # Concatenation is synthetic and is explicitly not a natural human session.
    for group in range(min(4,len(first)-7)):
        selected=first[group:group+8]
        state=PaperAgentState(f'probe-{group}')
        for row in selected:
            pred=predictions[row['turn_id']]
            state.messages.extend([{'role':'user','content':row['question']},
                                   {'role':'assistant','content':pred['answer']}])
            memory=next(s['output'] for s in pred['trace']['stages'] if s['stage']=='Memory' and s['status']=='ok')
            state.memory.entities.update(memory['entities'])
            state.memory.facts.update(memory['facts'])
        if args.semantic:
            from paper_agent.compaction import semantic_compact_state
            summary_reasoner=CompatibleReasoner(config,f'summary-{group}')
            semantic_state=copy.deepcopy(state)
            try:
                compressed=semantic_compact_state(semantic_state,output/'archives',summary_reasoner)
            except Exception as exc:
                write_json(output/f'summary-{group}-error.json',{'error':str(exc),'calls':summary_reasoner.calls})
                raise
            write_json(output/f'summary-{group}.json',{'compaction':compressed,'calls':summary_reasoner.calls})
            summary_calls.extend(summary_reasoner.calls)
        for kind,target in [('earliest',selected[0]),('latest',selected[-1])]:
            question='请继续解释本次会话'+('最早' if kind=='earliest' else '最后')+'讨论的那篇论文的实验方法，并只针对那篇论文检索。'
            for condition in (['semantic'] if args.semantic else ['full','compact']):
                working=copy.deepcopy(semantic_state if args.semantic else state)
                compression=compressed if args.semantic else compact_state(working,output/'archives') if condition=='compact' else {'compressed':False}
                cases.append({'case_id':f'{group}-{kind}-{condition}','pair_id':f'{group}-{kind}',
                    'condition':condition,'expected_paper_id':target['source_paper_ids'][0],
                    'compression':compression,'payload':{'question':question,'paper_catalog':catalog,
                        'memory':asdict(working.memory),'recent_messages':working.messages}})
    if len(cases)+len(summary_calls)>base_config.max_calls:
        raise ValueError('Probe exceeds configured request cap')
    def one(case):
        path=output/f"{case['case_id']}.json"
        if path.exists():
            import json
            return json.loads(path.read_text(encoding='utf-8'))
        reasoner=CompatibleReasoner(config,'memory-'+case['case_id'])
        record={k:v for k,v in case.items() if k!='payload'}
        record['payload_hash']=digest(case['payload'])
        try:
            result=reasoner.structured('nlu',case['payload'])
            record.update(result=result,correct=result['paper_ids']==[case['expected_paper_id']] and not result['needs_clarification'])
        except Exception as exc:
            record.update(error=str(exc),correct=False)
        record['calls']=reasoner.calls
        write_json(path,record)
        print('Memory probe',case['case_id'],record['correct'],flush=True)
        return record
    with ThreadPoolExecutor(max_workers=3) as executor:
        records=list(executor.map(one,cases))
    summary={}
    for condition in (['semantic'] if args.semantic else ['full','compact']):
        matching=[r for r in records if r['condition']==condition]
        calls=[c for r in matching for c in r['calls']]
        summary[condition]={'cases':len(matching),'correct':sum(r['correct'] for r in matching),
            'accuracy':sum(r['correct'] for r in matching)/len(matching),
            'all_usage_complete':all(c.get('usage_complete',False) for c in calls),
            'tokens':sum(sum(c.get('usage',{}).values()) for c in calls)}
    if args.semantic:
        import json
        baseline=json.loads((Path(args.baseline)/'summary.json').read_text(encoding='utf-8'))
        for record in records:
            prior=json.loads((Path(args.baseline)/(record['pair_id']+'-full.json')).read_text(encoding='utf-8'))
            if prior['expected_paper_id']!=record['expected_paper_id']:
                raise ValueError('Memory comparison cases changed')
        summary['summary_tokens']=sum(sum(c.get('usage',{}).values()) for c in summary_calls)
        summary['semantic']['total_tokens_including_summary']=summary['semantic']['tokens']+summary['summary_tokens']
        summary['full']=baseline['full']
        summary['token_reduction_including_summary']=1-summary['semantic']['total_tokens_including_summary']/summary['full']['tokens']
    else:
        summary['token_reduction']=1-summary['compact']['tokens']/summary['full']['tokens']
    summary['scope']='Eight paired earliest/latest reference tasks over synthetic concatenations of real answers; NLU only; no end-to-end memory or summary quality claim.'
    write_json(output/'summary.json',summary)


if __name__=='__main__':
    main()
