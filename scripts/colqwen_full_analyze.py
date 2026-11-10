"""Select page RRF on development and evaluate full-corpus MaxSim results."""
import argparse
import json
import statistics
from pathlib import Path

from colqwen_smoke import rows, sha
from irpapers_optimize import read_frozen
from paper_agent.page_fusion import fuse_page_rankings


def metrics(rs, key):
    n=len(rs); ranks=[r[key] for r in rs]
    return {**{f'hit{k}':sum(x is not None and x<=k for x in ranks)/n for k in (1,5,20)},
            'mrr':sum(1/x if x else 0 for x in ranks)/n}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['choose','evaluate'])
    for name in ('baseline','full','candidates','candidate-dev','candidate-test','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--plan'); a=p.parse_args()
    root, full = Path(a.baseline), Path(a.full); read_frozen(root)
    summary=json.loads((full/'summary-dev.json').read_text(encoding='utf-8'))
    if summary['status']!='complete' or summary['results_sha256']!=sha(full/'results-dev.jsonl'):
        raise ValueError('Full corpus results incomplete')
    base={x['case_id']:x for x in rows(root/'text-results.jsonl')}
    visual={x['case_id']:x for x in rows(root/'visual-results.jsonl')}
    candidate_summary=json.loads(Path(a.candidates+'.summary.json').read_text(encoding='utf-8'))
    if candidate_summary.get('results_sha256')!=sha(a.candidates):
        raise ValueError('Frozen candidate results changed')
    candidate={x['case_id']:x for x in rows(a.candidates)}
    for path, partition in ((a.candidate_dev,'dev'),) + (((a.candidate_test,'test'),) if a.mode=='evaluate' else ()):
        prior_summary=json.loads(Path(path+'.summary.json').read_text(encoding='utf-8'))
        if (prior_summary.get('status')!='complete' or prior_summary.get('partition')!=partition
                or prior_summary.get('results_sha256')!=sha(path)):
            raise ValueError('Candidate evaluation results changed')
    prior_rows=rows(a.candidate_dev)+(rows(a.candidate_test) if a.mode=='evaluate' else [])
    prior={x['case_id']:x for x in prior_rows}
    full_rows=rows(full/'results-dev.jsonl')
    if a.mode=='evaluate':
        test_summary=json.loads((full/'summary-test.json').read_text(encoding='utf-8'))
        if (test_summary['status']!='complete' or test_summary['results_sha256']!=sha(full/'results-test.jsonl')
                or not a.plan or test_summary['selection_sha256']!=sha(a.plan)):
            raise ValueError('Holdout score is incomplete or not bound to selection')
        full_rows+=rows(full/'results-test.jsonl')
    expected={k for k,v in base.items() if a.mode=='evaluate' or v['partition']=='dev'}
    if len(full_rows)!=len(expected) or {x['case_id'] for x in full_rows}!=expected or set(base)!=set(candidate) or set(prior)!=expected:
        raise ValueError('Case coverage mismatch')
    def calculate(weight, partition=None):
        result=[]
        for row in full_rows:
            if partition and row['partition']!=partition: continue
            original=base[row['case_id']]; old=prior[row['case_id']]; ca=candidate[row['case_id']]
            ranked=fuse_page_rankings([original['hybrid']['pages'],row['pages']],[1,weight],k=60)
            gold=original['gold_page']; valid=original['gold_status']=='consistent'
            fused=ranked.index(gold)+1 if valid and gold in ranked else None
            allowed=set(ca['pages'])
            result.append({'case_id':row['case_id'],'group':row['group'],'partition':row['partition'],
                'gold_status':row['gold_status'],'full_rank':row['gold_rank'],'fused_rank':fused,
                'text_rank':original['hybrid']['gold_rank'],'clip_rank':visual[row['case_id']]['clip']['gold_rank'],
                'candidate_rank':old['candidate_rank'],'candidate_covered':old['candidate_hit'],
                'full_top1_candidate_overlap':sum(x in allowed for x in row['pages'][:1]),
                'full_top5_candidate_overlap':sum(x in allowed for x in row['pages'][:5]),
                'full_top20_candidate_overlap':sum(x in allowed for x in row['pages'][:20]),
                'full_score_seconds':row['score_seconds'],'query_encode_seconds':row['query_encode_seconds'],
                'candidate_read_and_score_seconds':old['rerank_seconds'],
                'prior_text_search_seconds':old['prior_text_search_seconds'],'prior_clip_search_seconds':old['prior_clip_search_seconds']})
        return result
    if a.mode=='choose':
        choices=[]
        for w in (0,.05,.1,.25,.5,1):
            rs=calculate(w,'dev'); groups={g:metrics([r for r in rs if r['group']==g],'fused_rank') for g in ('standard','visual')}
            choices.append({'weight':w,'metrics':groups})
        reference=choices[0]['metrics']
        eligible=[x for x in choices if all(x['metrics'][g]['hit5']>=reference[g]['hit5'] for g in ('standard','visual'))]
        best=max(eligible,key=lambda x:(sum(x['metrics'][g]['mrr'] for g in ('standard','visual')),-x['weight']))
        out={'status':'frozen_after_development','weight':best['weight'],'k':60,'options':choices,
             'dev_results_sha256':sha(full/'results-dev.jsonl'),'baseline_freeze_sha256':sha(root/'BASELINE-FROZEN.json'),
             'eval_plan_sha256':sha(full/'plan.json'),
             'script_sha256':sha(__file__),'strict_blind_test':False}
    else:
        if not a.plan: raise ValueError('Development selection required')
        plan=json.loads(Path(a.plan).read_text(encoding='utf-8'))
        if (plan.get('status')!='frozen_after_development' or plan['dev_results_sha256']!=sha(full/'results-dev.jsonl')
                or plan['eval_plan_sha256']!=sha(full/'plan.json')
                or plan['baseline_freeze_sha256']!=sha(root/'BASELINE-FROZEN.json') or plan['script_sha256']!=sha(__file__)):
            raise ValueError('Selection changed')
        result=calculate(plan['weight']); stats={}
        for group in ('standard','visual'):
            for partition in ('all','dev','test'):
                rs=[r for r in result if r['group']==group and (partition=='all' or r['partition']==partition)]
                item={'n':len(rs),'candidate_coverage':sum(r['candidate_covered'] for r in rs)/len(rs)}
                for k in (1,5,20):
                    item[f'candidate_coverage_of_full_top{k}']=sum(r[f'full_top{k}_candidate_overlap'] for r in rs)/(k*len(rs))
                for timing in ('full_score_seconds','query_encode_seconds','candidate_read_and_score_seconds',
                               'prior_text_search_seconds','prior_clip_search_seconds'):
                    values=sorted(r[timing] for r in rs)
                    item[timing]={'median':statistics.median(values),'p95':values[min(len(values)-1,int(.95*len(values)))]}
                for key in ('text_rank','clip_rank','candidate_rank','full_rank','fused_rank'): item[key]=metrics(rs,key)
                for key in ('full_rank','fused_rank'):
                    hit=lambda x:x is not None and x<=5
                    item[key+'_paired_hit5']={'improved':[r['case_id'] for r in rs if hit(r[key]) and not hit(r['text_rank'])],
                                             'regressed':[r['case_id'] for r in rs if hit(r['text_rank']) and not hit(r[key])]}
                item['full_vs_candidate_hit5']={'full_only':[r['case_id'] for r in rs if hit(r['full_rank']) and not hit(r['candidate_rank'])],
                    'candidate_only':[r['case_id'] for r in rs if hit(r['candidate_rank']) and not hit(r['full_rank'])]}
                stats[group+'_'+partition]=item
        out={'plan_sha256':sha(a.plan),'weight':plan['weight'],'strict_blind_test':False,'metrics':stats,'cases':result,
             'note':'Full MaxSim uses all pages; prior candidate costs were separately measured, not live end-to-end.'}
    with Path(a.output).open('x',encoding='utf-8') as f: json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps({'output':a.output,'sha256':sha(a.output),'weight':out.get('weight')}))


if __name__=='__main__': main()
