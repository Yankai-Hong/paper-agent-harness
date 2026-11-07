"""Select page-level fusion only on development data, freeze, then score once."""
import argparse
import hashlib
import json
import time
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def load_dev(path):
    with Path(path).open(encoding='utf8') as f:
        for line in f:
            row=json.loads(line)
            if row['partition']=='dev':yield row


def ranked(lanes,weights,k=60):
    scores={}
    for lane,weight in zip(lanes,weights):
        if weight==0:continue
        seen=set()
        for i,key in enumerate(lane,1):
            if key in seen:continue
            seen.add(key);scores[key]=scores.get(key,0.)+weight/(k+i)
    return sorted(scores,key=lambda key:(-scores[key],key))


def score(rows):
    n=len(rows)
    return {'n':n,'hit1':sum(r is not None and r<=1 for r in rows)/n if n else None,
            'hit5':sum(r is not None and r<=5 for r in rows)/n if n else None,
            'hit20':sum(r is not None and r<=20 for r in rows)/n if n else None,
            'mrr':sum(1/r if r else 0 for r in rows)/n if n else None}


def rank_gold(pages,row):
    return pages.index(row['gold_page'])+1 if row['gold_status']=='consistent' and row['gold_page'] in pages else None


def read_frozen(root):
    root=Path(root); frozen=json.loads((root/'BASELINE-FROZEN.json').read_text(encoding='utf8'))
    if frozen['status']!='frozen_before_optimization':raise ValueError('Baseline not frozen')
    for name,expected in frozen['files'].items():
        if sha(root/name)!=expected:raise ValueError('Frozen baseline changed: '+name)
    return frozen


def choose(args):
    root=Path(args.baseline); frozen=read_frozen(root)
    text={x['case_id']:x for x in load_dev(root/'text-results.jsonl')}
    visual={x['case_id']:x for x in load_dev(root/'visual-results.jsonl')}
    if set(text)!=set(visual):raise ValueError('Development rows mismatch')
    rows=[(text[k],visual[k]) for k in sorted(text)]
    if any(a['hybrid']['status']!='ok' or a['bm25']['status']!='ok' or a['dense']['status']!='ok' or b['clip']['status']!='ok' or b['fusion']['status']!='ok' for a,b in rows):
        raise ValueError('Development baseline has failed branches')
    text_options={}
    for name in ('chunk_rrf','page_rrf'):
        ranks=[]
        for a,_ in rows:
            pages=a['hybrid']['pages'] if name=='chunk_rrf' else ranked([a['bm25']['pages'],a['dense']['pages']],[1.,1.])
            ranks.append(rank_gold(pages,a))
        text_options[name]=score(ranks)
    # Page-level voting is adopted only if it improves MRR and does not reduce Hit@5.
    chosen_text=('page_rrf' if text_options['page_rrf']['mrr']>text_options['chunk_rrf']['mrr']
                 and text_options['page_rrf']['hit5']>=text_options['chunk_rrf']['hit5'] else 'chunk_rrf')
    prepared=[]
    for a,b in rows:
        text_pages=a['hybrid']['pages'] if chosen_text=='chunk_rrf' else ranked([a['bm25']['pages'],a['dense']['pages']],[1.,1.])
        prepared.append((a,b,text_pages))
    weights=[0.,.05,.1,.25,.5,1.]
    old_by_group={g:score([rank_gold(b['fusion']['pages'],a) for a,b,_ in prepared if a['group']==g]) for g in ('standard','visual')}
    candidates=[]
    for weight in weights:
        ranks=[];groups={'standard':[],'visual':[]}
        for a,b,text_pages in prepared:
            result=ranked([text_pages,b['clip']['pages']],[1.,weight])
            r=rank_gold(result,a);ranks.append(r);groups[a['group']].append(r)
        stats={'weight':weight,'all':score(ranks),'standard':score(groups['standard']),'visual':score(groups['visual'])}
        stats['eligible']=all(stats[g]['hit5']>=old_by_group[g]['hit5'] for g in groups)
        candidates.append(stats)
    eligible=[x for x in candidates if x['eligible']]
    if not eligible:raise ValueError('No development candidate meets the predeclared safeguard')
    winner=max(eligible,key=lambda x:(x['all']['mrr'],-x['weight']))
    plan={'status':'frozen_before_test_evaluation','baseline_freeze_sha256':sha(root/'BASELINE-FROZEN.json'),
          'baseline_files':frozen['files'],'script_sha256':sha(__file__),
          'development_cases':len(rows),'selection':'development only; choose page RRF only if MRR improves without losing Hit@5; then choose weight with best combined MRR and neither group below equal-fusion Hit@5',
          'text_options':text_options,'text_strategy':chosen_text,'weight_candidates':candidates,
          'clip_weight':winner['weight'],'rrf_k':60,'test_rows_inspected_for_selection':False}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf8') as f:json.dump(plan,f,ensure_ascii=False,indent=2)
    print(json.dumps({'optimization_plan':str(out),'sha256':sha(out),'development_cases':len(rows),
                      'text_strategy':chosen_text,'clip_weight':winner['weight'],'dev_metrics':winner['all']}))


def evaluate(args):
    root=Path(args.baseline);frozen=read_frozen(root); plan=json.loads(Path(args.plan).read_text(encoding='utf8'))
    if (plan['status']!='frozen_before_test_evaluation' or plan['baseline_freeze_sha256']!=sha(root/'BASELINE-FROZEN.json')
            or plan['script_sha256']!=sha(__file__) or plan['baseline_files']!=frozen['files']):raise ValueError('Frozen optimization mismatch')
    with (root/'text-results.jsonl').open(encoding='utf8') as f:text=[json.loads(x) for x in f if x.strip()]
    with (root/'visual-results.jsonl').open(encoding='utf8') as f:visual=[json.loads(x) for x in f if x.strip()]
    if len(text)!=len(visual):raise ValueError('Baseline counts differ')
    out=[];start=time.monotonic()
    for a,b in zip(text,visual):
        if a['case_id']!=b['case_id']:raise ValueError('Baseline row mismatch')
        if any(x['status']!='ok' for x in (a['bm25'],a['dense'],a['hybrid'],b['clip'],b['fusion'])):
            raise ValueError('Baseline branch failed; do not hide failed cases')
        tp=a['hybrid']['pages'] if plan['text_strategy']=='chunk_rrf' else ranked([a['bm25']['pages'],a['dense']['pages']],[1.,1.])
        combined=ranked([tp,b['clip']['pages']],[1.,plan['clip_weight']])
        out.append({'case_id':a['case_id'],'group':a['group'],'partition':a['partition'],'gold_page':a['gold_page'],
                    'gold_status':a['gold_status'],'baseline_text_rank':a['hybrid']['gold_rank'],
                    'baseline_fusion_rank':b['fusion']['gold_rank'],'optimized_text_rank':rank_gold(tp,a),
                    'optimized_fusion_rank':rank_gold(combined,a),'optimized_text_top20':tp[:20],
                    'optimized_fusion_top20':combined[:20]})
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf8') as f:
        for r in out:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    report={}
    for label,group,partition in [('standard_all','standard',None),('standard_dev','standard','dev'),('standard_test','standard','test'),
                                   ('visual_all','visual',None),('visual_dev','visual','dev'),('visual_test','visual','test')]:
        subset=[r for r in out if r['group']==group and (partition is None or r['partition']==partition)]
        row={k:score([r[k] for r in subset]) for k in ('baseline_text_rank','optimized_text_rank','baseline_fusion_rank','optimized_fusion_rank')}
        for k in (1,5,20):
            old=lambda r:r['baseline_fusion_rank'] is not None and r['baseline_fusion_rank']<=k
            new=lambda r:r['optimized_fusion_rank'] is not None and r['optimized_fusion_rank']<=k
            row[f'paired_hit{k}']={'improved':sum(not old(r) and new(r) for r in subset),
                                   'regressed':sum(old(r) and not new(r) for r in subset)}
        report[label]=row
    summary={'plan_sha256':sha(args.plan),'baseline_freeze_sha256':sha(root/'BASELINE-FROZEN.json'),
             'results_sha256':sha(output),'rank_fusion_seconds_for_all':time.monotonic()-start,'metrics':report,
             'test_use':'first formal evaluation after frozen optimization selection; no subsequent tuning'}
    target=Path(str(output)+'.summary.json')
    with target.open('x',encoding='utf8') as f:json.dump(summary,f,ensure_ascii=False,indent=2)
    print(json.dumps({'output':str(output),'cases':len(out),'rank_fusion_seconds':summary['rank_fusion_seconds_for_all'],
                      'test':report['standard_test'],'visual_test':report['visual_test']}))


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='cmd',required=True)
    a=s.add_parser('choose');a.add_argument('--baseline',required=True);a.add_argument('--output',required=True)
    a=s.add_parser('evaluate');a.add_argument('--baseline',required=True);a.add_argument('--plan',required=True);a.add_argument('--output',required=True)
    args=p.parse_args();(choose if args.cmd=='choose' else evaluate)(args)


if __name__=='__main__':main()
